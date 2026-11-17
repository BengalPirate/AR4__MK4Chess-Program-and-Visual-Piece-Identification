"""Chess game state, move selection and vision-to-move inference.

Board representation, legal-move generation and all rules (check, castling
rights, en passant, promotion, 50-move / repetition draws) come from the
python-chess library (https://github.com/niklasf/python-chess, GPL-3.0).
Moves are chosen by Stockfish over UCI when it is installed
(https://stockfishchess.org, GPL-3.0), otherwise by a small built-in
alpha-beta search (engine fallback below, our own code).

The game state is kept here, not re-derived from the image: turn, castling
rights, en-passant square, move counters and the captured-piece tray survive
any single bad frame. Vision only has to tell us *which legal move* was made.
"""
import math
import os
import shutil

import chess
import chess.engine

import config

LABELS = [".", "P", "N", "B", "R", "Q", "K", "p", "n", "b", "r", "q", "k"]
LABEL_INDEX = {c: i for i, c in enumerate(LABELS)}


def square_labels(board):
    """64 labels ('.' or FEN piece letter), index = python-chess square (a1=0 .. h8=63)."""
    out = ["."] * 64
    for sq, pc in board.piece_map().items():
        out[sq] = pc.symbol()
    return out


def board_from_labels(labels, turn=chess.WHITE):
    """Build a board from 64 labels (castling rights inferred from piece placement)."""
    b = chess.Board(None)
    for sq, c in enumerate(labels):
        if c != ".":
            b.set_piece_at(sq, chess.Piece.from_symbol(c))
    b.turn = turn
    rights = ""
    if b.piece_at(chess.E1) == chess.Piece(chess.KING, chess.WHITE):
        if b.piece_at(chess.H1) == chess.Piece(chess.ROOK, chess.WHITE):
            rights += "K"
        if b.piece_at(chess.A1) == chess.Piece(chess.ROOK, chess.WHITE):
            rights += "Q"
    if b.piece_at(chess.E8) == chess.Piece(chess.KING, chess.BLACK):
        if b.piece_at(chess.H8) == chess.Piece(chess.ROOK, chess.BLACK):
            rights += "k"
        if b.piece_at(chess.A8) == chess.Piece(chess.ROOK, chess.BLACK):
            rights += "q"
    b.set_castling_fen(rights or "-")
    return b


# ---------------------------------------------------------------- engine
_PIECE_VALUE = {chess.PAWN: 100, chess.KNIGHT: 320, chess.BISHOP: 330,
                chess.ROOK: 500, chess.QUEEN: 900, chess.KING: 0}
_CENTER_BONUS = [0, 1, 2, 3, 3, 2, 1, 0]


def _evaluate(b):
    """Material + small centralisation bonus, from the side to move's view."""
    if b.is_checkmate():
        return -100000
    if b.is_stalemate() or b.is_insufficient_material():
        return 0
    s = 0
    for sq, pc in b.piece_map().items():
        v = _PIECE_VALUE[pc.piece_type]
        if pc.piece_type in (chess.KNIGHT, chess.BISHOP, chess.PAWN):
            v += 4 * (_CENTER_BONUS[chess.square_file(sq)] + _CENTER_BONUS[chess.square_rank(sq)])
        s += v if pc.color == b.turn else -v
    return s


def _negamax(b, depth, alpha, beta):
    if depth == 0 or b.is_game_over():
        return _evaluate(b), None
    best_move = None
    # captures first: better pruning
    moves = sorted(b.legal_moves, key=lambda m: not b.is_capture(m))
    for m in moves:
        b.push(m)
        score = -_negamax(b, depth - 1, -beta, -alpha)[0]
        b.pop()
        if score > alpha or best_move is None:
            alpha, best_move = max(alpha, score), m
        if alpha >= beta:
            break
    return alpha, best_move


class Engine:
    """Stockfish if available, else the built-in search."""

    def __init__(self):
        self.sf = None
        path = shutil.which("stockfish") or next(
            (p for p in config.STOCKFISH_PATHS if os.path.exists(p)), None)
        if path:
            try:
                self.sf = chess.engine.SimpleEngine.popen_uci(path)
            except Exception as e:      # noqa: BLE001 - fall back on any engine failure
                print(f"Stockfish failed to start ({e}); using built-in engine")
        self.name = f"Stockfish ({path})" if self.sf else \
            f"built-in alpha-beta, depth {config.FALLBACK_DEPTH}"

    def choose(self, board):
        if board.is_game_over():
            return None
        if self.sf:
            return self.sf.play(board, chess.engine.Limit(time=config.ENGINE_TIME_S)).move
        return _negamax(board.copy(), config.FALLBACK_DEPTH, -math.inf, math.inf)[1]

    def close(self):
        if self.sf:
            self.sf.quit()


# ---------------------------------------------------------------- game
class Game:
    """The authoritative game state plus the off-board tray of captured pieces."""

    def __init__(self, fen=chess.STARTING_FEN):
        self.board = chess.Board(fen)
        self.tray = []          # captured piece symbols in slot order (index = slot)

    # ---- moves
    def parse_move(self, text):
        """UCI ('e2e4', 'e7e8q') or SAN ('Nf3', 'O-O'). Returns a legal Move or raises
        ValueError with a reason (illegal moves are rejected, never applied)."""
        text = text.strip()
        b = self.board
        try:
            m = chess.Move.from_uci(text.lower())
        except ValueError:
            try:
                return b.parse_san(text)
            except ValueError:
                raise ValueError(f"'{text}' is not a move (use e2e4 or Nf3)")
        if m in b.legal_moves:
            return m
        # promotion without a piece letter -> default queen
        if b.piece_type_at(m.from_square) == chess.PAWN and \
                chess.square_rank(m.to_square) in (0, 7) and m.promotion is None:
            m2 = chess.Move(m.from_square, m.to_square, chess.QUEEN)
            if m2 in b.legal_moves:
                return m2
        raise ValueError(self.why_illegal(m))

    def why_illegal(self, m):
        b = self.board
        pc = b.piece_at(m.from_square)
        name = chess.square_name
        if pc is None:
            return f"illegal: no piece on {name(m.from_square)}"
        if pc.color != b.turn:
            return f"illegal: it is {'White' if b.turn else 'Black'}'s turn"
        if m in b.pseudo_legal_moves:
            return f"illegal: {m.uci()} leaves the king in check"
        return f"illegal: {pc.symbol()} on {name(m.from_square)} cannot move to {name(m.to_square)}"

    def push(self, m):
        b = self.board
        if m not in b.legal_moves:
            raise ValueError(self.why_illegal(m))
        cap = self.captured_square(m)
        if cap is not None:
            self.tray.append(b.piece_at(cap).symbol())
        b.push(m)

    def undo(self):
        if not self.board.move_stack:
            return None
        m = self.board.pop()
        if self.captured_square(m) is not None:
            self.tray.pop()
        return m

    def captured_square(self, m, board=None):
        """Square of the piece a move captures (differs from m.to_square for en passant)."""
        b = board or self.board
        if b.is_en_passant(m):
            return chess.square(chess.square_file(m.to_square), chess.square_rank(m.from_square))
        if b.is_capture(m):
            return m.to_square
        return None

    # ---- state for display
    def status(self):
        b = self.board
        side = "White" if b.turn else "Black"
        if b.is_checkmate():
            return f"Checkmate - {'Black' if b.turn else 'White'} wins"
        if b.is_stalemate():
            return "Stalemate"
        if b.is_insufficient_material():
            return "Draw - insufficient material"
        if b.can_claim_draw():
            return f"{side} to move (draw claimable)"
        return f"{side} to move" + (" - CHECK" if b.is_check() else "")

    def rights_text(self):
        b = self.board
        ep = chess.square_name(b.ep_square) if b.ep_square is not None and \
            b.has_legal_en_passant() else "-"
        return f"castling {b.castling_xfen() or '-'}  en passant {ep}  " \
               f"halfmove {b.halfmove_clock}  move {b.fullmove_number}"

    # ---- vision -> move
    def infer_move(self, logp):
        """Which legal move best explains the scanned board?

        logp: 64 x 13 log-probabilities from the piece classifier (rows = squares,
        columns = LABELS). Every legal move, and 'no move', is a hypothesis about
        the full board; its score is the sum over squares of log p(expected label).
        Returns dict(move, score, margin, ranked, surprise, worst_square) where move
        is None for 'no move'. The caller pauses for correction when the margin to
        the runner-up is small or some square strongly disagrees (surprise)."""
        b = self.board
        hyps = [(None, square_labels(b))]
        for m in b.legal_moves:
            b.push(m)
            hyps.append((m, square_labels(b)))
            b.pop()
        scored = []
        for m, labels in hyps:
            per_sq = [-logp[sq][LABEL_INDEX[c]] for sq, c in enumerate(labels)]
            scored.append((-sum(per_sq), m, per_sq))
        scored.sort(key=lambda t: t[0], reverse=True)
        best, runner = scored[0], scored[1] if len(scored) > 1 else None
        worst = max(range(64), key=lambda s: best[2][s])
        return {
            "move": best[1],
            "score": best[0],
            "margin": best[0] - runner[0] if runner else math.inf,
            "ranked": [(m, s) for s, m, _ in scored[:5]],
            "surprise": best[2][worst],
            "worst_square": worst,
        }

    def confident(self, inf):
        return inf["margin"] >= config.MOVE_MIN_MARGIN and \
            inf["surprise"] <= config.MOVE_MAX_SQUARE_SURPRISE

# Test log – Checkpoint 2

All arm tests at 10 % speed with an operator at the E-stop. Dates 2026-11-16.

## 1. Vision

| Test | Data | Result |
|---|---|---|
| Phone-photo test split | 483 squares from photo groups never trained on | 97.1 % exact |
| Webcam, start positions | 11 boards (4 from the first, lower camera angle) | 96.2 % exact; 98.7 % at the final camera angle |
| Webcam, changed positions | 3 mid-game boards: after 1.d4 e5 2.Bf4 Nc6 3.Nf3 Nf6; developed pieces with captures; both sides castled | 95.3 % each |
| Webcam, other arrangements | 5 positions set up from chess_demo's training setups (colours swapped, two scrambled, sparse endgame), 3 frames each, saved after the model was trained | 96.9–100 % per frame; every error flagged uncertain |
| All held-out webcam frames | 38 frames (`python tests/webcam_report.py`) | 98.1 % exact, occupancy 99.5 % |
| Colour | all 1,120 occupied squares on the 38 frames | 100 % on squares detected as occupied |
| Move inference | 88 simulated moves (knight moves out and back on the start boards) | 56 correct and accepted, 32 paused, 0 wrong accepted |

Error analysis (mid-game boards, 9 errors): all are pieces partly or fully
hidden behind a taller piece in the camera's line of sight (h2 pawn behind
the g1 king, f8 rook behind the g8 king, e8 king / e7 queen one behind the
other, c7 / f7 pawns behind pieces on rank 8 or 6). 7 of the 9 were flagged
below the confidence threshold. None of them can be explained by a legal
move, so during a game they make the scan pause rather than produce a move.

Ambiguous case found: the first 4 test boards came from a lower camera angle
where the e2 pawn hides behind the e1 king; every move test on them paused
(safe, but unusable). Raising the camera fixed it (83 % → 98.7 %).

## 2. Arm and gripper

| Test | Result | Change made |
|---|---|---|
| Hover over a1, h1, d4 at grip height + 15 mm | centred (operator check) | – |
| Reach map of all 64 squares | b8–g8 unreachable with the gripper straight down (J3 would need 55–63°, limit +52°; no joint combination avoids it) | rank 8 worked with the gripper tilted back on J5 |
| Tilted hover / grip on e8 | first misses (tool length measured from the wrong reference); 30° tilt with the corrected 64 mm tool length grips the king | `TOOL_LENGTH_MM = 64`, `RANK8_TILT_DEG = 30` |
| Grip each piece type: lift 20 mm, put back | all six gripped after height tuning | heights P 23, N 35, B 45, R 35, Q 57, K 68 mm |
| King with its cross parallel to the jaws | slipped (thin grip) | close until 18° or a motor stall: holds both cross orientations |
| Gripper "felt" signal (motor current) | false "felt" with empty jaws when the servo was loaded; one run picked nothing but reported success | position-based grip with stall stop; jam check before each grip |
| Opening after the Nano restarts (servo snaps shut on the foam pads) | stall on opening | jump straight out of the squeezed range; retry |
| Returning from 49° to 45° | jaws stayed loaded | approach 45° from below; wide opening 47° |
| Move all 16 white pieces two ranks forward | all moved after the fixes above | – |
| Move all 16 black pieces two ranks forward (rank 8 tilted) | c7 pawn pushed into the board at 20 mm (irregular pawn) | pawn grip 23 mm |
| | rank 8 picks 2–5 mm off-centre; piece placed straight after a tilted pick sat crooked | per-rank / per-square corrections; a piece is placed at the tilt it was picked with |
| Jaw rotation 45° vs straight | straight jaws clear the neighbours | all jaws straight |

## 3. Software tests

`pytest tests` – 25 tests, all pass.

- Chess logic: turn tracking and illegal moves; pinned piece rejected; castling
  rights preserved beyond the image; en passant window; promotion defaults to
  queen and uses the tray; undo restores the tray; engine returns a legal move.
- Move inference: simple move; no move; survives one misread square; castling
  and capture; an inconsistent board is not confident (pauses).
- Planning: square mapping; tray slots; quiet move, capture, en passant,
  castling, promotion from reserve or tray; illegal move rejected.
- Motion: gripper down with fixed jaw direction and J5 inside its window;
  vertical descent; unreachable plan refused before moving; J5 outside its
  window refused; interlock blocks a second arm.

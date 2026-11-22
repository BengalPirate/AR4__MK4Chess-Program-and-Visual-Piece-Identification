"""Summarise the live-game logs (results/games/*.log) and the grip log for the report.

Run from the repository root:  python docs/game_stats.py
Writes results/games/summary.md and prints the same numbers. Counts come only from
lines the running program printed (chess_demo.py / motion.Executor), so every number
can be traced back to a log line.
"""
import csv
import glob
import json
import os
import re

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GAMES = os.path.join(ROOT, "results", "games")


def parse_log(path):
    g = {"file": os.path.basename(path), "moves": [], "robot_moves": [], "player_moves": [],
         "captures_robot": [], "tray_places": 0, "place_ok": 0, "place_fix": 0,
         "retries": 0, "rank8_back": 0, "stops": [], "illegal": 0, "paused_unsure": 0,
         "verified": 0, "start": None}
    for ln in open(path, errors="replace"):
        ln = ln.rstrip()
        m = re.search(r"Game set from the board: (\S+) - (\w+) to move", ln)
        if m and g["start"] is None:
            g["start"] = (m.group(1), m.group(2))
        m = re.search(r"^(\S+) played \((robot|vision)\)", ln)
        if m:
            san, who = m.group(1), m.group(2)
            g["moves"].append((san, who))
            (g["robot_moves"] if who == "robot" else g["player_moves"]).append(san)
            if who == "robot" and "x" in san:
                g["captures_robot"].append(san)
        if re.search(r"place \w at tray:", ln):
            g["tray_places"] += 1
        if "placement check" in ln and ln.endswith("OK"):
            g["place_ok"] += 1
        if re.search(r"FIX \d", ln):
            g["place_fix"] += 1
        if re.search(r"RETRY \d", ln):
            g["retries"] += 1
        if "rank 8, two misses" in ln:
            g["rank8_back"] += 1
        if ln.startswith("ROBOT STOPPED"):
            g["stops"].append(ln[len("ROBOT STOPPED: "):].split(" - AUTO PAUSED")[0])
        if ln.startswith("ILLEGAL"):
            g["illegal"] += 1
        if ln.startswith("PAUSED - unsure"):
            g["paused_unsure"] += 1
        if ln.startswith("Board matches the game"):
            g["verified"] += 1
    return g


GAME_DAYS = ("2026-11-20", "2026-11-22")      # days with live games (grips.csv rows of those days)


def grip_stats(days=GAME_DAYS):
    if isinstance(days, str):
        days = (days,)
    rows = [r for r in csv.DictReader(open(os.path.join(ROOT, "results", "grips.csv")))
            if r["time"][:10] in days]
    first = [r for r in rows if r["attempt"] == "0"]
    ok = lambda rs: sum(r["square_seen"] == "empty" for r in rs)   # noqa: E731
    r8 = [r for r in first if r["square"].endswith("8")]
    return {"grips": len(rows), "first_tries": len(first), "first_ok": ok(first),
            "rank8_first": len(r8), "rank8_first_ok": ok(r8),
            "picks_ok": ok(rows)}


def scan_stats():
    kinds = {}
    for r in csv.reader(open(os.path.join(ROOT, "results", "scans.csv"))):
        if len(r) > 1 and r[0][:4].isdigit():
            kinds[r[1]] = kinds.get(r[1], 0) + 1
    return kinds


def main():
    games = [parse_log(p) for d in GAME_DAYS for p in sorted(glob.glob(os.path.join(GAMES, f"{d}_game*.log")))]
    tot = {k: sum(len(g[k]) if isinstance(g[k], list) else g[k] for g in games)
           for k in ("robot_moves", "player_moves", "captures_robot", "tray_places", "place_ok",
                     "place_fix", "retries", "rank8_back", "stops", "illegal", "paused_unsure", "verified")}
    gs, sc = grip_stats(), scan_stats()
    lines = ["# Live games 2026-11-20 and 2026-11-22 - summary (docs/game_stats.py)", "",
             "| Log | Start | Moves (robot = Black) | Robot captures | Stops |", "|---|---|---|---|---|"]
    for g in games:
        mv = " ".join(f"{s}{'*' if w == 'robot' else ''}" for s, w in g["moves"]) or "-"
        st = "standard" if g["start"] and g["start"][0].startswith("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP") \
            else (g["start"][0] if g["start"] else "not started")
        lines.append(f"| {g['file']} | {st} | {mv} | {', '.join(g['captures_robot']) or '-'} | "
                     f"{'; '.join(g['stops']) or '-'} |")
    lines += ["", "`*` = move made by the robot.", "", "| Measure | Value |", "|---|---|"]
    lines += [f"| {k} | {v} |" for k, v in tot.items()]
    lines += [f"| grips logged (grips.csv, {', '.join(GAME_DAYS)}) | {gs['grips']} |",
              f"| first-try picks confirmed (square seen empty) | {gs['first_ok']} / {gs['first_tries']} |",
              f"| rank-8 first-try picks confirmed | {gs['rank8_first_ok']} / {gs['rank8_first']} |",
              f"| scans.csv entries by kind (all sessions) | {json.dumps(sc)} |"]
    os.makedirs(GAMES, exist_ok=True)
    open(os.path.join(GAMES, "summary.md"), "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))
    return games, tot, gs, sc


if __name__ == "__main__":
    main()

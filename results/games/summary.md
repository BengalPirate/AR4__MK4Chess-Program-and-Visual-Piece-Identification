# Live games 2026-11-20 - summary (docs/game_stats.py)

| Log | Start | Moves (robot = Black) | Robot captures | Stops |
|---|---|---|---|---|
| 2026-11-20_game2.log | not started | - | - | - |
| 2026-11-20_game3.log | standard | e4 c5* Nf3 d6* Nc3 e5* Nd5 Nf6* | - | - |
| 2026-11-20_game4.log | standard | e4 c5* Nf3 d6* Nc3 e5* Nd5 Nf6* Nxf6+ Qxf6* Bd3 Be7* h3 | Qxf6 | No reply from the Nano to 'TG' - wrong port, sketch not loaded, or board unpowered |
| 2026-11-20_game5.log | standard | e4 c5* Nf3 d6* Ng5 Nf6* Bb5+ | - | Grip missed c8 3 times (camera still sees the piece there) - re-centre it by hand, then continue |
| 2026-11-20_game6.log | rnbqkb1r/pp2pppp/3p1n2/1Bp3N1/4P3/8/PPPP1PPP/RNBQK2R | Bd7* Bxd7+ Qxd7* Qf3 h6* Nh3 Nc6* d4 Nxd4* Qc3 Qg4* f3 Qxg2* | Qxd7, Nxd4, Qxg2 | - |

`*` = move made by the robot.

| Measure | Value |
|---|---|
| robot_moves | 20 |
| player_moves | 21 |
| captures_robot | 4 |
| tray_places | 4 |
| place_ok | 20 |
| place_fix | 8 |
| retries | 6 |
| rank8_back | 1 |
| stops | 2 |
| illegal | 0 |
| paused_unsure | 14 |
| verified | 130 |
| grips logged (grips.csv, 2026-11-20) | 61 |
| first-try picks confirmed (square seen empty) | 46 / 52 |
| rank-8 first-try picks confirmed | 9 / 11 |
| scans.csv entries by kind (all sessions) | {"verify": 395, "move": 29, "uncertain": 90} |

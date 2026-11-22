# Live games 2026-11-20 and 2026-11-22 - summary (docs/game_stats.py)

| Log | Start | Moves (robot = Black) | Robot captures | Stops |
|---|---|---|---|---|
| 2026-11-20_game2.log | not started | - | - | - |
| 2026-11-20_game3.log | standard | e4 c5* Nf3 d6* Nc3 e5* Nd5 Nf6* | - | - |
| 2026-11-20_game4.log | standard | e4 c5* Nf3 d6* Nc3 e5* Nd5 Nf6* Nxf6+ Qxf6* Bd3 Be7* h3 | Qxf6 | No reply from the Nano to 'TG' - wrong port, sketch not loaded, or board unpowered |
| 2026-11-20_game5.log | standard | e4 c5* Nf3 d6* Ng5 Nf6* Bb5+ | - | Grip missed c8 3 times (camera still sees the piece there) - re-centre it by hand, then continue |
| 2026-11-20_game6.log | rnbqkb1r/pp2pppp/3p1n2/1Bp3N1/4P3/8/PPPP1PPP/RNBQK2R | Bd7* Bxd7+ Qxd7* Qf3 h6* Nh3 Nc6* d4 Nxd4* Qc3 Qg4* f3 Qxg2* | Qxd7, Nxd4, Qxg2 | - |
| 2026-11-22_game10_part1_raw.log | standard | h4 e5* Rh3 d5* Re3 Nc6* a4 Nf6* | - | - |
| 2026-11-22_game10_part2_raw.log | r1bqkb1r/ppp2ppp/2n2n2/3pp3/P6P/R3R3/1PPPPPP1/1NBQKBN1 | Ng4* Rac3 d4* Rxc6 dxe3* fxe3 | dxe3 | Cannot confirm the pick from h4 (still undecided from aside) - arm stopped, gripper closed: check the jaws and the square |
| 2026-11-22_game10_part3_raw.log | r1bqkb1r/ppp2ppp/2R5/4p3/P5nP/4P3/1PPPP1P1/1NBQKBN1 | - | - | The piece is NOT on h4 after the set-down (the jaws were probably empty) - check the gripper and the board |
| 2026-11-22_game10e.log | r1b1kb1r/ppp2ppp/2R5/4p3/P5nq/4P1P1/1PPPP3/1NBQKBN1 | - | - | - |
| 2026-11-22_game10f.log | r1b1kb1r/ppp2ppp/2R5/4p3/P5nq/4P1P1/1PPPP3/1NBQKBN1 | Qxg3#* | Qxg3# | - |
| 2026-11-22_game9_raw.log | not started | - | - | - |

`*` = move made by the robot.

| Measure | Value |
|---|---|
| robot_moves | 28 |
| player_moves | 28 |
| captures_robot | 6 |
| tray_places | 7 |
| place_ok | 28 |
| place_fix | 10 |
| retries | 8 |
| rank8_back | 2 |
| stops | 4 |
| illegal | 0 |
| paused_unsure | 16 |
| verified | 173 |
| grips logged (grips.csv, 2026-11-20, 2026-11-22) | 79 |
| first-try picks confirmed (square seen empty) | 60 / 68 |
| rank-8 first-try picks confirmed | 11 / 14 |
| scans.csv entries by kind (all sessions) | {"verify": 442, "move": 38, "uncertain": 92} |

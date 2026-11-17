# Webcam recognition accuracy (held-out boards)

Generated 2026-11-16 23:21 with models/piece_cnn.pt (trained 2026-11-16 15:11). Boards in data_webcam_test/ were never used for training. One frame per board.

### All held-out webcam boards (14 boards)

- squares: 896 (442 occupied)
- exact label (empty / type+colour): **96.0%**
- occupancy: 98.7%
- colour, on occupied squares detected as occupied: 100.0%
- piece type, on occupied squares: 92.1%

| true \ pred | . | P | N | B | R | Q | K | p | n | b | r | q | k |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **.** | 453 |  |  |  |  |  |  | 1 |  |  |  |  |  |
| **P** | 6 | 105 |  |  |  | 1 |  |  |  |  |  |  |  |
| **N** |  |  | 24 |  | 4 |  |  |  |  |  |  |  |  |
| **B** |  | 2 |  | 24 |  |  |  |  |  |  |  |  |  |
| **R** |  |  |  |  | 28 |  |  |  |  |  |  |  |  |
| **Q** |  |  |  |  |  | 14 |  |  |  |  |  |  |  |
| **K** |  |  |  |  |  |  | 14 |  |  |  |  |  |  |
| **p** | 4 |  |  |  |  |  |  | 93 |  |  | 1 | 8 | 4 |
| **n** |  |  |  |  |  |  |  |  | 26 |  |  |  |  |
| **b** |  |  |  |  |  |  |  |  |  | 27 |  | 1 |  |
| **r** |  |  |  |  |  |  |  | 2 |  |  | 26 |  |  |
| **q** |  |  |  |  |  |  |  |  |  |  |  | 13 | 1 |
| **k** | 1 |  |  |  |  |  |  |  |  |  |  |  | 13 |

### Start positions (11 boards)

- squares: 704 (352 occupied)
- exact label (empty / type+colour): **96.2%**
- occupancy: 99.1%
- colour, on occupied squares detected as occupied: 100.0%
- piece type, on occupied squares: 92.3%

| true \ pred | . | P | N | B | R | Q | K | p | n | b | r | q | k |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **.** | 352 |  |  |  |  |  |  |  |  |  |  |  |  |
| **P** | 4 | 83 |  |  |  | 1 |  |  |  |  |  |  |  |
| **N** |  |  | 18 |  | 4 |  |  |  |  |  |  |  |  |
| **B** |  | 2 |  | 20 |  |  |  |  |  |  |  |  |  |
| **R** |  |  |  |  | 22 |  |  |  |  |  |  |  |  |
| **Q** |  |  |  |  |  | 11 |  |  |  |  |  |  |  |
| **K** |  |  |  |  |  |  | 11 |  |  |  |  |  |  |
| **p** | 2 |  |  |  |  |  |  | 73 |  |  | 1 | 8 | 4 |
| **n** |  |  |  |  |  |  |  |  | 22 |  |  |  |  |
| **b** |  |  |  |  |  |  |  |  |  | 22 |  |  |  |
| **r** |  |  |  |  |  |  |  | 1 |  |  | 21 |  |  |
| **q** |  |  |  |  |  |  |  |  |  |  |  | 11 |  |
| **k** |  |  |  |  |  |  |  |  |  |  |  |  | 11 |

### Mid-game positions (3 boards)

- squares: 192 (90 occupied)
- exact label (empty / type+colour): **95.3%**
- occupancy: 96.9%
- colour, on occupied squares detected as occupied: 100.0%
- piece type, on occupied squares: 91.1%

| true \ pred | . | P | N | B | R | Q | K | p | n | b | r | q | k |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **.** | 101 |  |  |  |  |  |  | 1 |  |  |  |  |  |
| **P** | 2 | 22 |  |  |  |  |  |  |  |  |  |  |  |
| **N** |  |  | 6 |  |  |  |  |  |  |  |  |  |  |
| **B** |  |  |  | 4 |  |  |  |  |  |  |  |  |  |
| **R** |  |  |  |  | 6 |  |  |  |  |  |  |  |  |
| **Q** |  |  |  |  |  | 3 |  |  |  |  |  |  |  |
| **K** |  |  |  |  |  |  | 3 |  |  |  |  |  |  |
| **p** | 2 |  |  |  |  |  |  | 20 |  |  |  |  |  |
| **n** |  |  |  |  |  |  |  |  | 4 |  |  |  |  |
| **b** |  |  |  |  |  |  |  |  |  | 5 |  | 1 |  |
| **r** |  |  |  |  |  |  |  | 1 |  |  | 5 |  |  |
| **q** |  |  |  |  |  |  |  |  |  |  |  | 2 | 1 |
| **k** | 1 |  |  |  |  |  |  |  |  |  |  |  | 2 |

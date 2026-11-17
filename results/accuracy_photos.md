# Piece recognition accuracy (phone-photo training)

Generated 2026-11-16 13:29. Photos: train 1079, validation 192, test 250 (split by groups of 10 consecutive shots). C=0.05 chosen on validation (87.9%).

### Held-out phone photos (test split)

- squares: 483 (250 occupied)
- exact label (empty / type+colour): **90.1%**
- occupancy: 99.0%
- colour, on occupied squares detected as occupied: 100.0%
- piece type, on occupied squares: 81.6%

| true \ pred | . | P | N | B | R | Q | K | p | n | b | r | q | k |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **.** | 231 |  |  |  |  | 1 | 1 |  |  |  |  |  |  |
| **P** | 3 | 12 |  |  |  | 5 |  |  |  |  |  |  |  |
| **N** |  |  | 21 |  | 2 |  | 1 |  |  |  |  |  |  |
| **B** |  | 2 |  | 16 |  | 1 |  |  |  |  |  |  |  |
| **R** |  |  | 4 |  | 16 |  |  |  |  |  |  |  |  |
| **Q** |  |  | 1 | 3 | 1 | 23 | 2 |  |  |  |  |  |  |
| **K** |  | 1 |  | 2 |  | 2 | 15 |  |  |  |  |  |  |
| **p** |  |  |  |  |  |  |  | 18 |  | 1 |  |  |  |
| **n** |  |  |  |  |  |  |  |  | 19 |  | 1 |  |  |
| **b** |  |  |  |  |  |  |  |  | 1 | 15 | 1 | 1 | 2 |
| **r** |  |  |  |  |  |  |  |  |  | 1 | 14 |  | 3 |
| **q** |  |  |  |  |  |  |  |  |  |  |  | 17 | 3 |
| **k** |  |  |  |  |  |  |  |  |  |  |  | 2 | 18 |

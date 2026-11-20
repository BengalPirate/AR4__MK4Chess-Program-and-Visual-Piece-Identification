# Piece recognition accuracy (fine-tuned ResNet-18)

Generated 2026-11-17 21:19; best validation accuracy 96.5%; 6 epochs.

### Held-out phone photos (test split)

- squares: 483 (250 occupied)
- exact label (empty / type+colour): **96.9%**
- occupancy: 99.2%
- colour, on occupied squares detected as occupied: 100.0%
- piece type, on occupied squares: 94.0%

| true \ pred | . | P | N | B | R | Q | K | p | n | b | r | q | k |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **.** | 233 |  |  |  |  |  |  |  |  |  |  |  |  |
| **P** | 3 | 17 |  |  |  |  |  |  |  |  |  |  |  |
| **N** |  | 1 | 23 |  |  |  |  |  |  |  |  |  |  |
| **B** |  | 1 |  | 18 |  |  |  |  |  |  |  |  |  |
| **R** | 1 |  |  |  | 19 |  |  |  |  |  |  |  |  |
| **Q** |  |  |  |  |  | 30 |  |  |  |  |  |  |  |
| **K** |  |  |  | 1 | 1 | 1 | 17 |  |  |  |  |  |  |
| **p** |  |  |  |  |  |  |  | 19 |  |  |  |  |  |
| **n** |  |  |  |  |  |  |  |  | 20 |  |  |  |  |
| **b** |  |  |  |  |  |  |  |  | 2 | 18 |  |  |  |
| **r** |  |  |  |  |  |  |  |  |  |  | 18 |  |  |
| **q** |  |  |  |  |  |  |  |  |  | 2 |  | 16 | 2 |
| **k** |  |  |  |  |  |  |  |  |  |  |  |  | 20 |

### Webcam boards (40, never trained on)

- squares: 2560 (1184 occupied)
- exact label (empty / type+colour): **99.3%**
- occupancy: 99.6%
- colour, on occupied squares detected as occupied: 100.0%
- piece type, on occupied squares: 98.6%

| true \ pred | . | P | N | B | R | Q | K | p | n | b | r | q | k |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **.** | 1375 |  |  |  |  |  |  | 1 |  |  |  |  |  |
| **P** | 7 | 283 |  |  |  |  |  |  |  |  |  |  |  |
| **N** |  |  | 77 |  |  |  |  |  |  |  |  |  |  |
| **B** |  |  |  | 75 |  |  |  |  |  |  |  |  |  |
| **R** |  |  |  |  | 74 |  |  |  |  |  |  |  |  |
| **Q** |  |  |  |  |  | 37 |  |  |  |  |  |  |  |
| **K** |  |  |  |  |  |  | 40 |  |  |  |  |  |  |
| **p** | 2 |  |  |  |  |  |  | 280 |  |  | 5 |  | 1 |
| **n** |  |  |  |  |  |  |  |  | 75 |  |  |  |  |
| **b** |  |  |  |  |  |  |  |  |  | 74 |  |  |  |
| **r** |  |  |  |  |  |  |  |  |  |  | 77 |  |  |
| **q** |  |  |  |  |  |  |  |  |  |  |  | 36 | 1 |
| **k** | 1 |  |  |  |  |  |  |  |  |  |  |  | 39 |

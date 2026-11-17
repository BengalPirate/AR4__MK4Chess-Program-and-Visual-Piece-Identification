# Piece recognition accuracy (fine-tuned ResNet-18)

Generated 2026-11-16 13:53; best validation accuracy 97.0%; 10 epochs.

### Held-out phone photos (test split)

- squares: 483 (250 occupied)
- exact label (empty / type+colour): **97.1%**
- occupancy: 99.0%
- colour, on occupied squares detected as occupied: 100.0%
- piece type, on occupied squares: 94.4%

| true \ pred | . | P | N | B | R | Q | K | p | n | b | r | q | k |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **.** | 233 |  |  |  |  |  |  |  |  |  |  |  |  |
| **P** | 3 | 17 |  |  |  |  |  |  |  |  |  |  |  |
| **N** | 1 |  | 23 |  |  |  |  |  |  |  |  |  |  |
| **B** |  |  |  | 19 |  |  |  |  |  |  |  |  |  |
| **R** | 1 |  |  |  | 19 |  |  |  |  |  |  |  |  |
| **Q** |  |  |  |  |  | 30 |  |  |  |  |  |  |  |
| **K** |  |  |  |  |  | 3 | 17 |  |  |  |  |  |  |
| **p** |  |  |  |  |  |  |  | 19 |  |  |  |  |  |
| **n** |  |  |  |  |  |  |  |  | 20 |  |  |  |  |
| **b** |  |  |  |  |  |  |  | 1 | 1 | 18 |  |  |  |
| **r** |  |  |  |  |  |  |  |  |  |  | 17 |  | 1 |
| **q** |  |  |  |  |  |  |  |  |  |  |  | 17 | 3 |
| **k** |  |  |  |  |  |  |  |  |  |  |  |  | 20 |

### Webcam boards (4, never trained on)

- squares: 256 (128 occupied)
- exact label (empty / type+colour): **57.4%**
- occupancy: 80.1%
- colour, on occupied squares detected as occupied: 100.0%
- piece type, on occupied squares: 23.4%

| true \ pred | . | P | N | B | R | Q | K | p | n | b | r | q | k |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **.** | 117 |  |  |  |  |  |  |  | 3 |  |  | 8 |  |
| **P** | 8 |  |  | 16 |  | 4 | 4 |  |  |  |  |  |  |
| **N** |  |  |  | 8 |  |  |  |  |  |  |  |  |  |
| **B** |  |  |  | 8 |  |  |  |  |  |  |  |  |  |
| **R** | 1 |  |  | 6 |  | 1 |  |  |  |  |  |  |  |
| **Q** |  |  |  |  |  | 4 |  |  |  |  |  |  |  |
| **K** |  |  |  |  |  |  | 4 |  |  |  |  |  |  |
| **p** | 13 |  |  |  |  |  |  |  |  | 8 |  | 7 | 4 |
| **n** | 8 |  |  |  |  |  |  |  |  |  |  |  |  |
| **b** | 2 |  |  |  |  |  |  |  |  | 6 |  |  |  |
| **r** | 8 |  |  |  |  |  |  |  |  |  |  |  |  |
| **q** |  |  |  |  |  |  |  |  |  |  |  | 4 |  |
| **k** |  |  |  |  |  |  |  |  |  |  |  |  | 4 |

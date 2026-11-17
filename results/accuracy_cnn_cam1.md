# Piece recognition accuracy (fine-tuned ResNet-18)

Generated 2026-11-16 14:11; best validation accuracy 96.2%; 5 epochs.

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
| **K** |  | 2 |  |  |  | 1 | 17 |  |  |  |  |  |  |
| **p** |  |  |  |  |  |  |  | 19 |  |  |  |  |  |
| **n** |  |  |  |  |  |  |  |  | 20 |  |  |  |  |
| **b** |  |  |  |  |  |  |  |  | 1 | 19 |  |  |  |
| **r** |  |  |  |  |  |  |  |  |  |  | 17 |  | 1 |
| **q** |  |  |  |  |  |  |  |  |  |  |  | 17 | 3 |
| **k** |  |  |  |  |  |  |  |  |  |  |  | 1 | 19 |

### Webcam boards (4, never trained on)

- squares: 256 (128 occupied)
- exact label (empty / type+colour): **83.2%**
- occupancy: 88.7%
- colour, on occupied squares detected as occupied: 100.0%
- piece type, on occupied squares: 66.4%

| true \ pred | . | P | N | B | R | Q | K | p | n | b | r | q | k |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **.** | 128 |  |  |  |  |  |  |  |  |  |  |  |  |
| **P** | 10 | 22 |  |  |  |  |  |  |  |  |  |  |  |
| **N** |  | 4 | 4 |  |  |  |  |  |  |  |  |  |  |
| **B** |  |  |  | 8 |  |  |  |  |  |  |  |  |  |
| **R** |  |  |  |  | 8 |  |  |  |  |  |  |  |  |
| **Q** |  |  |  |  |  | 4 |  |  |  |  |  |  |  |
| **K** |  |  |  |  |  |  | 4 |  |  |  |  |  |  |
| **p** | 11 |  |  |  |  |  |  | 14 |  |  |  | 7 |  |
| **n** | 3 |  |  |  |  |  |  |  | 5 |  |  |  |  |
| **b** | 2 |  |  |  |  |  |  |  |  | 6 |  |  |  |
| **r** | 3 |  |  |  |  |  |  |  | 3 |  | 2 |  |  |
| **q** |  |  |  |  |  |  |  |  |  |  |  | 4 |  |
| **k** |  |  |  |  |  |  |  |  |  |  |  |  | 4 |

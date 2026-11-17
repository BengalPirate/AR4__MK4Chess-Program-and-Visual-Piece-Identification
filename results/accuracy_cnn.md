# Piece recognition accuracy (fine-tuned ResNet-18)

Generated 2026-11-16 15:11; best validation accuracy 96.7%; 5 epochs.

### Held-out phone photos (test split)

- squares: 483 (250 occupied)
- exact label (empty / type+colour): **97.1%**
- occupancy: 99.4%
- colour, on occupied squares detected as occupied: 100.0%
- piece type, on occupied squares: 94.4%

| true \ pred | . | P | N | B | R | Q | K | p | n | b | r | q | k |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **.** | 233 |  |  |  |  |  |  |  |  |  |  |  |  |
| **P** | 3 | 17 |  |  |  |  |  |  |  |  |  |  |  |
| **N** |  | 1 | 23 |  |  |  |  |  |  |  |  |  |  |
| **B** |  | 1 |  | 18 |  |  |  |  |  |  |  |  |  |
| **R** |  |  |  | 1 | 19 |  |  |  |  |  |  |  |  |
| **Q** |  |  |  |  |  | 30 |  |  |  |  |  |  |  |
| **K** |  |  |  |  |  | 3 | 17 |  |  |  |  |  |  |
| **p** |  |  |  |  |  |  |  | 19 |  |  |  |  |  |
| **n** |  |  |  |  |  |  |  |  | 20 |  |  |  |  |
| **b** |  |  |  |  |  |  |  |  | 2 | 18 |  |  |  |
| **r** |  |  |  |  |  |  |  |  |  |  | 18 |  |  |
| **q** |  |  |  |  |  |  |  |  |  |  |  | 18 | 2 |
| **k** |  |  |  |  |  |  |  |  |  |  |  | 1 | 19 |

### Webcam boards (11, never trained on)

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

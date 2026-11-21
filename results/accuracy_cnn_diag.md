# Piece recognition accuracy (fine-tuned ResNet-18)

Generated 2026-11-20 22:19; best validation accuracy 96.7%; 6 epochs.

### Held-out phone photos (test split)

- squares: 483 (250 occupied)
- exact label (empty / type+colour): **97.1%**
- occupancy: 99.4%
- colour, on occupied squares detected as occupied: 99.6%
- piece type, on occupied squares: 94.4%

| true \ pred | . | P | N | B | R | Q | K | p | n | b | r | q | k |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **.** | 233 |  |  |  |  |  |  |  |  |  |  |  |  |
| **P** | 3 | 17 |  |  |  |  |  |  |  |  |  |  |  |
| **N** |  | 1 | 23 |  |  |  |  |  |  |  |  |  |  |
| **B** |  | 1 |  | 18 |  |  |  |  |  |  |  |  |  |
| **R** |  |  |  |  | 19 |  |  | 1 |  |  |  |  |  |
| **Q** |  |  |  |  |  | 30 |  |  |  |  |  |  |  |
| **K** |  | 1 | 1 |  |  |  | 18 |  |  |  |  |  |  |
| **p** |  |  |  |  |  |  |  | 19 |  |  |  |  |  |
| **n** |  |  |  |  |  |  |  |  | 20 |  |  |  |  |
| **b** |  |  |  |  |  |  |  |  | 2 | 18 |  |  |  |
| **r** |  |  |  |  |  |  |  |  |  |  | 17 |  | 1 |
| **q** |  |  |  |  |  |  |  |  |  |  |  | 17 | 3 |
| **k** |  |  |  |  |  |  |  |  |  |  |  |  | 20 |

### Webcam boards (40, never trained on)

- squares: 2560 (1184 occupied)
- exact label (empty / type+colour): **99.4%**
- occupancy: 99.6%
- colour, on occupied squares detected as occupied: 100.0%
- piece type, on occupied squares: 98.6%

| true \ pred | . | P | N | B | R | Q | K | p | n | b | r | q | k |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **.** | 1376 |  |  |  |  |  |  |  |  |  |  |  |  |
| **P** | 8 | 282 |  |  |  |  |  |  |  |  |  |  |  |
| **N** |  |  | 77 |  |  |  |  |  |  |  |  |  |  |
| **B** |  |  |  | 75 |  |  |  |  |  |  |  |  |  |
| **R** |  |  |  |  | 74 |  |  |  |  |  |  |  |  |
| **Q** |  |  |  |  |  | 37 |  |  |  |  |  |  |  |
| **K** |  |  |  |  |  |  | 40 |  |  |  |  |  |  |
| **p** | 1 |  |  |  |  |  |  | 282 |  |  | 5 |  |  |
| **n** |  |  |  |  |  |  |  |  | 75 |  |  |  |  |
| **b** |  |  |  |  |  |  |  |  |  | 74 |  |  |  |
| **r** |  |  |  |  |  |  |  |  |  |  | 77 |  |  |
| **q** |  |  |  |  |  |  |  |  |  |  |  | 37 |  |
| **k** | 1 |  |  |  |  |  |  |  |  |  |  | 1 | 38 |

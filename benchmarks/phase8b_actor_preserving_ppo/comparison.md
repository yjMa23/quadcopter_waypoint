# P8B Formal Comparison

| Method | Episodes | Settled landing | Deck miss | Hard contact | Ground crash | Timeout |
|---|---:|---:|---:|---:|---:|---:|
| Frozen PPO teacher | 768 | 94.66% | 5.34% | 0.13% | 0.00% | 0.00% |
| BC epoch 0 | 768 | 86.20% | 13.80% | 0.00% | 0.00% | 0.00% |
| P7 ordinary BC+PPO | 768 | 76.69% | 23.05% | 0.91% | 0.00% | 0.00% |
| P8A metric-selected | 2304 | 91.67% | 8.33% | 0.30% | 0.00% | 0.00% |
| P8B metric-selected | 2304 | 96.74% | 3.17% | 0.09% | 0.00% | 0.04% |
| P8B reward-selected | 2304 | 95.40% | 4.51% | 0.13% | 0.00% | 0.04% |
| P8B epoch-200 last | 2304 | 92.40% | 7.55% | 0.04% | 0.00% | 0.04% |

# Extreme Condition Comparison

Both videos use the same song window, model, seed (`99`), DDIM steps (`50`), guidance scale (`2.5`), and target Ka ratio (`0.45`).

| Video | const | complex | HS | BPM rhythm | note type | density | peak | Notes | Don | Ka | Actual Ka |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `extreme_all_easy_const6.mp4` | 6.0 | 0 | 0 | 0 | 0 | 0 | 0 | 90 | 52 | 38 | 0.422 |
| `extreme_all_hard_const10.mp4` | 10.0 | 2 | 1 | 2 | 2 | 2 | 2 | 171 | 80 | 91 | 0.532 |

Both outputs passed their selected legal-grid mask. The hard setup generated 1.9 times as many notes as the easy setup.

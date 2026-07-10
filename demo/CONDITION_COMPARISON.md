# Condition Comparison

All four videos use the same song window, model, seed (`42`), DDIM steps (`50`), and guidance scale (`2.5`).
Only chart conditions change.

| Video | const | complex | note type | density | peak | target Ka | Notes | Don | Ka | Actual Ka |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `compare_a_easy_sparse.mp4` | 6.0 | 0 | 0 | 0 | 0 | 0.25 | 90 | 63 | 27 | 0.300 |
| `compare_b_simple_dense.mp4` | 8.0 | 0 | 1 | 2 | 1 | 0.40 | 171 | 84 | 87 | 0.509 |
| `compare_c_balanced.mp4` | 8.5 | 1 | 1 | 1 | 1 | 0.50 | 127 | 73 | 54 | 0.425 |
| `compare_d_hard_dense.mp4` | 10.0 | 2 | 2 | 2 | 2 | 0.65 | 171 | 74 | 97 | 0.567 |

`complex=0` uses the strict 16-slot-per-measure grid. `complex=1` uses the allowed 8/12/16/24/32 subdivisions. `complex=2` keeps the unrestricted active-frame grid. All decoded notes passed their selected legal-grid mask.

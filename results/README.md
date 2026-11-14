# Paper source data

| File | Content |
| --- | --- |
| `figure1.json` | Category F1, count MAE and signed or absolute errors for the two encoding routes |
| `figure2.json` | Counts, presence rates, first-token scores and the full-image zero-target control |
| `figure3.json` | Visual-token and key/value transfer results and the matched random-block control |
| `figure4.json` | Cross-model and cross-dataset summaries |
| `figure5.json` | Chinese- and English-trained expert results, with all 10,000 paired bootstrap draws for each panel and language |
| `supplementary_task_tables.json` | Numerical rows for the five-task scores and response coverage |
| `value_scale_ablation.json` | Object and spatial MAE for the four readout configurations in each language |

The encoding order is crop then encode, followed by encode then select. Four-condition arrays follow IVER, IVEM, IMER, IMEM. Expert arrays follow LoRA then LoRA plus expert. The `en` and `zh` keys describe the five-task training and original-image evaluation language. Regional expert queries use English in both groups.

For the full-image control in `figure2.json`, the [frozen region and mask list](../splits/visdrone_full_image_controls.json) also includes the positive-control predictions reported in the Supplementary Methods.

In `figure2.json`, the paired scores include all four conditions. `visible_internal_score_mean_difference` is the mean of IVER minus IVEM across the same 1,127 regions.

In `figure5.json`, `bootstrap_by_image` contains the regional denominators, summed count errors and affirmative-answer totals used for the published 7B intervals. Resample source groups for the primary intervals or images for the sensitivity intervals. For each task and grouping, initialise NumPy's default generator with the recorded seed. For each replicate, draw multinomial counts with uniform probabilities over the sorted group labels, pool regional numerators and denominators, and compute the paired contrast or relative MAE reduction. Take the 2.5th and 97.5th percentiles across replicates. The exterior contrast is IMER minus IMEM in percentage points. Reported changes are expert minus LoRA, and relative MAE reductions are `100 * (1 - expert_MAE / LoRA_MAE)`.

The Kimi rows in `figure5.json` and the supplementary tables use fusion weights trained with the native Kimi pixel normalisation. Object-count MAE uses 547 paired complete answers in Chinese and 548 in English. Both spatial-count comparisons use 548 paired complete answers.

# M2 Reward Term Summary

This file is generated from the existing S0/S1 TensorBoard logs only. No simulator or training is invoked.

`Episode/Episode_Reward/*` is logged by the environment as mean episodic sum divided by 10 s. The tables below multiply it by 10 s to recover the logger's approximate mean episodic contribution for that reset cohort.

Episode-length normalization is an aggregate diagnostic approximation. `episode_lengths/iter` is in environment steps; M2 uses 0.04 s per environment step. When ep30 has no exact episode-length scalar, the latest earlier scalar is shown with its source iteration.

## S0

| iteration | training reward | episode length (steps) | source | reward/step approx | top negative contributors | top positive contributors |
| ---: | ---: | ---: | ---: | ---: | --- | --- |
| 10 | -37.450 | 111.57 | r10/l10 | -0.33566 | crash_penalty -28.75; horizontal_error -9.30; center_precision_square -3.28; height_tracking -3.20; progress_to_pad -3.01 | post_align_descent +0.78; align_hold +0.32; align_bonus +0.11 |
| 20 | -50.518 | 144.72 | r20/l20 | -0.34909 | horizontal_error -32.09; crash_penalty -16.25; progress_to_pad -11.28; height_tracking -8.18; rel_vel -3.48 | align_hold +0.11; post_align_descent +0.09; align_bonus +0.05 |
| 30 | -60.918 | 161.93 | r30/l29 | -0.37619 | horizontal_error -16.77; crash_penalty -11.25; height_tracking -4.34; rel_vel -3.23; progress_to_pad -2.52 | none |

## S1

| iteration | training reward | episode length (steps) | source | reward/step approx | top negative contributors | top positive contributors |
| ---: | ---: | ---: | ---: | ---: | --- | --- |
| 10 | -28.909 | 131.90 | r10/l10 | -0.21917 | horizontal_error -16.15; crash_penalty -8.12; height_tracking -4.09; progress_to_pad -3.07; rel_vel -1.55 | post_align_descent +0.08; align_hold +0.07; align_bonus +0.04 |
| 20 | -53.961 | 151.36 | r20/l20 | -0.35650 | predicted_pad_error -28.62; contact_clearance -28.60; crash_penalty -20.00; horizontal_error -12.40; height_tracking -9.15 | align_hold +1.65; align_bonus +0.48 |
| 30 | -73.855 | 173.82 | r30/l29 | -0.42488 | crash_penalty -17.50; horizontal_error -13.10; height_tracking -6.14; contact_clearance -3.31; predicted_pad_error -2.71 | align_bonus +0.32; align_hold +0.31; post_align_descent +0.11 |

## Phase-gating diagnostics

| run | diagnostic | n | Pearson | Spearman |
| --- | --- | ---: | ---: | ---: |
| S0 | direct can_land negative magnitude vs align | 30 | 0.842 | 0.807 |
| S0 | phase-sensitive magnitude share vs align | 30 | 0.562 | 0.660 |
| S0 | length-normalized reward vs episode length | 29 | -0.922 | -0.863 |
| S1 | direct can_land negative magnitude vs align | 30 | 0.809 | 0.866 |
| S1 | phase-sensitive magnitude share vs align | 30 | 0.832 | 0.897 |
| S1 | length-normalized reward vs episode length | 29 | -0.877 | -0.984 |

The first two diagnostics use reward-term aggregates and alignment metrics emitted from the same environment reset logging path. A strong positive association means post-latch reward magnitude appears precisely in iterations where more reset episodes have latched alignment; it does not by itself prove the policy later drifted within those episodes.

## Reward-versus-behavior trend diagnostics

| run | diagnostic | n | Pearson | Spearman |
| --- | --- | ---: | ---: | ---: |
| S0 | align | 30 | 0.042 | 0.276 |
| S0 | crash | 30 | -0.449 | -0.218 |
| S0 | deck_miss | 29 | 0.173 | 0.246 |
| S0 | timeout | 30 | 0.287 | -0.091 |
| S0 | episode_length | 29 | -0.983 | -0.984 |
| S1 | align | 30 | -0.343 | -0.313 |
| S1 | crash | 30 | -0.597 | -0.488 |
| S1 | deck_miss | 29 | -0.410 | -0.378 |
| S1 | timeout | 30 | 0.497 | 0.353 |
| S1 | episode_length | 29 | -0.910 | -0.997 |

These correlations use approximately 30 optimizer iterations and are descriptive only. The reset-cohort logger signals for `Episode_Termination/crash` and `time_out` are not probabilities and can exceed 1; they must not be interpreted as percentage rates.

## Existing deterministic evaluator: post-latch terminal state

`align_success` is latched, so an aligned episode whose terminal horizontal error is >= 0.25 m is direct evidence that the episode later ended outside the horizontal part of the instantaneous alignment envelope. This still does not reconstruct exactly when alignment was lost.

| run | iter | aligned | aligned ending outside 0.25 m | aligned timeouts | timeout outside 0.25 m | timeout terminal xy mean | timeout clearance mean |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| S0 | 10 | 16/64 (25.00%) | 12/16 (75.00%) | 0 | 0/0 (0.00%) | 0.000 m | 0.000 m |
| S0 | 20 | 17/64 (26.56%) | 17/17 (100.00%) | 0 | 0/0 (0.00%) | 0.000 m | 0.000 m |
| S0 | 30 | 18/64 (28.12%) | 18/18 (100.00%) | 0 | 0/0 (0.00%) | 0.000 m | 0.000 m |
| S1 | 10 | 5/64 (7.81%) | 5/5 (100.00%) | 0 | 0/0 (0.00%) | 0.000 m | 0.000 m |
| S1 | 20 | 10/64 (15.62%) | 10/10 (100.00%) | 3 | 3/3 (100.00%) | 2.484 m | 1.818 m |
| S1 | 30 | 20/64 (31.25%) | 19/20 (95.00%) | 10 | 10/10 (100.00%) | 0.710 m | 0.425 m |

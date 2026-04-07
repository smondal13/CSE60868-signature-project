# Current Results

This note focuses on model results and their interpretation.

## 1. Main Siamese Result

The full CRC run shows that the Siamese network is learning a meaningful
verification space.

### Full validation result

- Best validation EER: `0.0755`
- Best validation threshold: `0.7842`
- Validation AUC at the last recorded evaluation: `0.9640`
- Last validation EER: `0.1011`

### Interpretation

- The model is clearly better than chance and is separating genuine and negative
  pairs well.
- An EER of about `7.5%` on validation is encouraging for this stage of the
  project.
- The best epoch occurred before the final epoch, since the last validation EER
  (`0.1011`) is worse than the best validation EER (`0.0755`).

The most important conclusion from this run is:

> The Siamese approach is working, and the model is learning useful
> writer-independent features for Bengali and Hindi signature verification.

## 2. Robustness Result

To study generalization, I evaluated the trained model under controlled
query-side perturbations. The reference image in each verification pair was kept
clean, while the query image was modified. This isolates how sensitive the model
is to scan or appearance changes at test time.

## 3. Small Held-Out Test Robustness Experiment

### Baseline

- AUC: `0.9370`
- EER: `0.1479`

### Perturbation results

| Condition | AUC | EER | Change in EER |
|---|---:|---:|---:|
| Baseline | 0.9370 | 0.1479 | +0.0000 |
| Rotate `+3°` | 0.9220 | 0.1701 | +0.0222 |
| Rotate `-3°` | 0.9169 | 0.1729 | +0.0250 |
| Resolution `50%` | 0.6157 | 0.4514 | +0.3035 |
| Thickness `+1` pixel | 0.7698 | 0.2694 | +0.1215 |
| Thickness `-1` pixel | 0.4883 | 0.5167 | +0.3688 |

## 4. Interpretation Of The Robustness Result

The robustness experiment suggests three things.

### 1. Small angle changes are not the main weakness

Rotations of `±3°` increase EER by only about `0.02` to `0.03`. This means the
model is somewhat tolerant to mild slant or scan angle changes.

### 2. Resolution is a major sensitivity

When the query image is downsampled to `50%` and resized back, EER rises from
`0.1479` to `0.4514`. That is a very large degradation. This suggests the model
depends strongly on fine local stroke detail.

### 3. Stroke-width changes are also a major sensitivity

Both thickening and thinning hurt performance, but thinning is especially
damaging:

- Thickness `+1`: EER rises to `0.2694`
- Thickness `-1`: EER rises to `0.5167`

This suggests the current representation is not yet robust to changes in pen
pressure, scan quality, or preprocessing effects that alter stroke appearance.

## 5. Overall Conclusion

The current evidence supports the following result-focused conclusion:

> The Siamese model is learning meaningful verification features, but its
> generalization is much stronger for mild geometric variation than for changes
> in image quality and stroke appearance.

In other words:

- the core algorithm works
- the baseline verification result is promising
- the main weakness is robustness to realistic appearance shifts

## 6. Most Important Next Result To Produce

The next key experiment is to repeat the same robustness evaluation on the full
held-out test setup using the full CRC-trained checkpoint. That will tell us
whether the same pattern holds at the report-quality scale:

- mild robustness to small rotation
- strong sensitivity to reduced resolution
- strong sensitivity to stroke-thickness change

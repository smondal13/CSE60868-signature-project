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

## 3. Full Validation Robustness Experiment

### Baseline

- Split: `validation`
- Number of pairs: `62,116`
- AUC: `0.9785`
- EER: `0.0756`
- FAR at locked threshold: `0.0762`
- FRR at locked threshold: `0.0747`

### Perturbation results

| Condition | AUC | EER | Change in EER |
|---|---:|---:|---:|
| Baseline | 0.9785 | 0.0756 | +0.0000 |
| Rotate `+3°` | 0.9553 | 0.1150 | +0.0394 |
| Rotate `-3°` | 0.9552 | 0.1173 | +0.0416 |
| Resolution `50%` | 0.6604 | 0.3786 | +0.3030 |
| Thickness `+1` pixel | 0.6536 | 0.3845 | +0.3089 |
| Thickness `-1` pixel | 0.4828 | 0.5120 | +0.4364 |

## 4. Interpretation Of The Robustness Result

The full validation robustness experiment suggests three things.

### 1. Small angle changes are not the main weakness

Rotations of `±3°` increase EER by about `0.04`. This is a real drop, but it is
still much smaller than the degradation caused by the appearance-based
perturbations. This means the model has some tolerance to mild slant or scan
angle changes.

### 2. Resolution is a major sensitivity

When the query image is downsampled to `50%` and resized back, EER rises from
`0.0756` to `0.3786`. That is a very large degradation. This suggests the model
depends strongly on fine local stroke detail and is not yet robust to lower scan
quality.

### 3. Stroke-width changes are also a major sensitivity

Both thickening and thinning hurt performance, but thinning is especially
damaging:

- Thickness `+1`: EER rises to `0.3845`
- Thickness `-1`: EER rises to `0.5120`

This suggests the current representation is not yet robust to changes in pen
pressure, scan quality, or preprocessing effects that alter stroke appearance.

An additional important detail is the behavior at the locked validation
threshold. Under the strong perturbations, FAR stays similar or even drops
slightly, but FRR becomes extremely large:

- Resolution `50%`: FRR rises to `0.8446`
- Thickness `+1`: FRR rises to `0.7993`
- Thickness `-1`: FRR rises to `0.9486`

This means the model becomes overly strict under these perturbations and rejects
many genuine signatures that have been visually altered.

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

The next key experiment is to retrain the model with mild targeted
augmentation and compare against this full validation robustness baseline.

The first retraining recipe should be conservative:

- mild rotation augmentation
- mild resolution jitter, not as strong as `50%`
- mild stroke-width jitter

The goal is not simply to improve clean validation EER, but to reduce the large
robustness gaps observed here, especially for:

- reduced resolution
- thicker strokes
- thinner strokes

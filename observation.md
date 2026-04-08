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

## 3. Full Validation Robustness Baseline

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

## 6. Augmented Retraining Result

After observing the robustness weakness above, I retrained the model with a
conservative augmentation profile (`mild_v1`).

### Mild augmentation profile used during training

Each training image is augmented independently with probability `0.6`, so `40%`
of images are left unchanged.

Among the augmented images:

- `45%` receive a small random rotation in `[-3°, 3°]`
- `35%` receive mild resolution jitter, with scale sampled from `[0.75, 1.0]`
- `20%` receive stroke-thickness jitter

So the effective per-image probabilities are:

- No augmentation: `40%`
- Rotation: `27%`
- Resolution jitter: `21%`
- Thickness jitter: `12%`

For the thickness jitter branch, the code chooses `+1` and `-1` with equal
probability, so the effective probabilities are:

- Thickness `+1`: `6%`
- Thickness `-1`: `6%`

### Clean validation result after augmentation

- Best validation EER: `0.0739`
- Best validation threshold: `0.7795`
- Last validation AUC: `0.9719`
- Last validation EER: `0.0862`

### Comparison to the earlier non-augmented full run

- Best validation EER improved from `0.0755` to `0.0739`
- Last validation EER improved from `0.1011` to `0.0862`
- Last validation AUC improved from `0.9640` to `0.9719`

This suggests that the mild augmentation did not hurt clean validation
performance. Instead, it appears to have slightly improved both the best
checkpoint and the stability of later epochs.

## 7. Current Interpretation

The current evidence now suggests:

- the base Siamese model works
- the original full validation robustness run exposed a real generalization
  weakness
- the first augmented retraining run improved clean validation performance

This is encouraging, but it does **not** yet prove that robustness improved,
because the validation numbers above are still from clean validation data.

## 8. Robustness Result After Augmented Retraining

I then ran the same full validation robustness sweep on the augmented
checkpoint. This is the most important comparison, because it tests whether the
augmentation actually improved the failure modes that appeared in the baseline
robustness experiment.

### Full validation robustness after augmentation

| Condition | AUC | EER | Change in EER from clean baseline |
|---|---:|---:|---:|
| Baseline | 0.9754 | 0.0739 | +0.0000 |
| Rotate `+3°` | 0.9756 | 0.0744 | +0.0005 |
| Rotate `-3°` | 0.9715 | 0.0819 | +0.0080 |
| Resolution `50%` | 0.9415 | 0.1291 | +0.0552 |
| Thickness `+1` pixel | 0.9324 | 0.1390 | +0.0651 |
| Thickness `-1` pixel | 0.8918 | 0.1840 | +0.1102 |

### Comparison against the earlier robustness baseline

| Condition | Old EER | New EER | Improvement |
|---|---:|---:|---:|
| Baseline | 0.0756 | 0.0739 | -0.0017 |
| Rotate `+3°` | 0.1150 | 0.0744 | -0.0406 |
| Rotate `-3°` | 0.1173 | 0.0819 | -0.0354 |
| Resolution `50%` | 0.3786 | 0.1291 | -0.2495 |
| Thickness `+1` pixel | 0.3845 | 0.1390 | -0.2455 |
| Thickness `-1` pixel | 0.5120 | 0.1840 | -0.3280 |

### Interpretation

This is a strong result.

- Clean validation did not get worse; it improved slightly.
- Rotation robustness improved substantially, especially for `+3°`.
- Resolution robustness improved dramatically.
- Stroke-thickness robustness improved dramatically.
- Thinning remained the hardest perturbation, but it is far better than before.

The most important change is not only the reduction in EER, but also the large
drop in false rejection under the locked threshold. In the earlier baseline, the
model became overly strict under strong perturbations and rejected many genuine
signatures. After augmentation, this behavior is much more controlled.

For example:

- Resolution `50%`: FRR dropped from `0.8446` to `0.1381`
- Thickness `+1`: FRR dropped from `0.7993` to `0.1821`
- Thickness `-1`: FRR dropped from `0.9486` to `0.2669`

The FAR at the locked threshold increased somewhat under the strong
perturbations, but the overall trade-off is much better balanced than before,
and the EER improvement is large.

## 9. Updated Conclusion

The evidence now supports a stronger conclusion than before:

> Mild targeted augmentation substantially improves generalization to realistic
> signature appearance shifts without harming clean validation performance.

At this point, the augmentation is not just a reasonable idea; it has empirical
support on the full validation split.

## 10. Most Important Next Result To Produce

The next most useful step is a small, focused hyperparameter search while
keeping the `mild_v1` augmentation profile fixed.

The goal now is not to rediscover augmentation, but to refine the model around a
training recipe that already works well.

The first search should stay small and practical, for example:

- learning rate
- contrastive margin
- early stopping patience

I would keep the search modest and continue selecting models based on validation
performance together with robustness behavior, not clean validation EER alone.

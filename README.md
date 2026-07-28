# Recognising happiness and sadness in children's drawings

Leakage-controlled comparison of transfer-learning architectures on the KIDO dataset.

This repository contains the code and derived results for the paper
*"Recognising happiness and sadness in children's drawings: a leakage-controlled
comparison of transfer-learning architectures on the KIDO dataset."*

## Summary

We evaluate emotion recognition (happiness vs. sadness) from children's drawings
in the public **KIDO** dataset, under a protocol that controls for two pitfalls:

1. **Data leakage** — duplicate images are removed (via perceptual hashing) and the
   train/validation/test split is performed by image identity so no duplicate
   drawing spans partitions.
2. **Label composition** — the distributed KIDO label field co-encodes gender and
   school level alongside emotion; we isolate only the two emotion classes by name.

Three frozen transfer-learning backbones are compared, each with a regularised
linear classifier, over five random seeds:

| Architecture     | Accuracy | Macro F1 | Cohen's kappa |
|------------------|----------|----------|---------------|
| ResNet-50        | 0.637    | 0.636    | 0.273         |
| EfficientNet-B0  | 0.666    | 0.666    | 0.332         |
| ViT-B/16         | 0.679    | 0.679    | 0.358         |

(Mean over five seeds; held-out test set of 1,629 drawings; balanced dataset of
10,860 unique drawings, 5,430 per emotion.)

The vision transformer performs best; its advantage over ResNet-50 is statistically
significant (McNemar, p = 0.012). All figures sit above chance (0.50) and below the
fine-tuned figure (~0.79) reported by the dataset's creators.

## Repository contents

```
scripts/
  kido_memsafe_dedup.py   # main experiment: 3 architectures, 5 seeds, dedup, leak-check, stats
  kido_light.py           # fast single-model (ViT) sanity run
  make_samples.py         # generate a sample-drawings figure from the data
results/
  results_summary.json    # the reported numbers (per architecture)
requirements.txt
LICENSE
CITATION.cff
```

## Data

This project uses the publicly available **KIDO** dataset:

> Çiftçi S, Karci H, Karaca N, Söylemez B, Koçakoğlu H (2025).
> Emotion analysis of children's drawings. *IEEE Access*, 13:159730–159748.
> https://doi.org/10.1109/ACCESS.2025.3606359

The dataset is available on Hugging Face: `serdarciftci/KIDO`.
It is not redistributed here; the scripts download it directly.

## Reproducing the results

1. Open the main script in Google Colab (or any machine with a CUDA GPU).
2. Select a GPU runtime (e.g. an NVIDIA T4).
3. Run `scripts/kido_memsafe_dedup.py`.

The script downloads KIDO, isolates the emotion classes, removes duplicate images,
performs a group-aware (leakage-free) split, extracts frozen features for each
architecture, trains a linear classifier over five seeds, and prints the summary
table, McNemar tests, and confusion matrices.

## Citation

If you use this code, please cite the paper (see `CITATION.cff`) and the KIDO dataset.

## License

Code released under the MIT License (see `LICENSE`). The KIDO dataset is subject to
its own license terms; please consult the dataset source before redistributing images.

# SIGIR2026 Submitted

This repository contains the code for:
Residual Dominance Drives Last-Item Reliance in Causal Self-Attention for Sequential Recommendation.

This repository runs norm-based analysis of SASRec on top of the `time-to-split` codebase.
For full experimental pipelines and reproducibility details, see `time-to-split/README.md`.


## Section 3.1 Results (all datasets)

The figure below shows results on nine datasets when inputs are shuffled at inference time.

![Shuffle histogram](time-to-split/images/shuffle_histogram.png)

## Section 3.2 Results (K=10)

The image below shows HRLI and HRL2I computed at K=10.

![HRLI vs HRL2I at K=10](time-to-split/images/hist_comparison_10.png)



## Structure
- `time-to-split/`: Research codebase (vendored from the original repository)

## SASRecAnalyze: How to Run the Analysis

SASRecAnalyze saves per-batch analysis `.npz` during prediction. The default config already enables
`seqrec_module.save_analysis_npz: true`.

### 1) Train/evaluate with SASRecAnalyze

Run training with the analysis model. Use any dataset/split supported by `time-to-split`:

```bash
cd time-to-split
python runs/train.py model=SASRecAnalyze split_type=leave-one-out dataset=Beauty
```

### 2) Locate analysis outputs

Analysis files are saved under:

```
$SEQ_SPLITS_DATA_PATH/results/analysis/SASRecAnalyze/<dataset>/<split_type>/seed_<seed>/
```

### 3) Generate analysis plots/statistics

Edit `time-to-split/src/analyze.py` and set:

```python
analysis_dir = "./data/results/analysis/SASRecAnalyze/Movielens-1m/global_timesplit/seed_17"
```

Then run:

```bash
python src/analyze.py
```

Outputs include mixing-ratio statistics and average heatmaps, saved under:
`figures_avg_recent15_sourceDown/` inside the analysis directory.

## Notes

- Detailed data prep, split strategies, and training options live in `time-to-split/README.md`.

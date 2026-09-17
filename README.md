# STRATUM

Mediation-constrained hierarchical federated learning for joint rhythm, rate and
activity inference on wearable recordings.

This repository contains the complete implementation behind the manuscript:
the corpus builder, the framework, every competing scheme, the experiment
driver, the statistical analysis and the figure scripts. Everything runs on CPU.

## What the code does

A federation of wearable nodes jointly detects arrhythmic beats, estimates heart
rate and recognises physical activity. Movement raises heart rate through
physiology and, at the same time, corrupts the electrode signal in a way that
resembles ectopy. The framework keeps the first route and closes the second:

* the rhythm head receives movement context **only** through the estimated rate;
* a conditional probe and a context-swap consistency term remove the residual
  dependence of the rhythm representation on activity;
* the shared representation, the context adapter and the device calibration are
  aggregated at the cloud, inside one cell and nowhere respectively;
* the cloud resolves conflicts between cell updates with a strength read off an
  analysis of variance of the update dispersion;
* rhythm confidence is rescaled per motion stratum.

## Data

Two public collections are required. Neither is redistributed here.

| Source | Where to get it | Expected location |
| --- | --- | --- |
| MIT-BIH Arrhythmia Database (v1.0.0) | PhysioNet, `mitdb` | `data/mitdb/` containing `100.dat`, `100.hea`, `100.atr`, ... |
| MHEALTH dataset | UCI Machine Learning Repository, dataset 319 | `data/mhealth/` containing `mHealth_subject1.log` ... `mHealth_subject10.log` |

Override the locations with the `MITDB_DIR` and `MHEALTH_DIR` environment
variables if the files live elsewhere.

## Install

```
pip install -r requirements.txt
```

## Reproduce

```
python corpus_stats.py                      # corpus description, writes results/corpus_stats.json
python run_experiments.py --stage main        --folds 5
python run_experiments.py --stage ablation    --folds 5
python run_experiments.py --stage sensitivity --folds 5
python run_experiments.py --stage topology    --folds 5
python dump_predictions.py                  # predictions used by the calibration figure
python fold_denoms.py                       # stratum sizes for the pooled alarm-rate estimate
python analysis_stats.py                    # Friedman, Wilcoxon with Holm correction, effect sizes
python make_figures.py                      # every figure, grayscale
```

Each stage appends one JSON record per run to `results/<stage>.jsonl` and skips
work that is already recorded, so an interrupted run can be resumed by issuing
the same command again.

## Files

| File | Purpose |
| --- | --- |
| `config.py` | paths, signal constants, label spaces, partitions, defaults |
| `signal_utils.py` | filtering, resampling, R-peak detection, windowing |
| `data_mitbih.py` | MIT-BIH ingestion, AAMI mapping, donor pool with prematurity |
| `data_mhealth.py` | MHEALTH ingestion, inertial context windows, beat templates |
| `build_corpus.py` | graft operator, confound coefficient, client and cell construction |
| `experiment_setup.py` | fold construction and the three evaluation sets |
| `models.py` | encoders, context adapter, device calibration, heads, probe |
| `federated.py` | device objective, importance weights, tier aggregation, projection |
| `calibration.py` | activity-conditional temperature scaling |
| `metrics.py` | rhythm, binary alarm, rate, activity and calibration metrics |
| `trainer.py` | hierarchical training loop and evaluation, one entry point per run |
| `run_experiments.py` | every reported configuration, resumable |
| `analysis_stats.py` | paired statistical analysis |
| `make_figures.py` | all figures |
| `dump_predictions.py` | prediction dump for the reliability figure |
| `corpus_stats.py` | corpus description table |
| `fold_denoms.py` | per-fold stratum sizes used to pool the stratified alarm rates |

## Notes on the corpus

No public collection carries expert beat annotation and simultaneous inertial
context from the same person, so the coupled corpus grafts annotated arrhythmic
morphology onto wearable recordings. Every ingredient is real: the host supplies
the activity label, the inertial window and the motion artifact, the donor
supplies the morphology, its AAMI annotation and its prematurity. Normal and
ectopic donors pass through the same operator, so the splice carries no class
information. Beats that receive no graft are marked unannotated for the rhythm
task rather than labelled by assumption. The association between movement and
ectopy is an explicit coefficient: positive in training, removed or reversed at
evaluation.

## Licence

MIT. See `LICENSE`.

# Baseline Repositories and Private-Dataset Runs

## Cloned baseline repositories (GitHub)

1. Transfer-Learning-Library
2. pytorch-adapt
3. transferlearning
4. DomainBed
5. CDAN
6. pytorch_DANN
7. PyTorch-Deep-CORAL
8. MMD_Loss.Pytorch

All repositories are cloned under this folder.

## Private-dataset baseline methods in this project

The private-runner script is:

baseline/private_runner/run_private_baseline.py

Supported methods:

- source_only
- dann
- cdan
- deepcoral
- mmd
- jan
- entropy_min
- mcc

Batch launcher:

baseline/private_runner/run_all_private_baselines.sh

## Logs and metrics

- Orchestration log:
  baseline/logs/run_all_baselines.log
- Per-method logs:
  baseline/logs/<method>.log
- Per-method metrics CSV (includes FAR and FDR columns):
  baseline/private_runs/<method>/metrics.csv

## FAR/FDR definition used

- FAR (Fault Alarm Rate): fraction of normal samples predicted as fault.
- FDR (Fault Detection Rate): fraction of fault samples correctly predicted as fault.

Normal labels are inferred as class names in {L0, NO_L, NORMAL, HEALTHY}.

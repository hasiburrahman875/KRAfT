# KRAfT

Code and reproducibility assets for UAVSwarm W2C fold evaluation.

## Included now
- `configs/w2c_fold1_test.yaml`
- `configs/w2c_fold2_test.yaml`
- `configs/w2c_fold3_test.yaml`
- `scripts/reproduce_w2c_folds.sh`

## Run
Run one fold:
```bash
PYTHON_BIN=/home/mrpk9/.conda/envs/yolov12_botsort/bin/python \
  bash scripts/reproduce_w2c_folds.sh 1
```

Run all folds:
```bash
PYTHON_BIN=/home/mrpk9/.conda/envs/yolov12_botsort/bin/python \
  bash scripts/reproduce_w2c_folds.sh 1 2 3
```

## Notes
- Local environment used for validation: `yolov12_botsort`.
- Fold1 tuned hyperparameters:
  - `high_thres: 0.25`
  - `med_thres: 0.30`
  - `low_thres: 0.10`
  - `lambda_kf: 1.00`
- Fold2/fold3 currently use:
  - `high_thres: 0.25`
  - `med_thres: 0.10`
  - `low_thres: 0.05`
  - `lambda_kf: 0.60`

## Weights
The tracking checkpoint (`_epoch2100.pt`) is about 149 MB. Keep it in local storage or publish via Git LFS/external hosting and link it here.

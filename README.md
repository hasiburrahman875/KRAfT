# KRAfT

Minimal reproducibility snapshot for UAVSwarm W2C fold evaluation.

## Included in this initial commit
- Fold-1 test config used in our validated reproduction:
  - `configs/w2c_fold1_test.yaml`

## Notes
- We validated this setup in local environment `yolov12_botsort`.
- Main fold1 hyperparameters:
  - `high_thres: 0.25`
  - `med_thres: 0.30`
  - `low_thres: 0.10`
  - `lambda_kf: 1.00`

## Large weights
The tracking checkpoint used for evaluation (`_epoch2100.pt`) is ~149 MB and should be tracked via Git LFS or hosted externally.

# KRAfT

**KRAfT: Kalman Residual Diffusion with Formation Awareness for UAV Swarm Tracking**  
28th International Conference on Pattern Recognition (ICPR 2026)

[![Paper](https://img.shields.io/badge/Paper-Springer-blue)](https://doi.org/10.1007/978-3-032-31654-7_2)
[![Dataset](https://img.shields.io/badge/Dataset-Zenodo-blue)](https://doi.org/10.5281/zenodo.20566871)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Official code and reproducibility resources for KRAfT, a multi-object tracking framework for UAV swarms. This repository includes the configurations and evaluation scripts used for the three-fold UAVSwarm-W2C evaluation.

## Resources

- **Paper:** [KRAfT: Kalman Residual Diffusion with Formation Awareness for UAV Swarm Tracking](https://doi.org/10.1007/978-3-032-31654-7_2)
- **UAVSwarm-W2C dataset:** [Zenodo record](https://zenodo.org/records/20566871)
- **Dataset DOI:** [10.5281/zenodo.20566871](https://doi.org/10.5281/zenodo.20566871)

## UAVSwarm-W2C Dataset

The UAVSwarm-W2C dataset used for the fold-based evaluation is publicly available on Zenodo:

**https://doi.org/10.5281/zenodo.20566871**

Please use the Zenodo DOI when referring to the dataset so that the dataset record remains persistently identifiable.

## Repository Contents

Key reproducibility files include:

```text
configs/
  w2c_fold1_test.yaml
  w2c_fold2_test.yaml
  w2c_fold3_test.yaml

scripts/
  reproduce_w2c_folds.sh

main.py
requirement.txt
```

## Installation

Clone the repository and install the Python dependencies:

```bash
git clone https://github.com/hasiburrahman875/KRAfT.git
cd KRAfT
pip install -r requirement.txt
```

## Reproducing the UAVSwarm-W2C Fold Evaluation

The provided script runs KRAfT and then evaluates the tracking results using TrackEval.

Run a single fold:

```bash
PYTHON_BIN=/path/to/python bash scripts/reproduce_w2c_folds.sh 1
```

Run all three folds:

```bash
PYTHON_BIN=/path/to/python bash scripts/reproduce_w2c_folds.sh 1 2 3
```

If the desired Python interpreter is already active, you can use:

```bash
PYTHON_BIN="$(which python)" bash scripts/reproduce_w2c_folds.sh 1 2 3
```

### Evaluation setup

The current reproduction script expects TrackEval under the companion tracking setup at:

```text
../YOLOv12-BoT-SORT-ReID-w2c/BoT-SORT/TrackEval
```

If your TrackEval installation is elsewhere, update `TRACK_EVAL_ROOT` in `scripts/reproduce_w2c_folds.sh` accordingly.

## Fold Configuration

The repository currently uses the following KRAfT settings for the UAVSwarm-W2C folds.

**Fold 1**

```text
high_thres: 0.25
med_thres:  0.30
low_thres:  0.10
lambda_kf:  1.00
```

**Folds 2 and 3**

```text
high_thres: 0.25
med_thres:  0.10
low_thres:  0.05
lambda_kf:  0.60
```

## Model Weights

The tracking checkpoint (`_epoch2100.pt`) is approximately 149 MB and is not currently stored directly in this repository. It can be hosted through Git LFS or external storage and linked here.

## Citation

If you use KRAfT in your research, please cite the ICPR paper:

```bibtex
@inproceedings{rahman2026kraft,
  title     = {KRAfT: Kalman Residual Diffusion with Formation Awareness for UAV Swarm Tracking},
  author    = {Rahman, Md. Hasibur and Madria, Sanjay},
  booktitle = {Pattern Recognition -- 28th International Conference on Pattern Recognition (ICPR 2026)},
  pages     = {15--30},
  publisher = {Springer},
  doi       = {10.1007/978-3-032-31654-7_2}
}
```

For use of **UAVSwarm-W2C**, please also cite the Zenodo dataset record:

**https://doi.org/10.5281/zenodo.20566871**

## License

This repository is released under the [MIT License](LICENSE).

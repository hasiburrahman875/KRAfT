# vim: expandtab:ts=4:sw=4
import os
import glob
import numpy as np
from torch.utils.data import Dataset

# ---- import YOUR KalmanFilter (the XYAH version you pasted) ----
# If it's in the same file, you can remove this import and use KalmanFilter directly.
from .kalman_filter import KalmanFilter

# -----------------------------
# XYWH <-> XYAH converters
# -----------------------------
_EPS_H = 1e-6  # avoid divide-by-zero for h

def xywh_to_xyah(xywh: np.ndarray) -> np.ndarray:
    """(x,y,w,h) -> (x,y,a,w/h) with safe h."""
    x, y, w, h = xywh[..., 0], xywh[..., 1], xywh[..., 2], np.clip(xywh[..., 3], _EPS_H, None)
    a = w / h
    return np.stack([x, y, a, h], axis=-1)

def xyah_to_xywh(xyah: np.ndarray) -> np.ndarray:
    """(x,y,a,h) -> (x,y,w,h) where w = a*h."""
    x, y, a, h = xyah[..., 0], xyah[..., 1], xyah[..., 2], xyah[..., 3]
    w = a * h
    return np.stack([x, y, w, h], axis=-1)

# -----------------------------
# KF wrapper: history -> 1-step motion in XYWH
# -----------------------------
def kf_one_step_motion_xywh(history_xywh: np.ndarray) -> np.ndarray:
    """
    Run your KalmanFilter over a history of XYWH boxes and predict one step ahead.
    Returns predicted motion in XYWH: pred_xywh - last_xywh.
    history_xywh: (T,4), T >= 2 preferred.
    """
    assert history_xywh.ndim == 2 and history_xywh.shape[1] == 4, "history_xywh must be (T,4)"
    xyah_hist = xywh_to_xyah(history_xywh).astype(np.float32)

    kf = KalmanFilter()
    mean, cov = kf.initiate(xyah_hist[0])

    # Feed historical measurements
    for t in range(1, len(xyah_hist)):
        mean, cov = kf.predict(mean, cov)
        mean, cov = kf.update(mean, cov, xyah_hist[t])

    # One-step ahead prediction
    mean, cov = kf.predict(mean, cov)
    meas_xyah, _ = kf.project(mean, cov)        # measurement space (x,y,a,h)
    pred_xywh = xyah_to_xywh(meas_xyah)         # convert to (x,y,w,h)
    last_xywh = history_xywh[-1]
    kf_motion = (pred_xywh - last_xywh).astype(np.float32)
    return kf_motion

# -----------------------------
# Dataset: residual-guided outputs
# -----------------------------
class DiffMOTDatasetResidual(Dataset):
    """
    Returns:
      - condition: (T-1, 8)  -> [history boxes (x,y,w,h) from 2..T,  history deltas]
      - kf_motion: (4,)      -> KF one-step motion in XYWH
      - residual : (4,)      -> (raw_motion - kf_motion) in XYWH
      - cur_bbox : (4,)      -> ground-truth next box (optional, useful for eval)
    Notes:
      Assumes each track file is loadable with np.loadtxt(...).reshape(-1,7)
      and boxes are columns [2:6] = (x,y,w,h) in your text format.
    """
    def __init__(self, path, config=None):
        super().__init__()
        self.config = config
        # original code used: interval = config.interval + 1
        # with T = self.interval, you'll get (T-1) history steps for condition
        self.interval = int(self.config.interval) + 1

        self.trackers = {}
        self.images = {}
        self.nframes = {}
        self.ntrackers = {}

        self.nsamples = {}
        self.nS = 0

        self.nds = {}
        self.cds = {}

        if os.path.isdir(path):
            self.seqs = sorted(os.listdir(path))
            lastindex = 0
            for seq in self.seqs:
                # tracker files (per your original layout)
                trackerPath = os.path.join(path, seq, "img1/*.txt")
                self.trackers[seq] = sorted(glob.glob(trackerPath))
                self.ntrackers[seq] = len(self.trackers[seq])

                # image paths (kept for parity; not used directly)
                if 'MOT' in seq:
                    imagePath = os.path.join(path, '../../images/train', seq, "img1/*.*")
                else:
                    imagePath = os.path.join(path, '../train', seq, "img1/*.*")
                self.images[seq] = sorted(glob.glob(imagePath))
                self.nframes[seq] = len(self.images[seq])

                # count samples per tracker file
                self.nsamples[seq] = {}
                for i, pa in enumerate(self.trackers[seq]):
                    arr = np.loadtxt(pa, dtype=np.float32).reshape(-1, 7)
                    # we need at least (interval + 1) rows to form one sample
                    n = arr.shape[0] - self.interval
                    n = max(0, n)
                    self.nsamples[seq][i] = n
                    self.nS += n

                # cumulative offsets
                self.nds[seq] = [x for x in self.nsamples[seq].values()]
                if len(self.nds[seq]) == 0:
                    self.cds[seq] = []
                else:
                    self.cds[seq] = [sum(self.nds[seq][:i]) + lastindex for i in range(len(self.nds[seq]))]
                    lastindex = (self.cds[seq][-1] + self.nds[seq][-1]) if len(self.nds[seq]) else lastindex

        print('=' * 80)
        print('dataset summary')
        print(self.nS)
        print('=' * 80)

    def __len__(self):
        return self.nS

    def _locate(self, files_index):
        """
        Map a global sample index to (seq, trk_idx, local_start_index).
        """
        ds, trk, start_index = None, None, None
        for seq in self.cds:
            if not self.cds[seq]:
                continue
            if files_index >= self.cds[seq][0]:
                ds = seq
                for j, c in enumerate(self.cds[seq]):
                    if files_index >= c:
                        trk = j
                        start_index = c
                    else:
                        break
        if ds is None or trk is None or start_index is None:
            raise IndexError(f"Index {files_index} out of range")
        return ds, trk, start_index

    def __getitem__(self, files_index):
        # locate which track file
        ds, trk, start_index = self._locate(files_index)
        track_path = self.trackers[ds][trk]
        track_gt = np.loadtxt(track_path, dtype=np.float32).reshape(-1, 7)

        # local index within this track
        init_index = files_index - start_index

        # history boxes (T frames) and the next box as target
        # boxes: shape (T, 4) in XYWH
        boxes = np.array([track_gt[init_index + i][2:6] for i in range(self.interval)], dtype=np.float32)
        cur_bbox = track_gt[init_index + self.interval][2:6].astype(np.float32)  # next box

        # raw motion to next frame in XYWH
        raw_motion = (cur_bbox - boxes[-1]).astype(np.float32)  # (4,)

        # KF motion anchor (XYWH)
        kf_motion = kf_one_step_motion_xywh(boxes)              # (4,)
        residual = (raw_motion - kf_motion).astype(np.float32)  # (4,)

        # condition: history boxes (excluding the very first) + history deltas
        # boxes[1:] -> (T-1, 4), deltas -> (T-1, 4)  => concat -> (T-1, 8)
        delt_boxes = (boxes[1:] - boxes[:-1]).astype(np.float32)
        conds = np.concatenate((boxes[1:], delt_boxes), axis=1).astype(np.float32)

        ret = {
            "condition": conds,     # (T-1, 8)
            "kf_motion": kf_motion, # (4,)
            "residual": residual,   # (4,)
            "cur_bbox": cur_bbox,   # (4,) optional, useful for eval
        }
        return ret

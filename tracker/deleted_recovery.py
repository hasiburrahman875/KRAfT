import numpy as np


def bbox_overlaps(a, b):
    num_a = a.shape[0]
    num_b = b.shape[0]
    overlaps = np.zeros((num_a, num_b), dtype=np.float32)

    for n_b in range(num_b):
        box_area = (b[n_b, 2] - b[n_b, 0] + 1) * (b[n_b, 3] - b[n_b, 1] + 1)
        for n_a in range(num_a):
            iw = min(a[n_a, 2], b[n_b, 2]) - max(a[n_a, 0], b[n_b, 0]) + 1
            if iw > 0:
                ih = min(a[n_a, 3], b[n_b, 3]) - max(a[n_a, 1], b[n_b, 1]) + 1
                if ih > 0:
                    ua = (a[n_a, 2] - a[n_a, 0] + 1) * (a[n_a, 3] - a[n_a, 1] + 1) + box_area - iw * ih
                    overlaps[n_a, n_b] = iw * ih / max(ua, 1e-6)
    return overlaps


def find_deleted_detections(dets_lowerset, dets_superset, iou_keep=0.97):
    a = np.ascontiguousarray(dets_lowerset[:, :4], dtype=np.float64)
    b = np.ascontiguousarray(dets_superset[:, :4], dtype=np.float64)

    if a.size == 0 or b.size == 0:
        return dets_superset.copy()

    ious = bbox_overlaps(a, b)
    keep = np.max(ious, axis=0) < float(iou_keep)
    return dets_superset[keep]

import numpy as np
from .utils import xywh_to_xyxy, iou_xyxy

def find_deleted_from_main_and_backup(dets_main_xywhs, dets_backup_xywhs, iou_keep=0.97):
    """
    TrackTrack idea simplified.
    Anything in backup that does not strongly overlap with main
    becomes a deleted recovery candidate.
    Both inputs are xywh score arrays.
    Returns xywh score array.
    """
    a = np.asarray(dets_main_xywhs, dtype=np.float32)
    b = np.asarray(dets_backup_xywhs, dtype=np.float32)

    if a.size == 0 or b.size == 0:
        return b.reshape(-1, 5) if b.size else np.zeros((0, 5), dtype=np.float32)

    a_xyxy = xywh_to_xyxy(a[:, :4])
    b_xyxy = xywh_to_xyxy(b[:, :4])

    ious = iou_xyxy(a_xyxy, b_xyxy)  # N M
    max_iou = ious.max(axis=0) if ious.size else np.zeros((b.shape[0],), dtype=np.float32)

    return b[max_iou < iou_keep]

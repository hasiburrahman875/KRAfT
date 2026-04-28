import numpy as np
from .utils import f32c, tlwh_to_xyxy, iou_xyxy_matrix


def track_aware_nms_before_births(
    candidate_tlwh,
    candidate_scores,
    active_tlwh,
    nms_thr=0.75,
    score_thr=0.0,
):
    candidate_tlwh = np.asarray(candidate_tlwh, dtype=np.float32).reshape(-1, 4)
    candidate_scores = np.asarray(candidate_scores, dtype=np.float32).reshape(-1)

    N = candidate_tlwh.shape[0]
    if N == 0:
        return np.zeros((0,), dtype=bool)

    allow = candidate_scores > float(score_thr)

    if active_tlwh is not None and len(active_tlwh):
        a_xyxy = tlwh_to_xyxy(f32c(np.asarray(active_tlwh, dtype=np.float32)))
        c_xyxy = tlwh_to_xyxy(f32c(candidate_tlwh))
        iou_to_tracks = iou_xyxy_matrix(c_xyxy, a_xyxy)
        for i in range(N):
            if not allow[i]:
                continue
            if iou_to_tracks.shape[1] and float(iou_to_tracks[i].max()) > nms_thr:
                allow[i] = False

    c_xyxy = tlwh_to_xyxy(f32c(candidate_tlwh))
    self_iou = iou_xyxy_matrix(c_xyxy, c_xyxy)

    order = np.argsort(-candidate_scores)
    for ii in range(N):
        i = order[ii]
        if not allow[i]:
            continue
        for jj in range(ii + 1, N):
            j = order[jj]
            if not allow[j]:
                continue
            if candidate_scores[i] >= candidate_scores[j] and float(self_iou[i, j]) > nms_thr:
                allow[j] = False

    return allow.astype(bool)

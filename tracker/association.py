import numpy as np

from .utils import (
    BIG, f32c, nan2big,
    tlwh_to_xyxy,
    iou_xyxy_matrix,
    center_distance_matrix,
)
from .formation import formation_cost_sparse, knn_idx
from .motion_cues import estimate_track_velocity_xy, angle_cost_matrix


def hungarian_min(cost, linear_assignment_fn=None):
    cost = np.asarray(cost, dtype=np.float32)
    if cost.size == 0 or cost.shape[0] == 0 or cost.shape[1] == 0:
        return np.empty((0, 2), dtype=int)
    cost = f32c(nan2big(cost))

    if linear_assignment_fn is not None:
        try:
            return linear_assignment_fn(cost)
        except Exception:
            pass

    try:
        from scipy.optimize import linear_sum_assignment
        r, c = linear_sum_assignment(cost)
        return np.stack([r, c], axis=1).astype(int)
    except Exception:
        return np.empty((0, 2), dtype=int)


def cosine_cost(strack_pool, detections):
    T = len(strack_pool)
    D = len(detections)
    if T == 0 or D == 0:
        return np.zeros((T, D), dtype=np.float32)

    idx_t = [i for i, st in enumerate(strack_pool) if getattr(st, "emb", None) is not None]
    idx_d = [j for j, dt in enumerate(detections) if getattr(dt, "emb", None) is not None]
    if not idx_t or not idx_d:
        return np.zeros((T, D), dtype=np.float32)

    U = np.asarray([strack_pool[i].emb for i in idx_t], dtype=np.float32)
    V = np.asarray([detections[j].emb for j in idx_d], dtype=np.float32)
    U /= (np.linalg.norm(U, axis=1, keepdims=True) + 1e-12)
    V /= (np.linalg.norm(V, axis=1, keepdims=True) + 1e-12)
    dist = 1.0 - (U @ V.T)

    C = np.zeros((T, D), dtype=np.float32)
    C[np.ix_(idx_t, idx_d)] = dist
    return f32c(C)


def confidence_consistency_cost(strack_pool, detections):
    T = len(strack_pool)
    D = len(detections)
    if T == 0 or D == 0:
        return np.zeros((T, D), dtype=np.float32)

    t_prev = []
    t_now = []
    for st in strack_pool:
        t_now.append(float(getattr(st, "score", 0.0)))
        prev = float(st.score_memory[-2]) if hasattr(st, "score_memory") and len(st.score_memory) >= 2 else float(getattr(st, "score", 0.0))
        t_prev.append(prev)

    t_prev = np.asarray(t_prev, dtype=np.float32)
    t_now = np.asarray(t_now, dtype=np.float32)
    proj = t_now + (t_now - t_prev)

    d_scores = np.asarray([float(getattr(d, "score", 0.0)) for d in detections], dtype=np.float32)
    dist = np.abs(proj[:, None] - d_scores[None, :])

    return f32c(dist)


def build_dynamic_candidate_mask(iou_matrix, cd_matrix, min_iou_gate, max_center_frac):
    T, D = iou_matrix.shape
    if T == 0 or D == 0:
        return np.zeros((T, D), dtype=bool)

    DYN_IOU_MARGIN = 0.08
    TOPK_PER_ROW = 6
    CENTER_GATE = max(max_center_frac * 1.5, 0.09)

    best_iou = iou_matrix.max(axis=1)
    dyn_thr = np.maximum(min_iou_gate, best_iou - DYN_IOU_MARGIN)

    cand = (iou_matrix >= dyn_thr[:, None]) | (cd_matrix <= CENTER_GATE)

    for i in range(T):
        if D <= TOPK_PER_ROW:
            cand[i, :] = True
        else:
            kk = np.argpartition(iou_matrix[i], -(TOPK_PER_ROW))[-TOPK_PER_ROW:]
            cand[i, kk] = True

    return cand


def anti_swap_refine_with_angle(
    fused_cost,
    matches,
    trk_tlwh,
    det_tlwh,
    vel_xy,
    angle_gain=0.10,
    gain=0.12,
    max_pairs=20,
):
    if len(matches) < 2:
        return matches

    M = matches.copy()
    T = trk_tlwh.shape[0]
    D = det_tlwh.shape[0]
    if T == 0 or D == 0:
        return M

    ang = angle_cost_matrix(trk_tlwh, det_tlwh, vel_xy)

    lim = min(len(M), max_pairs)
    for a in range(lim):
        i1, j1 = M[a]
        for b in range(a + 1, lim):
            i2, j2 = M[b]

            cur = fused_cost[i1, j1] + fused_cost[i2, j2]
            swp = fused_cost[i1, j2] + fused_cost[i2, j1]

            cur_ang = ang[i1, j1] + ang[i2, j2]
            swp_ang = ang[i1, j2] + ang[i2, j1]

            cur2 = cur + angle_gain * cur_ang
            swp2 = swp + angle_gain * swp_ang

            if swp2 + 1e-12 < cur2 * (1.0 - gain):
                M[a] = np.array([i1, j2])
                M[b] = np.array([i2, j1])

    return M


def stageA_high_association(
    strack_pool,
    detections,
    img_w,
    img_h,
    center_norm,
    min_iou_gate,
    max_center_frac,
    w_assoc_emb,
    fila_k,
    fila_proxy_frac,
    fila_huber_delta,
    anti_swap_gain,
    anti_swap_angle_gain,
    linear_assignment_fn=None,
):
    trk_tlwh = f32c(np.array([st.tlwh for st in strack_pool], dtype=np.float32)) if len(strack_pool) else np.zeros((0, 4), np.float32)
    det_tlwh = f32c(np.array([d.tlwh for d in detections], dtype=np.float32)) if len(detections) else np.zeros((0, 4), np.float32)

    trk_xyxy = tlwh_to_xyxy(trk_tlwh)
    det_xyxy = tlwh_to_xyxy(det_tlwh)

    iou_matrix = iou_xyxy_matrix(trk_xyxy, det_xyxy)
    cd_matrix = center_distance_matrix(trk_tlwh, det_tlwh, center_norm, img_w, img_h)

    trkC = np.zeros((0, 2), np.float32) if trk_tlwh.size == 0 else (trk_tlwh[:, :2] + 0.5 * trk_tlwh[:, 2:4])
    Nlist = knn_idx(trkC, fila_k) if len(strack_pool) else []

    cand_mask = build_dynamic_candidate_mask(iou_matrix, cd_matrix, min_iou_gate, max_center_frac)

    iou_cost = 1.0 - iou_matrix
    base_cost = np.full_like(iou_cost, BIG, dtype=np.float32)
    base_cost[cand_mask] = iou_cost[cand_mask]

    form = formation_cost_sparse(
        trk_tlwh, det_tlwh, img_w, img_h, Nlist,
        iou_matrix, cd_matrix, cand_mask,
        proxy_gate_frac=fila_proxy_frac,
        huber_delta=fila_huber_delta,
        topk=6,
    )

    emb = cosine_cost(strack_pool, detections) if w_assoc_emb > 0.0 else np.zeros_like(base_cost, np.float32)
    if emb.size:
        emb[~cand_mask] = 0.0

    fused = base_cost + 1.25 * form + (w_assoc_emb * emb)
    fused = f32c(nan2big(fused))

    matches = hungarian_min(fused, linear_assignment_fn=linear_assignment_fn)
    vel_xy = estimate_track_velocity_xy(strack_pool)
    if len(matches):
        matches = anti_swap_refine_with_angle(
            fused, matches,
            trk_tlwh, det_tlwh, vel_xy,
            angle_gain=anti_swap_angle_gain,
            gain=anti_swap_gain,
            max_pairs=20,
        )

    return matches, iou_matrix, cd_matrix, trk_tlwh, det_tlwh, Nlist


def stageB_center_rescue(
    remain_trk_idx,
    remain_det_idx,
    cd_matrix,
    max_center_frac,
    linear_assignment_fn=None,
):
    if len(remain_trk_idx) == 0 or len(remain_det_idx) == 0:
        return np.empty((0, 2), dtype=int)

    RESCUE_CENTER_FRAC = max(max_center_frac, 0.06)
    cd_sub = cd_matrix[np.ix_(remain_trk_idx, remain_det_idx)]
    gate = cd_sub <= RESCUE_CENTER_FRAC
    cost = np.where(gate, cd_sub, BIG)
    cost = f32c(nan2big(cost))

    sub = hungarian_min(cost, linear_assignment_fn=linear_assignment_fn)
    out = np.array([[remain_trk_idx[i], remain_det_idx[j]] for i, j in sub], dtype=int)
    return out


def stage2_medium_iou_with_conf(
    tracked_stracks,
    dets_med,
    conf_consistency_w=0.10,
    linear_assignment_fn=None,
):
    T = len(tracked_stracks)
    D = len(dets_med)
    if T == 0 or D == 0:
        return np.empty((0, 2), dtype=int), list(range(T)), list(range(D))

    trk_tlwh = f32c(np.array([t.tlwh for t in tracked_stracks], dtype=np.float32))
    det_tlwh = f32c(np.array([d.tlwh for d in dets_med], dtype=np.float32))

    iou = iou_xyxy_matrix(tlwh_to_xyxy(trk_tlwh), tlwh_to_xyxy(det_tlwh))
    iou_cost = 1.0 - iou

    conf_cost = confidence_consistency_cost(tracked_stracks, dets_med)
    cost = 0.90 * iou_cost + conf_consistency_w * conf_cost
    cost[iou <= 0.10] = 1.0
    cost = np.clip(cost, 0.0, 1.0).astype(np.float32)

    matches = hungarian_min(cost, linear_assignment_fn=linear_assignment_fn)

    matched_t = set(matches[:, 0].tolist()) if len(matches) else set()
    matched_d = set(matches[:, 1].tolist()) if len(matches) else set()

    u_tracks = [i for i in range(T) if i not in matched_t]
    u_dets = [j for j in range(D) if j not in matched_d]

    return matches, u_tracks, u_dets
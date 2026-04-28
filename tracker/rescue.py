import numpy as np
from .utils import f32c, compute_centers


def duplicate_detection_sharing(
    unmatched_trk_idx,
    matched_pairs,
    trk_tlwh,
    det_tlwh,
    iou_matrix,
    velocity_xy=None,
    max_share=1,
    iou_thr=0.73,
):
    det_tlwh = np.asarray(det_tlwh, dtype=np.float32)
    if det_tlwh.size == 0 or len(unmatched_trk_idx) == 0:
        return [], []

    inv = {}
    for ti, dj in matched_pairs.items():
        inv.setdefault(dj, []).append(ti)

    pseudo = []
    owners = []
    for i in unmatched_trk_idx:
        if iou_matrix.shape[1] == 0:
            continue
        j_best = int(np.argmax(iou_matrix[i]))
        if j_best < 0 or iou_matrix[i, j_best] < iou_thr:
            continue

        used = inv.get(j_best, [])
        if len(used) >= max_share:
            continue

        tlwh = det_tlwh[j_best].copy()
        if velocity_xy is not None and i < velocity_xy.shape[0]:
            v = velocity_xy[i]
            n = float(np.linalg.norm(v))
            if n > 1e-6:
                v = v / n
                tlwh[0] += 1.0 * v[0]
                tlwh[1] += 1.0 * v[1]

        pseudo.append(tlwh)
        owners.append(i)

    return pseudo, owners


def procrustes_sim(P, Q):
    if P.shape[0] == 0:
        return 1.0, np.eye(2, dtype=np.float32)
    H = P.T @ Q
    U, S, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T
    denom = (P ** 2).sum()
    s = float(S.sum() / max(denom, 1e-12))
    return s, R.astype(np.float32)


def formation_proxy_for_track(
    i,
    Nlist,
    matched_pairs,
    trkC,
    detC,
    trk_tlwh,
    img_w,
    img_h,
    k_min=2,
    huber_delta=0.02,
    kappa=2.0,
):
    neigh = [n for n in Nlist[i] if n in matched_pairs]
    if len(neigh) < k_min:
        return None

    diag = float(np.hypot(img_w, img_h))
    P_pts = trkC[neigh]
    Q_pts = detC[[matched_pairs[n] for n in neigh]]
    Pzc = P_pts - P_pts.mean(0, keepdims=True)
    Qzc = Q_pts - Q_pts.mean(0, keepdims=True)

    s, R = procrustes_sim(Pzc, Qzc)
    resid = np.linalg.norm(Qzc - s * (Pzc @ R.T), axis=1)
    med_resid = float(np.median(resid) / max(diag, 1e-6))
    if med_resid > kappa * huber_delta:
        return None

    c_i = trkC[i]
    c_hat = s * ((c_i - P_pts.mean(0)) @ R.T) + Q_pts.mean(0)
    if not (0.0 <= c_hat[0] <= float(img_w) and 0.0 <= c_hat[1] <= float(img_h)):
        return None

    w = trk_tlwh[i, 2]
    h = trk_tlwh[i, 3]
    return np.array([c_hat[0] - 0.5 * w, c_hat[1] - 0.5 * h, w, h], dtype=np.float32)


def spawn_fgpd(
    unmatched_trk_idx,
    Nlist,
    matched_pairs,
    trk_tlwh,
    det_tlwh,
    img_w,
    img_h,
    k_min=2,
    huber_delta=0.02,
    kappa=2.0,
):
    trkC = compute_centers(trk_tlwh)
    detC = compute_centers(det_tlwh) if det_tlwh.size else np.zeros((0, 2), np.float32)

    pseudo = []
    owners = []
    for i in unmatched_trk_idx:
        tlwh = formation_proxy_for_track(
            i,
            Nlist,
            matched_pairs,
            trkC,
            detC,
            trk_tlwh,
            img_w,
            img_h,
            k_min=k_min,
            huber_delta=huber_delta,
            kappa=kappa,
        )
        if tlwh is not None:
            pseudo.append(tlwh)
            owners.append(i)

    return pseudo, owners

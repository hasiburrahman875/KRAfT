import numpy as np
from .utils import f32c, compute_centers


def knn_idx(points, k):
    points = np.asarray(points, dtype=np.float32)
    N = points.shape[0]
    if N == 0:
        return []
    d2 = ((points[:, None, :] - points[None, :, :]) ** 2).sum(-1)
    np.fill_diagonal(d2, np.inf)
    k_eff = min(k, max(N - 1, 1))
    idx = np.argpartition(d2, kth=k_eff - 1, axis=1)[:, :k_eff]
    out = []
    for i in range(N):
        neigh = idx[i]
        order = np.argsort(d2[i, neigh])
        out.append(neigh[order].tolist())
    return out


def huber_scalar(x, d):
    ax = abs(float(x))
    return 0.5 * ax * ax if ax <= d else d * (ax - 0.5 * d)


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


def formation_cost_sparse(
    trk_tlwh,
    det_tlwh,
    img_w,
    img_h,
    Nlist,
    iou_matrix,
    cd_matrix,
    candidate_mask,
    proxy_gate_frac=0.05,
    huber_delta=0.02,
    topk=6,
):
    trk_tlwh = np.asarray(trk_tlwh, dtype=np.float32)
    det_tlwh = np.asarray(det_tlwh, dtype=np.float32)
    iou_matrix = np.asarray(iou_matrix, dtype=np.float32)
    cd_matrix = np.asarray(cd_matrix, dtype=np.float32)
    candidate_mask = candidate_mask.astype(bool) if candidate_mask.size else candidate_mask

    T = trk_tlwh.shape[0]
    D = det_tlwh.shape[0]
    form = np.zeros((T, D), dtype=np.float32)
    if T == 0 or D == 0:
        return form

    diag = float(np.hypot(img_w, img_h))
    tC = compute_centers(trk_tlwh)
    dC = compute_centers(det_tlwh)

    cd_all = np.sqrt(((tC[:, None, :] - dC[None, :, :]) ** 2).sum(-1))
    nearest_det = np.argmin(cd_all, axis=1)
    nearest_dist = cd_all[np.arange(T), nearest_det]
    proxyC = tC.copy()

    gate = proxy_gate_frac * diag
    use_det = nearest_dist <= gate
    proxyC[use_det] = dC[nearest_det[use_det]]

    for i in range(T):
        if not candidate_mask[i].any():
            continue

        neigh = Nlist[i]
        if len(neigh) < 2:
            continue

        P = (tC[i][None, :] - tC[neigh])
        Pz = P - P.mean(0, keepdims=True)
        denom = (Pz ** 2).sum()
        if denom <= 1e-12:
            continue

        neigh_proxy = proxyC[neigh]

        cand = np.where(candidate_mask[i])[0]
        if cand.size == 0:
            continue

        K = min(topk, cand.size)
        sel = cand[np.argpartition(cd_matrix[i, cand], K - 1)[:K]]

        for j in sel:
            Q = (dC[j][None, :] - neigh_proxy)
            Qz = Q - Q.mean(0, keepdims=True)

            H = Pz.T @ Qz
            U, S, Vt = np.linalg.svd(H)
            R = Vt.T @ U.T
            if np.linalg.det(R) < 0:
                Vt[-1, :] *= -1
                R = Vt.T @ U.T
            s = float(S.sum() / denom)

            resid = np.linalg.norm(Qz - s * (Pz @ R.T), axis=1)
            med = float(np.median(resid) / max(diag, 1e-6))
            form[i, j] = huber_scalar(med, huber_delta)

    return f32c(form)

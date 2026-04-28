import numpy as np
from .utils import f32c, compute_centers


def estimate_track_velocity_xy(stracks):
    v = []
    for st in stracks:
        if len(st.xywh_amemory) >= 2:
            a = st.xywh_amemory[-1][:2]
            b = st.xywh_amemory[-2][:2]
            v.append((a - b).astype(np.float32))
        else:
            v.append(np.zeros(2, dtype=np.float32))
    return f32c(np.stack(v, axis=0)) if len(v) else np.zeros((0, 2), np.float32)


def angle_cost_matrix(trk_tlwh, det_tlwh, vel_xy):
    trk_tlwh = np.asarray(trk_tlwh, dtype=np.float32)
    det_tlwh = np.asarray(det_tlwh, dtype=np.float32)
    T = trk_tlwh.shape[0]
    D = det_tlwh.shape[0]
    if T == 0 or D == 0:
        return np.zeros((T, D), dtype=np.float32)

    tC = compute_centers(trk_tlwh)
    dC = compute_centers(det_tlwh)

    disp = dC[None, :, :] - tC[:, None, :]
    disp_norm = np.linalg.norm(disp, axis=-1, keepdims=True) + 1e-12

    v = vel_xy.astype(np.float32)
    v_norm = np.linalg.norm(v, axis=-1, keepdims=True) + 1e-12
    v_unit = v / v_norm
    v_unit = v_unit[:, None, :]

    disp_unit = disp / disp_norm

    cos = (v_unit * disp_unit).sum(-1)
    cos = np.clip(cos, -1.0, 1.0)
    ang = np.arccos(cos) / np.pi

    bad_static = (v_norm[:, 0] < 1e-3).astype(np.float32)
    ang = ang * (1.0 - bad_static[:, None])

    return f32c(ang.astype(np.float32))

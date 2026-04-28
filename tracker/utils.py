import numpy as np

BIG = 1e6


def f32c(a):
    return np.ascontiguousarray(a, dtype=np.float32)


def nan2big(a, big=BIG):
    return np.nan_to_num(a, nan=big, posinf=big, neginf=big)


def tlwh_to_xyxy(tlwh):
    tlwh = np.asarray(tlwh, dtype=np.float32)
    if tlwh.size == 0:
        return tlwh.reshape(0, 4)
    x = tlwh[:, 0]
    y = tlwh[:, 1]
    w = tlwh[:, 2]
    h = tlwh[:, 3]
    out = np.stack([x, y, x + w, y + h], axis=1)
    return f32c(out)


def tlbr_to_tlwh(tlbr):
    t = np.asarray(tlbr, dtype=np.float32).copy()
    t[2] = t[2] - t[0]
    t[3] = t[3] - t[1]
    return t


def tlwh_to_tlbr(tlwh):
    t = np.asarray(tlwh, dtype=np.float32).copy()
    t[2] = t[0] + t[2]
    t[3] = t[1] + t[3]
    return t


def compute_centers(tlwh):
    tlwh = np.asarray(tlwh, dtype=np.float32)
    if tlwh.size == 0:
        return tlwh.reshape(0, 2)
    c = tlwh.copy()
    c[:, 0] = c[:, 0] + 0.5 * c[:, 2]
    c[:, 1] = c[:, 1] + 0.5 * c[:, 3]
    return f32c(c[:, :2])


def center_distance_matrix(trk_tlwh, det_tlwh, norm, img_w, img_h):
    trk_tlwh = np.asarray(trk_tlwh, dtype=np.float32)
    det_tlwh = np.asarray(det_tlwh, dtype=np.float32)
    T = trk_tlwh.shape[0]
    D = det_tlwh.shape[0]
    if T == 0 or D == 0:
        return np.zeros((T, D), dtype=np.float32)

    tC = compute_centers(trk_tlwh)
    dC = compute_centers(det_tlwh)
    diffs = tC[:, None, :] - dC[None, :, :]
    dist = np.sqrt((diffs ** 2).sum(-1)).astype(np.float32)

    if norm == "pairwise_box":
        t_hw = trk_tlwh[:, 2:4]
        d_hw = det_tlwh[:, 2:4]
        t_s = np.maximum(t_hw[:, 0], t_hw[:, 1])[:, None]
        d_s = np.maximum(d_hw[:, 0], d_hw[:, 1])[None, :]
        scale = np.maximum(np.maximum(t_s, d_s), 1e-6)
        return f32c(dist / scale)

    diag = float(np.hypot(img_w, img_h))
    return f32c(dist / max(diag, 1e-6))


def iou_xyxy_matrix(trk_xyxy, det_xyxy):
    trk_xyxy = np.asarray(trk_xyxy, dtype=np.float32)
    det_xyxy = np.asarray(det_xyxy, dtype=np.float32)

    T = trk_xyxy.shape[0]
    D = det_xyxy.shape[0]
    if T == 0 or D == 0:
        return np.zeros((T, D), dtype=np.float32)

    t = trk_xyxy[:, None, :]
    d = det_xyxy[None, :, :]

    inter_x1 = np.maximum(t[..., 0], d[..., 0])
    inter_y1 = np.maximum(t[..., 1], d[..., 1])
    inter_x2 = np.minimum(t[..., 2], d[..., 2])
    inter_y2 = np.minimum(t[..., 3], d[..., 3])

    inter_w = np.maximum(0.0, inter_x2 - inter_x1)
    inter_h = np.maximum(0.0, inter_y2 - inter_y1)
    inter = inter_w * inter_h

    area_t = (t[..., 2] - t[..., 0]) * (t[..., 3] - t[..., 1])
    area_d = (d[..., 2] - d[..., 0]) * (d[..., 3] - d[..., 1])
    union = np.maximum(area_t + area_d - inter, 1e-6)

    return f32c((inter / union).astype(np.float32))

import numpy as np

def to_det_array(dets, name="dets", allow_k=(4, 5, 6)):
    if dets is None:
        return np.zeros((0, 5), dtype=np.float32)

    arr = np.asarray(dets, dtype=np.float32)
    if arr.size == 0:
        return np.zeros((0, 5), dtype=np.float32)

    if arr.ndim == 2:
        if arr.shape[1] in allow_k:
            return arr
        arr = arr.reshape(-1)

    if arr.ndim == 1:
        for k in allow_k:
            if arr.size % k == 0:
                return arr.reshape(-1, k)

    raise ValueError(
        f"{name} has unexpected shape ndim {arr.ndim} size {arr.size}"
    )


def xywh_to_xyxy(xywh):
    xywh = np.asarray(xywh, dtype=np.float32)
    if xywh.size == 0:
        return xywh.reshape(0, 4)
    out = xywh.copy()
    out[:, 2] = out[:, 0] + out[:, 2]
    out[:, 3] = out[:, 1] + out[:, 3]
    return out


def iou_xyxy(a, b):
    # a N 4, b M 4
    if a.size == 0 or b.size == 0:
        return np.zeros((a.shape[0], b.shape[0]), dtype=np.float32)

    a = a[:, None, :]
    b = b[None, :, :]

    inter_x1 = np.maximum(a[..., 0], b[..., 0])
    inter_y1 = np.maximum(a[..., 1], b[..., 1])
    inter_x2 = np.minimum(a[..., 2], b[..., 2])
    inter_y2 = np.minimum(a[..., 3], b[..., 3])

    inter_w = np.maximum(0.0, inter_x2 - inter_x1)
    inter_h = np.maximum(0.0, inter_y2 - inter_y1)
    inter = inter_w * inter_h

    area_a = (a[..., 2] - a[..., 0]) * (a[..., 3] - a[..., 1])
    area_b = (b[..., 2] - b[..., 0]) * (b[..., 3] - b[..., 1])
    union = np.maximum(area_a + area_b - inter, 1e-6)

    return (inter / union).astype(np.float32)


def nms_xywh(dets_xywhs, iou_thr=0.95, max_keep=5000):
    """
    Class agnostic NMS on xywh score.
    dets shape N 5 with x y w h score.
    Uses a high iou_thr by default to keep near duplicates.
    """
    if dets_xywhs is None:
        return np.zeros((0, 5), dtype=np.float32)

    dets = np.asarray(dets_xywhs, dtype=np.float32)
    if dets.size == 0:
        return dets.reshape(0, 5)

    if dets.shape[1] != 5:
        raise ValueError("nms_xywh expects N x 5")

    scores = dets[:, 4]
    order = np.argsort(-scores)

    dets_xyxy = xywh_to_xyxy(dets[:, :4])

    keep = []
    while order.size > 0 and len(keep) < max_keep:
        i = order[0]
        keep.append(i)
        if order.size == 1:
            break

        rest = order[1:]
        ious = iou_xyxy(dets_xyxy[i:i+1], dets_xyxy[rest]).reshape(-1)
        order = rest[ious <= iou_thr]

    return dets[keep]

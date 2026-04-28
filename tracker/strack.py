import numpy as np
from collections import deque

from tracking_utils.kalman_filter import KalmanFilter
from .utils import f32c


class TrackState:
    New = 0
    Tracked = 1
    Lost = 2
    Removed = 3


class BaseTrack:
    _count = 0

    @classmethod
    def next_id(cls):
        cls._count += 1
        return cls._count

    def mark_lost(self):
        self.state = TrackState.Lost

    def mark_removed(self):
        self.state = TrackState.Removed


class STrack(BaseTrack):
    def __init__(self, tlwh, score, temp_feat=None, buffer_size=30):
        self._tlwh = np.asarray(tlwh, dtype=np.float32)

        self.score = float(score)
        self.score_memory = deque([], maxlen=buffer_size)
        self.score_memory.append(self.score)

        self.features = deque([], maxlen=buffer_size)
        self.emb = temp_feat

        self.xywh_omemory = deque([], maxlen=buffer_size)
        self.xywh_pmemory = deque([], maxlen=buffer_size)
        self.xywh_amemory = deque([], maxlen=buffer_size)
        self.conds = deque([], maxlen=5)

        self.track_id = 0
        self.state = TrackState.New
        self.is_activated = False
        self.frame_id = 0
        self.start_frame = 0
        self.end_frame = 0
        self.tracklet_len = 0

        self.unobserved = False
        self.miss_streak = 0

        self.kf_center = KalmanFilter()
        self.mean = None
        self.covariance = None
        self.kf_inited = False

    def update_features(self, feat, alpha=0.95):
        if feat is None:
            return
        if self.emb is None:
            self.emb = feat
        else:
            self.emb = alpha * self.emb + (1.0 - alpha) * feat
        n = np.linalg.norm(self.emb) + 1e-12
        self.emb = self.emb / n

    @property
    def tlwh(self):
        return self._tlwh.copy()

    @property
    def tlbr(self):
        t = self.tlwh
        t[2:] += t[:2]
        return t

    @property
    def xywh(self):
        t = self.tlwh
        t[:2] = t[:2] + t[2:] / 2
        return t

    @staticmethod
    def tlwh_to_xyah(tlwh):
        ret = np.asarray(tlwh, dtype=np.float32).copy()
        ret[:2] += ret[2:] / 2
        ret[2] /= (ret[3] + 1e-12)
        return ret

    def _kf_measure_xyah(self):
        cx, cy, w, h = map(float, self.xywh)
        a = w / max(h, 1e-12)
        return np.array([cx, cy, a, h], dtype=np.float32)

    def kf_update_with_measurement(self):
        z = self._kf_measure_xyah()
        if (not self.kf_inited) or (self.mean is None) or (self.covariance is None):
            self.mean, self.covariance = self.kf_center.initiate(z)
            self.kf_inited = True
        else:
            self.mean, self.covariance = self.kf_center.predict(self.mean.copy(), self.covariance.copy())
            self.mean, self.covariance = self.kf_center.update(self.mean, self.covariance, z)

    def kf_predict_delta_xywh(self):
        if not self.kf_inited or self.mean is None or self.covariance is None:
            return np.zeros(4, dtype=np.float32)
        mean1, cov1 = self.kf_center.predict(self.mean.copy(), self.covariance.copy())
        z_pred, _ = self.kf_center.project(mean1, cov1)
        cx_p, cy_p, a_p, h_p = map(float, z_pred[:4])
        w_p = a_p * h_p
        cx, cy, w, h = map(float, self.xywh)
        return np.array([cx_p - cx, cy_p - cy, w_p - w, h_p - h], dtype=np.float32)

    @staticmethod
    def multi_predict_kdf(stracks, model, img_w, img_h, gamma=1.0, lambda_kf=1.0, residual_is_normalized=False):
        N = len(stracks)
        if N == 0:
            return

        W = float(max(img_w, 1e-6))
        H = float(max(img_h, 1e-6))

        cur_xywh = np.asarray([st.xywh for st in stracks], dtype=np.float32)

        prev_xywh = np.asarray([(st.xywh_amemory[-1] if len(st.xywh_amemory) >= 1 else st.xywh)
                                for st in stracks], dtype=np.float32)
        prev2_xywh = np.asarray([(st.xywh_amemory[-2] if len(st.xywh_amemory) >= 2 else prev_xywh[i])
                                 for i, st in enumerate(stracks)], dtype=np.float32)
        prev_speed = np.linalg.norm(prev_xywh[:, :2] - prev2_xywh[:, :2], axis=1)

        kf_motion = np.vstack([st.kf_predict_delta_xywh() for st in stracks]).astype(np.float32)

        residual = np.zeros_like(cur_xywh, dtype=np.float32)
        try:
            conds_list = [np.asarray(st.conds, dtype=np.float32) if len(st.conds)
                          else np.zeros((1, 8), np.float32) for st in stracks]
            gen = model.generate(conds_list, sample=1, bestof=True, img_w=img_w, img_h=img_h)
            gen = np.asarray(gen, dtype=np.float32)
            if gen.ndim == 3:
                residual = gen.mean(0)
            elif gen.ndim == 2:
                residual = gen
            if residual.shape != cur_xywh.shape:
                residual = np.zeros_like(cur_xywh, dtype=np.float32)
        except Exception:
            residual = np.zeros_like(cur_xywh, dtype=np.float32)

        residual = f32c(residual)
        if residual_is_normalized:
            residual *= np.array([W, H, W, H], dtype=np.float32)

        delta = lambda_kf * kf_motion + gamma * residual
        fused_xywh = cur_xywh + delta

        max_move = np.maximum(3.0, 3.5 * prev_speed)[:, None]
        dc = fused_xywh[:, :2] - cur_xywh[:, :2]
        dc = np.clip(dc, -max_move, max_move)
        fused_xywh[:, :2] = cur_xywh[:, :2] + dc

        rel = 0.45
        fused_xywh[:, 2] = np.clip(fused_xywh[:, 2], (1.0 - rel) * cur_xywh[:, 2], (1.0 + rel) * cur_xywh[:, 2])
        fused_xywh[:, 3] = np.clip(fused_xywh[:, 3], (1.0 - rel) * cur_xywh[:, 3], (1.0 + rel) * cur_xywh[:, 3])

        fused_xywh[:, 2] = np.clip(fused_xywh[:, 2], 1e-3, W)
        fused_xywh[:, 3] = np.clip(fused_xywh[:, 3], 1e-3, H)
        fused_xywh[:, 0] = np.clip(fused_xywh[:, 0], 0.0, W)
        fused_xywh[:, 1] = np.clip(fused_xywh[:, 1], 0.0, H)

        tlwh = np.empty_like(fused_xywh, dtype=np.float32)
        tlwh[:, 0] = fused_xywh[:, 0] - 0.5 * fused_xywh[:, 2]
        tlwh[:, 1] = fused_xywh[:, 1] - 0.5 * fused_xywh[:, 3]
        tlwh[:, 2] = fused_xywh[:, 2]
        tlwh[:, 3] = fused_xywh[:, 3]

        for st, box in zip(stracks, tlwh):
            st._tlwh = box

            st.xywh_pmemory.append(st.xywh.copy())
            st.xywh_amemory.append(st.xywh.copy())

            if len(st.xywh_amemory) >= 2:
                tmp_delta = st.xywh.copy() - st.xywh_amemory[-2].copy()
            else:
                tmp_delta = np.zeros(4, dtype=np.float32)

            st.conds.append(np.concatenate((st.xywh.copy(), tmp_delta)))

    def activate(self, frame_id):
        self.track_id = self.next_id()
        self.state = TrackState.Tracked
        self.is_activated = True if frame_id == 1 else True
        self.frame_id = frame_id
        self.start_frame = frame_id
        self.end_frame = frame_id
        self.tracklet_len = 0

        self.xywh_omemory.append(self.xywh.copy())
        self.xywh_pmemory.append(self.xywh.copy())
        self.xywh_amemory.append(self.xywh.copy())

        delta_bbox = np.zeros_like(self.xywh.copy())
        self.conds.append(np.concatenate((self.xywh.copy(), delta_bbox)))

        self.unobserved = False
        self.miss_streak = 0

        self.kf_update_with_measurement()

    def re_activate(self, new_track, frame_id, new_id=False):
        self._tlwh = new_track.tlwh
        self.frame_id = frame_id
        self.end_frame = frame_id
        self.tracklet_len = 0

        self.state = TrackState.Tracked
        self.is_activated = True
        self.unobserved = False
        self.miss_streak = 0

        if new_id:
            self.track_id = self.next_id()

        self.score = float(new_track.score)
        self.score_memory.append(self.score)

        self.xywh_omemory.append(self.xywh.copy())
        self.xywh_amemory.append(self.xywh.copy())

        if len(self.xywh_amemory) >= 2:
            tmp_delta = self.xywh.copy() - self.xywh_amemory[-2].copy()
        else:
            tmp_delta = np.zeros_like(self.xywh.copy())

        self.conds.append(np.concatenate((self.xywh.copy(), tmp_delta)))
        self.kf_update_with_measurement()

    def update(self, new_track, frame_id, update_feature=True):
        self.frame_id = frame_id
        self.end_frame = frame_id
        self.tracklet_len += 1

        self._tlwh = new_track.tlwh
        self.state = TrackState.Tracked
        self.is_activated = True
        self.unobserved = False
        self.miss_streak = 0

        self.score = float(new_track.score)
        self.score_memory.append(self.score)

        self.xywh_omemory.append(self.xywh.copy())
        self.xywh_amemory.append(self.xywh.copy())

        if len(self.xywh_amemory) >= 2:
            tmp_delta = self.xywh.copy() - self.xywh_amemory[-2].copy()
        else:
            tmp_delta = np.zeros_like(self.xywh.copy())

        self.conds.append(np.concatenate((self.xywh.copy(), tmp_delta)))

        if update_feature and getattr(new_track, "emb", None) is not None:
            self.update_features(new_track.emb, alpha=0.95)

        self.kf_update_with_measurement()
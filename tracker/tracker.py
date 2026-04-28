import numpy as np

from .strack import STrack, TrackState
from .utils import f32c, tlbr_to_tlwh
from .association import (
    stageA_high_association,
    stageB_center_rescue,
    stage2_medium_iou_with_conf,
)
from .formation import knn_idx
from .rescue import duplicate_detection_sharing, spawn_fgpd
from .motion_cues import estimate_track_velocity_xy
from .track_aware_nms import track_aware_nms_before_births
from .deleted_recovery import find_deleted_detections


class KRAfTTracker:
    def __init__(self, config, embedder=None, frame_rate=30):
        self.config = config
        self.embedder = embedder

        self.tracked_stracks = []
        self.lost_stracks = []
        self.removed_stracks = []

        self.frame_id = 0
        self.det_thresh = float(getattr(config, "high_thres", 0.6))
        self.low_thres = float(getattr(config, "low_thres", 0.1))
        self.med_thres = float(getattr(config, "med_thres", 0.3))

        self.buffer_size = int(frame_rate / 30.0 * 30)
        self.max_time_lost = int(getattr(config, "max_time_lost", self.buffer_size))

        self.center_norm = getattr(config, "center_norm", "pairwise_box")
        self.max_center_frac = float(getattr(config, "max_center_frac", 0.06))
        self.min_iou_gate = float(getattr(config, "min_iou_gate", 0.10))

        self.w_assoc_emb = float(getattr(config, "w_assoc_emb", 0.0))

        self.fila_k = int(getattr(config, "fila_k", 3))
        self.fila_proxy_frac = float(getattr(config, "fila_proxy_frac", 0.05))
        self.fila_huber_delta = float(getattr(config, "fila_huber_delta", 0.02))

        self.anti_swap_gain = float(getattr(config, "anti_swap_gain", 0.12))
        self.anti_swap_angle_gain = float(getattr(config, "anti_swap_angle_gain", 0.10))

        self.dup_iou_thr = float(getattr(config, "dup_iou_thr", 0.73))
        self.dup_max_share = int(getattr(config, "dup_max_share", 1))

        self.fgpd_kmin = int(getattr(config, "fgpd_kmin", 2))
        self.fgpd_ttl = int(getattr(config, "fgpd_ttl", 6))
        self.fgpd_kappa = float(getattr(config, "fgpd_kappa", 2.0))

        self.support_zone_frac = float(getattr(config, "support_zone_frac", 0.06))
        self.support_iou_gate = float(getattr(config, "support_iou_gate", 0.10))
        self.support_bonus = float(getattr(config, "support_bonus", 0.10))

        self.track_aware_birth_thr = float(getattr(config, "tai_thr", 0.75))
        self.track_birth_score_thr = float(getattr(config, "init_thr", self.det_thresh))

        self.residual_is_normalized = bool(getattr(config, "residual_is_normalized", False))
        self.lambda_kf = float(getattr(config, "lambda_kf", 1.0))
        self.gamma = float(getattr(config, "gamma", 1.0))

        self.conf_consistency_w = float(getattr(config, "conf_consistency_w", 0.10))

        self.linear_assignment_fn = getattr(config, "linear_assignment_fn", None)

        # deleted recovery option 1 controls
        self.ultra_low_thres = float(getattr(config, "ultra_low_thres", 0.02))
        self.deleted_backup_iou = float(getattr(config, "deleted_backup_iou", 0.95))
        self.deleted_keep_iou = float(getattr(config, "deleted_keep_iou", 0.97))
        self.deleted_backup_max = int(getattr(config, "deleted_backup_max", 5000))

    # ------------------------------
    # small local helpers to keep this file robust
    # ------------------------------
    def _safe_reshape_dets(self, dets, name="dets"):
        if dets is None:
            return np.zeros((0, 5), dtype=np.float32)

        arr = np.asarray(dets, dtype=np.float32)
        if arr.size == 0:
            return np.zeros((0, 5), dtype=np.float32)

        if arr.ndim == 2:
            if arr.shape[1] == 5:
                return arr
            if arr.shape[1] == 6:
                # common layout: [cls,x,y,w,h,score] or [id,x,y,w,h,score]
                # caller in your eval already slices to 5, but keep safe
                return arr[:, 1:6]
            if arr.shape[1] == 4:
                scores = np.ones((arr.shape[0], 1), dtype=np.float32)
                return np.concatenate([arr, scores], axis=1)

            arr = arr.reshape(-1)

        if arr.ndim == 1:
            if arr.size % 5 == 0:
                return arr.reshape(-1, 5)
            if arr.size % 6 == 0:
                tmp = arr.reshape(-1, 6)
                return tmp[:, 1:6]
            if arr.size % 4 == 0:
                tmp = arr.reshape(-1, 4)
                scores = np.ones((tmp.shape[0], 1), dtype=np.float32)
                return np.concatenate([tmp, scores], axis=1)

        raise ValueError(f"{name} has unexpected shape with size {arr.size}")

    def _xywh_to_tlbr(self, dets_xywhs):
        dets = dets_xywhs.copy()
        if dets.size:
            dets[:, 2] = dets[:, 0] + dets[:, 2]
            dets[:, 3] = dets[:, 1] + dets[:, 3]
        return dets

    def _iou_tlbr_matrix(self, a, b):
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

        area_a = np.maximum(0.0, (a[..., 2] - a[..., 0])) * np.maximum(0.0, (a[..., 3] - a[..., 1]))
        area_b = np.maximum(0.0, (b[..., 2] - b[..., 0])) * np.maximum(0.0, (b[..., 3] - b[..., 1]))
        union = np.maximum(area_a + area_b - inter, 1e-6)

        return (inter / union).astype(np.float32)

    def _loose_nms_tlbr(self, dets_tlbr, iou_thr=0.95, max_keep=5000):
        if dets_tlbr is None:
            return np.zeros((0, 5), dtype=np.float32)

        dets = np.asarray(dets_tlbr, dtype=np.float32)
        if dets.size == 0:
            return dets.reshape(0, 5)

        scores = dets[:, 4]
        order = np.argsort(-scores)

        keep = []
        while order.size > 0 and len(keep) < max_keep:
            i = int(order[0])
            keep.append(i)
            if order.size == 1:
                break

            rest = order[1:]
            ious = self._iou_tlbr_matrix(dets[i:i+1, :4], dets[rest, :4]).reshape(-1)
            order = rest[ious <= iou_thr]

        return dets[keep]

    def _build_backup_from_main_xywh(self, dets_xywhs):
        if dets_xywhs.size == 0:
            return np.zeros((0, 5), dtype=np.float32)

        cand = dets_xywhs[dets_xywhs[:, 4] >= self.ultra_low_thres]
        if cand.size == 0:
            return np.zeros((0, 5), dtype=np.float32)

        cand_tlbr = self._xywh_to_tlbr(cand)
        # keep near duplicates, only remove extreme exact overlaps
        backup = self._loose_nms_tlbr(
            cand_tlbr,
            iou_thr=self.deleted_backup_iou,
            max_keep=self.deleted_backup_max,
        )
        return backup

    def _compute_embs(self, img, tlbr, tag):
        if self.embedder is None or img is None or tlbr.size == 0:
            return [None] * tlbr.shape[0]
        embs = self.embedder.compute_embedding(img, tlbr, tag)
        return [e for e in embs]

    # ------------------------------
    # main update
    # ------------------------------
    def update(self, dets_xywhs, model, img_w, img_h, tag="", img=None, dets_xywhs_95=None):
        self.frame_id += 1

        # 1) robust input handling
        dets_xywhs = self._safe_reshape_dets(dets_xywhs, name="dets_xywhs")

        if dets_xywhs.size:
            good = np.isfinite(dets_xywhs).all(axis=1)
            whp = (dets_xywhs[:, 2] > 0) & (dets_xywhs[:, 3] > 0)
            dets_xywhs = dets_xywhs[good & whp]

        dets_xywhs = f32c(dets_xywhs)

        # convert to tlbr for internal use
        dets_tlbr = self._xywh_to_tlbr(dets_xywhs)
        dets_all = dets_tlbr.copy()

        # 2) tier split
        dets_high_mask = dets_all[:, 4] >= self.det_thresh if dets_all.size else np.array([], dtype=bool)
        dets_med_mask = (
            (dets_all[:, 4] >= self.med_thres) & (dets_all[:, 4] < self.det_thresh)
        ) if dets_all.size else np.array([], dtype=bool)
        dets_low_mask = (
            (dets_all[:, 4] >= self.low_thres) & (dets_all[:, 4] < self.med_thres)
        ) if dets_all.size else np.array([], dtype=bool)

        dets_high = dets_all[dets_high_mask] if dets_all.size else np.zeros((0, 5), np.float32)
        dets_med = dets_all[dets_med_mask] if dets_all.size else np.zeros((0, 5), np.float32)
        dets_low = dets_all[dets_low_mask] if dets_all.size else np.zeros((0, 5), np.float32)

        # 3) deleted detection recovery option 1
        # build or normalize backup set in tlbr
        if dets_xywhs_95 is None:
            dets_95 = self._build_backup_from_main_xywh(dets_xywhs)
        else:
            dets_xywhs_95 = self._safe_reshape_dets(dets_xywhs_95, name="dets_xywhs_95")
            dets_xywhs_95 = f32c(dets_xywhs_95)
            dets_95 = self._xywh_to_tlbr(dets_xywhs_95)

        deleted_extra = np.zeros((0, 5), np.float32)
        if dets_95.size:
            # compare against the full main set or high set
            # using high is closer to TrackTrack spirit for recovery
            deleted_extra = find_deleted_detections(dets_high, dets_95, iou_keep=self.deleted_keep_iou)

            if deleted_extra.size:
                m = (deleted_extra[:, 4] >= self.low_thres) & (deleted_extra[:, 4] < self.med_thres)
                deleted_extra = deleted_extra[m]
            else:
                deleted_extra = np.zeros((0, 5), np.float32)

        dets_low_support = (
            np.concatenate([dets_low, deleted_extra], axis=0)
            if (dets_low.size or deleted_extra.size)
            else np.zeros((0, 5), np.float32)
        )

        # 4) embeddings
        need_emb_high = (self.w_assoc_emb > 0.0)
        embs_high = (
            self._compute_embs(img, dets_high[:, :4], tag + "_high")
            if (need_emb_high and dets_high.shape[0])
            else [None] * dets_high.shape[0]
        )
        embs_med = (
            self._compute_embs(img, dets_med[:, :4], tag + "_med")
            if (dets_med.shape[0] and self.embedder is not None)
            else [None] * dets_med.shape[0]
        )

        # 5) wrap as STrack detections
        detections_high = [
            STrack(tlbr_to_tlwh(d[:4]), float(d[4]), f, 30)
            for d, f in zip(dets_high, embs_high)
        ]
        detections_med = [
            STrack(tlbr_to_tlwh(d[:4]), float(d[4]), f, 30)
            for d, f in zip(dets_med, embs_med)
        ]
        detections_low_support = [
            STrack(tlbr_to_tlwh(d[:4]), float(d[4]), None, 30)
            for d in dets_low_support
        ]

        # 6) split tracked and unconfirmed
        unconfirmed = []
        tracked = []
        for t in self.tracked_stracks:
            if t.is_activated:
                tracked.append(t)
            else:
                unconfirmed.append(t)

        strack_pool = tracked + self.lost_stracks

        # 7) KDF prediction
        STrack.multi_predict_kdf(
            strack_pool,
            model,
            img_w,
            img_h,
            gamma=self.gamma,
            lambda_kf=self.lambda_kf,
            residual_is_normalized=self.residual_is_normalized,
        )

        activated = []
        refind = []
        lost = []
        removed = []

        # 8) Stage A high association with FPC and Anti Swap
        matches_A, iou_matrix, cd_matrix, trk_tlwh, det_tlwh, Nlist = stageA_high_association(
            strack_pool,
            detections_high,
            img_w,
            img_h,
            self.center_norm,
            self.min_iou_gate,
            self.max_center_frac,
            self.w_assoc_emb,
            self.fila_k,
            self.fila_proxy_frac,
            self.fila_huber_delta,
            self.anti_swap_gain,
            self.anti_swap_angle_gain,
            linear_assignment_fn=self.linear_assignment_fn,
        )

        picked_trk = set(matches_A[:, 0].tolist()) if len(matches_A) else set()
        picked_det = set(matches_A[:, 1].tolist()) if len(matches_A) else set()

        remain_trk_idx = [i for i in range(len(strack_pool)) if i not in picked_trk]
        remain_det_idx = [j for j in range(len(detections_high)) if j not in picked_det]

        # 9) Stage B center rescue
        matches_B = stageB_center_rescue(
            remain_trk_idx,
            remain_det_idx,
            cd_matrix,
            self.max_center_frac,
            linear_assignment_fn=self.linear_assignment_fn,
        )

        matched_indices = (
            np.vstack([matches_A, matches_B])
            if (len(matches_A) or len(matches_B))
            else np.empty((0, 2), dtype=int)
        )

        unmatched_dets_high = (
            [j for j in range(len(detections_high)) if j not in matched_indices[:, 1]]
            if matched_indices.size
            else list(range(len(detections_high)))
        )
        unmatched_trk = (
            [i for i in range(len(strack_pool)) if i not in matched_indices[:, 0]]
            if matched_indices.size
            else list(range(len(strack_pool)))
        )

        # 10) safety keep
        if iou_matrix.size and matched_indices.size:
            keep = []
            for m in matched_indices:
                if float(iou_matrix[m[0], m[1]]) < 0.10:
                    unmatched_dets_high.append(int(m[1]))
                    unmatched_trk.append(int(m[0]))
                else:
                    keep.append(m.reshape(1, 2))
            matched_indices = np.concatenate(keep, axis=0) if len(keep) else np.empty((0, 2), dtype=int)

        # 11) apply matches
        for itrk, idet in matched_indices:
            tr = strack_pool[itrk]
            det = detections_high[idet]
            if tr.state == TrackState.Tracked:
                tr.update(det, self.frame_id, update_feature=True)
                activated.append(tr)
            else:
                tr.re_activate(det, self.frame_id, new_id=False)
                refind.append(tr)

        # 12) medium stage with confidence consistency cue only
        remaining_tracked_idx = [i for i in unmatched_trk if strack_pool[i].state == TrackState.Tracked]
        remaining_tracked = [strack_pool[i] for i in remaining_tracked_idx]

        matches_med, u_trk_med, u_det_med = stage2_medium_iou_with_conf(
            remaining_tracked,
            detections_med,
            conf_consistency_w=self.conf_consistency_w,
            linear_assignment_fn=self.linear_assignment_fn,
        )

        used_med_det_global = set()
        for li, lj in matches_med:
            itrk = remaining_tracked_idx[li]
            det = detections_med[lj]
            tr = strack_pool[itrk]
            tr.update(det, self.frame_id, update_feature=True)
            activated.append(tr)
            used_med_det_global.add(lj)

        pending_unmatched = [remaining_tracked_idx[i] for i in u_trk_med]

        # 13) low support stage with low + deleted
        rescued_support = set()
        if len(pending_unmatched) and len(detections_low_support):
            trk_sub = [strack_pool[i] for i in pending_unmatched]
            trk_sub_tlwh = f32c(np.array([t.tlwh for t in trk_sub], dtype=np.float32))
            det_sup_tlwh = f32c(np.array([d.tlwh for d in detections_low_support], dtype=np.float32))

            from .utils import tlwh_to_xyxy, iou_xyxy_matrix, center_distance_matrix
            iou_sup = iou_xyxy_matrix(tlwh_to_xyxy(trk_sub_tlwh), tlwh_to_xyxy(det_sup_tlwh))
            cd_sup = center_distance_matrix(trk_sub_tlwh, det_sup_tlwh, self.center_norm, img_w, img_h)

            gate = (cd_sup <= self.support_zone_frac) | (iou_sup >= self.support_iou_gate)
            cost = np.where(gate, cd_sup, 1e6).astype(np.float32)

            from .association import hungarian_min
            m_sup = hungarian_min(cost, linear_assignment_fn=self.linear_assignment_fn)

            for li, lj in m_sup:
                if cost[li, lj] >= 1e6:
                    continue
                itrk = pending_unmatched[li]
                det = detections_low_support[lj]
                det.score = float(min(1.0, det.score + self.support_bonus))

                tr = strack_pool[itrk]
                tr.update(det, self.frame_id, update_feature=False)
                tr.unobserved = False
                tr.miss_streak = 0
                activated.append(tr)
                rescued_support.add(itrk)

        pending_unmatched = [i for i in pending_unmatched if i not in rescued_support]

        # 14) rescue with DDS then FGPD
        matched_pairs = {int(t): int(d) for t, d in matched_indices} if matched_indices.size else {}

        Vxy = estimate_track_velocity_xy(strack_pool)

        if trk_tlwh.size and det_tlwh.size and len(pending_unmatched):
            pseudo_dds, owners_dds = duplicate_detection_sharing(
                pending_unmatched,
                matched_pairs,
                trk_tlwh,
                det_tlwh,
                iou_matrix,
                velocity_xy=Vxy,
                max_share=self.dup_max_share,
                iou_thr=self.dup_iou_thr,
            )
        else:
            pseudo_dds, owners_dds = [], []

        rescued_by_pseudo = set()
        for tlwh, owner in zip(pseudo_dds, owners_dds):
            st = strack_pool[owner]
            pseudo = STrack(tlwh, score=self.low_thres + 1e-3, buffer_size=30)
            st.update(pseudo, self.frame_id, update_feature=False)
            st.unobserved = True
            st.miss_streak += 1
            activated.append(st)
            rescued_by_pseudo.add(owner)

        pending_unmatched = [i for i in pending_unmatched if i not in rescued_by_pseudo]

        fvpd_candidates = [i for i in pending_unmatched if strack_pool[i].miss_streak < self.fgpd_ttl]
        if len(fvpd_candidates):
            trkC_all = (trk_tlwh[:, :2] + 0.5 * trk_tlwh[:, 2:4]) if trk_tlwh.size else np.zeros((0, 2), np.float32)
            Nlist2 = knn_idx(trkC_all, self.fila_k) if trkC_all.shape[0] else []

            pseudo_fgpd, owners_fgpd = spawn_fgpd(
                fvpd_candidates,
                Nlist2,
                matched_pairs,
                trk_tlwh,
                det_tlwh,
                img_w,
                img_h,
                k_min=self.fgpd_kmin,
                huber_delta=self.fila_huber_delta,
                kappa=self.fgpd_kappa,
            )
        else:
            pseudo_fgpd, owners_fgpd = [], []

        for tlwh, owner in zip(pseudo_fgpd, owners_fgpd):
            st = strack_pool[owner]
            pseudo = STrack(tlwh, score=self.low_thres + 1e-4, buffer_size=30)
            st.update(pseudo, self.frame_id, update_feature=False)
            st.unobserved = True
            st.miss_streak += 1
            activated.append(st)
            rescued_by_pseudo.add(owner)

        remaining_after_rescue = [i for i in pending_unmatched if i not in rescued_by_pseudo]
        for owner in remaining_after_rescue:
            tr = strack_pool[owner]
            if tr.state != TrackState.Lost:
                tr.mark_lost()
                lost.append(tr)

        # 15) unconfirmed matching with high only
        if len(unconfirmed) and len(detections_high):
            from .utils import tlwh_to_xyxy, iou_xyxy_matrix
            u_tlwh = f32c(np.array([t.tlwh for t in unconfirmed], dtype=np.float32))
            d_tlwh = f32c(np.array([d.tlwh for d in detections_high], dtype=np.float32))
            iou_u = iou_xyxy_matrix(tlwh_to_xyxy(u_tlwh), tlwh_to_xyxy(d_tlwh))
            iou_cost_u = 1.0 - iou_u
            from .association import hungarian_min
            m_u = hungarian_min(iou_cost_u, linear_assignment_fn=self.linear_assignment_fn)

            matched_u_t = set()
            matched_u_d = set()
            for i, j in m_u:
                if float(iou_u[i, j]) < 0.10:
                    continue
                unconfirmed[i].update(detections_high[j], self.frame_id, update_feature=True)
                activated.append(unconfirmed[i])
                matched_u_t.add(i)
                matched_u_d.add(j)

            for i, t in enumerate(unconfirmed):
                if i not in matched_u_t:
                    t.mark_removed()
                    removed.append(t)

            unmatched_dets_high = [j for j in unmatched_dets_high if j not in matched_u_d]

        # 16) births with track aware NMS
        birth_candidates = [detections_high[j] for j in unmatched_dets_high] if len(unmatched_dets_high) else []

        active_for_nms = [t.tlwh for t in self.tracked_stracks if t.state == TrackState.Tracked]
        active_for_nms += [
            t.tlwh for t in self.lost_stracks
            if (self.frame_id - t.end_frame) <= max(1, self.buffer_size // 2)
        ]

        cand_tlwh = [d.tlwh for d in birth_candidates]
        cand_scores = [d.score for d in birth_candidates]

        allow_mask = track_aware_nms_before_births(
            cand_tlwh,
            cand_scores,
            active_for_nms,
            nms_thr=self.track_aware_birth_thr,
            score_thr=self.track_birth_score_thr,
        )

        for ok, det in zip(allow_mask.tolist(), birth_candidates):
            if not ok:
                continue
            if det.score < self.det_thresh:
                continue
            det.activate(self.frame_id)
            activated.append(det)

        # 17) retire old lost
        for tr in self.lost_stracks:
            if self.frame_id - tr.end_frame > self.max_time_lost:
                tr.mark_removed()
                removed.append(tr)

        # 18) merge states
        self.tracked_stracks = [t for t in self.tracked_stracks if t.state == TrackState.Tracked]
        self.tracked_stracks = self._joint(self.tracked_stracks, activated)
        self.tracked_stracks = self._joint(self.tracked_stracks, refind)

        self.lost_stracks = self._sub(self.lost_stracks, self.tracked_stracks)
        self.lost_stracks.extend(lost)
        self.lost_stracks = self._sub(self.lost_stracks, self.removed_stracks)

        self.removed_stracks.extend(removed)

        self.tracked_stracks, self.lost_stracks = self._remove_duplicates(self.tracked_stracks, self.lost_stracks)

        # 19) output only real observed tracks
        output = [t for t in self.tracked_stracks if t.is_activated and not t.unobserved]
        return output

    # ------------------------------
    # list utilities
    # ------------------------------
    def _joint(self, a, b):
        exists = {t.track_id: 1 for t in a}
        res = list(a)
        for t in b:
            if exists.get(t.track_id, 0) == 0:
                exists[t.track_id] = 1
                res.append(t)
        return res

    def _sub(self, a, b):
        mp = {t.track_id: t for t in a}
        for t in b:
            if t.track_id in mp:
                del mp[t.track_id]
        return list(mp.values())

    def _remove_duplicates(self, a, b):
        if len(a) == 0 or len(b) == 0:
            return a, b

        from .utils import tlwh_to_xyxy, iou_xyxy_matrix, f32c
        a_tlwh = f32c(np.array([t.tlwh for t in a], dtype=np.float32))
        b_tlwh = f32c(np.array([t.tlwh for t in b], dtype=np.float32))
        iou = iou_xyxy_matrix(tlwh_to_xyxy(a_tlwh), tlwh_to_xyxy(b_tlwh))

        pairs = np.where((1.0 - iou) < 0.15)
        dupa = []
        dupb = []
        for p, q in zip(*pairs):
            timep = a[p].frame_id - a[p].start_frame
            timeq = b[q].frame_id - b[q].start_frame
            if timep > timeq:
                dupb.append(q)
            else:
                dupa.append(p)

        a2 = [t for i, t in enumerate(a) if i not in set(dupa)]
        b2 = [t for i, t in enumerate(b) if i not in set(dupb)]
        return a2, b2
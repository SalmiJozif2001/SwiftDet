import torch
import torch.nn as nn
import torch.nn.functional as F


def make_grid_points(strides=(8, 16, 32), img_size=480, device="cpu"):
    centers, grid_strides, level_counts = [], [], []
    for s in strides:
        h = w = img_size // s
        yv, xv = torch.meshgrid(
            torch.arange(h, device=device, dtype=torch.float32),
            torch.arange(w, device=device, dtype=torch.float32),
            indexing="ij",
        )
        pts = torch.stack([(xv + 0.5) * s, (yv + 0.5) * s], dim=-1).reshape(-1, 2)
        centers.append(pts)
        grid_strides.append(torch.full((pts.shape[0], 1), s, device=device, dtype=torch.float32))
        level_counts.append(pts.shape[0])
    return torch.cat(centers, dim=0), torch.cat(grid_strides, dim=0), level_counts


def bbox_giou(boxes1, boxes2, eps=1e-7):
    x1 = torch.maximum(boxes1[:, 0], boxes2[:, 0])
    y1 = torch.maximum(boxes1[:, 1], boxes2[:, 1])
    x2 = torch.minimum(boxes1[:, 2], boxes2[:, 2])
    y2 = torch.minimum(boxes1[:, 3], boxes2[:, 3])
    inter = (x2 - x1).clamp(min=0) * (y2 - y1).clamp(min=0)
    area1 = (boxes1[:, 2] - boxes1[:, 0]).clamp(min=0) * (boxes1[:, 3] - boxes1[:, 1]).clamp(min=0)
    area2 = (boxes2[:, 2] - boxes2[:, 0]).clamp(min=0) * (boxes2[:, 3] - boxes2[:, 1]).clamp(min=0)
    union = area1 + area2 - inter + eps
    iou = inter / union
    cx1 = torch.minimum(boxes1[:, 0], boxes2[:, 0])
    cy1 = torch.minimum(boxes1[:, 1], boxes2[:, 1])
    cx2 = torch.maximum(boxes1[:, 2], boxes2[:, 2])
    cy2 = torch.maximum(boxes1[:, 3], boxes2[:, 3])
    c_area = (cx2 - cx1).clamp(min=0) * (cy2 - cy1).clamp(min=0) + eps
    return iou - (c_area - union) / c_area, iou


def pairwise_iou(boxes_a, boxes_b, eps=1e-7):
    tl = torch.maximum(boxes_a[:, None, :2], boxes_b[None, :, :2])
    br = torch.minimum(boxes_a[:, None, 2:], boxes_b[None, :, 2:])
    inter = (br - tl).clamp(min=0).prod(dim=-1)
    area_a = (boxes_a[:, 2] - boxes_a[:, 0]).clamp(min=0) * (boxes_a[:, 3] - boxes_a[:, 1]).clamp(min=0)
    area_b = (boxes_b[:, 2] - boxes_b[:, 0]).clamp(min=0) * (boxes_b[:, 3] - boxes_b[:, 1]).clamp(min=0)
    return inter / (area_a[:, None] + area_b[None, :] - inter + eps)


class SwiftDetLoss(nn.Module):
    def __init__(self, num_classes=20, reg_max=7, strides=(8, 16, 32), img_size=480,
                 vfl_w=1.0, giou_w=2.0, dfl_w=0.25, assigner="simota"):
        super().__init__()
        self.num_classes = num_classes
        self.reg_max = reg_max
        self.strides = strides
        self.img_size = img_size
        self.vfl_w = vfl_w
        self.giou_w = giou_w
        self.dfl_w = dfl_w
        self.assigner = assigner
        self.register_buffer("proj", torch.arange(reg_max + 1, dtype=torch.float32))
        self._grid_cache = {}

    def _grid(self, device):
        key = str(device)
        if key not in self._grid_cache:
            self._grid_cache[key] = make_grid_points(self.strides, self.img_size, device=device)
        return self._grid_cache[key]

    def decode_boxes(self, reg_dist, centers, grid_strides):
        b, n, _ = reg_dist.shape
        prob = F.softmax(reg_dist.reshape(b, n, 4, self.reg_max + 1).float(), dim=-1)
        ltrb = torch.matmul(prob, self.proj.to(reg_dist.device)) * grid_strides.unsqueeze(0)
        x1y1 = centers.unsqueeze(0) - ltrb[..., :2]
        x2y2 = centers.unsqueeze(0) + ltrb[..., 2:]
        return torch.cat([x1y1, x2y2], dim=-1)

    @staticmethod
    def varifocal_loss(pred_logits, target_score, alpha=0.75, gamma=2.0):
        pred_sigmoid = pred_logits.sigmoid()
        focal_weight = target_score * (target_score > 0.0).float() + \
                       alpha * (pred_sigmoid - target_score).abs().pow(gamma) * (target_score <= 0.0).float()
        return F.binary_cross_entropy_with_logits(pred_logits, target_score, reduction="none") * focal_weight

    def distribution_focal_loss(self, pred_dist, target_ltrb):
        target_ltrb = target_ltrb.clamp(0.0, self.reg_max - 0.01)
        tl = target_ltrb.long()
        tr = tl + 1
        wl = tr.float() - target_ltrb
        wr = 1.0 - wl
        flat = pred_dist.reshape(-1, self.reg_max + 1)
        loss_l = F.cross_entropy(flat, tl.reshape(-1), reduction="none").reshape(tl.shape)
        loss_r = F.cross_entropy(flat, tr.reshape(-1), reduction="none").reshape(tl.shape)
        return (loss_l * wl + loss_r * wr).mean(dim=-1)

    def _empty_assignment(self, num_anchors, device):
        return (torch.zeros((num_anchors, self.num_classes), device=device),
                torch.zeros((num_anchors,), dtype=torch.bool, device=device),
                torch.zeros((num_anchors, 4), device=device))

    @torch.no_grad()
    def simota_assign(self, cls_logits, pred_boxes, gt_boxes, gt_labels, centers, grid_strides):
        num_gt = gt_boxes.shape[0]
        num_anchors = centers.shape[0]
        if num_gt == 0:
            return self._empty_assignment(num_anchors, cls_logits.device)

        in_box = (centers[:, None, 0] > gt_boxes[None, :, 0]) & \
                 (centers[:, None, 0] < gt_boxes[None, :, 2]) & \
                 (centers[:, None, 1] > gt_boxes[None, :, 1]) & \
                 (centers[:, None, 1] < gt_boxes[None, :, 3])
        gt_centers = 0.5 * (gt_boxes[:, :2] + gt_boxes[:, 2:])
        r = 2.5 * grid_strides[:, 0:1]
        in_center = (torch.abs(centers[:, None, 0] - gt_centers[None, :, 0]) < r) & \
                    (torch.abs(centers[:, None, 1] - gt_centers[None, :, 1]) < r)
        fg_mask = (in_box | in_center).any(dim=1)
        is_in_boxes_and_center = in_box & in_center

        cand_idx = torch.where(fg_mask)[0]
        if cand_idx.numel() == 0:
            return self._empty_assignment(num_anchors, cls_logits.device)

        ious = pairwise_iou(pred_boxes[cand_idx], gt_boxes)
        iou_cost = -torch.log(ious.clamp(min=1e-8))
        cls_prob = cls_logits[cand_idx].float().sigmoid()
        gt_cls_prob = cls_prob[:, gt_labels]
        cls_cost = -torch.log(gt_cls_prob.clamp(min=1e-6))
        cost = cls_cost + 3.0 * iou_cost + 1e5 * (~is_in_boxes_and_center[cand_idx]).float()

        matching_matrix = torch.zeros_like(cost, dtype=torch.long)
        topk_ious, _ = torch.topk(ious, k=min(10, ious.shape[0]), dim=0)
        dynamic_ks = torch.clamp(topk_ious.sum(dim=0).int(), min=1)
        for gt_idx in range(num_gt):
            _, pos_idx = torch.topk(cost[:, gt_idx], k=min(int(dynamic_ks[gt_idx]), cost.shape[0]), largest=False)
            matching_matrix[pos_idx, gt_idx] = 1

        multi = matching_matrix.sum(dim=1) > 1
        if multi.any():
            min_gt = torch.argmin(cost[multi], dim=1)
            matching_matrix[multi] = 0
            matching_matrix[multi, min_gt] = 1

        pos_in_cand = matching_matrix.sum(dim=1) > 0
        pos_mask = torch.zeros((num_anchors,), dtype=torch.bool, device=cls_logits.device)
        pos_mask[cand_idx[pos_in_cand]] = True

        assigned_gt_idx = matching_matrix[pos_in_cand].argmax(dim=1)
        assigned_ious = ious[pos_in_cand, assigned_gt_idx]
        target_cls = torch.zeros((num_anchors, self.num_classes), device=cls_logits.device)
        target_cls[cand_idx[pos_in_cand], gt_labels[assigned_gt_idx]] = assigned_ious.clamp(min=0.0, max=1.0)
        target_boxes = torch.zeros((num_anchors, 4), device=cls_logits.device)
        target_boxes[cand_idx[pos_in_cand]] = gt_boxes[assigned_gt_idx]
        return target_cls, pos_mask, target_boxes

    @torch.no_grad()
    def atss_assign(self, pred_boxes, gt_boxes, gt_labels, centers, grid_strides, level_counts, topk=9):
        num_gt, num_anchors = gt_boxes.shape[0], centers.shape[0]
        if num_gt == 0:
            return self._empty_assignment(num_anchors, centers.device)

        # Pseudo-boxes of size 8*stride around grid centers for ATSS IoU statistics
        half = 4.0 * grid_strides
        pseudo_anchors = torch.cat([centers - half, centers + half], dim=-1)
        ious = pairwise_iou(pseudo_anchors, gt_boxes)
        gt_centers = 0.5 * (gt_boxes[:, :2] + gt_boxes[:, 2:])
        dists = torch.cdist(centers, gt_centers, p=2)

        cand_mask = torch.zeros((num_anchors, num_gt), dtype=torch.bool, device=centers.device)
        offset = 0
        for cnt in level_counts:
            _, idxs = torch.topk(dists[offset:offset + cnt], k=min(topk, cnt), dim=0, largest=False)
            cand_mask[offset + idxs, torch.arange(num_gt, device=centers.device)] = True
            offset += cnt

        nan_fill = torch.full_like(ious, float("nan"))
        cand_ious = torch.where(cand_mask, ious, nan_fill)
        mean_iou = torch.nanmean(cand_ious, dim=0, keepdim=True)
        std_iou = torch.sqrt(torch.nanmean((cand_ious - mean_iou) ** 2, dim=0, keepdim=True))
        thresh = mean_iou + std_iou

        in_box = (centers[:, None, 0] > gt_boxes[None, :, 0]) & (centers[:, None, 0] < gt_boxes[None, :, 2]) & \
                 (centers[:, None, 1] > gt_boxes[None, :, 1]) & (centers[:, None, 1] < gt_boxes[None, :, 3])
        is_pos = cand_mask & (ious >= thresh) & in_box
        ious_masked = torch.where(is_pos, ious, torch.full_like(ious, -1e6))
        max_ious, best_gt = ious_masked.max(dim=1)
        pos_mask = max_ious > -1e5

        target_cls = torch.zeros((num_anchors, self.num_classes), device=centers.device)
        target_boxes = torch.zeros((num_anchors, 4), device=centers.device)
        if pos_mask.any():
            pred_ious = pairwise_iou(pred_boxes[pos_mask], gt_boxes)
            assigned_gt = best_gt[pos_mask]
            rows = torch.arange(assigned_gt.numel(), device=centers.device)
            target_cls[pos_mask, gt_labels[assigned_gt]] = pred_ious[rows, assigned_gt].clamp(0.0, 1.0)
            target_boxes[pos_mask] = gt_boxes[assigned_gt]
        return target_cls, pos_mask, target_boxes

    def forward(self, preds, targets):
        """preds: (B, N, num_classes + 4*(reg_max+1)); always computed in FP32."""
        device = preds.device
        preds = preds.float()
        centers, grid_strides, level_counts = self._grid(device)
        cls_preds = preds[..., :self.num_classes]
        reg_preds = preds[..., self.num_classes:]
        pred_boxes_all = self.decode_boxes(reg_preds, centers, grid_strides)

        total_vfl = preds.new_zeros(())
        total_giou = preds.new_zeros(())
        total_dfl = preds.new_zeros(())
        total_pos = 0
        for i in range(preds.shape[0]):
            gt_boxes = targets[i]["boxes"].to(device)
            gt_labels = targets[i]["labels"].to(device)
            if self.assigner == "atss":
                t_cls, pos_mask, t_boxes = self.atss_assign(
                    pred_boxes_all[i].detach(), gt_boxes, gt_labels, centers, grid_strides, level_counts
                )
            else:
                t_cls, pos_mask, t_boxes = self.simota_assign(
                    cls_preds[i].detach(), pred_boxes_all[i].detach(), gt_boxes, gt_labels, centers, grid_strides
                )

            num_pos = int(pos_mask.sum().item())
            total_pos += max(num_pos, 1)
            total_vfl = total_vfl + self.varifocal_loss(cls_preds[i], t_cls).sum()

            if num_pos > 0:
                p_box = pred_boxes_all[i][pos_mask]
                g_box = t_boxes[pos_mask]
                giou, _ = bbox_giou(p_box, g_box)
                weights = t_cls[pos_mask].max(dim=1).values.clamp(min=0.1)
                total_giou = total_giou + ((1.0 - giou) * weights).sum()

                pos_centers = centers[pos_mask]
                pos_strides = grid_strides[pos_mask]
                t_ltrb = torch.cat([pos_centers - g_box[:, :2], g_box[:, 2:] - pos_centers], dim=-1) / pos_strides
                dfl_loss = self.distribution_focal_loss(
                    reg_preds[i][pos_mask].reshape(-1, 4, self.reg_max + 1), t_ltrb
                )
                total_dfl = total_dfl + (dfl_loss * weights).sum()

        norm = max(total_pos, 1)
        loss = (self.vfl_w * total_vfl + self.giou_w * total_giou + self.dfl_w * total_dfl) / norm
        return loss, {
            "vfl": (total_vfl / norm).detach(),
            "giou": (total_giou / norm).detach(),
            "dfl": (total_dfl / norm).detach(),
        }

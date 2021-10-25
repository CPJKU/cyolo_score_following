"""
adapted from https://github.com/ultralytics/yolov5/blob/master/utils/loss.py
"""
import torch

import numpy as np
import torch.nn as nn

from cyolo_score_following.utils.general import bbox_iou


def compute_loss(p, targets, model):  # predictions, targets, model
    device = targets.device
    lbox, lobj = torch.zeros(1, device=device), torch.zeros(1, device=device)
    tbox, indices, anchors = build_targets(p, targets, model)  # targets

    BCEobj = nn.BCEWithLogitsLoss().to(device)

    loss_dict = {}

    box_balance = np.asarray([1., 1., 1.])

    if model.loss_type == "mse" or model.loss_type == "mse_diff":
        obj_balance = np.asarray([10., 1., 1.])
        lbox_weight = 0.1
        lobj_weight = 1.0
    else:
        raise NotImplementedError

    box_balance *= lbox_weight
    obj_balance *= lobj_weight

    # Losses
    for i, pi in enumerate(p):  # layer index, layer predictions
        b, a, gj, gi = indices[i]  # image, anchor, gridy, gridx
        tobj = torch.zeros_like(pi[..., 0], device=device)  # target obj

        n = b.shape[0]  # number of targets
        if n:
            ps = pi[b, a, gj, gi]  # prediction subset corresponding to targets

            # Regression
            pxy = ps[:, :2].sigmoid() * 2. - 0.5
            pwh = (ps[:, 2:4].sigmoid() * 2) ** 2
            pbox = torch.cat((pxy, pwh * anchors[i]), 1).to(device)  # predicted box

            if model.loss_type == "mse":

                iou = bbox_iou(pbox.T, tbox[i])  # iou(prediction, target)
                if model.nc == 1:
                    lbox_i = ((pbox - tbox[i])**2).mean()

                else:
                    lbox_i = ((pbox[:, :2] - tbox[i][:, :2]) ** 2 +
                              (pbox[:, 2:].sqrt() - tbox[i][:, 2:].sqrt()) ** 2).mean()

            elif model.loss_type == "mse_diff":
                iou = bbox_iou(pbox.T, tbox[i])
                lbox_i = ((pxy - tbox[i][:, :2]) ** 2 + (pwh - tbox[i][:, 2:]/anchors[i]) ** 2).mean()

            else:
                raise NotImplementedError

            # Objectness
            tobj[b, a, gj, gi] = iou.detach().clamp(0).type(tobj.dtype)  # iou ratio
            lobj_i = BCEobj(pi[b.unique(), ..., 4], tobj[b.unique()])

            lbox_i *= box_balance[i]
            lobj_i *= obj_balance[i]

            lbox += lbox_i
            lobj += lobj_i

            loss_dict[f'box_loss_{i}'] = lbox_i
            loss_dict[f'obj_loss_{i}'] = lobj_i

        else:
            # dummy loss that should have zero gradient, otherwise DistributedDataParallel
            # can throw errors if not all parameters are used
            lobj += (pi * 0).mean()

    loss = lbox + lobj

    loss_dict['loss'] = loss
    loss_dict['box_loss'] = lbox
    loss_dict['obj_loss'] = lobj

    return loss_dict


def build_targets(p, targets, model):
    # Build targets for compute_loss(), input targets(image,class,x,y,w,h)

    det = model.model[-1] if hasattr(model, "model") else model.module.model[-1]  # Detect() module
    na, nt = det.na, targets.shape[0]  # number of anchors, targets
    tbox, indices, anch = [], [], []
    gain = torch.ones(7, device=targets.device)  # normalized to gridspace gain

    targets_ = []

    for class_idx in range(det.nc):
        target_c = targets[targets[:, 1] == class_idx]
        nt = target_c.shape[0]
        ai = torch.arange(na, device=targets.device).float().view(na, 1).repeat(1, nt)

        targets_.append(torch.cat((target_c.repeat(na, 1, 1), ai[:, :, None]), 2))

    anchor_t = 4.0  # anchor-multiple threshold
    g = 0.5  # bias
    off = torch.tensor([[0, 0],
                        [1, 0], [0, 1], [-1, 0], [0, -1],  # j,k,l,m
                        ], device=targets.device).float() * g  # offsets

    for i in range(det.nl):
        anchors = det.anchors[i]
        gain[2:6] = torch.tensor(p[i].shape)[[3, 2, 3, 2]]  # xyxy gain

        # Match targets to anchors
        t = targets_[i] * gain
        if t.shape[1] > 0:
            # Matching anchors
            r = t[:, :, 4:6] / anchors[:, None]  # wh ratio
            j = torch.max(r, 1. / r).max(2)[0] < anchor_t  # compare

            t = t[j]  # filter

            # Offsets
            gxy = t[:, 2:4]  # grid xy
            gxi = gain[[2, 3]] - gxy  # inverse
            j, k = ((gxy % 1. < g) & (gxy > 1.)).T
            l, m = ((gxi % 1. < g) & (gxi > 1.)).T
            j = torch.stack((torch.ones_like(j), j, k, l, m))
            t = t.repeat((5, 1, 1))[j]
            offsets = (torch.zeros_like(gxy)[None] + off[:, None])[j]

            # Define
            b, c = t[:, :2].long().T  # image, class
            gxy = t[:, 2:4]  # grid xy
            gwh = t[:, 4:6]  # grid wh
            gij = (gxy - offsets).long()
            gi, gj = gij.T  # grid xy indices

            # Append
            a = t[:, 6].long()  # anchor indices
            indices.append((b, a, gj, gi))  # image, anchor, grid indices
            tbox.append(torch.cat((gxy - gij, gwh), 1))  # box
            anch.append(anchors[a])  # anchors
        else:
            indices.append((torch.zeros(0, dtype=torch.long), torch.zeros(0), torch.zeros(0), torch.zeros(0)))
            tbox.append(None)
            anch.append(None)

    return tbox, indices, anch


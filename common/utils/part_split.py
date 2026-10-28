"""CPU-only component splitting; never discard or duplicate source alpha."""

import cv2
import numpy as np


def _component_masks(part_info, count):
    img = part_info['img']
    support = (part_info['mask'] > 0) & (img[..., -1] > 0)
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        support.astype(np.uint8), connectivity=8)
    if num_labels - 1 < count:
        return []

    # OpenCV reserves label zero for background, regardless of its area.
    ranked = np.argsort(-stats[1:, cv2.CC_STAT_AREA], kind='stable') + 1
    anchors = ranked[:count]
    label_owner = np.zeros(num_labels, dtype=np.uint8)
    for start in range(0, count, 2):
        pair = anchors[start:start + 2]
        centers = stats[pair, cv2.CC_STAT_LEFT] + stats[pair, cv2.CC_STAT_WIDTH] / 2
        anchors[start:start + 2] = pair[np.argsort(centers, kind='stable')]
    label_owner[anchors] = np.arange(1, count + 1)
    for component in ranked[count:]:
        distances = np.sum((centroids[anchors] - centroids[component]) ** 2, axis=1)
        label_owner[component] = np.argmin(distances) + 1
    owner = label_owner[labels]

    # The support mask uses an alpha threshold upstream. Preserve the faint
    # antialiasing outside it, assigning each pixel to exactly one component.
    fringe = (img[..., -1] > 0) & (owner == 0)
    if np.any(fringe):
        best = np.full(np.count_nonzero(fringe), np.inf, dtype=np.float32)
        for index in range(1, count + 1):
            distance = cv2.distanceTransform(
                ((owner != index) | fringe).astype(np.uint8),
                cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
            candidate = distance[fringe]
            closer = candidate < best
            fringe_owner = owner[fringe]
            fringe_owner[closer] = index
            owner[fringe] = fringe_owner
            best[closer] = candidate[closer]

    return [(owner == index).astype(np.uint8) for index in range(1, count + 1)]


def process_cuts(img, depth, src_xyxy, tgt_bbox, mask=None):
    tx1, ty1, width, height = tgt_bbox[:4]
    tx2, ty2 = tx1 + width, ty1 + height
    img = img[ty1:ty2, tx1:tx2].copy()
    depth = depth[ty1:ty2, tx1:tx2].copy()
    if mask is not None:
        mask = mask[ty1:ty2, tx1:tx2] > 0
        # Dilation here can include another child's pixels, duplicating alpha.
        img[..., -1] *= mask
        depth[~mask] = 1

    xyxy = [tx1 + src_xyxy[0], ty1 + src_xyxy[1],
            tx2 + src_xyxy[0], ty2 + src_xyxy[1]]
    return img, depth, xyxy


def _cut_parts(part_info, masks, tags):
    result = {}
    # Splitting changes geometry, not the parent's inter-layer occlusion rank.
    # A per-child median can put an iris behind its eyewhite on just one side.
    if 'depth_median' in part_info:
        parent_depth = part_info['depth_median']
    else:
        visible = part_info['img'][..., -1] > 10
        parent_depth = np.median(part_info['depth'][visible])
    for mask, tag in zip(masks, tags):
        bbox = cv2.boundingRect(mask)
        img, depth, xyxy = process_cuts(
            part_info['img'], part_info['depth'], part_info['xyxy'], bbox, mask=mask)
        x, y, width, height = bbox
        result[tag] = {'img': img, 'depth': depth, 'xyxy': xyxy,
                       'mask': mask[y:y + height, x:x + width] > 0,
                       'depth_median': parent_depth, 'tag': tag}
    return result


def part_lr_split(tag, part_info):
    masks = _component_masks(part_info, 2)
    if not masks:
        # Touching counterparts are not separable by connected components.
        return {tag: part_info}
    # Image-left is anatomical right, matching the existing layer convention.
    return _cut_parts(part_info, masks, [f'{tag}-r', f'{tag}-l'])


def eyes_split(part_info):
    masks = _component_masks(part_info, 4)
    if not masks:
        # Two or three components used to remove the entire legacy eyes layer.
        return {'eyes': part_info}
    # Preserve legacy area-based eye/brow grouping, without losing other islands.
    return _cut_parts(part_info, masks, ['eyer', 'eyel', 'browr', 'browl'])


def tag_lr_split(tag, tag2pinfo):
    if tag in tag2pinfo:
        replacement = part_lr_split(tag, tag2pinfo[tag])
        # Preserve the parent's insertion position for equal-depth sort ties.
        items = list(tag2pinfo.items())
        tag2pinfo.clear()
        for name, info in items:
            if name == tag:
                tag2pinfo.update(replacement)
            else:
                tag2pinfo[name] = info

"""Shared building blocks for skin-texture stitching: layout context and per-photo warping into the frontal atlas."""
from __future__ import annotations

import cv2
import numpy as np


def _area2(a, b, c):
    return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])


def layout_context(layout: dict, S: int) -> dict:
    """Prepare everything that depends only on the mesh layout, scaled to an S x S atlas."""
    canon = np.array(layout["points"], dtype=np.float64) * S
    tris = np.array(layout["triangles"], dtype=np.int32)
    N = len(canon)
    n_real = 478 if N >= 478 else N
    centres: dict[int, set[int]] = {}
    for t in tris:
        for v in t:
            if v >= n_real:
                centres.setdefault(int(v), set()).update(int(x) for x in t if x < n_real)
    sign0 = np.array([1 if _area2(canon[a], canon[b], canon[c]) >= 0 else -1 for a, b, c in tris])
    mesh_mask = np.zeros((S, S), np.uint8)
    for a, b, c in tris:
        cv2.fillConvexPoly(mesh_mask, np.round(canon[[a, b, c]]).astype(np.int32), 255)
    nostril = np.zeros((S, S), np.uint8)
    if n_real > 327:
        r = 0.2 * float(np.hypot(*(canon[98] - canon[327])))
        for a, b, c in tris:
            cen = canon[[a, b, c]].mean(axis=0)
            if np.hypot(*(cen - canon[98])) < r or np.hypot(*(cen - canon[327])) < r:
                cv2.fillConvexPoly(nostril, np.round(canon[[a, b, c]]).astype(np.int32), 255)
    return dict(S=S, canon=canon, tris=tris, N=N, n_real=n_real, centres=centres, sign0=sign0, mesh_mask=mesh_mask, nostril=nostril)


def prepare_photo(img: np.ndarray, lm_norm, ctx: dict):
    """Warp every visible, front-facing triangle of one photo into the atlas.

    img: BGR image. lm_norm: (>=478, 3) landmarks, x/y normalised to the image width/height, z in image-width units.
    Returns dict(atlas float32 BGR, w float32 weight map, yaw, sharp) or None.
    """
    S, canon, tris, N, n_real = ctx["S"], ctx["canon"], ctx["tris"], ctx["N"], ctx["n_real"]
    H, W = img.shape[:2]
    L = np.asarray(lm_norm, dtype=np.float64)
    if L.ndim != 2 or len(L) < n_real:
        return None
    P = np.zeros((N, 3))
    P[:n_real] = L[:n_real, :3] * [W, H, W]
    for ci, verts in ctx["centres"].items():
        P[ci] = P[sorted(verts)].mean(axis=0)

    signs = np.array([1 if _area2(P[a], P[b], P[c]) >= 0 else -1 for a, b, c in tris])
    orient = 1 if float((signs == ctx["sign0"]).mean()) >= 0.5 else -1  # a mirrored photo flips every winding
    front = (signs * orient) == ctx["sign0"]

    v1 = P[tris[:, 1]] - P[tris[:, 0]]
    v2 = P[tris[:, 2]] - P[tris[:, 0]]
    nrm = np.cross(v1, v2)
    facing = np.abs(nrm[:, 2]) / (np.linalg.norm(nrm, axis=1) + 1e-9)

    # visibility (z-buffer by painting triangle ids far to near)
    zs = P[tris, 2].mean(axis=1)
    order = [i for i in np.argsort(-zs) if front[i]]
    idmap = np.zeros((H, W, 3), np.uint8)
    for i in order:
        idv = int(i) + 1
        cv2.fillConvexPoly(idmap, np.round(P[tris[i], :2]).astype(np.int32), (idv & 255, (idv >> 8) & 255, (idv >> 16) & 255))
    visible = np.zeros(len(tris), bool)
    for i in order:
        a, b, c = tris[i]
        cen = P[[a, b, c], :2].mean(axis=0)
        for q in (cen, (cen + P[a, :2]) / 2, (cen + P[b, :2]) / 2, (cen + P[c, :2]) / 2):
            x, y = int(round(q[0])), int(round(q[1]))
            if 0 <= x < W and 0 <= y < H:
                px = idmap[y, x]
                if (int(px[0]) | (int(px[1]) << 8) | (int(px[2]) << 16)) == i + 1:
                    visible[i] = True
                    break

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    x0, y0 = np.floor(P[:n_real, :2].min(axis=0)).astype(int)
    x1, y1 = np.ceil(P[:n_real, :2].max(axis=0)).astype(int)
    crop = gray[max(0, y0): min(H, y1), max(0, x0): min(W, x1)]
    sharp = float(cv2.Laplacian(crop, cv2.CV_64F).var()) if crop.size else 0.0

    atlas = np.zeros((S, S, 3), np.float32)
    wmap = np.zeros((S, S), np.float32)
    for i, (a, b, c) in enumerate(tris):
        if not (front[i] and visible[i]) or facing[i] < 0.22:
            continue
        dst = canon[[a, b, c]]
        src = P[[a, b, c], :2]
        bx0 = max(0, int(np.floor(dst[:, 0].min())) - 2)
        by0 = max(0, int(np.floor(dst[:, 1].min())) - 2)
        bx1 = min(S, int(np.ceil(dst[:, 0].max())) + 3)
        by1 = min(S, int(np.ceil(dst[:, 1].max())) + 3)
        if bx1 - bx0 < 2 or by1 - by0 < 2 or bx1 - bx0 > 600 or by1 - by0 > 600:
            continue
        M = cv2.getAffineTransform(src.astype(np.float32), (dst - [bx0, by0]).astype(np.float32))
        warped = cv2.warpAffine(img, M, (bx1 - bx0, by1 - by0), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
        tmask = np.zeros((by1 - by0, bx1 - bx0), np.uint8)
        cen = (dst - [bx0, by0]).mean(axis=0)
        grown = (dst - [bx0, by0]) + np.sign((dst - [bx0, by0]) - cen) * 0.6
        cv2.fillConvexPoly(tmask, np.round(grown).astype(np.int32), 255)
        sel = tmask > 0
        atlas[by0:by1, bx0:bx1][sel] = warped[sel]
        wmap[by0:by1, bx0:bx1][sel] = float(facing[i] ** 4)  # the most front-on photo of each patch dominates
    return {"atlas": atlas, "w": wmap, "yaw": float(P[1, 0] - P[[33, 263], 0].mean()), "sharp": sharp}

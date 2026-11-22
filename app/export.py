"""Write the scanned face as an OBJ model into the profile's mesh folder (where skin_atlas.json and skin.png already are): 3D landmark
positions from the front-facing frame, texture coordinates from the skin atlas layout, and the blended skin texture as its material."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

# The face outline (MediaPipe's face-oval landmarks): their best-fit plane is the face's front plane.
FACE_OVAL = [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136, 172, 58, 132,
             93, 234, 127, 162, 21, 54, 103, 67, 109]


def frontalize(P: np.ndarray) -> np.ndarray:
    """Turn the face so it looks exactly straight ahead, whatever way the head was angled when the scan's front frame was taken.
    P: (n, 3) in model units (x right, y up, z toward the viewer). The front plane of the face (best fit through the outline) becomes
    facing the viewer, and the line through the outer corners of the eyes becomes level. The face keeps its centre."""
    if len(P) < 400:
        return P
    c = P[:478].mean(axis=0)
    oval = P[FACE_OVAL] - P[FACE_OVAL].mean(axis=0)
    z = np.linalg.svd(oval)[2][-1]                      # normal of the plane through the outline
    if np.dot(P[4] - P[FACE_OVAL].mean(axis=0), z) < 0:  # make it point the way the nose does (toward the viewer)
        z = -z
    x = P[263] - P[33]                                  # across the eyes
    x = x - np.dot(x, z) * z
    x /= np.linalg.norm(x) + 1e-12
    y = np.cross(z, x)
    R = np.stack([x, y, z])                             # rows: the face's own right / up / front directions
    return (P - c) @ R.T + c


def write_obj(d: Path) -> dict:
    layout = json.loads((d / "skin_atlas.json").read_text(encoding="utf-8"))
    pts3 = layout.get("landmarks3d")
    pts2 = layout.get("points")
    tris = layout.get("triangles")
    aspect = float(layout.get("aspect") or 16 / 9)
    if not (pts3 and pts2 and tris):
        raise ValueError("the scan has no 3D landmarks saved; scan again")
    # The layout adds one centre point per eye (to fill the eye openings the stock mesh leaves as holes). Those points have no measured 3D
    # position, so place each at the average of the eye-outline points it fans out to; the eyes then get their skin-atlas texture.
    pts3 = [list(p) for p in pts3]
    n = len(pts2)
    for c in range(len(pts3), n):
        ring = {v for t in tris if c in t for v in t if v != c and v < len(pts3)}
        if not ring:
            n = c  # a centre with no outline: leave it (and its triangles) out
            break
        pts3.append([sum(pts3[v][k] for v in ring) / len(ring) for k in range(3)])
    keep = [t for t in tris if max(t) < n]
    size = float(layout.get("size") or 2048)
    lines = ["# Generate 3D Mesh", "mtllib mesh.mtl", "o face"]
    # normalized picture units -> head-height units, y up, z toward the camera; then turned so the face looks perfectly straight ahead
    P = frontalize(np.array([[(x - 0.5) * aspect, -(y - 0.5), -z * aspect] for x, y, z in pts3[:n]], dtype=float))
    for x, y, z in P:
        lines.append(f"v {x:.5f} {y:.5f} {z:.5f}")
    for u, v in pts2[:n]:
        lines.append(f"vt {u:.5f} {1 - v:.5f}")
    lines.append("usemtl skin")
    for a, b, c in keep:
        lines.append(f"f {a + 1}/{a + 1} {b + 1}/{b + 1} {c + 1}/{c + 1}")
    uv = [[round(u, 5), round(1 - v, 5)] for u, v in pts2[:n]]
    verts = [[round(float(x), 5), round(float(y), 5), round(float(z), 5)] for x, y, z in P]
    m = d  # d is the profile's mesh folder: everything needed to load and view the model lives here
    (m / "mesh.json").write_text(json.dumps({"vertices": verts, "uvs": uv, "triangles": keep, "eye_start": len(layout["landmarks3d"])}), encoding="utf-8")  # mesh data
    (m / "mesh.obj").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (m / "mesh.mtl").write_text("newmtl skin\nKd 1 1 1\nmap_Kd skin.png\n", encoding="utf-8")
    return {"vertices": n, "triangles": len(keep), "texture": "skin.png", "size": size}

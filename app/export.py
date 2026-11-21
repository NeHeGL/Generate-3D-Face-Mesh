"""Write the scanned face as an OBJ model: 3D landmark positions from the front-facing frame, texture coordinates from the skin
atlas layout, and the blended skin texture as its material."""
from __future__ import annotations

import json
from pathlib import Path


def write_obj(d: Path) -> dict:
    layout = json.loads((d / "skin_atlas.json").read_text(encoding="utf-8"))
    pts3 = layout.get("landmarks3d")
    pts2 = layout.get("points")
    tris = layout.get("triangles")
    aspect = float(layout.get("aspect") or 16 / 9)
    if not (pts3 and pts2 and tris):
        raise ValueError("the scan has no 3D landmarks saved; scan again")
    n = len(pts3)  # the extra centre points the layout adds for the eyes have no 3D position: triangles using them are left out
    keep = [t for t in tris if max(t) < n]
    size = float(layout.get("size") or 2048)
    lines = ["# Generate 3D Mesh", "mtllib mesh.mtl", "o face"]
    for x, y, z in pts3:  # normalized picture units -> head-height units, y up, z toward the camera
        lines.append(f"v {(x - 0.5) * aspect:.5f} {-(y - 0.5):.5f} {-z * aspect:.5f}")
    for u, v in pts2[:n]:
        lines.append(f"vt {u:.5f} {1 - v:.5f}")
    lines.append("usemtl skin")
    for a, b, c in keep:
        lines.append(f"f {a + 1}/{a + 1} {b + 1}/{b + 1} {c + 1}/{c + 1}")
    (d / "mesh.obj").write_text("\n".join(lines) + "\n", encoding="utf-8")
    tex = "skin_stitched.png" if (d / "skin_stitched.png").exists() else "skin_atlas.png"
    (d / "mesh.mtl").write_text(f"newmtl skin\nKd 1 1 1\nmap_Kd {tex}\n", encoding="utf-8")
    return {"vertices": n, "triangles": len(keep), "texture": tex, "size": size}

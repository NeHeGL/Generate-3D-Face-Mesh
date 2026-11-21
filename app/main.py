"""Generate 3D Mesh: a small local web app that does one thing, builds a good 3D mesh and skin texture of a face.

You look at the camera and move your head around; the page tracks your face (MediaPipe, in the browser) and sends the best photo for
each head angle here. The server blends them into one seamless skin texture and writes everything for one person into output/<name>/:

    photos/*.jpg          the best photo for each pose, with its measured head angle in photos.json
    skin_atlas.json       the mesh layout (landmarks, triangles, texture coordinates) and the front-facing 3D landmarks
    skin_stitched.png     the blended skin texture (2048 px)
    mesh.obj / mesh.mtl   the face as an OBJ model that uses that texture (open it in Blender, MeshLab, Windows 3D Viewer...)
"""
from __future__ import annotations

import base64
import json
import re
from pathlib import Path
from uuid import uuid4

from fastapi import FastAPI
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from app import export, skin_live

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "output"
STATIC = ROOT / "static"
app = FastAPI(title="Generate 3D Mesh")


def folder(name: str) -> Path:
    safe = re.sub(r"[^A-Za-z0-9_-]+", "_", name.strip()).strip("_") or "face"
    d = OUT / safe
    d.mkdir(parents=True, exist_ok=True)
    return d


def _b64(data: str) -> bytes:
    return base64.b64decode(data.split(",", 1)[-1])


class StitchStart(BaseModel):
    canonical_name: str
    layout: dict


class StitchAdd(BaseModel):
    canonical_name: str
    image_base64: str
    landmarks: list[list[float]]


class StitchFinish(BaseModel):
    canonical_name: str
    tone: list[float] | None = None


class Frames(BaseModel):
    canonical_name: str
    frames: list[dict]


class Atlas(BaseModel):
    canonical_name: str
    atlas_base64: str
    layout: dict


@app.post("/api/stitch/start")
def stitch_start(b: StitchStart) -> dict:
    (folder(b.canonical_name) / "skin_atlas.json").write_text(json.dumps(b.layout), encoding="utf-8")
    return skin_live.start(b.canonical_name, b.layout)


@app.post("/api/stitch/add")
def stitch_add(b: StitchAdd) -> dict:
    return skin_live.add(b.canonical_name, _b64(b.image_base64), b.landmarks)


@app.get("/api/stitch/preview")
def stitch_preview(canonical_name: str, tone: str = "") -> Response:
    try:
        tg = [float(x) for x in tone.split(",")] if tone else None
    except ValueError:
        tg = None
    got = skin_live.preview(canonical_name, tg)
    if got is None:
        return Response(status_code=404)
    jpeg, version, cov, tri = got
    return Response(content=jpeg, media_type="image/jpeg",
                    headers={"X-Version": str(version), "X-Coverage": f"{cov:.4f}", "X-TriCov": base64.b64encode(tri).decode(), "Cache-Control": "no-store"})


@app.post("/api/stitch/finish")
def stitch_finish(b: StitchFinish) -> dict:
    d = folder(b.canonical_name)
    out = skin_live.finish(b.canonical_name, d, b.tone)
    if out.get("status") == "ok":
        out["url"] = f"/output/{d.name}/skin_stitched.png"
    return out


@app.get("/api/stitch/status")
def stitch_status(canonical_name: str) -> dict:
    s = skin_live.SESSIONS.get(canonical_name)
    return {"hires": s.hires if s else "none", "photos": s.live.photos if s else 0}


@app.post("/api/capture/scan-frames")
def scan_frames(b: Frames) -> dict:
    d = folder(b.canonical_name) / "photos"
    d.mkdir(exist_ok=True)
    for old in d.glob("*"):  # a new scan replaces the old one
        old.unlink()
    rows = []
    for f in b.frames:
        if len(f.get("image_base64") or "") < 50:
            continue
        name = f"angle_{int(f.get('angle_index', 0))}_{uuid4().hex[:6]}.jpg"
        (d / name).write_bytes(_b64(f["image_base64"]))
        rows.append({"file": name, **{k: f.get(k) for k in ("caption", "angle_index", "quality_score", "yaw_deg", "pitch_deg")}})
    (d.parent / "photos.json").write_text(json.dumps(rows, indent=1), encoding="utf-8")
    return {"status": "ok", "photos": len(rows), "profile": {"photos": len(rows)}}


@app.post("/api/capture/skin-atlas")
def skin_atlas(b: Atlas) -> dict:
    d = folder(b.canonical_name)
    (d / "skin_atlas.png").write_bytes(_b64(b.atlas_base64))
    (d / "skin_atlas.json").write_text(json.dumps(b.layout), encoding="utf-8")
    return {"status": "ok"}


@app.post("/api/mesh/export")
def mesh_export(b: StitchFinish) -> dict:
    d = folder(b.canonical_name)
    try:
        info = export.write_obj(d)
    except Exception as exc:  # noqa: BLE001
        return {"status": "error", "message": f"{type(exc).__name__}: {exc}"}
    return {"status": "ok", "folder": str(d), "obj": f"/output/{d.name}/mesh.obj", **info}


@app.get("/")
def index() -> FileResponse:
    return FileResponse(STATIC / "index.html", headers={"Cache-Control": "no-store"})


OUT.mkdir(exist_ok=True)
app.mount("/static", StaticFiles(directory=STATIC), name="static")
app.mount("/output", StaticFiles(directory=OUT), name="output")

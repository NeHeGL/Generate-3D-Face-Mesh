"""Generate 3D Mesh: a small local web app that does one thing, builds a good 3D mesh and skin texture of a face.

You look at the camera and move your head around; the page tracks your face (MediaPipe, in the browser) and sends the best photo for
each head angle here. The server blends them into one seamless skin texture and writes everything for one person into output/<name>/:

    profile.json          the name as typed (the folder is the lowercase name: "Jeff Molofee" -> jeff_molofee)
    photos/               the best photo for each pose, with its measured head angle in photos/photos.json
    mesh/                 the 3D model and everything needed to load and view it:
        mesh.obj, mesh.mtl  the face as an OBJ model (open it in Blender, MeshLab, Windows 3D Viewer...)
        skin.png            its texture map (the blended skin, 2048 px)
        mesh.json           the same mesh as plain data (vertices, uvs, triangles), used by the in-app viewer
        skin_atlas.json     the mesh layout (landmarks, triangles, texture coordinates) the model is built from
        preview.jpg         a small picture of the skin
Saving a scan builds the whole mesh folder automatically.
"""
from __future__ import annotations

import base64
import json
import re
import shutil
import sys
from pathlib import Path
from uuid import uuid4

from fastapi import FastAPI
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from app import export, skin_live

if getattr(sys, "frozen", False):  # packaged .exe: bundled files live in the temp unpack dir, output goes next to the .exe
    ROOT = Path(sys._MEIPASS)
    OUT = Path(sys.executable).resolve().parent / "output"
else:
    ROOT = Path(__file__).resolve().parent.parent
    OUT = ROOT / "output"
STATIC = ROOT / "static"
app = FastAPI(title="Generate 3D Mesh")


def slug(name: str) -> str:
    """'Jeff Molofee' -> 'jeff_molofee'"""
    return re.sub(r"[^a-z0-9_-]+", "_", name.strip().lower()).strip("_") or "face"


def mesh_dir(name: str) -> Path:
    d = folder(name) / "mesh"
    d.mkdir(exist_ok=True)
    return d


def folder(name: str) -> Path:
    d = OUT / slug(name)
    d.mkdir(parents=True, exist_ok=True)
    prof = d / "profile.json"
    if not prof.exists():
        prof.write_text(json.dumps({"name": name.strip()}), encoding="utf-8")
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


class ProfileDelete(BaseModel):
    canonical_name: str


@app.post("/api/profile/delete")
def delete_profile(b: ProfileDelete) -> dict:
    d = OUT / slug(b.canonical_name)
    out_root = OUT.resolve()
    resolved = d.resolve()
    if resolved.parent != out_root or not resolved.is_dir():
        return {"status": "not_found"}
    shutil.rmtree(resolved)
    skin_live.SESSIONS.pop(b.canonical_name, None)
    skin_live.SESSIONS.pop(slug(b.canonical_name), None)
    return {"status": "ok"}


@app.post("/api/stitch/start")
def stitch_start(b: StitchStart) -> dict:
    (mesh_dir(b.canonical_name) / "skin_atlas.json").write_text(json.dumps(b.layout), encoding="utf-8")
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
    m = mesh_dir(b.canonical_name)
    out = skin_live.finish(b.canonical_name, m, b.tone)
    if out.get("status") == "ok":
        out["url"] = f"/output/{m.parent.name}/mesh/skin.png"
        try:  # saving the scan also builds the 3D model, so the mesh folder is complete straight away
            out["mesh"] = export.write_obj(m)
        except Exception as exc:  # noqa: BLE001
            out["mesh_error"] = f"{type(exc).__name__}: {exc}"
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
    (d / "photos.json").write_text(json.dumps(rows, indent=1), encoding="utf-8")
    return {"status": "ok", "photos": len(rows), "profile": {"photos": len(rows)}}


@app.post("/api/capture/skin-atlas")
def skin_atlas(b: Atlas) -> dict:
    d = mesh_dir(b.canonical_name)
    (d / "skin_atlas.json").write_text(json.dumps(b.layout), encoding="utf-8")
    return {"status": "ok"}


@app.get("/api/users")
def users() -> list[dict]:
    rows = []
    for d in sorted(OUT.iterdir()) if OUT.exists() else []:
        if not d.is_dir():
            continue
        try:
            name = json.loads((d / "profile.json").read_text(encoding="utf-8")).get("name") or d.name
        except (OSError, ValueError):
            name = d.name.replace("_", " ").title()
        rows.append({"name": name, "folder": d.name, "has_scan": (d / "photos" / "photos.json").exists(),
                     "has_mesh": (d / "mesh" / "mesh.obj").exists() and (d / "mesh" / "mesh.json").exists()})
    return rows


@app.get("/")
def index() -> FileResponse:
    return FileResponse(STATIC / "index.html", headers={"Cache-Control": "no-store"})


OUT.mkdir(exist_ok=True)
app.mount("/static", StaticFiles(directory=STATIC), name="static")
app.mount("/output", StaticFiles(directory=OUT), name="output")

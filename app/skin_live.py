"""Incremental ("live") skin stitching: photos arrive one at a time during the scan and are blended into a running texture.

Every photo is warped into the frontal atlas, colour-matched to the mosaic so far, and added to Laplacian-pyramid
accumulators. A preview is rendered from the accumulators in about 100 ms, so the browser can show the blended skin as it grows.
On finish, a 1024 result is written at once and a 2048 version is rebuilt in the background from the stored frames.
"""
from __future__ import annotations

import threading
import time
from pathlib import Path

import cv2
import numpy as np

from app.skin_core import layout_context, prepare_photo

LEVELS = 7


def _gpyr(im, n):
    out = [im]
    for _ in range(n):
        out.append(cv2.pyrDown(out[-1]))
    return out


class LiveStitcher:
    def __init__(self, layout: dict, S: int = 1024):
        self.ctx = layout_context(layout, S)
        self.S = S
        self.num = [None] * (LEVELS + 1)
        self.den = [None] * (LEVELS + 1)
        self.mos_num = np.zeros((S, S, 3), np.float32)
        self.mos_den = np.zeros((S, S), np.float32)
        self.cover = np.zeros((S, S), np.float32)
        self.smax = 1.0
        self.photos = 0
        self.gains = []

    def add(self, img_bgr: np.ndarray, lm_norm) -> bool:
        p = prepare_photo(img_bgr, lm_norm, self.ctx)
        if p is None or float(p["w"].max()) <= 0:
            return False
        self.smax = max(self.smax, p["sharp"])
        if self.photos >= 3 and p["sharp"] < 0.5 * self.smax:
            return False  # motion-blurred or out-of-focus frame: blending it in is what makes the skin go soft at times
        w = p["w"] * float(np.clip(p["sharp"] / self.smax, 0.25, 1.0)) ** 3
        atlas = p["atlas"]
        mesh = self.ctx["mesh_mask"] > 0
        if self.photos > 0:
            ref = self.mos_num / np.maximum(self.mos_den, 1e-6)[..., None]
            ov = (w > 0.02) & (self.mos_den > 0.02) & mesh
            if int(ov.sum()) > 400 * (self.S / 1024) ** 2:
                g = np.array([np.median(ref[..., k][ov]) / max(np.median(atlas[..., k][ov]), 1.0) for k in range(3)])
                g = np.clip(g, 0.7, 1.4)
                # A smooth gain map: the local brightness ratio inside the overlap, blurred widely and fading to the global gain outside it.
                # This is what removes the soft bands where one photo's exposure takes over from another's.
                sig = 0.06 * self.S
                ovf = ov.astype(np.float32)
                den = cv2.GaussianBlur(ovf, (0, 0), sig)
                gmap = np.empty_like(atlas)
                for k in range(3):
                    ratio = np.clip(ref[..., k] / np.maximum(atlas[..., k], 1.0), 0.6, 1.6) * ovf
                    loc = cv2.GaussianBlur(ratio, (0, 0), sig) / np.maximum(den, 1e-4)
                    a = np.clip(den / 0.3, 0, 1)
                    gmap[..., k] = a * loc + (1 - a) * g[k]
                atlas = np.clip(atlas * gmap, 0, 255)
                self.gains.append([round(float(x), 3) for x in g])
        self.mos_num += atlas * w[..., None]
        self.mos_den += w
        self.cover += w
        gi = _gpyr(atlas, LEVELS)
        gw = _gpyr(w, LEVELS)
        for k in range(LEVELS + 1):
            lap = gi[k] - cv2.pyrUp(gi[k + 1], dstsize=(gi[k].shape[1], gi[k].shape[0])) if k < LEVELS else gi[k]
            w3 = gw[k][..., None]
            self.num[k] = lap * w3 if self.num[k] is None else self.num[k] + lap * w3
            self.den[k] = gw[k].copy() if self.den[k] is None else self.den[k] + gw[k]
        self.photos += 1
        return True

    def tri_coverage(self) -> np.ndarray:
        """How completely each mesh triangle has been seen (0..255), so the browser knows which parts are finished."""
        tris, canon, S = self.ctx["tris"], self.ctx["canon"], self.S
        out = np.zeros(len(tris), np.uint8)
        for i, (a, b, c) in enumerate(tris):
            pts = canon[[a, b, c]]
            x0, y0 = max(0, int(pts[:, 0].min()) - 1), max(0, int(pts[:, 1].min()) - 1)
            x1, y1 = min(S, int(pts[:, 0].max()) + 2), min(S, int(pts[:, 1].max()) + 2)
            if x1 - x0 < 2 or y1 - y0 < 2:
                continue
            m = np.zeros((y1 - y0, x1 - x0), np.uint8)
            cv2.fillConvexPoly(m, np.round(pts - [x0, y0]).astype(np.int32), 255)
            sel = m > 0
            if not sel.any():  # a sliver of a triangle (nostril rim): judge it by the pixel at its centre instead of calling it unseen
                cx, cy = int(round(pts[:, 0].mean())), int(round(pts[:, 1].mean()))
                if 0 <= cx < S and 0 <= cy < S:
                    out[i] = int(255 * float(np.clip(self.cover[cy, cx] / 0.08, 0, 1)))
                continue
            # 0.08 (not higher): steep triangles on the nose flanks only ever get a light weight, but they are still painted
            out[i] = int(255 * float(np.clip(self.cover[y0:y1, x0:x1][sel] / 0.08, 0, 1).mean()))
        return out

    def render(self, tone=None):
        """Current blended texture as BGR uint8 plus coverage (0..1 of the mesh area)."""
        S = self.S
        mask = self.ctx["mesh_mask"]
        if self.photos == 0:
            return np.full((S, S, 3), 40, np.uint8), 0.0
        res = [self.num[k] / np.maximum(self.den[k], 1e-6)[..., None] for k in range(LEVELS + 1)]
        out = res[LEVELS]
        for k in range(LEVELS - 1, -1, -1):
            out = cv2.pyrUp(out, dstsize=(res[k].shape[1], res[k].shape[0])) + res[k]
        out = np.clip(out, 0, 255)
        blur = cv2.GaussianBlur(out, (0, 0), 2.0 * S / 2048 + 0.6)
        out = np.clip(out + 0.7 * (out - blur), 0, 255)
        out8 = out.astype(np.uint8)
        covered = (self.cover > 0.04) & (mask > 0)  # any usable photo counts, including oblique ones (nose flanks)
        coverage = float(covered.sum() / max(1, (mask > 0).sum()))
        hole = ((~covered) & (mask > 0)).astype(np.uint8) * 255
        if hole.any() and covered.any():
            sm = 256
            small = cv2.resize(out8, (sm, sm), interpolation=cv2.INTER_AREA)
            smask = cv2.dilate(cv2.resize(hole, (sm, sm), interpolation=cv2.INTER_NEAREST), np.ones((3, 3), np.uint8))
            filled = cv2.inpaint(small, smask, 9, cv2.INPAINT_TELEA)
            filled = cv2.GaussianBlur(cv2.resize(filled, (S, S), interpolation=cv2.INTER_CUBIC), (0, 0), 2)
            soft = np.clip(cv2.GaussianBlur(hole.astype(np.float32) / 255, (0, 0), 1.5 * S / 1024) * 1.6, 0, 1)[..., None]  # feathered join, no visible rim
            out8 = np.clip(out8 * (1 - soft) + filled * soft, 0, 255).astype(np.uint8)
            # Unpainted nostril rim: the surrounding skin tone, darkened as a shadow. A flat dark colour reads as a black hole from the side.
            dark = (self.ctx["nostril"] > 0) & (hole > 0) & (self.cover < 0.01)
            out8[dark] = np.clip(filled[dark] * 0.62, 0, 255).astype(np.uint8)
        zone = self.ctx["zone"]
        if zone.any():  # the nostril rim is made of tiny steep triangles: smooth that patch so it reads as one soft shape, not shreds
            sm = cv2.GaussianBlur(out8, (0, 0), 2.2 * S / 1024 + 0.4)
            zs = np.clip(cv2.GaussianBlur(zone.astype(np.float32) / 255, (0, 0), 3.0 * S / 1024), 0, 1)[..., None]
            out8 = np.clip(out8 * (1 - zs) + sm * zs, 0, 255).astype(np.uint8)
        if tone and len(tone) == 3:
            out8 = np.clip(out8.astype(np.float32) * np.array([tone[2], tone[1], tone[0]], np.float32), 0, 255).astype(np.uint8)
        out8[mask == 0] = 0
        return out8, coverage


class Session:
    def __init__(self, layout: dict):
        self.layout = layout
        self.live = LiveStitcher(layout, 1024)
        self.frames: list[tuple[bytes, list]] = []  # kept for the high-resolution rebuild
        self.lock = threading.Lock()
        self.version = 0
        self.hires = "none"  # none | building | done | error


SESSIONS: dict[str, Session] = {}


def start(name: str, layout: dict) -> dict:
    SESSIONS[name] = Session(layout)
    return {"status": "ok"}


def add(name: str, jpeg: bytes, lm) -> dict:
    s = SESSIONS.get(name)
    if s is None:
        return {"status": "error", "message": "stitch session not started"}
    img = cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        return {"status": "error", "message": "bad image"}
    with s.lock:
        ok = s.live.add(img, lm)
        if ok:
            s.frames.append((jpeg, lm))
            s.version += 1
        return {"status": "ok", "added": ok, "photos": s.live.photos, "version": s.version}


def preview(name: str, tone=None):
    s = SESSIONS.get(name)
    if s is None:
        return None
    with s.lock:
        img, cov = s.live.render(tone)
        version = s.version
        tri = s.live.tri_coverage().tobytes()
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 88])
    return (buf.tobytes(), version, cov, tri) if ok else None


def _build_hires(name: str, scan_dir: Path, tone):
    s = SESSIONS[name]
    try:
        hi = LiveStitcher(s.layout, 2048)
        for jpeg, lm in list(s.frames):
            img = cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR)
            if img is not None:
                hi.add(img, lm)
        out, _ = hi.render(tone)
        rgba = cv2.cvtColor(out, cv2.COLOR_BGR2BGRA)
        rgba[..., 3] = hi.ctx["mesh_mask"]
        cv2.imwrite(str(scan_dir / "skin.png"), rgba)
        cv2.imwrite(str(scan_dir / "preview.jpg"), cv2.resize(out, (1024, 1024), interpolation=cv2.INTER_AREA), [cv2.IMWRITE_JPEG_QUALITY, 92])
        s.hires = "done"
    except Exception:  # noqa: BLE001 - background job: record failure, keep the 1024 result
        s.hires = "error"


def finish(name: str, scan_dir: Path, tone=None) -> dict:
    s = SESSIONS.get(name)
    if s is None or s.live.photos == 0:
        return {"status": "error", "message": "no photos were stitched"}
    scan_dir.mkdir(parents=True, exist_ok=True)
    with s.lock:
        out, cov = s.live.render(tone)
        rgba = cv2.cvtColor(out, cv2.COLOR_BGR2BGRA)
        rgba[..., 3] = s.live.ctx["mesh_mask"]
        cv2.imwrite(str(scan_dir / "skin.png"), rgba)
        cv2.imwrite(str(scan_dir / "preview.jpg"), out, [cv2.IMWRITE_JPEG_QUALITY, 92])
        photos = s.live.photos
    s.hires = "building"
    threading.Thread(target=_build_hires, args=(name, scan_dir, tone), daemon=True).start()
    return {"status": "ok", "photos_used": photos, "coverage_percent": round(100 * cov), "hires": "building", "t": time.time()}

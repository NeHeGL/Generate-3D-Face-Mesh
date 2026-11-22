/**
 * Face scan, Face-ID style: look at the camera and move your head around naturally — no chair turns.
 * Left half: your camera. Right half: your face being mapped, filling in as coverage grows.
 *
 * The scan tracks real head angles (yaw/pitch) and keeps the best photo for each cell of a grid of
 * poses. Accuracy % = how well that grid is covered by sharp, steady photos. Every saved photo carries
 * its measured angle so the 3D head builder can use it later.
 */
import { estimateHeadPose, scoreCapture } from "/static/face-pose.js?v=25";

const CAPTURE_PX = 640;
const SCAN_WIDE = 1.5; // saved scan photos include hair, ears and neck around the on-screen circle
const DISPLAY_WIDE = 1; // how much of that saved area the camera window shows: 1 = just the circle (a tight crop on your face, like the mask); 1.5 = the whole saved area
const FRONTAL_PX = 1024;
const FACE_PX_GOOD = 200; // the saved photos are cropped and enlarged around your face, so what counts is how many camera pixels it covers, not how big it looks
const BASELINE_FRAMES = 20;
const CAPTURE_INTERVAL_MS = 110;
const FRONTAL_INDEX = 100; // hero photo slot; grid cells use 0..44
const RAD = 180 / Math.PI;
const DONE_AT = 60; // % accuracy before "Done" unlocks

const YAWS = [-40, -30, -20, -10, 0, 10, 20, 30, 40]; // degrees of real head turn the outer columns stand for (the sides of the face need real turns)
const PITCHES = [16, 8, 0, -8, -16]; // top row = looking up (small range: big tilts are hard and the tracker loses you)
const YAW_TOL = 6.5;
const PITCH_TOL = 5.5;
const CELLS = [];
PITCHES.forEach((p, r) => YAWS.forEach((y, c) => CELLS.push({ yaw: y, pitch: p, r, c, w: 1 - 0.35 * Math.hypot(y / 40, p / 16) / Math.SQRT2 })));

const store = {
  get: (k, fallback = false) => {
    try {
      const v = localStorage.getItem(k);
      return v === null ? fallback : v === "1";
    } catch {
      return fallback;
    }
  },
  set: (k, v) => {
    try {
      localStorage.setItem(k, v ? "1" : "0");
    } catch {
      /* ignore */
    }
  },
};

export async function runFaceScan(root, canonicalName, api) {
  let stream = null;
  let framing = null;
  let settled = false;
  let finishing = false;
  let captureBusy = false;
  let framesSeen = 0;
  let lastCaptureAt = 0;
  let frontal = null;
  let frontalLm = null;
  let frontalAspect = 16 / 9;
  const flipYaw = false; // fixed: the old Flip buttons were never needed
  const flipPitch = true;
  let edges = null;
  let mod2 = null;
  let lastLm = null;
  // Your own reachable range, learned as you move. Grid rows/columns are scaled to it, so the top row
  // means "as far up as YOU can look" instead of an absolute angle that may be out of reach.
  const reach = { yaw: 0, up: 0, down: 0 };
  let tris = null; // triangles derived from the tessellation edges
  let canon = null; // frontal layout (atlas coordinates) per landmark, from the calibration frame
  const ATLAS = 2048;
  const atlas = document.createElement("canvas");
  atlas.width = ATLAS;
  atlas.height = ATLAS;
  const actx = atlas.getContext("2d");
  let best = null; // best facing score each triangle has been painted with
  let bq = null; // the same score without the face-size factor (used for the quality tint)
  let front = null; // +1/-1: winding of each triangle when seen from the front (used for back-face culling)
  const idCanvas = document.createElement("canvas"); // visibility (z-buffer) pass: each triangle drawn in its own colour
  idCanvas.width = 640;
  idCanvas.height = 640;
  const ictx = idCanvas.getContext("2d", { willReadFrequently: true });
  const normCanvas = document.createElement("canvas"); // face crop with colour matched to the calibration frame
  const nctx = normCanvas.getContext("2d", { willReadFrequently: true });
  const diag = { last: "waiting for calibration", painted: 0, rejected: 0 }; // why painting did or did not happen
  let sharpMax = 0; // sharpest frame seen so far
  let nbrs = null; // triangles that share an edge with each triangle
  let colr = null; // stored average colour of each painted triangle (3 numbers each)
  let pendN = null; // how many frames in a row have offered the same odd colour for a triangle
  let pendC = null; // that odd colour
  let lastCrop = null; // pixel data of the colour-matched face crop, for sampling colours
  let patchData = null; // per triangle: its uncorrected pixels, average colour and the colour gain applied to it
  let feature = null; // 1 = a real, persistent feature (birthmark, mole): never colour-corrected
  let lastHarmAt = 0;
  // Live stitching: good frames go to the server as they are captured; it blends them into one seamless skin texture.
  let stitchStarted = false;
  let stitchBusy = false;
  let stitchDirty = false;
  let previewBusy = false;
  let stitchCoverage = null;
  let showMesh = true; // right pane: live mapped mesh by default; the stitched skin is the option
  let previewTimer = null;
  let triCov = null; // Uint8Array: how complete each mesh triangle is in the stitched skin (from the server)
  let canonS = null; // canon points scaled to the stitched image
  let canonSk = 0;
  const snapCanvas = document.createElement("canvas");
  const snapCtx = snapCanvas.getContext("2d");
  let showWeak = false; // highlight weak (low quality) areas on the map; off by default because it looks messy
  const liveCanvas = document.createElement("canvas"); // current camera frame, used as the texture of the live map
  const lctx = liveCanvas.getContext("2d");
  const holeSmall = document.createElement("canvas"); // small copy used to interpolate skin into unpainted areas
  holeSmall.width = 256;
  holeSmall.height = 256;
  const hsx = holeSmall.getContext("2d", { willReadFrequently: true });
  const holeAtlas = document.createElement("canvas"); // the interpolated skin, same size as the atlas
  holeAtlas.width = ATLAS;
  holeAtlas.height = ATLAS;
  const hax = holeAtlas.getContext("2d");
  let holeReady = false;
  let lastInpaintAt = 0;
  let lastInpaintPainted = -1;
  let nostril = null; // 1 = triangle near a nostril: keeps its real (dark) colour, never colour-corrected or filled
  let userPink = 0; // -1..1: how much red to take out of the painted skin
  let userWarm = 0; // -1..1: cooler .. warmer
  const tcan = document.createElement("canvas"); // scratch canvas for one patch
  const tctx = tcan.getContext("2d", { willReadFrequently: true });
  let prevLm = null; // previous landmarks, to spot tracking jumps
  let skinTarget = null; // consensus skin colour (updated continuously from many frames)
  const skinSamples = []; // raw skin colour seen in recent good frames
  let skinFrames = 0;
  const SKIN_POINTS = [205, 425, 151, 9, 117, 346]; // cheeks, forehead, mid-cheeks
  let acc = null; // accumulated blend weight per triangle (so many sharp frames average together)
  let eyeLoops = []; // eye contours, filled as fans so the eyes are not black holes
  let nBase = 478;
  let lastPaintAt = 0;
  let lastDrawAt = 0;
  let pose = { yaw: 0, pitch: 0 };
  // Dead-ahead is found from the face itself: the nose sits midway between the two cheek edges only when you face the camera,
  // whatever the camera's position. Frames like that set the centre; the first frames are not assumed to be straight.
  const frontalYaws = [];
  const symOf = (lm) => {
    const l = lm[234], r = lm[454], n = lm[1];
    const w = Math.abs(r.x - l.x);
    return w > 0.02 ? (n.x - (l.x + r.x) / 2) / w : 1;
  };
  let liveParts = null; // size / steadiness / centring of the frame on screen right now
  let recenter = null; // frames still to average after the Re-center button
  const baseline = { yaw: 0, pitch: 0, n: 0, sy: 0, sp: 0, ready: false, lmSum: null }; // lmSum: running sum of the landmarks over the calibration frames
  const slots = Array(CELLS.length).fill(null);

  const wrap = document.createElement("section");
  wrap.className = "card face-scan-card orbit-card";
  wrap.innerHTML = `
    <div class="scan-head"><h2>Face Scan</h2><p class="scan-hint" id="scan-hint">Starting camera…</p></div>
    <div class="orbit-acc"><div class="orbit-acc-bar"><div id="acc-fill"></div></div><span id="acc-text">Scan accuracy: 0%</span></div>
    <div class="scan-cols">
      <div class="scan-left">
        <div class="orbit-pane"><div class="scan-video-wrap orbit-video" id="scan-video-wrap"><video id="scan-video" playsinline muted autoplay></video></div><p class="orbit-cap">Your camera</p></div>
        <div class="orbit-grid" id="orbit-grid" aria-label="Pose coverage"></div>
        <p class="orbit-legend"><span class="lg red"></span> not captured &nbsp; <span class="lg yellow"></span> low quality &nbsp; <span class="lg green"></span> good</p>
        <button type="button" class="btn-ghost recenter-btn" id="scan-recenter" title="Look straight at the camera, then click: that becomes the centre square">Re-center</button>
      </div>
      <div class="scan-right">
        <div class="scan-mid">
          <div class="orbit-pane"><canvas id="orbit-model" class="orbit-model" width="480" height="480"></canvas><img id="stitch-live" class="orbit-model" alt="Stitched skin" style="display:none"><p class="orbit-cap">Live mesh: your face as it is mapped</p></div>
          <div class="scan-texts">
            <div class="scan-live scan-stats" id="scan-live" aria-live="off"></div>
          </div>
        </div>
        <p class="orbit-tip" id="scan-tip"></p>
        <div class="step-actions step-actions--row scan-buttons">
          <button type="button" class="btn-ghost" id="scan-cancel">Cancel</button>
          <button type="button" class="btn-primary" id="scan-done" disabled>Save</button>
        </div>
      </div>
    </div>
  `;
  const video = wrap.querySelector("#scan-video");
  // The Re-center button lives in the page's footer line, bottom-left, on the same row as the credit text (the empty #hw-mode slot).
  const recenterBtn = wrap.querySelector("#scan-recenter");
  const footerSlot = document.getElementById("hw-mode");
  if (recenterBtn && footerSlot) { footerSlot.innerHTML = ""; recenterBtn.style.cssText = "margin:0;padding:0.3rem 0.9rem;font-size:0.85rem"; footerSlot.appendChild(recenterBtn); }
  const videoWrap = wrap.querySelector("#scan-video-wrap");
  const hint = wrap.querySelector("#scan-hint");
  const liveEl = wrap.querySelector("#scan-live");
  const accFill = wrap.querySelector("#acc-fill");
  const accText = wrap.querySelector("#acc-text");
  const tipEl = wrap.querySelector("#scan-tip");
  const liveImg = wrap.querySelector("#stitch-live");
  // The numbers beside the face map: one fixed row per fact (label left, value right), so nothing grows, shrinks or wraps.
  let statusMsg = "Starting…";
  let liveRows = null;
  let qualRows = [];
  const esc = (v) => String(v).replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
  function renderStats() {
    const rows = [["Status", statusMsg], ...(liveRows || []), ...(liveRows ? qualRows : [])];
    liveEl.innerHTML = rows.map(([k, v]) => `<span class="k">${esc(k)}</span><span class="v" title="${esc(v)}">${esc(v)}</span>`).join("");
  }
  function showStatus(msg) {
    statusMsg = msg;
    liveRows = null;
    renderStats();
  }
  const grid = wrap.querySelector("#orbit-grid");
  const btnDone = wrap.querySelector("#scan-done");
  const btnCancel = wrap.querySelector("#scan-cancel");
  const canvas = wrap.querySelector("#orbit-model");
  const g2 = canvas.getContext("2d");

  const cellEls = [];
  PITCHES.forEach((_, r) => {
    const row = document.createElement("div");
    row.className = "orbit-cells";
    YAWS.forEach((__, c) => {
      const el = document.createElement("span");
      el.className = "orbit-cell";
      row.appendChild(el);
      cellEls[r * YAWS.length + c] = el;
    });
    grid.appendChild(row);
  });
  let activeCell = -1;
  let needsCell = []; // per square: still circled (needs work)
  const Q_TARGET = 0.85; // photo quality that counts as 100% (quality is 0..1; see the formula where photos are scored)
  const cellPct = (i) => (slots[i] ? Math.min(1, slots[i].quality / Q_TARGET) : 0);

  /** 0..100: weighted coverage by quality, with partial credit from neighbouring cells (corners are hard to reach). */
  function accuracy() {
    let num = 0;
    let den = 0;
    CELLS.forEach((cell, i) => {
      let credit = slots[i] ? slots[i].quality : 0;
      for (const dr of [-1, 0, 1])
        for (const dc of [-1, 0, 1]) {
          if (!dr && !dc) continue;
          const r = cell.r + dr;
          const c = cell.c + dc;
          if (r < 0 || c < 0 || r >= PITCHES.length || c >= YAWS.length) continue;
          const nb = slots[r * YAWS.length + c];
          if (nb) credit = Math.max(credit, 0.55 * nb.quality);
        }
      num += cell.w * credit;
      den += cell.w;
    });
    return Math.min(100, Math.round((num / den) * 115)); // quality rarely hits 1.0, so scale up a little
  }

  /** What colour is the painted skin really? Compare its red/green/blue balance with typical skin and say so. */
  function toneReport() {
    if (!acc || !colr) return "";
    const tg = toneGain();
    let r = 0;
    let g = 0;
    let b = 0;
    let n = 0;
    for (let t = 0; t < acc.length; t++) {
      if (acc[t] === 0) continue;
      r += colr[t * 3] * tg[0];
      g += colr[t * 3 + 1] * tg[1];
      b += colr[t * 3 + 2] * tg[2];
      n++;
    }
    if (n < 40) return "";
    const gr = g / Math.max(r, 1);
    const br = b / Math.max(r, 1);
    const tag = `(green/red ${gr.toFixed(2)}, blue/red ${br.toFixed(2)}; typical skin about 0.70-0.80 and 0.58-0.70)`;
    if (gr < 0.58) return `Skin tone: very pink/red ${tag}.`;
    if (gr < 0.66) return `Skin tone: a bit pink ${tag}.`;
    if (gr > 0.9) return `Skin tone: washed out or greenish ${tag}.`;
    return `Skin tone: in the natural range ${tag}.`;
  }

  /** Plain-language reasons for the accuracy number: which ingredient of the photos is holding it down. */
  function qualitySummary() {
    const have = slots.filter((x) => x && x.parts);
    const tone = toneReport().replace(/^Skin tone: /, "").replace(/\s*\(green.*?\)/, "").replace(/\.$/, "");
    const toneRow = tone ? [["Skin tone", tone]] : [];
    if (!have.length && !liveParts) return toneRow;
    const avg = (k) => (liveParts ? liveParts[k] : have.reduce((a, x) => a + x.parts[k], 0) / have.length);
    const size = avg("size");
    const stab = avg("stab");
    const centre = avg("centre");
    const faceW = avg("faceW");
    const pct = (v) => Math.round(v * 100);
    const worst = Math.min(size, stab, centre);
    let advice = "";
    if (worst >= 0.85) advice = "Photo quality is good";
    else if (worst === size) advice = "Lean in a little, or add light";
    else if (worst === stab) advice = "Hold still for half a second at each position";
    else advice = "Keep your face in the middle of the view";
    return [
      ["Face width", `${Math.round(faceW)} px of a ${video.videoWidth || "?"}×${video.videoHeight || "?"} camera picture (${FACE_PX_GOOD}+ is plenty)`],
      ["Steadiness", `${pct(stab)}%`],
      ["Centring", `${pct(centre)}%`],
      ...toneRow,
      ["Tip", advice],
    ];
  }

  function refresh() {
    const acc = accuracy();
    accFill.style.width = `${acc}%`;
    accText.textContent = `Scan accuracy: ${acc}% · skin mapped: ${stitchCoverage ?? paintedPercent()}% · colour consistency: ${colourMatchPercent()}%`;
    // Every square is scored against a FIXED target (a good, sharp, well-centred photo taken at the middle of the square), never against
    // the other squares. So a square that reaches 100% stays at 100%, and only squares below 100% are circled.
    cellEls.forEach((el, i) => {
      const pct = cellPct(i);
      const needs = pct < 1;
      needsCell[i] = needs;
      // Pure green is reserved for 100%; anything less shades from red towards yellow-green.
      el.style.background = !slots[i] ? "rgba(200, 55, 45, 0.45)" : `hsl(${pct >= 1 ? 120 : Math.round(100 * pct)}, 70%, 38%)`;
      el.classList.toggle("need-work", needs);
      el.classList.toggle("dim", !needs);
      el.classList.toggle("active", i === activeCell);
    });
    qualRows = qualitySummary();
    btnDone.disabled = acc < DONE_AT || finishing;
    if (!framing?.isModelReady) hint.textContent = "Loading face tracking (the first time takes 10–20 s)…";
    else if (!baseline.ready) hint.textContent = "Look straight at the camera and hold still for a moment…";
    else if (acc < DONE_AT)
      hint.textContent = "Sit close, keep even light on your face, and move your head slowly, pausing half a second at each extreme.";
    else if (acc < 85) hint.textContent = "Good. Keep moving to fill the remaining dots.";
    else hint.textContent = "Excellent coverage.";
    return acc;
  }

  // ---- right pane: your face, texture-mapped as it is scanned ---------------------------------
  function buildTriangles() {
    if (!edges || tris) return;
    const adj = new Map();
    for (const e of edges) {
      if (!adj.has(e.start)) adj.set(e.start, new Set());
      if (!adj.has(e.end)) adj.set(e.end, new Set());
      adj.get(e.start).add(e.end);
      adj.get(e.end).add(e.start);
    }
    const out = [];
    for (const e of edges) {
      const a = Math.min(e.start, e.end);
      const b = Math.max(e.start, e.end);
      for (const c of adj.get(a)) if (c > b && adj.get(b).has(c)) out.push([a, b, c]);
    }
    // Eyes: the stock mesh has holes there. Fan-fill each eye contour from its centre.
    eyeLoops = [];
    for (const set of [mod2?.FaceLandmarker?.FACE_LANDMARKS_LEFT_EYE, mod2?.FaceLandmarker?.FACE_LANDMARKS_RIGHT_EYE]) {
      if (!set) continue;
      const verts = [...new Set(set.flatMap((e) => [e.start, e.end]))];
      const centre = nBase + eyeLoops.length;
      eyeLoops.push(verts);
      for (const e of set) out.push([centre, e.start, e.end]);
    }
    tris = out;
    best = new Float32Array(out.length);
    acc = new Float32Array(out.length);
    bq = new Float32Array(out.length);
    colr = new Float32Array(out.length * 3);
    pendN = new Int16Array(out.length);
    pendC = new Float32Array(out.length * 3);
    patchData = new Array(out.length).fill(null);
    feature = new Uint8Array(out.length);
    const byEdge = new Map();
    out.forEach(([a, b, c], t) => {
      for (const [u, v] of [[a, b], [b, c], [a, c]]) {
        const key = u < v ? u * 4096 + v : v * 4096 + u;
        if (!byEdge.has(key)) byEdge.set(key, []);
        byEdge.get(key).push(t);
      }
    });
    nbrs = out.map(() => []);
    for (const list of byEdge.values()) for (const t of list) for (const o of list) if (o !== t) nbrs[t].push(o);
  }

  /** Append one synthetic centre point per eye so the fan triangles have a vertex. */
  function withCentres(arr) {
    if (arr.length > nBase) return arr;
    const out = arr.slice();
    for (const verts of eyeLoops) {
      let x = 0;
      let y = 0;
      let z = 0;
      for (const i of verts) {
        x += arr[i].x;
        y += arr[i].y;
        z += arr[i].z || 0;
      }
      out.push({ x: x / verts.length, y: y / verts.length, z: z / verts.length });
    }
    return out;
  }

  /** Affine transform [a b c d e f] mapping source triangle s0,s1,s2 onto destination d0,d1,d2. */
  function affine(s0, s1, s2, d0, d1, d2) {
    const det = s0.x * (s1.y - s2.y) - s1.x * (s0.y - s2.y) + s2.x * (s0.y - s1.y);
    if (Math.abs(det) < 1e-9) return null;
    const a = (d0.x * (s1.y - s2.y) - d1.x * (s0.y - s2.y) + d2.x * (s0.y - s1.y)) / det;
    const c = (-d0.x * (s1.x - s2.x) + d1.x * (s0.x - s2.x) - d2.x * (s0.x - s1.x)) / det;
    const e = (d0.x * (s1.x * s2.y - s2.x * s1.y) - d1.x * (s0.x * s2.y - s2.x * s0.y) + d2.x * (s0.x * s1.y - s1.x * s0.y)) / det;
    const b = (d0.y * (s1.y - s2.y) - d1.y * (s0.y - s2.y) + d2.y * (s0.y - s1.y)) / det;
    const d = (-d0.y * (s1.x - s2.x) + d1.y * (s0.x - s2.x) - d2.y * (s0.x - s1.x)) / det;
    const f = (d0.y * (s1.x * s2.y - s2.x * s1.y) - d1.y * (s0.x * s2.y - s2.x * s0.y) + d2.y * (s0.x * s1.y - s1.x * s0.y)) / det;
    return [a, b, c, d, e, f];
  }

  function drawTri(ctx, img, s0, s1, s2, d0, d1, d2, grow = 1.3) {
    const m = affine(s0, s1, s2, d0, d1, d2);
    if (!m) return;
    const cx = (d0.x + d1.x + d2.x) / 3;
    const cy = (d0.y + d1.y + d2.y) / 3;
    const ex = (p) => ({ x: p.x + Math.sign(p.x - cx) * grow, y: p.y + Math.sign(p.y - cy) * grow }); // hides hairline seams
    const e0 = ex(d0);
    const e1 = ex(d1);
    const e2 = ex(d2);
    ctx.save();
    ctx.beginPath();
    ctx.moveTo(e0.x, e0.y);
    ctx.lineTo(e1.x, e1.y);
    ctx.lineTo(e2.x, e2.y);
    ctx.closePath();
    ctx.clip();
    ctx.setTransform(m[0], m[1], m[2], m[3], m[4], m[5]);
    ctx.drawImage(img, 0, 0);
    ctx.restore();
  }

  function setCanon(lm, ar) {
    let minX = 1e9;
    let minY = 1e9;
    let maxX = -1e9;
    let maxY = -1e9;
    for (const p of lm) {
      minX = Math.min(minX, p.x * ar);
      minY = Math.min(minY, p.y);
      maxX = Math.max(maxX, p.x * ar);
      maxY = Math.max(maxY, p.y);
    }
    const span = Math.max(maxX - minX, maxY - minY);
    const k = (0.9 * ATLAS) / span;
    const ox = (ATLAS - (maxX - minX) * k) / 2 - minX * k;
    const oy = (ATLAS - (maxY - minY) * k) / 2 - minY * k;
    canon = withCentres(lm.map((p) => ({ x: p.x * ar * k + ox, y: p.y * k + oy, z: 0 })));
    nostril = new Uint8Array(tris.length);
    if (canon[98] && canon[327]) {
      const w98 = canon[98];
      const w327 = canon[327];
      const reach = 0.2 * Math.hypot(w98.x - w327.x, w98.y - w327.y);
      tris.forEach(([a, b, c], t) => {
        const cx = (canon[a].x + canon[b].x + canon[c].x) / 3;
        const cy = (canon[a].y + canon[b].y + canon[c].y) / 3;
        if (Math.hypot(cx - w98.x, cy - w98.y) < reach || Math.hypot(cx - w327.x, cy - w327.y) < reach) nostril[t] = 1;
      });
    }
    front = new Int8Array(tris.length);
    tris.forEach(([a, b, c], t) => {
      front[t] = area2(canon[a], canon[b], canon[c]) >= 0 ? 1 : -1;
    });
  }

  const area2 = (A, B, C) => (B.x - A.x) * (C.y - A.y) - (B.y - A.y) * (C.x - A.x);

  /** Triangle indices that are front-facing (not back-facing) for projected points Q, far to near. */
  function frontTrianglesFarToNear(Q, mirrored, backFacing = false) {
    const list = [];
    for (let t = 0; t < tris.length; t++) {
      const [a, b, c] = tris[t];
      const ar = area2(Q[a], Q[b], Q[c]);
      if (Math.abs(ar) < 1e-6) continue;
      if ((Math.sign(ar) !== (mirrored ? -front[t] : front[t])) !== backFacing) continue; // back-facing ones are hidden behind the face (or, with backFacing, the only ones returned)
      list.push([t, (Q[a].z + Q[b].z + Q[c].z) / 3]);
    }
    list.sort((u, v) => v[1] - u[1]); // MediaPipe: smaller z = nearer, so larger z is drawn first
    return list.map((e) => e[0]);
  }

  /** Which triangles can the camera actually see right now (nothing in front of them)? Cheap z-buffer by painting ids. */
  function visibleSet(Q, order) {
    const W = idCanvas.width;
    let minX = 1e9;
    let minY = 1e9;
    let maxX = -1e9;
    let maxY = -1e9;
    for (const p of Q) {
      minX = Math.min(minX, p.x);
      minY = Math.min(minY, p.y);
      maxX = Math.max(maxX, p.x);
      maxY = Math.max(maxY, p.y);
    }
    const k = (W - 4) / Math.max(maxX - minX, maxY - minY, 1);
    const X = (p) => (p.x - minX) * k + 2;
    const Y = (p) => (p.y - minY) * k + 2;
    ictx.setTransform(1, 0, 0, 1, 0, 0);
    ictx.fillStyle = "#000";
    ictx.fillRect(0, 0, W, W);
    ictx.imageSmoothingEnabled = false;
    for (const t of order) {
      const [a, b, c] = tris[t];
      const id = t + 1;
      ictx.fillStyle = `rgb(${id & 255},${(id >> 8) & 255},${(id >> 16) & 255})`;
      ictx.beginPath();
      ictx.moveTo(X(Q[a]), Y(Q[a]));
      ictx.lineTo(X(Q[b]), Y(Q[b]));
      ictx.lineTo(X(Q[c]), Y(Q[c]));
      ictx.closePath();
      ictx.fill();
    }
    const px = ictx.getImageData(0, 0, W, W).data;
    const seen = new Set();
    for (const t of order) {
      const [a, b, c] = tris[t];
      const mx = (X(Q[a]) + X(Q[b]) + X(Q[c])) / 3;
      const my = (Y(Q[a]) + Y(Q[b]) + Y(Q[c])) / 3;
      // Centre plus three points halfway to each corner: tiny triangles often lose their exact centre pixel
      // to anti-aliasing, so any one of them showing the triangle's own colour counts as visible.
      for (const [sx, sy] of [[mx, my], [(mx + X(Q[a])) / 2, (my + Y(Q[a])) / 2], [(mx + X(Q[b])) / 2, (my + Y(Q[b])) / 2], [(mx + X(Q[c])) / 2, (my + Y(Q[c])) / 2]]) {
        const o = (Math.round(sy) * W + Math.round(sx)) * 4;
        if ((px[o] | (px[o + 1] << 8) | (px[o + 2] << 16)) === t + 1) {
          seen.add(t);
          break;
        }
      }
    }
    return seen;
  }

  /**
   * Crop the face out of the live frame and scale R, G, B so its skin colour matches the calibration frame.
   * Auto-exposure, auto white balance and your room's colour cast change from frame to frame; without this the
   * patches painted from different frames do not match.
   */
  function normalizedCrop(lm) {
    const vw = video.videoWidth;
    const vh = video.videoHeight;
    let minX = 1e9;
    let minY = 1e9;
    let maxX = -1e9;
    let maxY = -1e9;
    for (const p of lm) {
      minX = Math.min(minX, p.x * vw);
      minY = Math.min(minY, p.y * vh);
      maxX = Math.max(maxX, p.x * vw);
      maxY = Math.max(maxY, p.y * vh);
    }
    const padX = (maxX - minX) * 0.15;
    const padY = (maxY - minY) * 0.15;
    const ox = Math.max(0, Math.floor(minX - padX));
    const oy = Math.max(0, Math.floor(minY - padY));
    const cw = Math.min(vw - ox, Math.ceil(maxX - minX + 2 * padX));
    const ch = Math.min(vh - oy, Math.ceil(maxY - minY + 2 * padY));
    if (cw < 16 || ch < 16) return null;
    normCanvas.width = cw;
    normCanvas.height = ch;
    nctx.drawImage(video, ox, oy, cw, ch, 0, 0, cw, ch);
    const img = nctx.getImageData(0, 0, cw, ch);
    const d = img.data;
    let r = 0;
    let g = 0;
    let b = 0;
    let n = 0;
    const patches = [];
    for (const i of SKIN_POINTS) {
      const cx = Math.round(lm[i].x * vw - ox);
      const cy = Math.round(lm[i].y * vh - oy);
      let pr = 0;
      let pg = 0;
      let pb = 0;
      let pn = 0;
      for (let yy = -5; yy <= 5; yy++)
        for (let xx = -5; xx <= 5; xx++) {
          const x = cx + xx;
          const y = cy + yy;
          if (x < 0 || y < 0 || x >= cw || y >= ch) continue;
          const o = (y * cw + x) * 4;
          pr += d[o];
          pg += d[o + 1];
          pb += d[o + 2];
          pn++;
        }
      if (pn) patches.push([pr / pn, pg / pn, pb / pn]);
      r += pr;
      g += pg;
      b += pb;
      n += pn;
    }
    if (n < 30 || patches.length < 4) return { ox, oy, sharp: 0, faceW: maxX - minX, bad: true, why: "face too close to the frame edge" };
    r /= n;
    g /= n;
    b /= n;
    // Occlusion check: skin patches should all be similar. A hand, cup or hair across part of the face makes one or
    // more of them a different colour, or very different from the reference colour: reject the whole frame.
    // Compare chroma (the colour, brightness removed) strictly and brightness loosely: forehead, cheeks and nose are
    // legitimately lit differently, but they are all the same skin colour.
    const lumOf = (q) => (q[0] + q[1] + q[2]) / 3;
    const chromaOf = (q) => [q[0] - lumOf(q), q[1] - lumOf(q), q[2] - lumOf(q)];
    const medLum = patches.map(lumOf).sort((u, v) => u - v)[patches.length >> 1];
    const medChroma = [0, 1, 2].map((k) => patches.map((q) => chromaOf(q)[k]).sort((u, v) => u - v)[patches.length >> 1]);
    let odd = 0;
    for (const q of patches) {
      const ch2 = chromaOf(q);
      const dChroma = Math.hypot(ch2[0] - medChroma[0], ch2[1] - medChroma[1], ch2[2] - medChroma[2]);
      if (dChroma > 40 || Math.abs(lumOf(q) - medLum) > 110) odd++;
    }
    if (odd >= 2) return { ox, oy, sharp: 0, faceW: maxX - minX, bad: true, why: "something covers part of the face (spots differ in colour)" };
    if (skinTarget) {
      const ratio = (u, v) => u / Math.max(v, 1);
      const rr = ratio(skinTarget.r, r);
      const rg = ratio(skinTarget.g, g);
      const rb = ratio(skinTarget.b, b);
      if (Math.max(rr, rg, rb) > 2.1 || Math.min(rr, rg, rb) < 0.48) return { ox, oy, sharp: 0, faceW: maxX - minX, bad: true, why: "lighting changed a lot since calibration" };
    }
    // Reference colour = the MEDIAN skin colour over the last ~300 good frames, nudged halfway toward typical skin chroma
    // (G/R ~0.74, B/R ~0.62) when the camera makes it redder. One odd early frame cannot decide the colour of the whole face.
    const natural = (rr, gg, bb) => {
      const gr = gg / Math.max(rr, 1);
      const br = bb / Math.max(rr, 1);
      const grT = gr < 0.74 ? gr + 0.5 * (0.74 - gr) : gr;
      const brT = br < 0.62 ? br + 0.5 * (0.62 - br) : br;
      const r2 = gg / grT;
      const b2 = brT * r2;
      const k = (0.3 * rr + 0.59 * gg + 0.11 * bb) / (0.3 * r2 + 0.59 * gg + 0.11 * b2); // keep the same brightness
      return { r: r2 * k, g: gg * k, b: b2 * k };
    };
    skinSamples.push([r, g, b]);
    if (skinSamples.length > 300) skinSamples.shift();
    skinFrames += 1;
    if (!skinTarget) {
      skinTarget = natural(r, g, b);
    } else if (skinFrames % 12 === 0 && skinSamples.length >= 12) {
      const med = [0, 1, 2].map((k) => skinSamples.map((q) => q[k]).sort((u, v) => u - v)[skinSamples.length >> 1]);
      const tgt = natural(med[0], med[1], med[2]);
      skinTarget = { r: 0.7 * skinTarget.r + 0.3 * tgt.r, g: 0.7 * skinTarget.g + 0.3 * tgt.g, b: 0.7 * skinTarget.b + 0.3 * tgt.b };
    }
    const clampG = (v) => Math.max(0.65, Math.min(1.5, v));
    const gr = clampG(skinTarget.r / Math.max(r, 1));
    const gg = clampG(skinTarget.g / Math.max(g, 1));
    const gb = clampG(skinTarget.b / Math.max(b, 1));
    for (let i = 0; i < d.length; i += 4) {
      d[i] = Math.min(255, d[i] * gr);
      d[i + 1] = Math.min(255, d[i + 1] * gg);
      d[i + 2] = Math.min(255, d[i + 2] * gb);
    }
    // Sharpness of this frame (edge energy in the middle of the face crop): motion blur lowers it.
    let gsum = 0;
    let gcount = 0;
    const gx0 = Math.floor(cw * 0.2);
    const gx1 = Math.floor(cw * 0.8);
    const gy0 = Math.floor(ch * 0.2);
    const gy1 = Math.floor(ch * 0.8);
    for (let y = gy0; y < gy1; y += 3)
      for (let x = gx0; x < gx1; x += 3) {
        const o = (y * cw + x) * 4;
        const l = d[o] + d[o + 1] + d[o + 2];
        gsum += Math.abs(l - (d[o + 4] + d[o + 5] + d[o + 6])) + Math.abs(l - (d[o + cw * 4] + d[o + cw * 4 + 1] + d[o + cw * 4 + 2]));
        gcount++;
      }
    nctx.putImageData(img, 0, 0);
    lastCrop = { d, cw, ch };
    return { ox, oy, sharp: gcount ? gsum / gcount : 0, faceW: maxX - minX };
  }

  /** Blend every sharp, reasonably front-on, UNOCCLUDED view of each triangle into the atlas (weighted running average). */
  function paintAtlas(lm, jitter) {
    if (!tris || !canon || !front || !video.videoWidth) return;
    const vw = video.videoWidth;
    const vh = video.videoHeight;
    const crop = normalizedCrop(lm);
    if (!crop || crop.bad || !normCanvas.width) {
      diag.last = crop?.why || "no usable frame";
      return; // covered or badly lit frame: paint nothing
    }
    const P = withCentres(lm.map((p) => ({ x: p.x * vw - crop.ox, y: p.y * vh - crop.oy, z: p.z * vw })));
    if (prevLm) {
      let move = 0;
      for (let i = 0; i < lm.length; i += 6) move += Math.hypot(lm[i].x - prevLm[i].x, lm[i].y - prevLm[i].y);
      move /= Math.ceil(lm.length / 6);
      prevLm = lm.map((q) => ({ x: q.x, y: q.y }));
      if (move * vw > 0.09 * crop.faceW) {
        diag.last = "moving too fast / tracking jumped";
        return; // landmarks jumped: the tracker lost the face or something covered it
      }
    } else {
      prevLm = lm.map((q) => ({ x: q.x, y: q.y }));
    }
    const order = frontTrianglesFarToNear(P, false);
    const visible = visibleSet(P, order);
    // Quality of THIS frame: sharpness (relative to the sharpest frame seen) and how big your face is in it.
    sharpMax = Math.max(sharpMax, crop.sharp);
    const sharpN = Math.max(0.25, Math.min(1, crop.sharp / (sharpMax || 1)));
    const sizeN = Math.min(1.4, crop.faceW / 350);
    for (const t of order) {
      if (!visible.has(t)) continue; // something is in front of it (e.g. the nose hides the far cheek)
      const [a, b, c] = tris[t];
      const A = P[a];
      const B = P[b];
      const C = P[c];
      const ux = B.x - A.x;
      const uy = B.y - A.y;
      const uz = B.z - A.z;
      const wx = C.x - A.x;
      const wy = C.y - A.y;
      const wz = C.z - A.z;
      const nx = uy * wz - uz * wy;
      const ny = uz * wx - ux * wz;
      const nz = ux * wy - uy * wx;
      const len = Math.hypot(nx, ny, nz) || 1;
      const facing = Math.abs(nz) / len;
      if (facing < 0.2) continue; // edge-on: nothing useful to copy
      const score = facing * facing * sharpN * sizeN;
      // Keep the best photo of each patch ever seen. Replace only when a clearly better one arrives, never blend,
      // so colour and detail do not drift or get worse.
      if (acc[t] > 0 && score <= best[t] * 1.12) continue;
      const patch = renderRaw(t, normCanvas, A, B, C);
      if (!patch) continue;
      const decision = decideGain(t, patch.mean);
      if (nostril && nostril[t]) {
        // A nostril is really dark, but the mesh there is coarse and a fully dark block looks harsh: pull it only 40% of the
        // way toward the surrounding skin. It stays clearly dark without being plugged or blocky.
        decision.gain = decision.gain.map((g) => 1 + 0.4 * (g - 1));
        decision.feature = false;
        decision.flat = null;
      }
      if (decision.flat) {
        // Not skin (hair, deep shadow, hairline edge): paint it as plain skin in the neighbours' colour instead of a blotch.
        const px = patch.raw.data;
        for (let i = 0; i < px.length; i += 4) {
          px[i] = decision.flat[0];
          px[i + 1] = decision.flat[1];
          px[i + 2] = decision.flat[2];
        }
        patch.mean = decision.flat;
      }
      if (Math.max(...decision.gain.map((g) => Math.abs(g - 1))) > 0.08) diag.rejected += 1; // counted as "adjusted"
      diag.painted += 1;
      diag.last = "painting";
      patchData[t] = { ...patch, gain: decision.gain };
      feature[t] = decision.feature ? 1 : 0;
      colr[t * 3] = patch.mean[0] * decision.gain[0];
      colr[t * 3 + 1] = patch.mean[1] * decision.gain[1];
      colr[t * 3 + 2] = patch.mean[2] * decision.gain[2];
      best[t] = score;
      bq[t] = facing * facing * sharpN;
      acc[t] = 1;
      putPatch(t);
    }
  }

  /** Average colour of a small square in the colour-matched crop. */
  function sampleCrop(x, y) {
    if (!lastCrop) return null;
    const { d, cw, ch } = lastCrop;
    let r = 0;
    let g = 0;
    let b = 0;
    let n = 0;
    for (let yy = -2; yy <= 2; yy++)
      for (let xx = -2; xx <= 2; xx++) {
        const px = Math.round(x) + xx;
        const py = Math.round(y) + yy;
        if (px < 0 || py < 0 || px >= cw || py >= ch) continue;
        const o = (py * cw + px) * 4;
        r += d[o];
        g += d[o + 1];
        b += d[o + 2];
        n++;
      }
    return n ? [r / n, g / n, b / n] : null;
  }

  const cdist = (u, v) => Math.hypot(u[0] - v[0], u[1] - v[1], u[2] - v[2]);
  const median = (arr) => arr.slice().sort((x, y) => x - y)[arr.length >> 1];

  /** Colour of the painted neighbours of triangle t: median colour and how much they differ among themselves. */
  function neighbourStats(t) {
    const cols = [];
    for (const o of nbrs[t]) if (acc[o] > 0) cols.push([colr[o * 3], colr[o * 3 + 1], colr[o * 3 + 2]]);
    if (cols.length < 2) return null;
    const med = [0, 1, 2].map((k) => median(cols.map((q) => q[k])));
    const spread = median(cols.map((q) => cdist(q, med)));
    return { med, spread, n: cols.length };
  }

  /**
   * Should triangle t accept this candidate colour? It has to look like the patches around it. A real feature
   * (a birthmark, a mole, a scar) differs from its neighbours but shows up in every frame, so a colour that keeps
   * being offered several frames in a row is accepted. A hand, shadow or blur does not persist like that.
   */
  function fitsNeighbours(t, col) {
    const st = neighbourStats(t);
    if (!st) return true; // nothing to compare with yet
    const limit = Math.max(26, 3 * st.spread + 12);
    if (cdist(col, st.med) <= limit) {
      pendN[t] = 0;
      return true;
    }
    const pc = [pendC[t * 3], pendC[t * 3 + 1], pendC[t * 3 + 2]];
    if (pendN[t] > 0 && cdist(pc, col) < 20) pendN[t] += 1;
    else {
      pendN[t] = 1;
      pendC[t * 3] = col[0];
      pendC[t * 3 + 1] = col[1];
      pendC[t * 3 + 2] = col[2];
    }
    if (pendN[t] >= 10) {
      pendN[t] = 0;
      return true; // consistent across many frames: a real feature of your skin
    }
    return false;
  }

  /** Share of painted triangles whose colour matches the patches around them (ignores isolated real features). */
  function colourMatchPercent() {
    if (!tris || !nbrs) return 0;
    let ok = 0;
    let total = 0;
    for (let t = 0; t < tris.length; t++) {
      if (acc[t] === 0) continue;
      total++;
      const st = neighbourStats(t);
      if (!st) {
        ok++;
        continue;
      }
      const limit = Math.max(22, 3 * st.spread + 10);
      if (cdist([colr[t * 3], colr[t * 3 + 1], colr[t * 3 + 2]], st.med) <= limit) ok++;
    }
    return total ? Math.round((100 * ok) / total) : 0;
  }

  /** Draw source triangle (A,B,C) into a tight scratch canvas at atlas position; return its pixels and mean colour. */
  function renderRaw(t, srcImg, A, B, C) {
    const [a, b, c] = tris[t];
    const da = canon[a];
    const db = canon[b];
    const dc = canon[c];
    const x0 = Math.max(0, Math.floor(Math.min(da.x, db.x, dc.x)) - 2);
    const y0 = Math.max(0, Math.floor(Math.min(da.y, db.y, dc.y)) - 2);
    const x1 = Math.min(ATLAS, Math.ceil(Math.max(da.x, db.x, dc.x)) + 2);
    const y1 = Math.min(ATLAS, Math.ceil(Math.max(da.y, db.y, dc.y)) + 2);
    const w = x1 - x0;
    const h = y1 - y0;
    if (w <= 2 || h <= 2 || w > 400 || h > 400) return null;
    tcan.width = w;
    tcan.height = h;
    tctx.clearRect(0, 0, w, h);
    const areaC = Math.abs((db.x - da.x) * (dc.y - da.y) - (db.y - da.y) * (dc.x - da.x)) / 2;
    if (areaC < 80) {
      // Too small to carry a texture warp (e.g. beside the nostrils): paint it as a small flat patch of the colour it covers.
      const col = sampleCrop((A.x + B.x + C.x) / 3, (A.y + B.y + C.y) / 3);
      if (!col) return null;
      const mx = (da.x + db.x + dc.x) / 3 - x0;
      const my = (da.y + db.y + dc.y) / 3 - y0;
      const grow = (q) => {
        const dx = q.x - x0 - mx;
        const dy = q.y - y0 - my;
        const L = Math.hypot(dx, dy) || 1;
        return { x: q.x - x0 + (dx / L) * 1.6, y: q.y - y0 + (dy / L) * 1.6 };
      };
      const g0 = grow(da);
      const g1 = grow(db);
      const g2p = grow(dc);
      tctx.fillStyle = `rgb(${col[0] | 0},${col[1] | 0},${col[2] | 0})`;
      tctx.beginPath();
      tctx.moveTo(g0.x, g0.y);
      tctx.lineTo(g1.x, g1.y);
      tctx.lineTo(g2p.x, g2p.y);
      tctx.closePath();
      tctx.fill();
      return { x0, y0, w, h, raw: tctx.getImageData(0, 0, w, h), mean: col };
    }
    drawTri(tctx, srcImg, A, B, C, { x: da.x - x0, y: da.y - y0 }, { x: db.x - x0, y: db.y - y0 }, { x: dc.x - x0, y: dc.y - y0 });
    const raw = tctx.getImageData(0, 0, w, h);
    const d = raw.data;
    let r = 0;
    let g = 0;
    let bl = 0;
    let n = 0;
    for (let i = 0; i < d.length; i += 4) {
      if (d[i + 3] < 200) continue;
      r += d[i];
      g += d[i + 1];
      bl += d[i + 2];
      n++;
    }
    if (n < 6) return null;
    return { x0, y0, w, h, raw, mean: [r / n, g / n, bl / n] };
  }

  /** Overall colour gain from the two tone sliders. */
  function toneGain() {
    return [
      (1 - 0.25 * userPink) * (1 + 0.12 * userWarm),
      1 + 0.1 * userPink,
      (1 + 0.05 * userPink) * (1 - 0.15 * userWarm),
    ];
  }

  /** Rewrite every painted patch (after a slider moved). */
  function reapplyAll() {
    if (!patchData) return;
    for (let t = 0; t < patchData.length; t++) if (patchData[t] && acc[t] > 0) putPatch(t);
  }

  /** Write patch t into the atlas with its current colour gain. */
  function putPatch(t) {
    const p = patchData[t];
    const out = new ImageData(p.w, p.h);
    const src = p.raw.data;
    const o = out.data;
    const tg = toneGain();
    const g0 = p.gain[0] * tg[0];
    const g1 = p.gain[1] * tg[1];
    const g2c = p.gain[2] * tg[2];
    for (let i = 0; i < src.length; i += 4) {
      o[i] = Math.min(255, src[i] * g0);
      o[i + 1] = Math.min(255, src[i + 1] * g1);
      o[i + 2] = Math.min(255, src[i + 2] * g2c);
      o[i + 3] = src[i + 3];
    }
    tcan.width = p.w;
    tcan.height = p.h;
    tctx.putImageData(out, 0, 0);
    actx.drawImage(tcan, p.x0, p.y0);
  }

  const clampGain = (v) => Math.max(0.75, Math.min(1.33, v));
  const lumaOf = (c) => 0.3 * c[0] + 0.59 * c[1] + 0.11 * c[2];

  /**
   * Colour a patch should have: its neighbours' brightness (so real shadows and light direction stay) combined with the
   * face-wide consensus colour (so colour does not depend on whichever patches happened to be painted first).
   */
  function goalColor(med) {
    if (!skinTarget) return med;
    const ref = [skinTarget.r, skinTarget.g, skinTarget.b];
    const k = lumaOf(med) / Math.max(lumaOf(ref), 1);
    return med.map((v, i) => 0.55 * v + 0.45 * ref[i] * k);
  }

  /**
   * Colour gain for a freshly painted patch: match the patches around it. A real feature (birthmark, mole) differs from
   * its neighbours but is the same in every frame, so a colour offered about 10 frames in a row is left uncorrected.
   */
  function decideGain(t, mean) {
    const st = neighbourStats(t);
    if (!st) {
      pendN[t] = 0;
      return { gain: [1, 1, 1], feature: false };
    }
    const goal = goalColor(st.med);
    const desired = [0, 1, 2].map((k) => clampGain(goal[k] / Math.max(mean[k], 1)));
    const limit = Math.max(26, 3 * st.spread + 12);
    if (cdist(mean, st.med) > limit) {
      const pc = [pendC[t * 3], pendC[t * 3 + 1], pendC[t * 3 + 2]];
      if (pendN[t] > 0 && cdist(pc, mean) < 20) pendN[t] += 1;
      else {
        pendN[t] = 1;
        pendC[t * 3] = mean[0];
        pendC[t * 3 + 1] = mean[1];
        pendC[t * 3 + 2] = mean[2];
      }
      if (pendN[t] >= 10) {
        pendN[t] = 0;
        return { gain: [1, 1, 1], feature: true };
      }
    } else {
      pendN[t] = 0;
    }
    if (cdist(mean, st.med) > Math.max(60, limit * 2)) return { gain: [1, 1, 1], feature: false, flat: st.med.slice() };
    return { gain: desired, feature: false };
  }

  /**
   * Relax the colour field: nudge every patch halfway toward the median colour of its neighbours, so earlier mistakes get
   * pulled into line instead of staying wrong and then rejecting better photos.
   */
  function harmonise() {
    if (!tris || !patchData || !nbrs) return;
    let changed = 0;
    for (let t = 0; t < tris.length && changed < 220; t++) {
      if (acc[t] === 0 || feature[t]) continue;
      const p = patchData[t];
      if (!p) continue;
      const st = neighbourStats(t);
      if (!st) continue;
      const goal = goalColor(st.med);
      const desired = [0, 1, 2].map((k) => clampGain(goal[k] / Math.max(p.mean[k], 1)));
      const isNos = nostril && nostril[t];
      const want = isNos ? desired.map((v) => 1 + 0.4 * (v - 1)) : desired;
      const g = p.gain.map((v, k) => 0.5 * v + 0.5 * want[k]);
      if (Math.max(...g.map((v, k) => Math.abs(v - p.gain[k]))) < 0.015) continue;
      p.gain = g;
      colr[t * 3] = p.mean[0] * g[0];
      colr[t * 3 + 1] = p.mean[1] * g[1];
      colr[t * 3 + 2] = p.mean[2] * g[2];
      putPatch(t);
      changed++;
    }
  }

  /** Interpolate the painted skin outward into unpainted areas, smoothly (no flat polygons). */
  function inpaintHoles() {
    const N = 256;
    hsx.clearRect(0, 0, N, N);
    hsx.drawImage(atlas, 0, 0, N, N);
    const d = hsx.getImageData(0, 0, N, N).data;
    const has = new Uint8Array(N * N);
    const R = new Float32Array(N * N);
    const G = new Float32Array(N * N);
    const B = new Float32Array(N * N);
    for (let i = 0; i < N * N; i++)
      if (d[i * 4 + 3] > 128) {
        has[i] = 1;
        R[i] = d[i * 4];
        G[i] = d[i * 4 + 1];
        B[i] = d[i * 4 + 2];
      }
    const idx = [];
    const rr = [];
    const gg = [];
    const bb = [];
    for (let pass = 0; pass < 90; pass++) {
      idx.length = 0;
      rr.length = 0;
      gg.length = 0;
      bb.length = 0;
      for (let y = 1; y < N - 1; y++)
        for (let x = 1; x < N - 1; x++) {
          const i = y * N + x;
          if (has[i]) continue;
          let r = 0;
          let g = 0;
          let b = 0;
          let n = 0;
          for (const j of [i - 1, i + 1, i - N, i + N])
            if (has[j]) {
              r += R[j];
              g += G[j];
              b += B[j];
              n++;
            }
          if (n) {
            idx.push(i);
            rr.push(r / n);
            gg.push(g / n);
            bb.push(b / n);
          }
        }
      if (!idx.length) break;
      for (let k = 0; k < idx.length; k++) {
        const i = idx[k];
        has[i] = 1;
        R[i] = rr[k];
        G[i] = gg[k];
        B[i] = bb[k];
      }
    }
    const out = hsx.createImageData(N, N);
    for (let i = 0; i < N * N; i++) {
      out.data[i * 4] = R[i];
      out.data[i * 4 + 1] = G[i];
      out.data[i * 4 + 2] = B[i];
      out.data[i * 4 + 3] = 255;
    }
    hsx.putImageData(out, 0, 0);
    hax.imageSmoothingEnabled = true;
    hax.imageSmoothingQuality = "high";
    hax.drawImage(holeSmall, 0, 0, ATLAS, ATLAS);
  }

  function inpaintIfNeeded() {
    const now = performance.now();
    let painted = 0;
    if (acc) for (let t = 0; t < acc.length; t++) if (acc[t] > 0) painted++;
    if (now - lastInpaintAt < 1500 || painted === lastInpaintPainted || painted < 40) return;
    lastInpaintAt = now;
    lastInpaintPainted = painted;
    inpaintHoles();
    holeReady = true;
  }

  /** Fill any triangle still unpainted with the interpolated skin (nostrils dark), so the saved skin map has no gaps. */
  function fillHolesInAtlas() {
    if (!tris || !canon || !acc) return;
    inpaintHoles();
    holeReady = true;
    for (let t = 0; t < tris.length; t++) {
      if (acc[t] > 0) continue;
      const [a, b, c] = tris[t];
      if (nostril && nostril[t]) {
        actx.fillStyle = "rgb(70, 38, 34)";
        actx.beginPath();
        actx.moveTo(canon[a].x, canon[a].y);
        actx.lineTo(canon[b].x, canon[b].y);
        actx.lineTo(canon[c].x, canon[c].y);
        actx.closePath();
        actx.fill();
      } else {
        drawTri(actx, holeAtlas, canon[a], canon[b], canon[c], canon[a], canon[b], canon[c], 0.8);
      }
    }
  }

  async function startStitch(lm0) {
    if (stitchStarted || !canon || !tris) return;
    stitchStarted = true;
    try {
      const r = await api("/api/stitch/start", {
        canonical_name: canonicalName,
        layout: {
          size: ATLAS,
          points: canon.map((q) => [q.x / ATLAS, q.y / ATLAS]),
          triangles: tris,
          // The 3D face as measured at calibration (x, y in units of the picture width, z toward/away), so the avatar can build the mesh.
          neutral3d: lm0 ? lm0.slice(0, 478).map((q) => [q.x, q.y / ((video.videoWidth || 16) / (video.videoHeight || 9)), q.z]) : null,
        },
      });
      if (r?.status !== "ok") stitchStarted = false;
    } catch (e) {
      stitchStarted = false;
      console.warn("stitch start failed:", e);
    }
  }

  /** Send the current camera frame and the face points found in it. The server blends it into the running skin texture. */
  async function sendStitch(lm) {
    if (!stitchStarted || stitchBusy || !video.videoWidth) return;
    stitchBusy = true;
    try {
      snapCanvas.width = video.videoWidth;
      snapCanvas.height = video.videoHeight;
      snapCtx.drawImage(video, 0, 0);
      const r = await api("/api/stitch/add", {
        canonical_name: canonicalName,
        image_base64: snapCanvas.toDataURL("image/jpeg", 0.9),
        landmarks: lm.slice(0, 478).map((q) => [q.x, q.y, q.z]),
      });
      if (r?.added) stitchDirty = true;
    } catch (e) {
      console.warn("stitch add failed:", e);
    } finally {
      stitchBusy = false;
    }
  }

  /** Fetch the current blended texture and show it in the right pane. */
  async function pullPreview() {
    if (!stitchStarted || !stitchDirty || previewBusy || settled) return;
    previewBusy = true;
    stitchDirty = false;
    try {
      const tg = toneGain().map((v) => v.toFixed(3)).join(",");
      const res = await fetch(`/api/stitch/preview?canonical_name=${encodeURIComponent(canonicalName)}&tone=${tg}&t=${Date.now()}`);
      if (res.ok) {
        stitchCoverage = Math.round(100 * Number(res.headers.get("X-Coverage") || 0));
        const tc = res.headers.get("X-TriCov");
        if (tc) {
          const bin = atob(tc);
          const arr = new Uint8Array(bin.length);
          for (let i = 0; i < bin.length; i++) arr[i] = bin.charCodeAt(i);
          triCov = arr;
        }
        const url = URL.createObjectURL(await res.blob());
        const old = liveImg.src;
        liveImg.src = url;
        if (old && old.startsWith("blob:")) URL.revokeObjectURL(old);
        liveImg.style.display = showMesh ? "none" : "block";
        canvas.style.display = showMesh ? "block" : "none";
        const caps = wrap.querySelectorAll(".orbit-cap");
        if (caps[1]) caps[1].textContent = showMesh ? "Live mesh — your face as it is mapped" : "Your skin, stitched as you scan";
      }
    } catch (e) {
      console.warn("preview failed:", e);
    } finally {
      previewBusy = false;
    }
  }

  function paintedPercent() {
    if (!best || !best.length) return 0;
    let n = 0;
    for (let i = 0; i < best.length; i++) if (acc[i] > 0) n++;
    return Math.round((100 * n) / best.length);
  }

  function drawModel() {
    const W = canvas.width;
    const H = canvas.height;
    g2.setTransform(1, 0, 0, 1, 0, 0);
    g2.clearRect(0, 0, W, H);
    g2.fillStyle = "rgba(0,0,0,0.35)";
    g2.fillRect(0, 0, W, H);
    if (!lastLm) {
      g2.fillStyle = "rgba(255,255,255,0.5)";
      g2.font = "16px system-ui, sans-serif";
      g2.textAlign = "center";
      g2.fillText("Waiting for your face…", W / 2, H / 2);
      return;
    }
    const lm = lastLm;
    const ar = (video.videoWidth || 16) / (video.videoHeight || 9); // landmark x is 0..1 of the width, y of the height
    let minX = 1e9;
    let minY = 1e9;
    let maxX = -1e9;
    let maxY = -1e9;
    for (const p of lm) {
      minX = Math.min(minX, p.x * ar);
      minY = Math.min(minY, p.y);
      maxX = Math.max(maxX, p.x * ar);
      maxY = Math.max(maxY, p.y);
    }
    const cx = (minX + maxX) / 2;
    const cy = (minY + maxY) / 2;
    const s = (0.88 * Math.min(W, H)) / Math.max(maxX - minX, maxY - minY, 0.01);
    const D = withCentres(lm.map((p) => ({ x: W / 2 - (p.x * ar - cx) * s, y: H / 2 + (p.y - cy) * s, z: p.z }))); // mirrored like the preview
    // The live camera frame is the texture: one consistent lighting, so no faceted patchwork.
    let S3 = null;
    const lvw = video.videoWidth;
    const lvh = video.videoHeight;
    if (lvw && lvh) {
      if (liveCanvas.width !== lvw || liveCanvas.height !== lvh) {
        liveCanvas.width = lvw;
        liveCanvas.height = lvh;
      }
      lctx.drawImage(video, 0, 0, lvw, lvh);
      S3 = withCentres(lm.map((q) => ({ x: q.x * lvw, y: q.y * lvh, z: q.z * lvw })));
    }
    if (tris && canon && front) {
      // Hide back-facing triangles (the far side of the nose, etc.) and draw the rest far to near.
      const order = frontTrianglesFarToNear(D, true);
      g2.lineWidth = 1;
      g2.strokeStyle = "rgba(255,255,255,0.10)";
      const useSkin = triCov && triCov.length === tris.length && liveImg.complete && liveImg.naturalWidth > 0;
      if (useSkin) {
        const k = liveImg.naturalWidth / ATLAS;
        if (!canonS || canonSk !== k) {
          canonS = canon.map((q) => ({ x: q.x * k, y: q.y * k }));
          canonSk = k;
        }
      }
      // Under everything: the triangles that face away (inside of the nostrils, far side of the nose). Seen at an angle they would otherwise
      // leave a black hole in the face; the front-facing triangles are drawn over them.
      for (const t of frontTrianglesFarToNear(D, true, true)) {
        const [a, b, c] = tris[t];
        if (useSkin && triCov[t] > 50) drawTri(g2, liveImg, canonS[a], canonS[b], canonS[c], D[a], D[b], D[c]);
      }
      for (const t of order) {
        const [a, b, c] = tris[t];
        if (useSkin) {
          if (triCov[t] > 50) drawTri(g2, liveImg, canonS[a], canonS[b], canonS[c], D[a], D[b], D[c]); // finished sections: the seamless stitched skin
        } else if (S3 && acc[t] > 0) {
          drawTri(g2, liveCanvas, S3[a], S3[b], S3[c], D[a], D[b], D[c]); // before the first stitched preview exists
        }
      }
      g2.setTransform(1, 0, 0, 1, 0, 0);
      for (const t of order) {
        const [a, b, c] = tris[t];
        let alpha;
        let outline = false;
        if (useSkin) {
          const cv = triCov[t] / 255;
          if (cv >= 0.6) continue; // finished: no overlay
          alpha = cv <= 0.2 ? 0.9 : 0.1 + 0.55 * ((0.6 - cv) / 0.4); // partly seen: fading in
          outline = cv <= 0.2;
        } else {
          if (acc[t] > 0) continue;
          alpha = 0.7;
          outline = true;
        }
        g2.fillStyle = `rgba(40, 46, 60, ${alpha.toFixed(2)})`;
        g2.beginPath();
        g2.moveTo(D[a].x, D[a].y);
        g2.lineTo(D[b].x, D[b].y);
        g2.lineTo(D[c].x, D[c].y);
        g2.closePath();
        g2.fill();
        if (outline) g2.stroke();
      }
    } else {
      for (const p of D) {
        g2.fillStyle = "rgba(255,255,255,0.3)";
        g2.fillRect(p.x - 1, p.y - 1, 2, 2);
      }
    }
  }

  /** Tell the user which part of the face still has no skin painted on it, and how to bring it to the camera. */
  let lastTipAt = 0;
  /** What to do to clear the circle on the square you are in: the biggest thing holding its photo quality down right now. */
  function cellAdvice() {
    const i = activeCell;
    if (i < 0) return "";
    const s = slots[i];
    if (cellPct(i) >= 1) return "This cell is at 100%.";
    if (!s) return "This square has no photo yet. Hold your head right here, still and facing the camera, for a second.";
    const lp = liveParts || s.parts;
    const gains = [  // only things you can actually act on: where inside a square you are is too fine to steer
      ["size", 0.4 * (1 - lp.size)],
      ["stab", 0.26 * (1 - lp.stab)],
      ["centre", 0.34 * (1 - lp.centre)],
    ].sort((a, b) => b[1] - a[1]);
    const [kind, gain] = gains[0];
    if (gain < 0.02) return "Stay here and hold perfectly still: this square improves on its own as better frames come in.";
    if (kind === "size") return `Your face is only ${Math.round(lp.faceW ?? 0)} px wide: lean a little closer to the camera (or add light), then hold still.`;
    if (kind === "stab") return "Hold still for half a second: movement is blurring the photo for this square.";
    return "Keep your face in the middle of the camera view, then hold still.";
  }

  function updateTip() {
    const now = performance.now();
    if (now - lastTipAt < 600) return;
    lastTipAt = now;
    const adv = cellAdvice();
    if (adv) {
      tipEl.textContent = adv;
      return;
    }
    if (!tris || !canon || !acc || !baseline.ready) {
      tipEl.textContent = "";
      return;
    }
    const cx = ATLAS / 2;
    const cy = ATLAS / 2;
    const span = 0.9 * ATLAS;
    let left = 0;
    let right = 0;
    let top = 0;
    let bottom = 0;
    let total = 0;
    for (let t = 0; t < tris.length; t++) {
      total++;
      if (acc[t] > 0) continue;
      const [a, b, c] = tris[t];
      const x = (canon[a].x + canon[b].x + canon[c].x) / 3;
      const y = (canon[a].y + canon[b].y + canon[c].y) / 3;
      if (x > cx + 0.16 * span) left++; // larger camera x is the LEFT of the mirrored preview
      else if (x < cx - 0.16 * span) right++;
      if (y < cy - 0.2 * span) top++;
      else if (y > cy + 0.2 * span) bottom++;
    }
    // The server's blended skin map is the real measure of what is painted. When it says the skin is (nearly) all mapped, the few
    // triangles left over here are ones no head angle can reach (under the jaw, edge of the neck): do not keep telling you to chase them.
    if (stitchCoverage != null && stitchCoverage >= 97) {
      tipEl.textContent = "Skin is fully mapped.";
      return;
    }
    const need = Math.max(4, total * 0.04);
    const opts = [
      [left, "The left side of your face map is not painted yet — turn your face toward the RIGHT of the screen until that cheek faces the camera, then hold still."],
      [right, "The right side of your face map is not painted yet — turn your face toward the LEFT of the screen until that cheek faces the camera, then hold still."],
      [top, "Your forehead is not fully painted — tilt your chin DOWN so your forehead faces the camera, then hold still."],
      [bottom, "Your jaw and under-chin are not fully painted — lift your chin UP, then hold still."],
    ].filter((o) => o[0] >= need);
    if (!opts.length) {
      let weak = 0;
      for (let t = 0; t < tris.length; t++) if (acc[t] > 0 && bq[t] < 0.2) weak++;
      if (weak > total * 0.1)
        tipEl.textContent = "Some painted patches are low quality (orange/yellow on the map) — turn toward them slowly, get closer to the camera, and hold still for half a second.";
      else tipEl.textContent = paintedPercent() >= 85 ? "Nearly everything is painted. Tap Done when you are happy." : "Almost there — slow, small movements, pausing half a second at each end.";
      return;
    }
    opts.sort((p, q) => q[0] - p[0]);
    tipEl.textContent = opts[0][1];
  }

  // ---- capture ---------------------------------------------------------------------------------
  /** Brief white flash on a square each time a frame is evaluated for it, so you can see it is being worked on. */
  function flashCell(i) {
    const el = cellEls[i];
    if (!el) return;
    el.classList.remove("flash");
    void el.offsetWidth; // restart the animation
    el.classList.add("flash");
  }

  function nearestCell(yaw, pitch) {
    let best = -1;
    let bestD = 1e9;
    CELLS.forEach((c, i) => {
      const d = Math.hypot((yaw - c.yaw) / YAW_TOL, (pitch - c.pitch) / PITCH_TOL);
      if (d < bestD) {
        bestD = d;
        best = i;
      }
    });
    return { index: best, dist: bestD };
  }

  function keep(index, jpeg, quality, yaw, pitch, parts) {
    if (!jpeg) return;
    const prev = slots[index];
    if (prev && quality <= prev.quality + 0.002) return;
    slots[index] = { jpeg, quality, yaw, pitch, parts, caption: `Scan yaw ${Math.round(yaw)} pitch ${Math.round(pitch)}` };
    refresh();
  }

  function grab(index, quality, yaw, pitch, parts) {
    if (captureBusy || settled || !framing) return;
    captureBusy = true;
    requestAnimationFrame(() => {
      try {
        keep(index, framing?.captureFramedJpeg(0.92, CAPTURE_PX, SCAN_WIDE), quality, yaw, pitch, parts);
      } finally {
        captureBusy = false;
      }
    });
  }

  function onFrame(lm, matrix) {
    if (!lm) {
      lastLm = null;
      showStatus(`No face detected — look back toward the camera (frames seen: ${framesSeen}).`);
      drawModel();
      return;
    }
    lastLm = lm;
    framesSeen += 1;
    if (!tris) nBase = lm.length;
    buildTriangles();
    const { box, yaw: yr, pitch: pr } = estimateHeadPose(lm, matrix);
    const rawYaw = yr * RAD;
    const rawPitch = pr * RAD;

    if (!baseline.ready) {
      if (!framing?.isSettled?.()) {
        baseline.n = 0;
        baseline.sy = 0;
        baseline.sp = 0;
        baseline.lmSum = null;
        showStatus("Centering…");
        drawModel();
        return;
      }
      if (Math.abs(symOf(lm)) > 0.05) {
        baseline.n = 0; baseline.sy = 0; baseline.sp = 0; baseline.lmSum = null;
        showStatus("Look straight at the camera and hold still…");
        drawModel();
        return;
      }
      baseline.sy += rawYaw;
      baseline.sp += rawPitch;
      baseline.n += 1;
      if (!baseline.lmSum || baseline.lmSum.length !== lm.length) baseline.lmSum = lm.map(() => [0, 0, 0]);
      lm.forEach((q, i) => { baseline.lmSum[i][0] += q.x; baseline.lmSum[i][1] += q.y; baseline.lmSum[i][2] += q.z; });
      showStatus(`Calibrating ${baseline.n}/${BASELINE_FRAMES} — look straight at the camera`);
      if (baseline.n >= BASELINE_FRAMES) {
        baseline.yaw = baseline.sy / baseline.n;
        baseline.pitch = baseline.sp / baseline.n;
        baseline.ready = true;
        frontal = { jpeg: framing?.captureFramedJpeg?.(0.95, FRONTAL_PX, SCAN_WIDE) ?? null };
        frontalAspect = (video.videoWidth || 16) / (video.videoHeight || 9);
        // The front-facing 3D landmarks for the 3D model: the AVERAGE over all the calibration frames, so the model's resting shape has no
        // single-frame jitter in it. (The skin layout below still comes from the last frame, which is the one the first photo was taken from.)
        frontalLm = baseline.lmSum.map((q) => q.map((v) => v / baseline.n));
        setCanon(lm, frontalAspect);
        startStitch(lm).then(() => sendStitch(lm));
        framing?.lockFrame?.();
        refresh();
      }
      drawModel();
      return;
    }

    if (Math.abs(symOf(lm)) < 0.03 && Math.abs(rawYaw - baseline.yaw) < 40) {
      // a frame where you really face the camera: keep the centre tied to the median of the latest such frames
      frontalYaws.push(rawYaw);
      if (frontalYaws.length > 40) frontalYaws.shift();
      if (frontalYaws.length >= 8) {
        const s = [...frontalYaws].sort((a, b) => a - b);
        baseline.yaw = s[s.length >> 1];
      }
    }
    if (recenter) {
      recenter.sy += rawYaw;
      recenter.sp += rawPitch;
      if (++recenter.n >= 8) {
        baseline.yaw = recenter.sy / recenter.n;
        baseline.pitch = recenter.sp / recenter.n;
        reach.yaw = reach.up = reach.down = 0; // the range learned so far was measured from the wrong centre
        recenter = null;
      }
    }
    const yaw = (flipYaw ? -1 : 1) * (rawYaw - baseline.yaw);
    const pitch = (flipPitch ? -1 : 1) * (rawPitch - baseline.pitch);
    pose = { yaw, pitch };
    // Learn the reachable range (ignore tracker glitches).
    if (Math.abs(yaw) < 70) reach.yaw = Math.max(reach.yaw, Math.abs(yaw));
    if (pitch > 0 && pitch < 45) reach.up = Math.max(reach.up, pitch);
    if (pitch < 0 && pitch > -45) reach.down = Math.max(reach.down, -pitch);
    const clampN = (v, lo, hi) => Math.max(lo, Math.min(hi, v));
    // The grid is stretched to fit how far YOU can turn, but never so much that a small turn reaches the outer squares: the outer column
    // needs at least ~30 degrees of real turn (it used to be as little as ~18), so the sides of the head really get photographed.
    const yawS = yaw * (40 / clampN(reach.yaw * 0.9, 30, 40));
    const pitchS = pitch >= 0 ? pitch * (16 / clampN(reach.up * 0.7, 4, 16)) : pitch * (16 / clampN(reach.down * 0.7, 4, 16));
    const { index, dist } = nearestCell(yawS, pitchS);
    activeCell = index;
    {
      // the photo-quality text describes the live picture, so it moves as you move closer, hold still or re-centre
      const vw = video.videoWidth || 1280, vh = video.videoHeight || 720;
      const ff = box.w * vw; // the face's width in camera pixels
      liveParts = {
        size: Math.min(1, ff / FACE_PX_GOOD),
        stab: 1 - Math.min(1, (framing?.getJitter?.() ?? 0) * 3.2),
        centre: Math.max(0, Math.min(1, 1 - Math.hypot(1 - box.cx - 0.5, box.cy - 0.5) * 1.4)),
        faceW: ff,
      };
    }
    refresh();
    const tnow = performance.now();
    const jit = framing?.getJitter?.() ?? 0;
    if (tnow - lastPaintAt > 90 && jit < 0.8) {
      lastPaintAt = tnow;
      try {
        paintAtlas(lm, jit);
      } catch (err) {
        diag.last = "error: " + (err?.message || err);
        console.error(err);
      }
    } else if (jit >= 0.8) {
      diag.last = "hold still for a moment";
    }
    if (tnow - lastHarmAt > 700) {
      lastHarmAt = tnow;
      try {
        harmonise();
        inpaintIfNeeded();
      } catch (err) {
        diag.last = "error: " + (err?.message || err);
        console.error(err);
      }
    }
    if (tnow - lastDrawAt > 40) {
      lastDrawAt = tnow;
      drawModel();
    }
    updateTip();
    const here = Math.floor(100 * cellPct(index)); // fixed scale: never drops because another square got better
    statusMsg = diag.last === "painting" ? "Scanning" : diag.last;
    liveRows = [
      ["Square", `row ${CELLS[index].r + 1}, column ${CELLS[index].c + 1}` + (slots[index] ? ` (${here}% quality)` : " (no photo yet)")],
      ["Head", `${yaw >= 0 ? "right" : "left"} ${Math.abs(yaw).toFixed(0)}° · ${pitch >= 0 ? "up" : "down"} ${Math.abs(pitch).toFixed(0)}°`],
      ["Your range", `sideways ±${reach.yaw.toFixed(0)}° · up ${reach.up.toFixed(0)}° · down ${reach.down.toFixed(0)}°`],
      ["Skin painted", `${diag.painted} patches (${diag.rejected} colour-matched)`],
    ];
    renderStats();

    const c = CELLS[index];
    const jitter = framing?.getJitter?.() ?? 0;
    const inTol = dist <= 1.35; // near enough to this square (was a strict box: some squares never triggered)
    const now = performance.now();
    if (inTol && jitter < 0.9 && now - lastCaptureAt >= CAPTURE_INTERVAL_MS && !captureBusy) {
      lastCaptureAt = now;
      const vw0 = video.videoWidth || 1280;
      const vh0 = video.videoHeight || 720;
      const faceFrac = box.w * vw0; // the face's width in camera pixels
      const sizePart = Math.min(1, faceFrac / FACE_PX_GOOD); // this many pixels is plenty of detail
      const stabPart = 1 - Math.min(1, jitter * 3.2);
      const centrePart = Math.max(0, Math.min(1, 1 - Math.hypot(1 - box.cx - 0.5, box.cy - 0.5) * 1.4));
      const q = Math.max(0.05, (0.34 * centrePart + 0.4 * sizePart + 0.26 * stabPart) * (1 - 0.25 * Math.min(1, dist)));
      flashCell(index);
      if ((!slots[index] || q > slots[index].quality + 0.002) && stabPart > 0.6) void sendStitch(lm); // never send a frame taken while moving
      grab(index, q, yaw, pitch, { size: sizePart, stab: stabPart, centre: centrePart, faceW: faceFrac });
    }
  }

  // ---- lifecycle -------------------------------------------------------------------------------
  function stopCamera() {
    framing?.stop();
    framing = null;
    if (stream) {
      stream.getTracks().forEach((t) => t.stop());
      stream = null;
    }
  }

  let resolveScan = null;
  function completeAndExit(value) {
    if (settled) return;
    settled = true;
    const hb = document.getElementById("logout-btn");
    if (hb) hb.style.display = ""; // the header button comes back
    if (previewTimer) clearInterval(previewTimer);
    stopCamera();
    resolveScan?.(value);
  }

  async function startCamera() {
    try {
      const mod = await import("/static/vendor/mediapipe/vision_bundle.mjs");
      edges = mod.FaceLandmarker?.FACE_LANDMARKS_TESSELATION || null;
      mod2 = mod;
    } catch {
      edges = null; // falls back to dots
    }
    if (!stream) {
      stream = await navigator.mediaDevices.getUserMedia({
        video: { facingMode: "user", width: { ideal: 1280 }, height: { ideal: 720 } },
        audio: false,
      });
      video.srcObject = stream;
      await video.play();
    }
    const { createCameraFraming } = await import("/static/camera-framing.js?v=42");
    framing = await createCameraFraming(video, videoWrap, null, {
      mode: "face",
      steadyFrame: false,
      scanMode: true,
      fillWide: SCAN_WIDE,
      persistKey: "scan", // remember the framing that worked: the next scan starts with it
      aimY: 0.58, // the face points run brow to chin, so the face sits a little below the middle to leave room for hair above it
      targetFrac: 0.5, // the face fills about this much of the camera window: centred and cropped in, like the mask beside it
      onLandmarks: onFrame,
      onStatus: (msg) => {
        showStatus(msg);
      },
    });
    // Show the whole area that gets saved (SCAN_WIDE x the circle), not just its centre: same box size, wider view.
    const panEl = videoWrap.querySelector(".scan-video-pan");
    if (DISPLAY_WIDE > 1 && panEl && !panEl.parentElement.classList.contains("scan-wide-view")) {
      const wideView = document.createElement("div");
      wideView.className = "scan-wide-view";
      wideView.style.cssText = `position:absolute;inset:0;transform:scale(${1 / DISPLAY_WIDE});transform-origin:center center`;
      panEl.parentElement.insertBefore(wideView, panEl);
      wideView.appendChild(panEl);
    }
    const poll = setInterval(() => {
      if (settled) return clearInterval(poll);
      if (framing?.isModelReady) {
        refresh();
        clearInterval(poll);
      }
    }, 400);
    refresh();
    drawModel();
    if (!previewTimer) previewTimer = setInterval(pullPreview, 600);
  }

  async function finishScan() {
    if (finishing || settled) return;
    finishing = true;
    btnDone.disabled = true;
    const frames = slots
      .map((s, i) =>
        s ? { image_base64: s.jpeg, caption: s.caption, angle_index: i, quality_score: Math.min(1, s.quality), yaw_deg: s.yaw, pitch_deg: s.pitch, tracked: true } : null
      )
      .filter(Boolean);
    if (frontal?.jpeg) {
      frames.push({ image_base64: frontal.jpeg, caption: "Center", angle_index: FRONTAL_INDEX, quality_score: 1, yaw_deg: 0, pitch_deg: 0, tracked: true });
    }
    hint.textContent = "Saving your scan…";
    stopCamera();
    try {
      const result = await api("/api/capture/scan-frames", { canonical_name: canonicalName, frames });
      if (result.status === "error") throw new Error(result.message || "Save failed");
      if (canon && tris && acc) {
        fillHolesInAtlas();
        try {
          await api("/api/capture/skin-atlas", {
            canonical_name: canonicalName,
            atlas_base64: atlas.toDataURL("image/png"),
            layout: { landmarks3d: frontalLm, aspect: frontalAspect, size: ATLAS, points: canon.map((p) => [p.x / ATLAS, p.y / ATLAS]), triangles: tris, painted: Array.from(acc), score: Array.from(best) },
          });
        } catch (e) {
          console.warn("skin atlas not saved:", e);
        }
      }
      // The skin texture has been blending while you scanned: finishing is instant, a higher-resolution copy follows.
      hint.textContent = "Finishing…";
      try {
        const st = await api("/api/stitch/finish", { canonical_name: canonicalName, tone: toneGain() });
        if (st?.status === "ok") {
          completeAndExit(result); // straight back to the update page; the high-resolution copy finishes in the background
          return;
        }
        console.warn("stitch finish failed:", st?.message);
      } catch (e2) {
        console.warn("stitch finish failed:", e2);
      }
      completeAndExit(result);
    } catch (e) {
      alert("Could not save scan: " + e.message);
      finishing = false;
      settled = false;
      framesSeen = 0;
      await startCamera();
    }
  }

  return new Promise((resolve) => {
    root.innerHTML = "";
    root.appendChild(wrap);
    resolveScan = resolve;
    btnCancel.onclick = () => completeAndExit(null);
    const hb = document.getElementById("logout-btn");
    if (hb) hb.style.display = "none"; // the scan has its own Cancel and Save
    btnDone.onclick = () => void finishScan();
    if (recenterBtn) recenterBtn.onclick = () => { recenter = { n: 0, sy: 0, sp: 0 }; };
    (async () => {
      try {
        await startCamera();
      } catch (err) {
        hint.textContent = "Camera blocked or unavailable.";
        console.error(err);
      }
    })();
  });
}

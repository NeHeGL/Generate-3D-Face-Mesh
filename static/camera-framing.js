/**
 * Live camera: fill the frame and smoothly pan/zoom to keep the face centered.
 */

const WASM = "/static/vendor/mediapipe/wasm"; // served from this project: no internet needed, nothing for an ad blocker to block
const MODEL = "/static/vendor/mediapipe/face_landmarker.task";

function ensurePanLayer(container, video) {
  let pan = container.querySelector(".scan-video-pan");
  if (!pan) {
    pan = document.createElement("div");
    pan.className = "scan-video-pan";
    container.appendChild(pan);
    pan.appendChild(video);
  }
  return pan;
}

export async function createCameraFraming(video, container, hintEl, options = {}) {
  const mode = options.mode || "face";
  const steadyFrame = options.steadyFrame === true;
  const scanMode = options.scanMode === true;
  const targetFaceWidth = mode === "body" ? 0.32 : 0.36;
  const pan = ensurePanLayer(container, video);
  let landmarker = null;
  let rafId = 0;
  let lastTransform = { scale: 1, tx: 0, ty: 0 };
  let faceDetected = false;
  let latestLandmarks = null;
  let poseJitter = 0;
  let lastNose = null;
  let onLandmarks = options.onLandmarks ?? null;

  // fillWide > 1: the view shows fillWide x the box, so never zoom out or slide the picture far enough to expose its edges
  const fillWide = options.fillWide > 1 ? options.fillWide : 0;
  function applyTransform(scale, tx, ty) {
    if (fillWide) {
      const ov = mode === "body" ? 1.2 : 1.22;
      scale = Math.max(scale, fillWide / ov);
      const k = (ov * scale - fillWide) / 2;
      const mx = k * container.clientWidth;
      const my = k * container.clientHeight;
      tx = Math.min(mx, Math.max(-mx, tx));
      ty = Math.min(my, Math.max(-my, ty));
    }
    lastTransform = { scale, tx, ty };
    pan.style.transform = `translate(${tx}px, ${ty}px) scale(${scale})`;
  }

  function layoutVideo() {
    const w = container.clientWidth;
    const h = container.clientHeight;
    const overscan = mode === "body" ? 1.2 : 1.22;
    video.style.width = `${w * overscan}px`;
    video.style.height = `${h * overscan}px`;
    video.style.maxWidth = "none";
    video.style.maxHeight = "none";
    video.style.objectFit = "cover";
    video.style.transform = "scaleX(-1)";
    video.style.transformOrigin = "center center";
    pan.style.width = `${w}px`;
    pan.style.height = `${h}px`;
    pan.style.left = "50%";
    pan.style.top = "50%";
    pan.style.marginLeft = `${-w / 2}px`;
    pan.style.marginTop = `${-h / 2}px`;
  }

  const status = (msg) => {
    try {
      options.onStatus?.(msg);
    } catch {
      /* ignore */
    }
  };
  let currentDelegate = "";
  let failCount = 0;
  let gpuFailTotal = 0;      // failures on the GPU delegate that a later good frame does NOT reset (intermittent errors)
  let tickCount = 0;
  let reiniting = false;
  let locked = false;
  let settledFrames = 0;
  let goodRun = 0; // consecutive frames that were properly centred
  let scanFrames = 0; // frames spent trying to centre in scan mode
  let lastVideoTime = -1;
  let lastTs = 0;
  // Some browsers/cameras make MediaPipe throw "index out of bounds" when given the raw <video>. After the first such error
  // we feed it a downscaled canvas copy of the frame instead (same picture, plain pixel buffer).
  let useCanvasInput = false;
  const detectCanvas = document.createElement("canvas");
  const detectCtx = detectCanvas.getContext("2d", { willReadFrequently: true });

  async function initLandmarker(delegates = ["GPU", "CPU"]) {
    const withTimeout = (promise, ms, what) =>
      Promise.race([promise, new Promise((_, rej) => setTimeout(() => rej(new Error(`${what} timed out after ${ms / 1000}s`)), ms))]);
    for (const delegate of delegates) {
      status(`Loading face tracking (${delegate})…`);
      try {
        const { FaceLandmarker, FilesetResolver } = await import(
          "/static/vendor/mediapipe/vision_bundle.mjs"
        );
        const vision = await FilesetResolver.forVisionTasks(WASM);
        landmarker = await withTimeout(FaceLandmarker.createFromOptions(vision, {
          baseOptions: { modelAssetPath: MODEL, delegate },
          runningMode: "VIDEO",
          numFaces: 1,
          minFaceDetectionConfidence: 0.25,
          minFacePresenceConfidence: 0.25,
          minTrackingConfidence: 0.25,
          outputFacialTransformationMatrixes: true,
        }), 25000, `Face tracking (${delegate})`);
        currentDelegate = delegate;
        return true;
      } catch (e) {
        console.warn(`Face landmarker (${delegate}) failed:`, e);
        status(`Face landmarker (${delegate}) failed: ${e?.message || e}`);
      }
    }
    return false;
  }

  function bboxFromLandmarks(landmarks) {
    let minX = 1,
      minY = 1,
      maxX = 0,
      maxY = 0;
    for (const p of landmarks) {
      minX = Math.min(minX, p.x);
      minY = Math.min(minY, p.y);
      maxX = Math.max(maxX, p.x);
      maxY = Math.max(maxY, p.y);
    }
    return {
      cx: (minX + maxX) / 2,
      cy: (minY + maxY) / 2,
      w: maxX - minX,
      h: maxY - minY,
    };
  }

  function tick() {
    layoutVideo();
    tickCount += 1;
    if (!landmarker || video.readyState < 2 || video.videoWidth === 0) {
      if (tickCount % 30 === 0) {
        status(`Waiting for video… readyState ${video.readyState}, ${video.videoWidth}x${video.videoHeight}, paused=${video.paused}`);
      }
      if (video.paused && video.srcObject) video.play().catch(() => {});
      rafId = requestAnimationFrame(tick);
      return;
    }
    // Only run detection on a new video frame (rAF can outrun the camera).
    if (video.currentTime === lastVideoTime) {
      rafId = requestAnimationFrame(tick);
      return;
    }
    lastVideoTime = video.currentTime;
    // MediaPipe requires strictly increasing integer-ms timestamps.
    lastTs = Math.max(lastTs + 1, Math.round(performance.now()));
    let results;
    try {
      let input = video;
      if (useCanvasInput) {
        const k = Math.min(1, 960 / video.videoWidth);
        detectCanvas.width = Math.max(2, Math.round(video.videoWidth * k) & ~1);
        detectCanvas.height = Math.max(2, Math.round(video.videoHeight * k) & ~1);
        detectCtx.drawImage(video, 0, 0, detectCanvas.width, detectCanvas.height);
        input = detectCanvas;
      }
      results = landmarker.detectForVideo(input, lastTs);
      failCount = 0;
    } catch (err) {
      failCount += 1;
      if (!useCanvasInput && /index out of bounds/i.test(String(err?.message || err))) useCanvasInput = true;
      if (currentDelegate === "GPU") gpuFailTotal += 1;
      console.warn("detectForVideo failed:", err);
      status(`Detection error (${currentDelegate}): ${err?.message || err}`);
      // The GPU delegate throws "index out of bounds" on some drivers, often only now and then: three failures IN A ROW
      // never happened, so it stayed on the broken delegate. Switch to the CPU after the first such error or any 3 failures.
      const gpuBroken =
        currentDelegate === "GPU" &&
        (gpuFailTotal >= 3 || failCount >= 3 || /index out of bounds/i.test(String(err?.message || err)));
      if (gpuBroken && !reiniting) {
        reiniting = true;
        status("GPU detection failing — switching to CPU…");
        const old = landmarker;
        landmarker = null;
        try {
          old?.close?.();
        } catch {
          /* ignore */
        }
        initLandmarker(["CPU"]).finally(() => {
          reiniting = false;
          failCount = 0;
          gpuFailTotal = 0;
        });
      }
      rafId = requestAnimationFrame(tick);
      return;
    }
    const lm = results.faceLandmarks?.[0];
    const w = container.clientWidth;
    const h = container.clientHeight;

    if (!lm) {
      faceDetected = false;
      if (!locked) settledFrames = 0; // once the view is locked, losing the face for a moment must not undo it
      if (onLandmarks) {
        try {
          onLandmarks(null, null);
        } catch (err) {
          console.error("Scan frame handler error:", err);
        }
      }
      // In scan mode never reset pan/zoom when the face is lost (locked or not).
      if (!scanMode) applyTransform(1, 0, 0);
      if (hintEl) {
        hintEl.textContent = "Center your face in the circle.";
      }
      rafId = requestAnimationFrame(tick);
      return;
    }

    faceDetected = true;
    latestLandmarks = lm;
    const nose = lm[1];
    if (lastNose) {
      const jw = scanMode ? 0.08 : 0.15;
      poseJitter =
        poseJitter * 0.85 +
        Math.hypot(nose.x - lastNose.x, nose.y - lastNose.y) * jw;
    }
    lastNose = { x: nose.x, y: nose.y };
    const mat = results.facialTransformationMatrixes?.[0];
    const matrix = mat?.data ?? mat;
    if (onLandmarks) {
      try {
        onLandmarks(lm, matrix);
      } catch (err) {
        console.error("Scan frame handler error:", err);
      }
    }
    if (locked) {
      rafId = requestAnimationFrame(tick);
      return;
    }
    const box = bboxFromLandmarks(lm);
    const visualCx = 1 - box.cx;
    const errX = (visualCx - 0.5) * w;
    const errY = (box.cy - 0.5) * h;
    const smooth = steadyFrame ? 0.12 : 0.22;
    const prev = lastTransform;
    let scaleTarget = targetFaceWidth / Math.max(box.w, 0.12);
    if (scanMode) {
      scaleTarget = Math.min(2.2, Math.max(0.9, scaleTarget));
    } else {
      if (box.w > 0.4) scaleTarget = Math.min(scaleTarget, 0.95);
      scaleTarget = Math.min(1.18, Math.max(0.92, scaleTarget));
    }
    if (scanMode) {
      // Closed loop: measure where the face ACTUALLY is in the circle (same
      // geometry the JPEG capture uses), then nudge pan/zoom to fix what's left.
      const g = framedGeometry();
      const fx = g.centerX - (-g.drawW / 2 + ((box.cx * g.vw - g.sx) / g.sw) * g.drawW);
      const fy = g.centerY + (-g.drawH / 2 + ((box.cy * g.vh - g.sy) / g.sh) * g.drawH);
      const offX = fx - g.cw / 2; // + = face is right of the circle center
      const offY = fy - g.ch * (options.aimY ?? 0.5); // + = face is below where it should be (aimY: 0.5 = the middle; larger leaves headroom for hair)
      const faceFrac = ((box.w * g.vw) / g.sw) * g.drawW / g.cw; // face width / circle width
      const TARGET_FRAC = options.targetFrac ?? 0.48; // how much of the window the face fills (the saved photos still include extra margin around it)
      const newScale = Math.min(3, Math.max(0.9, prev.scale * (TARGET_FRAC / Math.max(faceFrac, 0.05))));
      let scale2 = prev.scale * (1 - smooth) + newScale * smooth;
      if (fillWide) {
        // The picture may only slide as far as the zoom leaves spare room (so no empty edges show). A face that is off to one side
        // needs that room: zoom in just enough to be able to centre it, instead of getting stuck.
        const ov = mode === "body" ? 1.2 : 1.22;
        // The offset of the face from the middle grows in proportion to the zoom, so measure it in picture units (offset / zoom) and solve for the
        // zoom that leaves exactly enough room: room = (ov * zoom - fillWide) / 2 must cover the offset (ov / 2 - d) ... which gives zoom >= fillWide / (ov - 2d).
        const d = Math.max(Math.abs(prev.tx - offX) / g.cw, Math.abs(prev.ty - offY) / g.ch) / Math.max(prev.scale, 0.5); // the TOTAL shift needed (already applied + what is left), not just what is left
        const den = ov - 2 * 1.1 * d;
        const need = den > 0.2 ? fillWide / den : 3;
        scale2 = Math.min(3, Math.max(scale2, need));
      }
      applyTransform(scale2, prev.tx - offX * smooth, prev.ty - offY * smooth);
      const centered2 = Math.hypot(offX / g.cw, offY / g.ch) < 0.03;
      const zoomed2 = Math.abs(faceFrac - TARGET_FRAC) < 0.06 || (fillWide && lastTransform.scale <= fillWide / (mode === "body" ? 1.2 : 1.22) + 0.01 && faceFrac > TARGET_FRAC); // at the widest allowed view already
      scanFrames++;
      // "Properly centred" = centred, and the face at least about the target size (it may be bigger: a head high in the picture needs extra zoom for room).
      const good = centered2 && faceFrac > TARGET_FRAC - 0.06 && faceFrac < 0.95;
      goodRun = good ? goodRun + 1 : 0;
      if (goodRun >= 12) saveDefault(); // 12 steady frames properly centred: this framing becomes the default for next time
      settledFrames = good || scanFrames > 150 ? settledFrames + 1 : 0; // never hang on "Centering": after ~3 s, carry on (but that is not saved)
      if (settledFrames >= 12) locked = true; // centred once, at the start: from here the view never moves again, whatever your head does
      rafId = requestAnimationFrame(tick);
      return;
    }
    const scale = prev.scale * (1 - smooth) + scaleTarget * smooth;
    const tx = steadyFrame ? 0 : prev.tx * (1 - smooth) - errX * smooth;
    const ty = steadyFrame ? 0 : prev.ty * (1 - smooth) - errY * smooth;
    applyTransform(scale, tx, ty);

    if (hintEl) {
      const off = Math.hypot(errX, errY);
      if (off > w * 0.1) {
        hintEl.textContent = "Auto-framing… move until you're centered.";
      } else if (box.w < 0.2) {
        hintEl.textContent = "Move a little closer.";
      } else if (box.w > 0.48) {
        hintEl.textContent = "Move back slightly — you're close.";
      } else {
        hintEl.textContent = "Looking good — hold still.";
      }
    } else if (options.quiet) {
      /* scan mode sets its own hints */
    }
    rafId = requestAnimationFrame(tick);
  }

  // The framing worked out for this camera (zoom and shift) is remembered, so the next scan starts centred straight away.
  const DEFAULT_KEY = options.persistKey ? `cameraFraming:${options.persistKey}` : "";
  function saveDefault() {
    if (!DEFAULT_KEY || locked || !container.clientWidth) return;
    try { localStorage.setItem(DEFAULT_KEY, JSON.stringify({ ...lastTransform, cw: container.clientWidth, ch: container.clientHeight })); } catch { /* storage unavailable */ }
  }
  layoutVideo();
  if (DEFAULT_KEY) {
    try {
      const s = JSON.parse(localStorage.getItem(DEFAULT_KEY) || "null");
      if (s && s.scale > 0 && s.cw && s.ch && container.clientWidth) applyTransform(s.scale, (s.tx * container.clientWidth) / s.cw, (s.ty * container.clientHeight) / s.ch);
    } catch { /* ignore a bad saved value */ }
  }
  const hasMl = await initLandmarker();
  const onResize = () => layoutVideo();
  window.addEventListener("resize", onResize);

  if (hasMl) {
    if (hintEl) hintEl.textContent = "Centering you in the frame…";
    rafId = requestAnimationFrame(tick);
  } else {
    applyTransform(1.08, 0, 0);
    if (hintEl) {
      hintEl.textContent =
        "Auto-center unavailable — line your face up with the circle.";
    }
  }

  function stop() {
    cancelAnimationFrame(rafId);
    window.removeEventListener("resize", onResize);
    landmarker?.close?.();
    landmarker = null;
  }

  function framedGeometry() {
    const vw = video.videoWidth || 640;
    const vh = video.videoHeight || 480;
    const cw = container.clientWidth;
    const ch = container.clientHeight;
    const { scale, tx, ty } = lastTransform;
    const overscan = mode === "body" ? 1.2 : 1.22;
    const drawW = cw * overscan * scale;
    const drawH = ch * overscan * scale;
    const centerX = cw / 2 + tx;
    const centerY = ch / 2 + ty;
    const videoAspect = vw / vh;
    const destAspect = drawW / drawH;
    let sx = 0,
      sy = 0,
      sw = vw,
      sh = vh;
    if (videoAspect > destAspect) {
      sw = vh * destAspect;
      sx = (vw - sw) / 2;
    } else {
      sh = vw / destAspect;
      sy = (vh - sh) / 2;
    }
    return { cw, ch, vw, vh, sx, sy, sw, sh, drawW, drawH, centerX, centerY };
  }

  /** Nose (or any landmark) in container space, -1..1 from circle center; +x = right on screen. */
  function landmarkToContainerNorm(landmarks, index = 1) {
    const p = landmarks[index];
    if (!p) return null;
    const g = framedGeometry();
    const { cw, ch, vw, vh, sx, sy, sw, sh, drawW, drawH, centerX, centerY } = g;
    if (!cw || !ch || !vw || !vh) return null;
    const px = p.x * vw;
    const py = p.y * vh;
    const u = (px - sx) / sw;
    const v = (py - sy) / sh;
    const drawX = -drawW / 2 + u * drawW;
    const drawY = -drawH / 2 + v * drawH;
    const containerX = centerX - drawX;
    const containerY = centerY + drawY;
    const radius = Math.min(cw, ch) / 2;
    return {
      x: (containerX - cw / 2) / radius,
      y: (containerY - ch / 2) / radius,
    };
  }

  /** outPx: output width/height in pixels (default = on-screen size). Higher = sharper. */
  /** wide > 1: also keep the area around the on-screen circle (hair, ears, neck) and skip the circular mask. */
  function captureFramedJpeg(quality = 0.9, outPx = 0, wide = 1) {
    const g = framedGeometry();
    const { cw, ch, vw, vh, sx, sy, sw, sh, drawW, drawH, centerX, centerY } = g;
    if (!cw || !ch || !vw || !vh) return null;
    const k = outPx ? outPx / Math.min(cw, ch) : 1;
    const canvas = document.createElement("canvas");
    const ctx = canvas.getContext("2d");
    if (wide > 1) {
      const side = Math.min(cw, ch) * wide;
      canvas.width = Math.round(side * k);
      canvas.height = Math.round(side * k);
      ctx.fillStyle = "#000";
      ctx.fillRect(0, 0, canvas.width, canvas.height);
      ctx.imageSmoothingQuality = "high";
      ctx.scale(k, k);
      ctx.translate(side / 2 - cw / 2, side / 2 - ch / 2); // keep the circle's centre at the middle
    } else {
      canvas.width = Math.round(cw * k);
      canvas.height = Math.round(ch * k);
      ctx.imageSmoothingQuality = "high";
      ctx.scale(k, k);
    }

    ctx.save();
    if (wide <= 1) {
      ctx.beginPath();
      ctx.arc(cw / 2, ch / 2, Math.min(cw, ch) / 2, 0, Math.PI * 2);
      ctx.clip();
    }
    ctx.translate(centerX, centerY);
    ctx.scale(-1, 1);
    ctx.drawImage(video, sx, sy, sw, sh, -drawW / 2, -drawH / 2, drawW, drawH);
    ctx.restore();

    return canvas.toDataURL("image/jpeg", quality);
  }

  return {
    stop,
    isSettled: () => settledFrames >= 12,
    lockFrame: () => {
      locked = true;
    },
    unlockFrame: () => {
      locked = false;
    },
    captureFramedJpeg,
    layoutVideo,
    landmarkToContainerNorm,
    getJitter: () => poseJitter,
    isTracking: () => Boolean(landmarker),
    get isModelReady() {
      return Boolean(landmarker);
    },
    set onLandmarks(cb) {
      onLandmarks = cb;
    },
  };
}
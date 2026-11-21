/**
 * Ring layout (8 wedges, clock from top):
 * 0 top | 1 top-right | 2 right | 3 bottom-right
 * 4 bottom | 5 bottom-left | 6 left | 7 top-left
 * Slot 8 = center (straight at camera) — no wedge, inner ring glow
 */

export const RING_SEGMENTS = 32;
export const CENTER_SLOT = -1; // "facing camera" — not a capture slot
export const TOTAL_SLOTS = RING_SEGMENTS;

/**
 * Raw head direction from landmarks only (no magic constants, no matrix sign guessing).
 * x: + = user turned toward screen-right (mirror view). y: + = looking down, - = looking up.
 * Subtract a neutral baseline (user looking at the camera) before using.
 */
export function rawCompass(lm) {
  const nose = lm[1];
  const le = lm[33];
  const re = lm[263];
  const chin = lm[152];
  const eyeMidX = (le.x + re.x) / 2;
  const eyeMidY = (le.y + re.y) / 2;
  const eyeW = Math.max(0.04, Math.abs(re.x - le.x));
  const eyeToChin = Math.max(0.05, chin.y - eyeMidY);
  return {
    x: -(nose.x - eyeMidX) / eyeW,
    y: (nose.y - eyeMidY) / eyeToChin,
  };
}

export const WEDGE_STEP = (Math.PI * 2) / RING_SEGMENTS;
export const WEDGE_ORIGIN = -Math.PI / 2 - WEDGE_STEP / 2;

export function bboxFromLandmarks(landmarks) {
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

function matrixToArray(matrix) {
  if (!matrix) return null;
  if (Array.isArray(matrix) || ArrayBuffer.isView(matrix)) {
    return matrix.length >= 16 ? matrix : null;
  }
  if (matrix.data && (Array.isArray(matrix.data) || ArrayBuffer.isView(matrix.data))) {
    return matrix.data;
  }
  return null;
}

/** MediaPipe 4×4 facial transform → radians (selfie / mirror yaw). */
export function poseFromMatrix(matrix) {
  const m = matrixToArray(matrix);
  if (!m) return null;
  const yaw = Math.atan2(m[8], m[0]);
  const pitch = Math.asin(Math.max(-1, Math.min(1, -m[9])));
  return {
    yaw: Math.max(-1.2, Math.min(1.2, -yaw)),
    pitch: Math.max(-1.2, Math.min(1.2, pitch)),
  };
}

export function estimateHeadPose(landmarks, transformMatrix) {
  const box = bboxFromLandmarks(landmarks);
  const nose = landmarks[1];
  const le = landmarks[33];
  const re = landmarks[263];
  const chin = landmarks[152];
  const forehead = landmarks[10];

  const fromMat = poseFromMatrix(transformMatrix);
  let yaw;
  let pitch;

  if (fromMat) {
    yaw = fromMat.yaw;
    pitch = fromMat.pitch;
  } else {
    const eyeMidX = (le.x + re.x) / 2;
    const eyeW = Math.max(0.08, Math.abs(re.x - le.x));
    yaw = (-(nose.x - eyeMidX) / eyeW) * 1.1;
    const eyeMidY = (le.y + re.y) / 2;
    const faceH = Math.max(0.1, chin.y - forehead.y);
    pitch = (nose.y - eyeMidY) / (faceH * 0.9);
  }

  const faceH = Math.max(0.1, chin.y - forehead.y);
  const chinOpen = (chin.y - nose.y) / faceH;

  return { yaw, pitch, chinOpen, box };
}

/** Neutral pose from first ~1s so “looking at camera” = center, not bottom. */
export function createPoseBaseline() {
  return {
    yaw: 0,
    pitch: 0,
    n: 0,
    ready: false,
  };
}

const BASELINE_FRAMES = 10;

export function feedPoseBaseline(baseline, landmarks, matrix) {
  if (baseline.ready) return baseline;
  const { yaw, pitch } = estimateHeadPose(landmarks, matrix);
  baseline.yaw += yaw;
  baseline.pitch += pitch;
  baseline.n += 1;
  if (baseline.n >= BASELINE_FRAMES) {
    baseline.yaw /= baseline.n;
    baseline.pitch /= baseline.n;
    baseline.ready = true;
  }
  return baseline;
}

/** Resting pose offset — works mid-calibration after a few frames. */
export function effectiveBaseline(baseline) {
  if (!baseline.n) return { yaw: 0, pitch: 0, centerR: 0.38 };
  if (baseline.ready) return { yaw: baseline.yaw, pitch: baseline.pitch, centerR: 0.22 };
  return {
    yaw: baseline.yaw / baseline.n,
    pitch: baseline.pitch / baseline.n,
    centerR: 0.3,
  };
}

const MAIN_NAMES = ["Top", "Top right", "Right", "Bottom right", "Bottom", "Bottom left", "Left", "Top left"];

/** Wedges run clockwise from the top; every 4th wedge is a main compass direction. */
export function segmentCaption(i) {
  if (i < 0 || i >= RING_SEGMENTS) return "Center";
  const per = RING_SEGMENTS / MAIN_NAMES.length;
  const deg = Math.round((i * 360) / RING_SEGMENTS * 10) / 10;
  if (i % per === 0) return MAIN_NAMES[i / per];
  const base = Math.floor(i / per);
  const near = i % per < per / 2 ? base : (base + 1) % MAIN_NAMES.length;
  return `${MAIN_NAMES[near]} (${deg}° clockwise from top)`;
}

export function facingLabel(seg) {
  return segmentCaption(seg);
}

export function segmentFromNoseOffset(nx, ny, centerRadius = 0.2) {
  const dist = Math.hypot(nx, ny);
  if (dist < centerRadius) return CENTER_SLOT;

  const angle = Math.atan2(ny, nx);
  let t = (angle - WEDGE_ORIGIN) / WEDGE_STEP;
  while (t < 0) t += RING_SEGMENTS;
  while (t >= RING_SEGMENTS) t -= RING_SEGMENTS;
  return Math.floor(t) % RING_SEGMENTS;
}

function clamp12(v) {
  return Math.max(-1.2, Math.min(1.2, v));
}

/**
 * Fixed face proportions (forward ≈ center). Looking down raises chin/nose vs eyes.
 */
function headCompassVector(landmarks, transformMatrix) {
  const nose = landmarks[1];
  const le = landmarks[33];
  const re = landmarks[263];
  const chin = landmarks[152];
  const forehead = landmarks[10];

  const eyeMidY = (le.y + re.y) / 2;
  const faceH = Math.max(0.08, chin.y - forehead.y);

  const chinVsEyes = (chin.y - eyeMidY) / faceH;
  const noseVsEyes = (nose.y - eyeMidY) / faceH;
  const browVsEyes = (forehead.y - eyeMidY) / faceH;

  let pitchY =
    (chinVsEyes - 0.78) / 0.1 + (noseVsEyes - 0.34) / 0.06 - (browVsEyes + 0.52) / 0.08;
  pitchY *= 0.22;

  const noseX = 1 - nose.x;
  const leX = 1 - le.x;
  const reX = 1 - re.x;
  const eyeMidX = (leX + reX) / 2;
  const eyeW = Math.max(0.06, Math.abs(reX - leX));
  let yawX = (noseX - eyeMidX) / (eyeW * 0.32);

  const fromMat = poseFromMatrix(transformMatrix);
  if (fromMat) {
    yawX = yawX * 0.35 + fromMat.yaw / 0.18;
    pitchY = pitchY * 0.35 + fromMat.pitch / 0.16;
  }

  const x = clamp12(yawX);
  const y = clamp12(pitchY);
  const centerR = 0.13;

  return { x, y, mag: Math.hypot(x, y), centerR };
}

export function segmentFromLandmarks(landmarks, transformMatrix) {
  const { x, y, centerR } = headCompassVector(landmarks, transformMatrix);
  return segmentFromNoseOffset(x, y, centerR);
}

export function facingVector(landmarks, transformMatrix) {
  const { x, y, mag } = headCompassVector(landmarks, transformMatrix);
  return { x, y, mag };
}

export function headPoseDebug(landmarks, transformMatrix) {
  const { x, y } = headCompassVector(landmarks, transformMatrix);
  return { yaw: x, pitch: y };
}

export function scoreCapture(box, jitter = 0, facingMag = 0, seg = CENTER_SLOT) {
  const visCx = 1 - box.cx;
  const centerPen = 1 - Math.hypot(visCx - 0.5, box.cy - 0.5) * 1.4;
  const sizePen = Math.min(1, box.w / 0.42);
  const stabPen = 1 - Math.min(1, jitter * 3.2);
  let posePen =
    seg === CENTER_SLOT
      ? Math.max(0, 1 - facingMag * 2.5)
      : Math.min(1, facingMag * 1.2);

  return Math.max(
    0,
    Math.min(
      1,
      0.28 * centerPen + 0.34 * sizePen + 0.26 * stabPen + 0.12 * posePen
    )
  );
}
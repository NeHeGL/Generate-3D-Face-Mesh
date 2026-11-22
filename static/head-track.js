// Camera face control for the viewer. Whoever is in front of the camera drives the saved mask: the mask keeps its own face shape, and only
// the MOVEMENT is transferred. Each of the 478 camera points is matched to the same numbered point on the mask.
//
// Per frame: (1) line today's face up with the same person's resting face (rotation, scale, shift) using the points that barely move;
// that rotation is the head turn/tilt/roll. (2) What is left over at every point is the expression (jaw, lips, brows, lids).
const WASM = "/static/vendor/mediapipe/wasm";
const MODEL = "/static/vendor/mediapipe/face_landmarker.task";
const N_REAL = 478;
const STABLE = [10, 168, 6, 197, 195, 5, 4, 33, 133, 362, 263, 234, 454, 127, 356, 151, 9]; // forehead, nose bridge, eye corners, cheek edges

const tr = (a) => [a[0], a[3], a[6], a[1], a[4], a[7], a[2], a[5], a[8]];
function inv(a) {
  const [m00, m01, m02, m10, m11, m12, m20, m21, m22] = a;
  const c00 = m11 * m22 - m12 * m21, c01 = m12 * m20 - m10 * m22, c02 = m10 * m21 - m11 * m20;
  const det = m00 * c00 + m01 * c01 + m02 * c02 || 1e-12;
  return [c00 / det, (m02 * m21 - m01 * m22) / det, (m01 * m12 - m02 * m11) / det, c01 / det, (m00 * m22 - m02 * m20) / det, (m02 * m10 - m00 * m12) / det,
    c02 / det, (m01 * m20 - m00 * m21) / det, (m00 * m11 - m01 * m10) / det];
}
/** Nearest rotation to M (polar decomposition by Newton iteration, scaled first so it converges). */
function nearestRotation(M) {
  const f = Math.sqrt(M.reduce((t, v) => t + v * v, 0)) || 1;
  let R = M.map((v) => v / f);
  for (let i = 0; i < 30; i++) { const it = tr(inv(R)); R = R.map((v, k) => 0.5 * (v + it[k])); }
  return R;
}
const mul = (a, b) => { const o = new Array(9); for (let i = 0; i < 3; i++) for (let j = 0; j < 3; j++) o[i * 3 + j] = a[i * 3] * b[j] + a[i * 3 + 1] * b[3 + j] + a[i * 3 + 2] * b[6 + j]; return o; };
const F = [1, 0, 0, 0, -1, 0, 0, 0, -1];    // camera picture axes (x right, y down, z away) -> model axes (x right, y up, z toward you)
const MIRROR = [-1, 0, 0, 0, 1, 0, 0, 0, 1]; // the face is shown like a mirror

/** Starts the camera. Calls onFace(rotation, delta) every frame once calibrated:
 *  rotation: 3x3 (row-major, model axes, mirrored) head turn/tilt/roll relative to the resting face.
 *  delta: Float32Array(478*3), how far each point has moved from rest, in face heights, in model axes (x right, y up, z toward you), not mirrored.
 *  onStatus(text) says "look straight ahead" while it learns the resting face, then "" . Returns stop(). */
export async function startHeadTracking(onFace, onStatus = () => {}) {
  const stream = await navigator.mediaDevices.getUserMedia({ video: { width: 640, height: 480 }, audio: false });
  const video = document.createElement("video");
  video.muted = true;
  video.playsInline = true;
  video.srcObject = stream;
  await video.play();
  let landmarker;
  try {
    const { FaceLandmarker, FilesetResolver } = await import("/static/vendor/mediapipe/vision_bundle.mjs");
    const vision = await FilesetResolver.forVisionTasks(WASM);
    const make = (delegate) => FaceLandmarker.createFromOptions(vision, { baseOptions: { modelAssetPath: MODEL, delegate }, runningMode: "VIDEO", numFaces: 1 });
    try { landmarker = await make("GPU"); } catch { landmarker = await make("CPU"); }
  } catch (e) {
    stream.getTracks().forEach((t) => t.stop());
    throw e;
  }
  let alive = true, raf = 0, lastT = -1;
  let n = 0, acc = null, neutral = null, faceH = 1;
  let Rs = null; // smoothed head rotation
  let sm = null; // smoothed expression
  const tick = () => {
    if (!alive) return;
    raf = requestAnimationFrame(tick);
    if (video.readyState < 2 || video.currentTime === lastT) return;
    lastT = video.currentTime;
    let res;
    try { res = landmarker.detectForVideo(video, performance.now()); } catch { return; }
    const lm = res.faceLandmarks?.[0];
    if (!lm || lm.length < N_REAL) return;
    const W = video.videoWidth, H = video.videoHeight;
    const L = new Float32Array(N_REAL * 3); // pixels (depth is in picture-width units, scaled to pixels like x)
    for (let i = 0; i < N_REAL; i++) { L[i * 3] = lm[i].x * W; L[i * 3 + 1] = lm[i].y * H; L[i * 3 + 2] = lm[i].z * W; }
    if (!neutral) { // the first frames: this person's resting face
      if (!acc) acc = new Float32Array(N_REAL * 3);
      for (let i = 0; i < L.length; i++) acc[i] += L[i];
      onStatus("Look straight ahead with a relaxed face…");
      if (++n >= 15) {
        neutral = acc.map((v) => v / n);
        faceH = Math.hypot(neutral[30] - neutral[152 * 3], neutral[31] - neutral[152 * 3 + 1], neutral[32] - neutral[152 * 3 + 2]) || 1;
        onStatus("");
      }
      return;
    }
    const cN = [0, 0, 0], cL = [0, 0, 0];
    for (const i of STABLE) for (let k = 0; k < 3; k++) { cN[k] += neutral[i * 3 + k] / STABLE.length; cL[k] += L[i * 3 + k] / STABLE.length; }
    const M = [0, 0, 0, 0, 0, 0, 0, 0, 0];
    let sN = 0, sL = 0;
    for (const i of STABLE) {
      const a = [0, 1, 2].map((k) => neutral[i * 3 + k] - cN[k]);
      const b = [0, 1, 2].map((k) => L[i * 3 + k] - cL[k]);
      for (let r = 0; r < 3; r++) for (let c = 0; c < 3; c++) M[r * 3 + c] += a[r] * b[c];
      sN += a[0] * a[0] + a[1] * a[1] + a[2] * a[2];
      sL += b[0] * b[0] + b[1] * b[1] + b[2] * b[2];
    }
    const R = nearestRotation(M); // today's pose -> resting pose
    const sc = Math.sqrt(sN / (sL || 1));
    if (!Number.isFinite(sc) || sc < 0.5 || sc > 2 || R.some((v) => !Number.isFinite(v))) return; // a bad frame
    if (!sm) sm = new Float32Array(N_REAL * 3);
    // Expression: what is left at every point after undoing the head movement (face heights). Webcam depth is noisy, so it counts for
    // little and nothing may ever move far: one bad reading can never blow the model up. Then to model axes, mirrored (x right, y up, z toward you).
    for (let i = 0; i < N_REAL; i++) {
      const v = [0, 1, 2].map((k) => L[i * 3 + k] - cL[k]);
      const d = [0, 1, 2].map((r) => {
        const aligned = sc * (R[r * 3] * v[0] + R[r * 3 + 1] * v[1] + R[r * 3 + 2] * v[2]) + cN[r];
        const lim = r === 2 ? 0.05 : 0.3;
        const x = ((aligned - neutral[i * 3 + r]) / faceH) * (r === 2 ? 0.4 : 1);
        return Math.max(-lim, Math.min(lim, Number.isFinite(x) ? x : 0));
      });
      // Camera axes (x right, y down, z away) -> model axes (x right, y up, z toward you). NOT mirrored: every point keeps its own number and side
      // (the model is not flipped, so the corner of the mouth that moves out in the camera must move out on the model). Only the head turn is mirrored.
      const target = [d[0], -d[1], -d[2]];
      for (let k = 0; k < 3; k++) sm[i * 3 + k] += (target[k] - sm[i * 3 + k]) * 0.55;
    }
    // Head pose = the inverse of the alignment (resting -> today), in model axes and mirrored.
    const Pm = mul(mul(MIRROR, mul(F, mul(tr(R), F))), MIRROR);
    Rs = Rs ? nearestRotation(Rs.map((v, k) => v + (Pm[k] - v) * 0.45)) : Pm;
    onFace(Rs, sm);
  };
  raf = requestAnimationFrame(tick);
  return () => {
    alive = false;
    cancelAnimationFrame(raf);
    stream.getTracks().forEach((t) => t.stop());
    try { landmarker.close(); } catch { /* ignore */ }
  };
}

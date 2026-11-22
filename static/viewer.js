// Minimal WebGL viewer: loads output/<folder>/mesh.json + the skin texture, drag to rotate, wheel to zoom.
// MediaPipe's inner-lip loop (the edge of the mouth opening): the stock face mesh leaves the inside of the mouth empty, so it is filled with a dark patch.
const MOUTH = [78, 191, 80, 81, 82, 13, 312, 311, 310, 415, 308, 324, 318, 402, 317, 14, 87, 178, 88, 95];

// Lines of points along the lips and jaw. Their movement is smoothed along each line (the two ends stay put), so one point cannot run far ahead of its
// neighbours and pull the outline into a point: the mouth opening and the chin stay round.
const CHAINS = [
  [78, 191, 80, 81, 82, 13, 312, 311, 310, 415, 308],               // upper inner lip
  [78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308],               // lower inner lip
  [61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291],                // upper outer lip
  [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291],              // lower outer lip
  [58, 172, 136, 150, 149, 176, 148, 152, 377, 400, 378, 379, 365, 397, 288], // jaw and chin
];
function smoothChains(d, passes = 3) {
  for (const chain of CHAINS) {
    for (let p = 0; p < passes; p++) {
      const prev = chain.map((v) => [d[v * 3], d[v * 3 + 1], d[v * 3 + 2]]);
      for (let i = 1; i < chain.length - 1; i++) for (let t = 0; t < 3; t++) d[chain[i] * 3 + t] = 0.25 * prev[i - 1][t] + 0.5 * prev[i][t] + 0.25 * prev[i + 1][t];
    }
  }
}

export async function showMesh(canvas, folder) {
  const base = `/output/${folder}/mesh/`;
  const mesh = await (await fetch(base + "mesh.json?t=" + Date.now())).json();
  const gl = canvas.getContext("webgl");
  if (!gl) throw new Error("WebGL is not available");
  const V = mesh.vertices;
  const lo = [1e9, 1e9, 1e9], hi = [-1e9, -1e9, -1e9];
  for (const p of V) for (let i = 0; i < 3; i++) { lo[i] = Math.min(lo[i], p[i]); hi[i] = Math.max(hi[i], p[i]); }
  const c = lo.map((l, i) => (l + hi[i]) / 2);
  const scale = 2 / (hi[1] - lo[1]); // head height becomes 2 (y from -1 to 1)
  const halfW = (hi[0] - lo[0]) * scale / 2;
  const hasMouth = V.length > 415;
  const nV = V.length + (hasMouth ? 1 : 0); // one extra vertex in the middle of the mouth opening
  const mc = V.length;
  const pos = new Float32Array(nV * 3), uv = new Float32Array(nV * 2);
  V.forEach((p, i) => {
    for (let k = 0; k < 3; k++) pos[i * 3 + k] = (p[k] - c[k]) * scale;
    uv[i * 2] = mesh.uvs[i][0]; uv[i * 2 + 1] = 1 - mesh.uvs[i][1]; // mesh.json uvs are OBJ-style (v up); the image is top-down
  });
  const mouthCentre = () => { // the mouth point sits at the average of the lip loop around it
    for (let t = 0; t < 3; t++) pos[mc * 3 + t] = MOUTH.reduce((sum, v) => sum + pos[v * 3 + t], 0) / MOUTH.length;
  };
  if (hasMouth) { mouthCentre(); uv[mc * 2] = uv[MOUTH[0] * 2]; uv[mc * 2 + 1] = uv[MOUTH[0] * 2 + 1]; }
  const mouthTris = new Uint16Array(hasMouth ? MOUTH.flatMap((v, k) => [mc, v, MOUTH[(k + 1) % MOUTH.length]]) : []);
  // The saved model always has the eyes. eye_start is the first eye-fill vertex; the viewer can hide the triangles that use them.
  const eyeStart = mesh.eye_start ?? V.length;
  const idxAll = new Uint16Array(mesh.triangles.flat());
  const idxNoEyes = new Uint16Array(mesh.triangles.filter((t) => Math.max(...t) < eyeStart).flat());
  let idx = idxAll; // eyes are shown by default (the Show eyes box in the page starts ticked)
  const prog = gl.createProgram();
  const sh = (type, src) => { const s = gl.createShader(type); gl.shaderSource(s, src); gl.compileShader(s); gl.attachShader(prog, s); };
  sh(gl.VERTEX_SHADER, "attribute vec3 p; attribute vec2 t; uniform mat4 m; varying vec2 v; void main(){ v=t; gl_Position=m*vec4(p,1.0); }");
  sh(gl.FRAGMENT_SHADER, "precision mediump float; varying vec2 v; uniform sampler2D s; uniform float solid; void main(){ gl_FragColor = solid > 0.5 ? vec4(0.13, 0.04, 0.05, 1.0) : texture2D(s, v); }");
  gl.linkProgram(prog); gl.useProgram(prog);
  const buf = (data, name, size) => {
    gl.bindBuffer(gl.ARRAY_BUFFER, gl.createBuffer()); gl.bufferData(gl.ARRAY_BUFFER, data, gl.STATIC_DRAW);
    const l = gl.getAttribLocation(prog, name); gl.enableVertexAttribArray(l); gl.vertexAttribPointer(l, size, gl.FLOAT, false, 0, 0);
  };
  buf(pos, "p", 3);
  const posBuf = gl.getParameter(gl.ARRAY_BUFFER_BINDING);
  const pos0 = new Float32Array(pos); // resting positions; face tracking moves the live ones away from these
  const mh = V.length > 152 ? Math.hypot(V[10][0] - V[152][0], V[10][1] - V[152][1], V[10][2] - V[152][2]) : 1; // model face height
  const rings = []; // the eye-fill points follow the average of the eyelid points around them
  for (let e = eyeStart; e < V.length; e++) rings.push([e, [...new Set(mesh.triangles.filter((t) => t.includes(e)).flat().filter((v) => v !== e && v < eyeStart))]]);
  buf(uv, "t", 2);
  const eb = (data) => { const b = gl.createBuffer(); gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, b); gl.bufferData(gl.ELEMENT_ARRAY_BUFFER, data, gl.STATIC_DRAW); return b; };
  const ebAll = eb(idxAll), ebNoEyes = eb(idxNoEyes), ebMouth = eb(mouthTris);
  let ebCur = ebAll;
  const img = new Image();
  await new Promise((ok, bad) => { img.onload = ok; img.onerror = () => bad(new Error("texture missing")); img.src = base + "skin.png?t=" + Date.now(); });
  gl.bindTexture(gl.TEXTURE_2D, gl.createTexture());
  gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, gl.RGBA, gl.UNSIGNED_BYTE, img);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
  gl.enable(gl.DEPTH_TEST);
  const mLoc = gl.getUniformLocation(prog, "m");
  const solidLoc = gl.getUniformLocation(prog, "solid");
  let yaw = 0, pitch = 0, zoom = 1, drag = null, alive = true;
  const draw = () => {
    if (!alive || !canvas.isConnected) return;
    const w = canvas.clientWidth, h = canvas.clientHeight;
    if (canvas.width !== w || canvas.height !== h) { canvas.width = w; canvas.height = h; }
    gl.viewport(0, 0, w, h); gl.clearColor(0.05, 0.07, 0.1, 1); gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);
    const Y = yaw, P = pitch;
    const cy = Math.cos(Y), sy = Math.sin(Y), cp = Math.cos(P), sp = Math.sin(P);
    const a = h / w;
    // fill the canvas height (or width, if the window is tall and narrow) with a margin, camera distance is 4
    const s = zoom * 4 * 0.84 * Math.min(1, 1 / (a * halfW * 1.1));
    // rotation rows (yaw about y, then pitch about x) -> clip-space matrix, column-major, with a mild perspective (camera on +z)
    const r = [[cy, 0, sy], [sy * sp, cp, -cy * sp], [-sy * cp, sp, cy * cp]]; // r[row][col]
    const m = new Float32Array(16);
    for (let col = 0; col < 3; col++) {
      m[col * 4] = r[0][col] * s * a;
      m[col * 4 + 1] = r[1][col] * s;
      m[col * 4 + 2] = -r[2][col] * 0.2;
      m[col * 4 + 3] = -r[2][col];
    }
    m[15] = 4;
    gl.uniformMatrix4fv(mLoc, false, m);
    gl.uniform1f(solidLoc, 0);
    gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, ebCur);
    gl.drawElements(gl.TRIANGLES, idx.length, gl.UNSIGNED_SHORT, 0);
    if (mouthTris.length) { // the inside of the mouth: a flat dark patch behind the lips
      gl.uniform1f(solidLoc, 1);
      gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, ebMouth);
      gl.drawElements(gl.TRIANGLES, mouthTris.length, gl.UNSIGNED_SHORT, 0);
    }
    requestAnimationFrame(draw);
  };
  canvas.onpointerdown = (e) => { drag = [e.clientX, e.clientY]; canvas.setPointerCapture(e.pointerId); };
  canvas.onpointerup = () => { drag = null; };
  canvas.onpointermove = (e) => {
    if (!drag) return;
    yaw += (e.clientX - drag[0]) * 0.01;
    pitch = Math.max(-1.5, Math.min(1.5, pitch + (e.clientY - drag[1]) * 0.01));
    drag = [e.clientX, e.clientY];
  };
  canvas.onwheel = (e) => { e.preventDefault(); zoom = Math.max(0.4, Math.min(4, zoom * (e.deltaY < 0 ? 1.1 : 0.9))); };
  draw();
  const stop = () => { alive = false; };
  stop.resetView = () => { yaw = 0; pitch = 0; zoom = 1; };  // back to the default, front-on view
  // Camera control: the mask keeps its own face; the person in front of the camera only drives it. delta is how far each of the 478 points has
  // moved from rest (face heights, model axes), R is the head turn/tilt/roll. null goes back to the still mesh.
  stop.setLive = (R, delta) => {
    if (!R) { pos.set(pos0); gl.bindBuffer(gl.ARRAY_BUFFER, posBuf); gl.bufferSubData(gl.ARRAY_BUFFER, 0, pos); return; }
    const n = Math.min(478, eyeStart, V.length);
    const k = mh * scale;
    const d = Float32Array.from(delta);
    smoothChains(d);
    for (let i = 0; i < n; i++) for (let t = 0; t < 3; t++) pos[i * 3 + t] = pos0[i * 3 + t] + d[i * 3 + t] * k;
    for (const [e, ring] of rings) for (let t = 0; t < 3; t++) pos[e * 3 + t] = ring.length ? ring.reduce((sum, v) => sum + pos[v * 3 + t], 0) / ring.length : pos0[e * 3 + t];
    if (hasMouth) mouthCentre();
    for (let i = 0; i < nV; i++) { // then turn the whole mask with the head
      const x = pos[i * 3], y = pos[i * 3 + 1], z = pos[i * 3 + 2];
      pos[i * 3] = R[0] * x + R[1] * y + R[2] * z;
      pos[i * 3 + 1] = R[3] * x + R[4] * y + R[5] * z;
      pos[i * 3 + 2] = R[6] * x + R[7] * y + R[8] * z;
    }
    gl.bindBuffer(gl.ARRAY_BUFFER, posBuf);
    gl.bufferSubData(gl.ARRAY_BUFFER, 0, pos);
  };
  stop.setEyes = (on) => { idx = on ? idxAll : idxNoEyes; ebCur = on ? ebAll : ebNoEyes; };  // the page's Show eyes box starts ticked
  return stop;
}

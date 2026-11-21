import { runFaceScan } from "/static/orbit-scan.js?v=1";

const root = document.getElementById("root");

async function api(path, body) {
  const res = await fetch(path, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) });
  if (!res.ok) throw new Error(`${path} failed (${res.status})`);
  return res.json();
}

const esc = (s) => String(s).replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));

function start() {
  const last = (() => { try { return localStorage.getItem("meshName") || ""; } catch { return ""; } })();
  root.innerHTML = `
    <section class="card home-card">
      <h2>Scan a face</h2>
      <p class="home-hint">Look at the camera, then move your head slowly around. Everything is saved in the output folder under this name.</p>
      <input type="text" class="step-input" id="name" placeholder="Name" value="${esc(last)}" autocomplete="off" />
      <div class="step-actions step-actions--solo"><button type="button" class="btn-primary" id="go">Start scan</button></div>
    </section>`;
  const name = document.getElementById("name");
  const go = document.getElementById("go");
  name.focus();
  const begin = async () => {
    const n = name.value.trim();
    if (!n) return;
    try { localStorage.setItem("meshName", n); } catch { /* ignore */ }
    const result = await runFaceScan(root, n, api);
    if (result) done(n, result);
    else start();
  };
  go.onclick = begin;
  name.addEventListener("keydown", (e) => e.key === "Enter" && begin());
}

async function done(name, result) {
  const safe = name.trim().replace(/[^A-Za-z0-9_-]+/g, "_").replace(/^_+|_+$/g, "") || "face";
  root.innerHTML = `
    <section class="card home-card">
      <h2>${esc(name)}</h2>
      <p class="home-hint">Scan saved: ${esc(result.photos ?? "")} photos. The blended skin texture is below; the 3D model uses it.</p>
      <img id="skin" alt="Skin texture" style="max-width:360px;width:100%;border-radius:12px;background:#000" src="/output/${safe}/skin_stitched_preview.jpg?t=${Date.now()}" />
      <p class="home-hint" id="msg"></p>
      <div class="step-actions step-actions--row">
        <button type="button" class="btn-ghost" id="again">Scan again</button>
        <button type="button" class="btn-primary" id="export">Export 3D model</button>
      </div>
    </section>`;
  document.getElementById("again").onclick = start;
  document.getElementById("export").onclick = async () => {
    const msg = document.getElementById("msg");
    msg.textContent = "Building the model…";
    const r = await api("/api/mesh/export", { canonical_name: name });
    msg.textContent = r.status === "ok"
      ? `Saved ${r.vertices} points and ${r.triangles} triangles. Open mesh.obj in this folder: ${r.folder}`
      : `Could not export: ${r.message}`;
  };
}

start();

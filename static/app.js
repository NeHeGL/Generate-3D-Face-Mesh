import { runFaceScan } from "/static/orbit-scan.js?v=25";
import { showMesh } from "/static/viewer.js?v=31";
import { startHeadTracking } from "/static/head-track.js?v=7";

const root = document.getElementById("root");

async function api(path, body) {
  const res = await fetch(path, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) });
  if (!res.ok) throw new Error(`${path} failed (${res.status})`);
  return res.json();
}

const esc = (s) => String(s).replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
// "Jeff Molofee" -> "jeff_molofee" (same rule as the server)
const slug = (n) => n.trim().toLowerCase().replace(/[^a-z0-9_-]+/g, "_").replace(/^_+|_+$/g, "") || "face";

const profileBtn = document.getElementById("logout-btn");
profileBtn.textContent = "Change profile";
profileBtn.className = "btn-ghost";
profileBtn.onclick = start;

let stopViewer = null;
let stopTrack = null;
function killViewer() {  // leaving the viewer: stop the model and release the camera
  if (stopTrack) { stopTrack(); stopTrack = null; }
  if (stopViewer) { stopViewer(); stopViewer = null; }
}

function show(html) {
  const hw = document.getElementById("hw-mode");
  if (hw) hw.innerHTML = "";  // the Back button (viewer only) lives in the footer line: clear it whenever the screen changes
  document.body.classList.remove("cinematic");
  root.classList.remove("step-enter");
  root.innerHTML = html;
  requestAnimationFrame(() => root.classList.add("step-enter"));
}

async function getUsers() {
  try { return await (await fetch("/api/users")).json(); } catch { return []; }
}

async function start() {
  killViewer();
  profileBtn.hidden = true;  // this is the profile screen
  const users = await getUsers();
  const has = users.length > 0;
  // With people saved, a drop-down (pick one, or "New profile…" to type a name); without, just the name box.
  show(`
    <section class="step-card step-card--center">
      <h2 class="step-title">Profile Management</h2>
      ${has ? `<select class="step-input step-select" id="profile-select">
        <option value="" disabled selected>Select a profile…</option>
        ${users.map((u) => `<option value="${esc(u.name)}">${esc(u.name)}</option>`).join("")}
        <option value="__new__">＋ New profile…</option>
      </select>` : ""}
      <input type="text" class="step-input" id="name-input" placeholder="${has ? "Type the new profile's full name" : "What's your name?"}" autocomplete="off" ${has ? "hidden" : ""} />
      <div class="step-actions step-actions--solo">
        <button type="button" class="btn-primary" id="name-ok">OK</button>
      </div>
    </section>`);
  const input = document.getElementById("name-input");
  const select = document.getElementById("profile-select");
  const picked = () => Boolean(select && select.value && select.value !== "__new__");
  const submit = () => {
    const n = picked() ? select.value : input.value.trim();
    if (!n) { (select && select.value !== "__new__" ? select : input).focus(); return; }
    const existing = users.find((u) => u.name.toLowerCase() === n.toLowerCase());
    if (existing && (existing.has_scan || existing.has_mesh)) choose(existing.name, existing);
    else scan(existing ? existing.name : n);
  };
  if (select) {
    select.focus();
    select.addEventListener("change", () => {
      const makingNew = select.value === "__new__";
      input.hidden = !makingNew;
      if (makingNew) input.focus();
    });
    select.addEventListener("keydown", (e) => e.key === "Enter" && select.value && submit());
  } else {
    input.focus();
  }
  document.getElementById("name-ok").onclick = submit;
  input.addEventListener("keydown", (e) => e.key === "Enter" && submit());
}

async function scan(n) {
  killViewer();
  profileBtn.hidden = true;
  document.body.classList.add("cinematic");
  const result = await runFaceScan(root, n, api);
  if (result) view(n, { has_mesh: true });
  else start();
}

function choose(n, u) {
  profileBtn.hidden = false;
  show(`
    <section class="card" style="height:465px;box-sizing:border-box;display:flex;flex-direction:column">
      <h2>${esc(n)}</h2>
      <p class="step-copy">${esc(n)} already has a saved scan${u.has_mesh ? " and 3D model" : ""}. What would you like to do?</p>
      <div class="step-actions step-actions--row">
        <button type="button" class="btn-ghost" id="new">Create a new mesh</button>
        <button type="button" class="btn-primary" id="view">View the mesh</button>
      </div>
      <div class="step-actions" style="margin-top:1rem">
        <button type="button" class="btn-danger" id="delete-profile">Delete profile</button>
      </div>
      <p class="step-copy" id="msg"></p>
    </section>`);
  document.getElementById("new").onclick = () => scan(n);
  document.getElementById("view").onclick = () => view(n, u);
  document.getElementById("delete-profile").onclick = async () => {
    if (!confirm(`Delete the profile "${n}" and all of its saved scans and mesh data? This cannot be undone.`)) return;
    try {
      await api("/api/profile/delete", { canonical_name: n });
      start();
    } catch (e) {
      document.getElementById("msg").textContent = `Could not delete profile: ${e.message}`;
    }
  };
}

async function view(n, u) {
  profileBtn.hidden = false;
  const folder = slug(n);
  if (!u.has_mesh) {  // an old scan saved before models were built on save: it has to be scanned again
    document.getElementById("msg").textContent = "This profile has no 3D model yet. Choose Create a new mesh.";
    return;
  }
  show(`
    <section class="card" style="height:465px;box-sizing:border-box;display:flex;flex-direction:column;position:relative">
      <h2>${esc(n)}</h2>
      <div style="flex:1;min-height:0;position:relative"><canvas id="view3d" style="position:absolute;inset:0;width:100%;height:100%;border-radius:12px;touch-action:none;cursor:grab"></canvas></div>
      <div class="step-copy" style="margin:0.4rem auto 0;display:flex;gap:1.6rem;align-items:center;justify-content:center;flex:none">
        <label style="display:flex;gap:0.5rem;align-items:center;cursor:pointer"><input type="checkbox" id="eyes" checked /> Show eyes</label>
        <label style="display:flex;gap:0.5rem;align-items:center;cursor:pointer"><input type="checkbox" id="track" /> Control with camera</label>
      </div>
      <p class="step-copy" id="msg" style="margin:0.3rem auto 0;text-align:center;flex:none;height:1.5em;line-height:1.5em;white-space:nowrap;overflow:hidden">Control with the mouse: drag to turn, scroll to zoom.</p>
    </section>`);
  // Back: bottom-left of the page, on the same line as the credit text on the right; returns to "Create a new mesh / View the mesh".
  const hw = document.getElementById("hw-mode");
  hw.innerHTML = '<button type="button" class="btn-ghost" id="back" style="padding:0.3rem 0.8rem;font-size:0.85rem">‹ Back</button>';
  document.getElementById("back").onclick = () => { killViewer(); choose(n, u); };
  const msg = document.getElementById("msg");
  const load = async () => {
    killViewer();
    try { stopViewer = await showMesh(document.getElementById("view3d"), folder); }
    catch (e) { msg.textContent = `Could not show the mesh: ${e.message}`; }
  };
  document.getElementById("eyes").onchange = (e) => stopViewer && stopViewer.setEyes(e.target.checked);  // viewing only: the saved model always includes the eyes
  document.getElementById("track").onchange = async (e) => {
    if (stopTrack) { stopTrack(); stopTrack = null; }
    if (stopViewer) stopViewer.setLive(null);
    if (!e.target.checked) { msg.textContent = "Control with the mouse: drag to turn, scroll to zoom."; return; }
    if (stopViewer) stopViewer.resetView();  // start from the default, front-on view
    msg.textContent = "Starting the camera: look straight ahead, relaxed.";
    e.target.disabled = true;
    try {
      stopTrack = await startHeadTracking((R, d) => { if (stopViewer) stopViewer.setLive(R, d); }, (t) => { msg.textContent = t || "Control by moving your face: turn, tilt, open your mouth, blink."; });
      msg.textContent = "Control by moving your face: turn, tilt, open your mouth, blink.";
    } catch (err) {
      e.target.checked = false;
      msg.textContent = `Could not start head tracking: ${err.message || err}`;
    }
    e.target.disabled = false;
  };
  await load();
}

start();

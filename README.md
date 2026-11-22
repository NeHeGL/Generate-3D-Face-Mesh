# Generate 3D Mesh

A small local app that does one thing: builds a good 3D mesh of your face from your webcam.

Look at the camera and move your head slowly around. The page tracks your face in the browser (MediaPipe, bundled, no internet
needed) and keeps the best photo for each head angle. A grid shows which angles are covered. When coverage is good, Save blends the
photos into one seamless skin texture. **Export 3D model** then writes an OBJ model that uses it.

Everything for a person goes in `output/<name>/`:

Saving a scan builds the whole `mesh/` folder automatically.

| File | What |
|---|---|
| `profile.json` | the name as typed (the folder is the lowercase name: Jeff Molofee → `jeff_molofee`) |
| `photos/` | the best photo for each pose, with `photos.json` (measured head angle for each) |
| `mesh/mesh.obj`, `mesh.mtl`, `skin.png` | the 3D model and its skin texture: everything a 3D tool needs |
| `mesh/mesh.json` | the same mesh as plain data, used by the in-app viewer |
| `mesh/skin_atlas.json`, `preview.jpg` | the layout the model is built from, and a small picture of the skin |

## Run

Double-click **Start Generate 3D Mesh App.bat** (own window) or **Start Generate 3D Mesh Web.bat** (browser tab). The first run
creates a Python environment and installs the few packages in `requirements.txt` (needs Python 3.10+). The app listens on
`http://localhost:8780`.

## Notes

The 3D model export is new and untested against a live scan: it uses the 468 face landmarks (with depth) from the front-facing frame.

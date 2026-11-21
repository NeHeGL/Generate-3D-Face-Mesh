# Generate 3D Mesh

A small local app that does one thing: builds a good 3D mesh of your face from your webcam.

Look at the camera and move your head slowly around. The page tracks your face in the browser (MediaPipe, bundled, no internet
needed) and keeps the best photo for each head angle. A grid shows which angles are covered. When coverage is good, Save blends the
photos into one seamless skin texture. **Export 3D model** then writes an OBJ model that uses it.

Everything for a person goes in `output/<name>/`:

| File | What |
|---|---|
| `photos/` + `photos.json` | the best photo for each pose, with its measured head angle |
| `skin_atlas.json` | mesh layout: landmarks, triangles, texture coordinates, front-facing 3D landmarks |
| `skin_stitched.png` | the blended 2048 px skin texture |
| `mesh.obj`, `mesh.mtl` | the face model (open in Blender, MeshLab, Windows 3D Viewer) |

## Run

Double-click **Start Generate 3D Mesh App.bat** (own window) or **Start Generate 3D Mesh Web.bat** (browser tab). The first run
creates a Python environment and installs the few packages in `requirements.txt` (needs Python 3.10+). The app listens on
`http://localhost:8780`.

## Notes

This is the face scan from Avatar Studio, moved out into its own project. Avatar Studio now only needs a single good photo.
The 3D model export is new and untested against a live scan: it uses the 468 face landmarks (with depth) from the front-facing frame.

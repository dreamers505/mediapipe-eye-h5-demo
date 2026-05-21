# MediaPipe Eye H5 Demo

This demo validates camera-based cursor control outside the WeChat mini program runtime.

It uses MediaPipe Face Landmarker through `@mediapipe/tasks-vision` and reads the iris landmarks:

- Right iris: `468-472`
- Left iris: `473-477`

## Run

Use a local web server. Camera access will not work from a plain `file://` URL.

```bash
node mediapipe-eye-h5-demo/server.mjs
```

Open:

```text
http://localhost:5174
```

For phone testing, open the local network URL shown by the server, or use a tunnel.

## GitHub Pages

This is a static app. It can be hosted directly from the repository root with GitHub Pages.

After enabling Pages, open the generated HTTPS URL on your phone and allow camera access.

## Calibration

1. Click `Start`.
2. Look at center and click `Center`.
3. Look left, right, up, down and click the matching calibration buttons.
4. The cursor snaps to the 3 x 3 grid based on iris position.

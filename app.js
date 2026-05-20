import {
  FaceLandmarker,
  FilesetResolver
} from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision"

const video = document.querySelector("#video")
const overlay = document.querySelector("#overlay")
const ctx = overlay.getContext("2d")
const statusEl = document.querySelector("#status")
const startButton = document.querySelector("#startButton")
const cursor = document.querySelector("#cursor")
const gridButtons = [...document.querySelectorAll("[data-cell]")]
const calibrationButtons = [...document.querySelectorAll("[data-calibrate]")]

const metrics = {
  fps: document.querySelector("#fps"),
  landmarks: document.querySelector("#landmarks"),
  iris: document.querySelector("#iris"),
  cell: document.querySelector("#cell"),
  x: document.querySelector("#xValue"),
  y: document.querySelector("#yValue")
}

const state = {
  landmarker: null,
  running: false,
  lastVideoTime: -1,
  frames: 0,
  fpsAt: performance.now(),
  activeCell: 5,
  candidateCell: 5,
  cursor: { x: 0.5, y: 0.5 },
  calibration: {}
}

const targets = {
  center: { x: 0.5, y: 0.5 },
  left: { x: 0.18, y: 0.5 },
  right: { x: 0.82, y: 0.5 },
  up: { x: 0.5, y: 0.18 },
  down: { x: 0.5, y: 0.82 }
}

startButton.addEventListener("click", start)
document.querySelector("#resetCalibration").addEventListener("click", () => {
  state.calibration = {}
  calibrationButtons.forEach((button) => button.classList.remove("done"))
})

calibrationButtons.forEach((button) => {
  button.addEventListener("click", () => {
    if (!state.lastFeature) return
    const name = button.dataset.calibrate
    state.calibration[name] = { ...state.lastFeature }
    button.classList.add("done")
  })
})

async function start() {
  try {
    startButton.disabled = true
    startButton.textContent = "Starting"
    setStatus("Loading model")

    const fileset = await withTimeout(
      FilesetResolver.forVisionTasks(
        "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@latest/wasm"
      ),
      20000,
      "MediaPipe WASM load timed out"
    )

    state.landmarker = await withTimeout(
      createLandmarker(fileset),
      30000,
      "Face model load timed out"
    )

    setStatus("Requesting camera")
    if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
      throw new Error("Camera API is unavailable. Use HTTPS or localhost.")
    }

    const stream = await navigator.mediaDevices.getUserMedia({
      video: {
        facingMode: "user",
        width: { ideal: 960 },
        height: { ideal: 720 }
      },
      audio: false
    })

    setStatus("Starting video")
    video.srcObject = stream
    await video.play()
    state.running = true
    setStatus("Running")
    startButton.textContent = "Running"
    requestAnimationFrame(loop)
  } catch (error) {
    console.error(error)
    startButton.disabled = false
    startButton.textContent = "Retry"
    setStatus(error && error.message ? error.message : String(error), true)
  }
}

function withTimeout(promise, ms, message) {
  return Promise.race([
    promise,
    new Promise((_, reject) => {
      setTimeout(() => reject(new Error(message)), ms)
    })
  ])
}

async function createLandmarker(fileset) {
  const baseOptions = {
    modelAssetPath:
      "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task"
  }

  try {
    return await FaceLandmarker.createFromOptions(fileset, {
      baseOptions: { ...baseOptions, delegate: "GPU" },
      runningMode: "VIDEO",
      numFaces: 1
    })
  } catch (error) {
    setStatus("GPU failed, trying CPU")
    return FaceLandmarker.createFromOptions(fileset, {
      baseOptions,
      runningMode: "VIDEO",
      numFaces: 1
    })
  }
}

function setStatus(message, failed = false) {
  statusEl.textContent = message
  statusEl.classList.toggle("failed", failed)
}

function loop() {
  if (!state.running) return

  if (video.currentTime !== state.lastVideoTime) {
    state.lastVideoTime = video.currentTime
    const result = state.landmarker.detectForVideo(video, performance.now())
    handleResult(result)
  }

  requestAnimationFrame(loop)
}

function handleResult(result) {
  resizeOverlay()
  ctx.clearRect(0, 0, overlay.width, overlay.height)

  const landmarks = result.faceLandmarks && result.faceLandmarks[0]
  metrics.landmarks.textContent = landmarks ? landmarks.length : 0
  metrics.iris.textContent = landmarks && landmarks.length >= 478 ? "Yes" : "No"

  if (!landmarks || landmarks.length < 478) {
    statusEl.textContent = "No iris landmarks"
    return
  }

  drawLandmarks(landmarks)
  const feature = extractEyeFeature(landmarks)
  state.lastFeature = feature
  const point = calibratedPoint(feature)
  snapCursor(point)
  updateFps()
}

function extractEyeFeature(landmarks) {
  const rightEye = eyeFeature(landmarks, {
    outer: 33,
    inner: 133,
    top: 159,
    bottom: 145,
    iris: [468, 469, 470, 471, 472]
  })
  const leftEye = eyeFeature(landmarks, {
    outer: 263,
    inner: 362,
    top: 386,
    bottom: 374,
    iris: [473, 474, 475, 476, 477]
  })

  return {
    x: (rightEye.x + leftEye.x) / 2,
    y: (rightEye.y + leftEye.y) / 2
  }
}

function eyeFeature(landmarks, indices) {
  const outer = landmarks[indices.outer]
  const inner = landmarks[indices.inner]
  const top = landmarks[indices.top]
  const bottom = landmarks[indices.bottom]
  const iris = average(indices.iris.map((index) => landmarks[index]))
  const center = {
    x: (outer.x + inner.x) / 2,
    y: (top.y + bottom.y) / 2
  }
  const width = Math.max(0.001, distance(outer, inner))
  const height = Math.max(0.001, distance(top, bottom))

  return {
    x: (iris.x - center.x) / width,
    y: (iris.y - center.y) / height
  }
}

function calibratedPoint(feature) {
  const c = state.calibration

  let x = 0.5 + feature.x * 2.1
  let y = 0.5 + feature.y * 1.35

  if (c.center) {
    const dx = feature.x - c.center.x
    const dy = feature.y - c.center.y
    const leftSpan = c.left ? Math.abs(c.left.x - c.center.x) : 0.08
    const rightSpan = c.right ? Math.abs(c.right.x - c.center.x) : 0.08
    const upSpan = c.up ? Math.abs(c.up.y - c.center.y) : 0.08
    const downSpan = c.down ? Math.abs(c.down.y - c.center.y) : 0.08
    const xSpan = dx < 0 ? leftSpan : rightSpan
    const ySpan = dy < 0 ? upSpan : downSpan
    x = 0.5 + dx / Math.max(0.025, xSpan) * 0.32
    y = 0.5 + dy / Math.max(0.025, ySpan) * 0.32
  }

  return {
    x: clamp(x, 0.04, 0.96),
    y: clamp(y, 0.06, 0.94)
  }
}

function snapCursor(point) {
  const cell = cellFromPoint(point)
  const column = (cell - 1) % 3
  const row = Math.floor((cell - 1) / 3)
  state.cursor.x = (column + 0.5) / 3
  state.cursor.y = (row + 0.5) / 3
  state.candidateCell = cell
  state.activeCell = cell

  cursor.style.left = `${state.cursor.x * 100}%`
  cursor.style.top = `${state.cursor.y * 100}%`
  gridButtons.forEach((button) => {
    const isCell = Number(button.dataset.cell) === cell
    button.classList.toggle("candidate", isCell)
    button.classList.toggle("active", isCell)
  })

  metrics.cell.textContent = String(cell)
  metrics.x.textContent = point.x.toFixed(2)
  metrics.y.textContent = point.y.toFixed(2)
}

function cellFromPoint(point) {
  const column = clamp(Math.floor(point.x * 3), 0, 2)
  const row = clamp(Math.floor(point.y * 3), 0, 2)
  return row * 3 + column + 1
}

function drawLandmarks(landmarks) {
  const irisIds = [468, 469, 470, 471, 472, 473, 474, 475, 476, 477]
  ctx.save()
  ctx.scale(-1, 1)
  ctx.translate(-overlay.width, 0)

  for (const index of irisIds) {
    const point = landmarks[index]
    ctx.beginPath()
    ctx.arc(point.x * overlay.width, point.y * overlay.height, 4, 0, Math.PI * 2)
    ctx.fillStyle = "#fbbf24"
    ctx.fill()
  }

  ctx.restore()
}

function resizeOverlay() {
  const rect = video.getBoundingClientRect()
  const width = Math.max(1, Math.round(rect.width * devicePixelRatio))
  const height = Math.max(1, Math.round(rect.height * devicePixelRatio))
  if (overlay.width !== width || overlay.height !== height) {
    overlay.width = width
    overlay.height = height
  }
}

function updateFps() {
  state.frames += 1
  const now = performance.now()
  if (now - state.fpsAt > 1000) {
    metrics.fps.textContent = String(state.frames)
    state.frames = 0
    state.fpsAt = now
  }
}

function average(points) {
  const total = points.reduce((sum, point) => ({
    x: sum.x + point.x,
    y: sum.y + point.y
  }), { x: 0, y: 0 })
  return {
    x: total.x / points.length,
    y: total.y / points.length
  }
}

function distance(a, b) {
  return Math.hypot(a.x - b.x, a.y - b.y)
}

function clamp(value, min, max) {
  return Math.max(min, Math.min(max, value))
}

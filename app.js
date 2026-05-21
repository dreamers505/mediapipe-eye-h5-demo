import {
  FaceLandmarker,
  FilesetResolver,
  HandLandmarker
} from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision"

const video = document.querySelector("#video")
const overlay = document.querySelector("#overlay")
const ctx = overlay.getContext("2d")
const statusEl = document.querySelector("#status")
const startButton = document.querySelector("#startButton")
const modeButton = document.querySelector("#modeButton")
const noseCenterButton = document.querySelector("#noseCenter")
const noseLessButton = document.querySelector("#noseLess")
const noseMoreButton = document.querySelector("#noseMore")
const cursor = document.querySelector("#cursor")
const centerTarget = document.querySelector("#centerTarget")
const gridButtons = [...document.querySelectorAll("[data-cell]")]

const metrics = {
  fps: document.querySelector("#fps"),
  landmarks: document.querySelector("#landmarks"),
  action: document.querySelector("#action"),
  cell: document.querySelector("#cell"),
  x: document.querySelector("#xValue"),
  y: document.querySelector("#yValue")
}

const actionTitle = document.querySelector("#actionTitle")
const actionText = document.querySelector("#actionText")
const actionFill = document.querySelector("#actionFill")

const state = {
  handLandmarker: null,
  faceLandmarker: null,
  mode: "hand",
  running: false,
  lastVideoTime: -1,
  frames: 0,
  fpsAt: performance.now(),
  activeCell: 5,
  confirmedCell: 0,
  cursor: { x: 0.5, y: 0.5 },
  handCursor: { x: 0.5, y: 0.5 },
  noseCursor: { x: 0.5, y: 0.5 },
  actionCount: 0,
  actionAfter: 4,
  dwellCell: 5,
  dwellCount: 0,
  dwellAfter: 18,
  clickCooldown: 0,
  noseCenter: { x: 0.5, y: 0.5, ready: false },
  lastNose: { x: 0.5, y: 0.5 },
  noseGain: 1.6,
  noseCalibrating: true,
  calibrationSamples: [],
  calibrationCount: 0,
  calibrationAfter: 24
}

startButton.addEventListener("click", start)
modeButton.addEventListener("click", () => {
  state.mode = state.mode === "hand" ? "nose" : "hand"
  resetAction()
  if (state.mode === "nose") startNoseCalibration()
  updateModeUI()
})
noseCenterButton.addEventListener("click", () => {
  startNoseCalibration()
})
noseLessButton.addEventListener("click", () => {
  state.noseGain = Math.max(0.6, state.noseGain - 0.2)
  setStatus(`Nose gain ${state.noseGain.toFixed(1)}`)
})
noseMoreButton.addEventListener("click", () => {
  state.noseGain = Math.min(2.2, state.noseGain + 0.2)
  setStatus(`Nose gain ${state.noseGain.toFixed(1)}`)
})

updateModeUI()

async function start() {
  try {
    startButton.disabled = true
    modeButton.disabled = true
    startButton.textContent = "Starting"
    setStatus("Loading models")

    const fileset = await withTimeout(
      FilesetResolver.forVisionTasks(
        "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@latest/wasm"
      ),
      20000,
      "MediaPipe WASM load timed out"
    )

    state.handLandmarker = await withTimeout(createHandLandmarker(fileset), 30000, "Hand model load timed out")
    state.faceLandmarker = await withTimeout(createFaceLandmarker(fileset), 30000, "Face model load timed out")

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
    modeButton.disabled = false
    requestAnimationFrame(loop)
  } catch (error) {
    console.error(error)
    startButton.disabled = false
    modeButton.disabled = false
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

async function createHandLandmarker(fileset) {
  const baseOptions = {
    modelAssetPath:
      "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task"
  }
  try {
    return await HandLandmarker.createFromOptions(fileset, {
      baseOptions: { ...baseOptions, delegate: "GPU" },
      runningMode: "VIDEO",
      numHands: 1
    })
  } catch (error) {
    setStatus("Hand GPU failed, trying CPU")
    return HandLandmarker.createFromOptions(fileset, {
      baseOptions,
      runningMode: "VIDEO",
      numHands: 1
    })
  }
}

async function createFaceLandmarker(fileset) {
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
    setStatus("Face GPU failed, trying CPU")
    return FaceLandmarker.createFromOptions(fileset, {
      baseOptions,
      runningMode: "VIDEO",
      numFaces: 1
    })
  }
}

function loop() {
  if (!state.running) return

  if (video.currentTime !== state.lastVideoTime) {
    state.lastVideoTime = video.currentTime
    if (state.mode === "hand") {
      handleHand(state.handLandmarker.detectForVideo(video, performance.now()))
    } else {
      handleNose(state.faceLandmarker.detectForVideo(video, performance.now()))
    }
  }

  requestAnimationFrame(loop)
}

function handleHand(result) {
  resizeOverlay()
  ctx.clearRect(0, 0, overlay.width, overlay.height)

  const hand = result.landmarks && result.landmarks[0]
  metrics.landmarks.textContent = hand ? hand.length : 0

  if (!hand) {
    setStatus("Show one hand")
    updateAction(false, "open", 0)
    updateFps()
    return
  }

  setStatus("Hand detected")
  drawHand(hand)

  const indexTip = hand[8]
  const thumbTip = hand[4]
  const wrist = hand[0]
  const middleBase = hand[9]
  const handScale = Math.max(0.001, distance(wrist, middleBase))
  const pinchDistance = distance(indexTip, thumbTip) / handScale
  const pinching = pinchDistance < 0.72

  const point = {
    x: clamp(1 - indexTip.x, 0.02, 0.98),
    y: clamp(indexTip.y, 0.02, 0.98)
  }
  state.handCursor.x = state.handCursor.x * 0.72 + point.x * 0.28
  state.handCursor.y = state.handCursor.y * 0.72 + point.y * 0.28
  moveCursor(state.handCursor)
  updateAction(pinching, pinching ? "pinch" : "open", state.actionAfter)
  updateFps()
}

function handleNose(result) {
  resizeOverlay()
  ctx.clearRect(0, 0, overlay.width, overlay.height)

  const face = result.faceLandmarks && result.faceLandmarks[0]
  metrics.landmarks.textContent = face ? face.length : 0

  if (!face || !face[1]) {
    setStatus("Show your face")
    updateAction(false, "dwell", 0)
    updateFps()
    return
  }

  setStatus("Nose detected")
  const nose = face[1]
  const rawNose = {
    x: clamp(1 - nose.x, 0.02, 0.98),
    y: clamp(nose.y, 0.02, 0.98)
  }
  state.lastNose = rawNose

  if (state.noseCalibrating || !state.noseCenter.ready) {
    drawNose(face)
    updateNoseCalibration(rawNose)
    updateFps()
    return
  }

  drawNose(face)
  const nosePoint = mapNoseToCursor(rawNose)
  state.noseCursor.x = state.noseCursor.x * 0.38 + nosePoint.x * 0.62
  state.noseCursor.y = state.noseCursor.y * 0.32 + nosePoint.y * 0.68
  moveCursor(state.noseCursor)
  updateDwell()
  updateFps()
}

function mapNoseToCursor(nose) {
  const center = state.noseCenter.ready ? state.noseCenter : { x: 0.5, y: 0.5 }
  const dx = nose.x - center.x
  const dy = nose.y - center.y
  const deadX = 0.004
  const deadY = 0.004
  const shapedX = shapeAxis(Math.abs(dx) < deadX ? 0 : dx, 10.5 * state.noseGain)
  const shapedY = shapeAxis(Math.abs(dy) < deadY ? 0 : dy, 16 * state.noseGain)

  return {
    x: clamp(0.5 + shapedX, 0.02, 0.98),
    y: clamp(0.5 + shapedY, 0.02, 0.98)
  }
}

function shapeAxis(value, gain) {
  const sign = value < 0 ? -1 : 1
  const magnitude = Math.abs(value)
  return sign * Math.min(0.48, (magnitude * gain) + (magnitude * magnitude * gain * 18))
}

function startNoseCalibration() {
  state.noseCalibrating = true
  state.noseCenter = { x: 0.5, y: 0.5, ready: false }
  state.calibrationSamples = []
  state.calibrationCount = 0
  state.noseCursor = { x: 0.5, y: 0.5 }
  state.confirmedCell = 0
  moveCursor({ x: 0.5, y: 0.5 })
  actionTitle.textContent = "Centering"
  actionText.textContent = "hold"
  actionFill.style.width = "0%"
  centerTarget.classList.add("visible")
  centerTarget.classList.remove("inside")
  setStatus("Put your nose at center")
}

function updateNoseCalibration(nose) {
  const preview = previewNoseCentering(nose)
  moveCursor(preview.point)

  state.calibrationSamples.push(nose)
  if (state.calibrationSamples.length > 8) state.calibrationSamples.shift()

  const stable = preview.inside && isStable(state.calibrationSamples, 0.012)
  if (stable) {
    state.calibrationCount = Math.min(state.calibrationCount + 1, state.calibrationAfter)
  } else {
    state.calibrationCount = Math.max(0, state.calibrationCount - 2)
  }

  const percent = Math.round((state.calibrationCount / state.calibrationAfter) * 100)
  actionFill.style.width = `${percent}%`
  actionText.textContent = `${percent}%`

  if (state.calibrationCount >= state.calibrationAfter) {
    const center = averagePoint(state.calibrationSamples)
    state.noseCenter = { ...center, ready: true }
    state.noseCalibrating = false
    state.calibrationSamples = []
    state.calibrationCount = 0
    actionTitle.textContent = "Dwell"
    actionText.textContent = "dwell"
    actionFill.style.width = "0%"
    centerTarget.classList.remove("visible")
    centerTarget.classList.remove("inside")
    setStatus(`Nose centered · gain ${state.noseGain.toFixed(1)}`)
  } else {
    setStatus(preview.inside ? "Hold steady" : "Move nose to center")
  }
}

function previewNoseCentering(nose) {
  const dx = nose.x - 0.5
  const dy = nose.y - 0.5
  const point = {
    x: clamp(0.5 + dx * 1.8, 0.18, 0.82),
    y: clamp(0.5 + dy * 2.2, 0.18, 0.82)
  }
  const inside = Math.abs(point.x - 0.5) < 0.085 && Math.abs(point.y - 0.5) < 0.085

  centerTarget.classList.add("visible")
  centerTarget.classList.toggle("inside", inside)

  return { point, inside }
}

function isStable(samples, tolerance) {
  if (samples.length < 5) return false
  const center = averagePoint(samples)
  return samples.every((point) => (
    Math.abs(point.x - center.x) <= tolerance &&
    Math.abs(point.y - center.y) <= tolerance
  ))
}

function averagePoint(points) {
  const total = points.reduce((sum, point) => ({
    x: sum.x + point.x,
    y: sum.y + point.y
  }), { x: 0, y: 0 })

  return {
    x: total.x / points.length,
    y: total.y / points.length
  }
}

function moveCursor(point) {
  state.cursor.x = point.x
  state.cursor.y = point.y

  cursor.style.left = `${state.cursor.x * 100}%`
  cursor.style.top = `${state.cursor.y * 100}%`

  const cell = cellFromPoint(state.cursor)
  state.activeCell = cell

  gridButtons.forEach((button) => {
    const value = Number(button.dataset.cell)
    button.classList.toggle("candidate", value === cell)
    button.classList.toggle("active", value === cell)
    button.classList.toggle("confirmed", value === state.confirmedCell)
  })

  metrics.cell.textContent = String(cell)
  metrics.x.textContent = state.cursor.x.toFixed(2)
  metrics.y.textContent = state.cursor.y.toFixed(2)
}

function updateAction(active, text, threshold) {
  if (state.clickCooldown > 0) state.clickCooldown -= 1

  if (active) {
    state.actionCount = Math.min(state.actionCount + 1, threshold)
  } else {
    state.actionCount = 0
  }

  if (active && state.actionCount >= threshold && state.clickCooldown <= 0) {
    state.confirmedCell = state.activeCell
    state.clickCooldown = 12
  }

  const percent = threshold ? Math.round((state.actionCount / threshold) * 100) : 0
  actionText.textContent = text
  actionFill.style.width = `${percent}%`
  metrics.action.textContent = active ? "Yes" : "No"
}

function updateDwell() {
  if (state.dwellCell === state.activeCell) {
    state.dwellCount = Math.min(state.dwellCount + 1, state.dwellAfter)
  } else {
    state.dwellCell = state.activeCell
    state.dwellCount = 1
  }

  if (state.dwellCount >= state.dwellAfter) {
    state.confirmedCell = state.activeCell
  }

  const percent = Math.round((state.dwellCount / state.dwellAfter) * 100)
  actionText.textContent = "dwell"
  actionFill.style.width = `${percent}%`
  metrics.action.textContent = percent >= 100 ? "Yes" : "No"
}

function resetAction() {
  state.actionCount = 0
  state.dwellCount = 0
  state.dwellCell = state.activeCell
  actionFill.style.width = "0%"
}

function updateModeUI() {
  modeButton.textContent = state.mode === "hand" ? "Hand" : "Nose"
  actionTitle.textContent = state.mode === "hand" ? "Pinch" : (state.noseCalibrating ? "Centering" : "Dwell")
  actionText.textContent = state.mode === "hand" ? "open" : (state.noseCalibrating ? "hold" : "dwell")
  setStatus(state.mode === "hand" ? "Hand mode" : `Nose mode gain ${state.noseGain.toFixed(1)}`)
}

function cellFromPoint(point) {
  const column = clamp(Math.floor(point.x * 3), 0, 2)
  const row = clamp(Math.floor(point.y * 3), 0, 2)
  return row * 3 + column + 1
}

function drawHand(hand) {
  ctx.save()
  ctx.scale(-1, 1)
  ctx.translate(-overlay.width, 0)

  const links = [
    [0, 1], [1, 2], [2, 3], [3, 4],
    [0, 5], [5, 6], [6, 7], [7, 8],
    [5, 9], [9, 10], [10, 11], [11, 12],
    [9, 13], [13, 14], [14, 15], [15, 16],
    [13, 17], [17, 18], [18, 19], [19, 20],
    [0, 17]
  ]

  ctx.lineWidth = 3
  ctx.strokeStyle = "rgba(45, 212, 191, 0.9)"
  for (const [a, b] of links) {
    ctx.beginPath()
    ctx.moveTo(hand[a].x * overlay.width, hand[a].y * overlay.height)
    ctx.lineTo(hand[b].x * overlay.width, hand[b].y * overlay.height)
    ctx.stroke()
  }

  hand.forEach((point, index) => {
    ctx.beginPath()
    ctx.arc(point.x * overlay.width, point.y * overlay.height, index === 8 ? 7 : 4, 0, Math.PI * 2)
    ctx.fillStyle = index === 8 ? "#fbbf24" : "#ffffff"
    ctx.fill()
  })

  ctx.restore()
}

function drawNose(face) {
  ctx.save()
  ctx.scale(-1, 1)
  ctx.translate(-overlay.width, 0)

  const noseIds = [1, 4, 5, 6, 168]
  for (const index of noseIds) {
    const point = face[index]
    if (!point) continue
    ctx.beginPath()
    ctx.arc(point.x * overlay.width, point.y * overlay.height, index === 1 ? 8 : 4, 0, Math.PI * 2)
    ctx.fillStyle = index === 1 ? "#fbbf24" : "#ffffff"
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

function setStatus(message, failed = false) {
  statusEl.textContent = message
  statusEl.classList.toggle("failed", failed)
}

function distance(a, b) {
  return Math.hypot(a.x - b.x, a.y - b.y)
}

function clamp(value, min, max) {
  return Math.max(min, Math.min(max, value))
}

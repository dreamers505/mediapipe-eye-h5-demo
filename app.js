import {
  FilesetResolver,
  HandLandmarker
} from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision"

const video = document.querySelector("#video")
const overlay = document.querySelector("#overlay")
const ctx = overlay.getContext("2d")
const statusEl = document.querySelector("#status")
const startButton = document.querySelector("#startButton")
const cursor = document.querySelector("#cursor")
const gridButtons = [...document.querySelectorAll("[data-cell]")]

const metrics = {
  fps: document.querySelector("#fps"),
  landmarks: document.querySelector("#landmarks"),
  pinch: document.querySelector("#pinch"),
  cell: document.querySelector("#cell"),
  x: document.querySelector("#xValue"),
  y: document.querySelector("#yValue")
}

const pinchText = document.querySelector("#pinchText")
const pinchFill = document.querySelector("#pinchFill")

const state = {
  landmarker: null,
  running: false,
  lastVideoTime: -1,
  frames: 0,
  fpsAt: performance.now(),
  activeCell: 5,
  confirmedCell: 0,
  cursor: { x: 0.5, y: 0.5 },
  pinchCount: 0,
  pinchAfter: 4,
  clickCooldown: 0
}

startButton.addEventListener("click", start)

async function start() {
  try {
    startButton.disabled = true
    startButton.textContent = "Starting"
    setStatus("Loading hand model")

    const fileset = await withTimeout(
      FilesetResolver.forVisionTasks(
        "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@latest/wasm"
      ),
      20000,
      "MediaPipe WASM load timed out"
    )

    state.landmarker = await withTimeout(
      createHandLandmarker(fileset),
      30000,
      "Hand model load timed out"
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
    setStatus("GPU failed, trying CPU")
    return HandLandmarker.createFromOptions(fileset, {
      baseOptions,
      runningMode: "VIDEO",
      numHands: 1
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

  const hand = result.landmarks && result.landmarks[0]
  metrics.landmarks.textContent = hand ? hand.length : 0

  if (!hand) {
    setStatus("Show one hand")
    updatePinch(false, 0)
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

  moveCursor(point)
  updatePinch(pinching, pinchDistance)
  updateFps()
}

function moveCursor(point) {
  state.cursor.x = state.cursor.x * 0.72 + point.x * 0.28
  state.cursor.y = state.cursor.y * 0.72 + point.y * 0.28

  cursor.style.left = `${state.cursor.x * 100}%`
  cursor.style.top = `${state.cursor.y * 100}%`

  const cell = cellFromPoint(state.cursor)
  state.activeCell = cell

  gridButtons.forEach((button) => {
    const isCell = Number(button.dataset.cell) === cell
    button.classList.toggle("candidate", isCell)
    button.classList.toggle("active", isCell)
    button.classList.toggle("confirmed", Number(button.dataset.cell) === state.confirmedCell)
  })

  metrics.cell.textContent = String(cell)
  metrics.x.textContent = state.cursor.x.toFixed(2)
  metrics.y.textContent = state.cursor.y.toFixed(2)
}

function updatePinch(pinching, pinchDistance) {
  if (state.clickCooldown > 0) state.clickCooldown -= 1

  if (pinching) {
    state.pinchCount = Math.min(state.pinchCount + 1, state.pinchAfter)
  } else {
    state.pinchCount = 0
  }

  if (pinching && state.pinchCount >= state.pinchAfter && state.clickCooldown <= 0) {
    state.confirmedCell = state.activeCell
    state.clickCooldown = 12
  }

  const percent = Math.round((state.pinchCount / state.pinchAfter) * 100)
  pinchText.textContent = pinching ? "pinch" : "open"
  pinchFill.style.width = `${percent}%`
  metrics.pinch.textContent = pinching ? "Yes" : "No"
  metrics.x.title = `pinch distance ${pinchDistance.toFixed(2)}`
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

function distance(a, b) {
  return Math.hypot(a.x - b.x, a.y - b.y)
}

function clamp(value, min, max) {
  return Math.max(min, Math.min(max, value))
}

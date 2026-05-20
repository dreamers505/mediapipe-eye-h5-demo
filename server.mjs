import { createServer } from "node:http"
import { readFile } from "node:fs/promises"
import { extname, join, normalize } from "node:path"
import { fileURLToPath } from "node:url"

const root = fileURLToPath(new URL(".", import.meta.url))
const port = 5174

const types = {
  ".html": "text/html; charset=utf-8",
  ".css": "text/css; charset=utf-8",
  ".js": "text/javascript; charset=utf-8",
  ".md": "text/plain; charset=utf-8"
}

createServer(async (request, response) => {
  try {
    const url = new URL(request.url || "/", `http://${request.headers.host}`)
    const pathname = url.pathname === "/" ? "/index.html" : url.pathname
    const filePath = normalize(join(root, pathname))
    if (!filePath.startsWith(normalize(root))) {
      response.writeHead(403)
      response.end("Forbidden")
      return
    }

    const content = await readFile(filePath)
    response.writeHead(200, {
      "Content-Type": types[extname(filePath)] || "application/octet-stream"
    })
    response.end(content)
  } catch (error) {
    response.writeHead(404)
    response.end("Not found")
  }
}).listen(port, () => {
  console.log(`MediaPipe demo: http://localhost:${port}`)
})

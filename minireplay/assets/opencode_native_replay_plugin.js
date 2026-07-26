import { mkdir, readFile, rename, writeFile } from "node:fs/promises"
import { readFileSync } from "node:fs"
import path from "node:path"

const endpoint = process.env.NATIVE_REPLAY_BOUNDARY_URL
const token = process.env.NATIVE_REPLAY_BOUNDARY_TOKEN
const actor = process.env.NATIVE_REPLAY_ACTOR_ID
const invocation = process.env.NATIVE_REPLAY_INVOCATION_ID || actor
const role = "coral-opencode"
const reservations = new Map()
const dispatchReservations = new Map()
const startPromises = new Map()
const completed = new Set()
const completionPromises = new Map()
const sessionNames = new Map()
const sessionParents = new Map()
const childCounters = new Map()
const activeTasks = new Map()
const toolArguments = new Map()
const toolNames = new Map()
const subprocessTools = new Set(["bash", "shell", "glob", "grep", "skill"])
let nativeWorkspace
let gatePromise
const observationMetadata = {
  bash: ["output"],
  shell: ["output"],
  edit: ["diff", "filediff"],
  read: ["preview", "display"],
  task: ["parentSessionId", "sessionId"],
}
function recordedResultContract(tool) {
  const pointers = ["/output", "/error"]
  for (const field of observationMetadata[tool] || []) pointers.push(`/metadata/${field}`)
  return {
    schema_version: "native-agent-replay.result-contract/v2",
    kind: "recorded-observation",
    fields: pointers.map((json_pointer) => ({json_pointer, optional: true})),
  }
}

function applyRecordedObservation(target, recorded, tool) {
  for (const field of ["output", "error"]) {
    if (field in recorded) target[field] = recorded[field]
  }
  const metadata = recorded.metadata
  if (!metadata || typeof metadata !== "object") return
  target.metadata ||= {}
  for (const field of observationMetadata[tool] || []) {
    if (field in metadata) target.metadata[field] = metadata[field]
  }
}

const hostMonotonicOffsetNs = (() => {
  const uptime = Number(readFileSync("/proc/uptime", "utf8").trim().split(/\s+/)[0])
  if (!Number.isFinite(uptime) || uptime < 0) {
    throw new Error("native replay could not read the Linux monotonic clock")
  }
  return BigInt(Math.round(uptime * 1e9)) - process.hrtime.bigint()
})()

function required(name, value) {
  if (!value) throw new Error(`native replay requires ${name}`)
  return value
}

function monotonicBigIntNs() {
  return hostMonotonicOffsetNs + process.hrtime.bigint()
}

function monotonicNs() {
  return Number(monotonicBigIntNs())
}

async function post(route, payload) {
  const response = await fetch(`${required("boundary endpoint", endpoint)}${route}`, {
    method: "POST",
    headers: {
      Authorization: `Bearer ${required("boundary token", token)}`,
      "Content-Type": "application/json",
    },
    body: JSON.stringify(payload),
  })
  const body = await response.json()
  if (!response.ok) throw new Error(body.error || `native replay boundary HTTP ${response.status}`)
  return body
}

function stableSession(sessionID) {
  if (sessionNames.has(sessionID)) return sessionNames.get(sessionID)
  const value = `${required("invocation", invocation)}/root-${sessionNames.size}`
  sessionNames.set(sessionID, value)
  return value
}

async function signalReadyAndWait() {
  const readyDir = required("ready directory", process.env.NATIVE_REPLAY_READY_DIR)
  const gate = required("start gate", process.env.NATIVE_REPLAY_START_GATE)
  const runID = required("run ID", process.env.NATIVE_REPLAY_RUN_ID)
  const gateActors = JSON.parse(process.env.NATIVE_REPLAY_GATE_ACTORS || "[]")
  const participates = gateActors.includes(actor)
  if (participates) {
    await mkdir(readyDir, { recursive: true })
    const destination = path.join(readyDir, `${actor}.json`)
    let gateAlreadyOpen = false
    try {
      await readFile(gate)
      gateAlreadyOpen = true
    } catch (error) {
      if (error.code !== "ENOENT") throw error
    }
    if (gateAlreadyOpen) {
      let prior
      try {
        prior = JSON.parse(await readFile(destination, "utf8"))
      } catch (error) {
        throw new Error(`native replay actor started after gate without prior readiness: ${error}`)
      }
      if (prior.run_id !== runID || prior.actor_id !== actor) {
        throw new Error("native replay restart readiness belongs to another actor or run")
      }
    } else {
      const temporary = `${destination}.tmp.${process.pid}`
      await writeFile(temporary, `${JSON.stringify({
        schema_version: "native-agent-replay.actor-ready/v1",
        run_id: runID,
        actor_id: actor,
        process_role: role,
        pid: process.pid,
        ready_at_ns: monotonicNs(),
        runtime: "opencode-plugin",
      })}\n`)
      await rename(temporary, destination)
    }
  }
  const deadline = Date.now() + Number(process.env.NATIVE_REPLAY_GATE_TIMEOUT_S || "1800") * 1000
  let payload
  while (Date.now() < deadline) {
    try {
      payload = JSON.parse(await readFile(gate, "utf8"))
      break
    } catch (error) {
      if (error.code !== "ENOENT") throw error
      await Bun.sleep(5)
    }
  }
  if (!payload) throw new Error("timed out waiting for native replay start gate")
  if (payload.run_id !== runID) throw new Error("native replay start gate belongs to another run")
  const offsets = JSON.parse(process.env.NATIVE_REPLAY_ARRIVAL_OFFSETS || "{}")
  const offsetNs = BigInt(Math.round(Number(offsets[actor] || 0) * 1e9))
  const opened = BigInt(payload.opened_at_ns_decimal || String(payload.opened_at_ns))
  const release = opened + offsetNs
  while (monotonicBigIntNs() < release) {
    const remainingMs = Number(release - monotonicBigIntNs()) / 1e6
    await Bun.sleep(Math.max(1, Math.min(5, remainingMs)))
  }
}

function ensureGate() {
  if (!gatePromise) gatePromise = signalReadyAndWait()
  return gatePromise
}

function eventStartArguments(part) {
  if (part.state?.status === "running") return part.state.input || {}
  if (part.tool !== "task" || part.state?.status !== "pending") return null
  if (
    part.state.input
    && typeof part.state.input === "object"
    && Object.keys(part.state.input).length
  ) {
    return part.state.input
  }
  // The built-in task tool can bypass tool.execute.before. Its pending part is
  // streamed repeatedly, so wait until the raw argument object is complete
  // before reserving the native operation.
  try {
    const parsed = JSON.parse(part.state.raw)
    return parsed && typeof parsed === "object" ? parsed : null
  } catch {
    return null
  }
}

async function runningTaskForChild(
  client,
  directory,
  parentSessionID,
  childSessionID,
) {
  // Some OpenCode versions persist the task's running state and create the
  // child session without publishing tool.execute.before (or even a usable
  // pending-part event). Backfill from the framework's own message store before
  // the child can issue its first LLM request.
  for (let attempt = 0; attempt < 50; attempt += 1) {
    const response = await client.session.messages({
      path: { id: parentSessionID },
      query: { directory },
    })
    const messages = Array.isArray(response.data) ? response.data : []
    const candidates = messages
      .flatMap((message) => Array.isArray(message.parts) ? message.parts : [])
      .filter((part) => (
        part?.type === "tool"
        && part.tool === "task"
        && ["pending", "running"].includes(part.state?.status)
      ))
    const exact = candidates.find((part) => {
      const metadata = part.state?.metadata
      return metadata?.sessionId === childSessionID || metadata?.sessionID === childSessionID
    })
    if (exact) return exact
    if (candidates.length === 1) return candidates[0]
    await Bun.sleep(20)
  }
  return null
}

async function startTool(callID, tool, sessionID, args) {
  if (reservations.has(callID)) return null
  const existing = startPromises.get(callID)
  if (existing) return await existing

  const pending = (async () => {
    await ensureGate()
    toolNames.set(callID, tool)
    const dispatch = await post("/v1/boundary/start", {
      kind: "dispatch",
      actor_id: actor,
      session_id: stableSession(sessionID),
      process_role: role,
      parent_span_id: sessionParents.get(sessionID) || null,
      started_at_ns: monotonicNs(),
      origin: {
        kind: "llm_structured",
        trigger_id: "auto",
        model_call_id: callID,
      },
      parser_identity: "opencode.message.part.tool",
      dispatcher_identity: "opencode.tool.execute.before",
      native_call_id: callID,
      name: tool,
      arguments: args,
      workspace_path: required("native workspace", nativeWorkspace),
    })
    if (dispatch.execution_arguments) {
      for (const key of Object.keys(args)) delete args[key]
      Object.assign(args, dispatch.execution_arguments)
    }
    toolArguments.set(callID, args)
    dispatchReservations.set(callID, dispatch)
    const reservation = await post("/v1/boundary/start", {
      kind: "tool",
      actor_id: actor,
      process_role: role,
      parent_span_id: dispatch.span_id,
      started_at_ns: monotonicNs(),
      dispatch_id: dispatch.record_id,
      name: tool,
      implementation: process.env.NATIVE_REPLAY_OPENCODE_IDENTITY || "opencode:unknown",
      arguments: args,
      workspace_path: required("native workspace", nativeWorkspace),
      result_contract: recordedResultContract(tool),
    })
    reservations.set(callID, reservation)
    if (tool === "task") {
      const values = activeTasks.get(sessionID) || []
      values.push(reservation)
      activeTasks.set(sessionID, values)
    }
    return reservation
  })()
  startPromises.set(callID, pending)
  try {
    return await pending
  } finally {
    startPromises.delete(callID)
  }
}

async function finish(callID, status, result) {
  if (completed.has(callID)) return null
  const existing = completionPromises.get(callID)
  if (existing) return await existing

  const pending = (async () => {
    const reservation = reservations.get(callID)
    if (!reservation) throw new Error(`OpenCode tool completion has no start: ${callID}`)
    const completion = await post("/v1/boundary/complete", {
      reservation_id: reservation.reservation_id,
      ended_at_ns: monotonicNs(),
      status,
      result,
      logical_frames: [],
      side_effects: {},
      child_processes: subprocessTools.has(toolNames.get(callID))
        ? [{
            kind: "native-subprocess",
            owner_actor: actor,
            launcher: `opencode.${toolNames.get(callID)}`,
            command: toolArguments.get(callID) || {},
            cwd: null,
            shell: ["bash", "shell"].includes(toolNames.get(callID)),
            executable: null,
          }]
        : [],
      native_execution: true,
    })
    const dispatch = dispatchReservations.get(callID)
    if (!dispatch) throw new Error(`OpenCode tool completion has no dispatch: ${callID}`)
    await post("/v1/boundary/complete", {
      reservation_id: dispatch.reservation_id,
      ended_at_ns: monotonicNs(),
      status: "executed",
      execution_call_id: reservation.record_id,
    })
    completed.add(callID)
    reservations.delete(callID)
    dispatchReservations.delete(callID)
    toolArguments.delete(callID)
    toolNames.delete(callID)
    return completion
  })()
  completionPromises.set(callID, pending)
  try {
    return await pending
  } finally {
    completionPromises.delete(callID)
  }
}

const NativeReplayPlugin = async ({ client, directory }) => {
  nativeWorkspace = directory
  delete process.env.BUN_CONFIG_REGISTRY
  delete process.env.npm_config_registry
  return {
    event: async ({ event }) => {
      if (event.type === "session.created" || event.type === "session.created.1") {
        const info = event.properties.info
        if (info.parentID) {
          const parent = stableSession(info.parentID)
          const index = childCounters.get(parent) || 0
          childCounters.set(parent, index + 1)
          sessionNames.set(info.id, `${parent}/child-${index}`)
          const candidates = activeTasks.get(info.parentID) || []
          let reservation = candidates.shift()
          if (!reservation) {
            const taskPart = await runningTaskForChild(
              client,
              directory,
              info.parentID,
              info.id,
            )
            if (!taskPart) {
              throw new Error(
                `native replay could not bind child session ${info.id} to its parent task`,
              )
            }
            reservation = await startTool(
              taskPart.callID,
              taskPart.tool,
              taskPart.sessionID,
              eventStartArguments(taskPart) || {},
            )
            const queued = activeTasks.get(info.parentID) || []
            const queuedIndex = queued.indexOf(reservation)
            if (queuedIndex >= 0) queued.splice(queuedIndex, 1)
          }
          if (reservation) sessionParents.set(info.id, reservation.span_id)
        } else {
          stableSession(info.id)
        }
      }
      if (
        event.type === "message.part.updated"
        || event.type === "message.part.updated.1"
      ) {
        const part = event.properties.part
        // OpenCode's built-in task/subagent tool can publish its running state
        // without invoking tool.execute.before. Capture that state transition so
        // the parent operation exists before the child session begins. The
        // shared start promise also makes this safe for ordinary tools whose hook
        // and event arrive concurrently.
        const startArguments = part.type === "tool" ? eventStartArguments(part) : null
        if (startArguments && !reservations.has(part.callID)) {
          await startTool(
            part.callID,
            part.tool,
            part.sessionID,
            startArguments,
          )
        }
        if (part.type === "tool" && part.state?.status === "error") {
          if (!reservations.has(part.callID)) {
            await startTool(
              part.callID,
              part.tool,
              part.sessionID,
              part.state.input || {},
            )
          }
          const tool = toolNames.get(part.callID)
          const completion = await finish(part.callID, "error", {
            error: part.state.error,
            metadata: part.state.metadata || {},
          })
          if (completion?.result_replay_required === true) {
            const recorded = completion.framework_result
            if (!recorded || typeof recorded !== "object") {
              throw new Error("OpenCode replay has no recorded framework error")
            }
            applyRecordedObservation(part.state, recorded, tool)
          }
        }
      }
    },
    "chat.headers": async (input, output) => {
      await ensureGate()
      output.headers["X-Native-Replay-Actor"] = actor
      output.headers["X-Native-Replay-Session"] = stableSession(input.sessionID)
      output.headers["X-Native-Replay-Role"] = sessionParents.has(input.sessionID)
        ? "coral-subagent"
        : "coral-agent"
      output.headers["X-Native-Replay-Target"] = process.env.NATIVE_REPLAY_TARGET_ID || "default"
      const parentSpan = sessionParents.get(input.sessionID)
      if (parentSpan) output.headers["X-Native-Replay-Parent-Span"] = parentSpan
    },
    "tool.execute.before": async (input, output) => {
      await startTool(input.callID, input.tool, input.sessionID, output.args)
    },
    "tool.execute.after": async (input, output) => {
      const completion = await finish(input.callID, "ok", {
        title: output.title,
        output: output.output,
        metadata: output.metadata || {},
      })
      if (completion?.result_replay_required === true) {
        const recorded = completion.framework_result
        if (!recorded || typeof recorded !== "object") {
          throw new Error("OpenCode replay has no recorded framework output")
        }
        applyRecordedObservation(output, recorded, input.tool)
      }
    },
    "shell.env": async (input, output) => {
      delete output.env.BUN_CONFIG_REGISTRY
      delete output.env.npm_config_registry
      const args = toolArguments.get(input.callID) || {}
      const command = String(args.command || "")
      const reservation = reservations.get(input.callID)
      output.env.NATIVE_REPLAY_ADAPTER = /(^|\s)coral\s+(eval|wait|log)(\s|$)/.test(command)
        ? "coral"
        : ""
      output.env.NATIVE_REPLAY_PROCESS_ROLE = "coral-tool-child"
      output.env.NATIVE_REPLAY_ACTOR_ID = actor
      output.env.NATIVE_REPLAY_SESSION_ID = stableSession(input.sessionID || actor)
      if (reservation) output.env.NATIVE_REPLAY_PARENT_SPAN_ID = reservation.span_id
    },
    dispose: async () => {
      if (reservations.size) throw new Error(`${reservations.size} OpenCode tools never completed`)
      if (dispatchReservations.size) {
        throw new Error(`${dispatchReservations.size} OpenCode dispatches never completed`)
      }
    },
  }
}

export default NativeReplayPlugin

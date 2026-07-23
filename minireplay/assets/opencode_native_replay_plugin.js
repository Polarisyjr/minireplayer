import { mkdir, readFile, rename, writeFile } from "node:fs/promises"
import { readFileSync } from "node:fs"
import path from "node:path"

const endpoint = process.env.NATIVE_REPLAY_BOUNDARY_URL
const token = process.env.NATIVE_REPLAY_BOUNDARY_TOKEN
const actor = process.env.NATIVE_REPLAY_ACTOR_ID
const role = "coral-opencode"
const reservations = new Map()
const dispatchReservations = new Map()
const completed = new Set()
const completionPromises = new Map()
const sessionNames = new Map()
const sessionParents = new Map()
const childCounters = new Map()
const activeTasks = new Map()
const toolArguments = new Map()
const toolNames = new Map()
const subprocessTools = new Set(["bash", "shell", "glob", "grep", "skill"])
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
  const value = `${required("actor", actor)}/root-${sessionNames.size}`
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

export const NativeReplayPlugin = async () => {
  delete process.env.BUN_CONFIG_REGISTRY
  delete process.env.npm_config_registry
  return {
    event: async ({ event }) => {
      if (event.type === "session.created") {
        const info = event.properties.info
        if (info.parentID) {
          const parent = stableSession(info.parentID)
          const index = childCounters.get(parent) || 0
          childCounters.set(parent, index + 1)
          sessionNames.set(info.id, `${parent}/child-${index}`)
          const candidates = activeTasks.get(info.parentID) || []
          if (candidates.length) sessionParents.set(info.id, candidates.shift().span_id)
        } else {
          stableSession(info.id)
        }
      }
      if (event.type === "message.part.updated") {
        const part = event.properties.part
        if (part.type === "tool" && part.state.status === "error") {
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
      await ensureGate()
      toolNames.set(input.callID, input.tool)
      const dispatch = await post("/v1/boundary/start", {
        kind: "dispatch",
        actor_id: actor,
        session_id: stableSession(input.sessionID),
        process_role: role,
        parent_span_id: sessionParents.get(input.sessionID) || null,
        started_at_ns: monotonicNs(),
        origin: {
          kind: "llm_structured",
          trigger_id: "auto",
          model_call_id: input.callID,
        },
        parser_identity: "opencode.message.part.tool",
        dispatcher_identity: "opencode.tool.execute.before",
        native_call_id: input.callID,
        name: input.tool,
        arguments: output.args,
      })
      if (dispatch.execution_arguments) {
        for (const key of Object.keys(output.args)) delete output.args[key]
        Object.assign(output.args, dispatch.execution_arguments)
      }
      toolArguments.set(input.callID, output.args)
      dispatchReservations.set(input.callID, dispatch)
      const reservation = await post("/v1/boundary/start", {
        kind: "tool",
        actor_id: actor,
        process_role: role,
        parent_span_id: dispatch.span_id,
        started_at_ns: monotonicNs(),
        dispatch_id: dispatch.record_id,
        name: input.tool,
        implementation: process.env.NATIVE_REPLAY_OPENCODE_IDENTITY || "opencode:unknown",
        arguments: output.args,
        result_contract: recordedResultContract(input.tool),
      })
      reservations.set(input.callID, reservation)
      if (input.tool === "task") {
        const values = activeTasks.get(input.sessionID) || []
        values.push(reservation)
        activeTasks.set(input.sessionID, values)
      }
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

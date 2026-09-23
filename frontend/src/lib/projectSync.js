/**
 * Keeps the user's project on the server until they press New project.
 *
 * The app's state hooks (usePersisted, usePlotHistory) read localStorage on
 * first render and write it on every change. So the server copy is loaded into
 * localStorage before React mounts, and every change schedules a debounced
 * save of the project keys. localStorage stays a fast local cache.
 */
import { deleteProject, fetchProjectState, saveProjectState } from '../api/client'

const PREFIX = 'sst_v1_'
// UI preferences (theme, panel width, preview size, think mode) stay per browser.
const PROJECT_KEYS = [
  'activeStep', 'completedSteps', 'fileName', 'datasetId', 'datasetVersionId',
  'targetVariable', 'selectedFeatures', 'modelMetrics', 'versionHistory',
  'chatMessages', 'plotHistory',
]
const OWNER_KEY = 'sst_owner'
const SAVE_DELAY_MS = 1500
const RETRY_MS = 15_000
// The server accepts 5 MB; plots are dropped oldest first to stay under it.
const MAX_STATE_CHARS = 4.5 * 1024 * 1024

let enabled = false
let timer = null
let saving = null
let dirty = false

function readLocal() {
  const values = {}
  let savedAt = 0
  for (const key of PROJECT_KEYS) {
    try {
      const raw = localStorage.getItem(PREFIX + key)
      if (!raw) continue
      const { value, ts } = JSON.parse(raw)
      values[key] = value
      savedAt = Math.max(savedAt, ts || 0)
    } catch {
      // unreadable entry: leave it out
    }
  }
  return { values, savedAt }
}

function clearLocal() {
  for (const key of PROJECT_KEYS) localStorage.removeItem(PREFIX + key)
}

function writeLocal(state) {
  clearLocal()
  for (const [key, value] of Object.entries(state.values || {})) {
    if (!PROJECT_KEYS.includes(key)) continue
    try {
      localStorage.setItem(PREFIX + key, JSON.stringify({ value, ts: state.savedAt }))
    } catch {
      // localStorage full: the server copy still has it
    }
  }
}

function snapshot() {
  const { values } = readLocal()
  const state = { version: 1, savedAt: Date.now(), values }
  while (JSON.stringify(state).length > MAX_STATE_CHARS && values.plotHistory?.plots?.length) {
    values.plotHistory = { ...values.plotHistory, plots: values.plotHistory.plots.slice(1) }
  }
  return state
}

async function saveNow() {
  clearTimeout(timer)
  timer = null
  if (!enabled) return
  if (saving) {
    dirty = true
    return
  }
  saving = saveProjectState(snapshot())
    .catch((err) => console.warn('Could not save the project', err))
    .finally(() => {
      saving = null
      if (dirty) {
        dirty = false
        scheduleSave()
      }
    })
  await saving
}

function scheduleSave(delay = SAVE_DELAY_MS) {
  if (!enabled) return
  clearTimeout(timer)
  timer = setTimeout(saveNow, delay)
}

/** Called by the persisted-state hooks after each write. */
export function notifyProjectChange() {
  scheduleSave()
}

/**
 * Adopt the server copy unless this browser holds a newer one of the same user.
 * Returns true when localStorage was replaced.
 */
/** The server's `{ user_id, state }`; anything else (e.g. an HTML fallback page) throws. */
async function loadServerProject() {
  const data = await fetchProjectState()
  const userId = data?.user_id
  const state = data?.state
  if (typeof userId !== 'string' || !userId || (state !== null && typeof state !== 'object')) {
    throw new Error('Unexpected /project/state response')
  }
  return { userId, state }
}

function reconcile(userId, serverState) {
  enabled = true
  const previousOwner = localStorage.getItem(OWNER_KEY)
  let replaced = false
  if (previousOwner && previousOwner !== userId) {
    clearLocal()
    replaced = true
  }
  localStorage.setItem(OWNER_KEY, userId)
  const local = readLocal()
  if (serverState?.values && (serverState.savedAt || 0) >= local.savedAt) {
    writeLocal(serverState)
    return replaced || serverState.savedAt !== local.savedAt
  }
  if (Object.keys(local.values).length) scheduleSave(0)
  return replaced
}

async function retryLoad() {
  try {
    const { userId, state } = await loadServerProject()
    if (reconcile(userId, state)) window.location.reload()
  } catch {
    setTimeout(retryLoad, RETRY_MS)
  }
}

/** Load the saved project into localStorage. Call before rendering the app. */
export async function bootstrapProject() {
  // An owner of "undefined" came from a malformed response; it names no user.
  if (localStorage.getItem(OWNER_KEY) === 'undefined') localStorage.removeItem(OWNER_KEY)
  try {
    const { userId, state } = await loadServerProject()
    reconcile(userId, state)
  } catch (err) {
    // The Space may be waking up: work from this browser's copy, save once it answers.
    console.warn('Saved project unavailable; retrying in the background', err)
    setTimeout(retryLoad, RETRY_MS)
  }
  document.addEventListener('visibilitychange', () => {
    if (document.visibilityState === 'hidden' && timer) saveNow()
  })
}

/** Delete the project everywhere (server rows, files, saved state, this browser). */
export async function resetProject() {
  const wasEnabled = enabled
  enabled = false
  clearTimeout(timer)
  timer = null
  await saving
  try {
    await deleteProject()
  } catch (err) {
    enabled = wasEnabled
    throw err
  }
  clearLocal()
}

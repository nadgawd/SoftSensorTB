import axios from 'axios'
import { authHeaders } from '../lib/supabase'

// Empty by default so requests stay same-origin and flow through the Vite dev
// proxy (see vite.config.js). Set VITE_API_BASE_URL when the API is deployed
// somewhere other than the origin serving the frontend.
const API_BASE_URL = (import.meta.env.VITE_API_BASE_URL || '').replace(/\/$/, '')

const api = axios.create({
  baseURL: API_BASE_URL,
  timeout: 180_000,
})

api.interceptors.request.use(async (config) => {
  Object.assign(config.headers, await authHeaders())
  return config
})

/** The server's error message from a failed response, else a generic one. */
export function errorDetail(err, fallback = 'Request failed.') {
  const detail = err?.response?.data?.detail
  if (typeof detail === 'string') return detail
  return err?.message || fallback
}

export async function postChat({ message, history, datasetVersionId, stepHint, uiContext, think, signal }, onEvent) {
  const response = await fetch(`${API_BASE_URL}/chat`, {
    method: 'POST',
    signal,
    headers: {
      'Content-Type': 'application/json',
      'Accept': 'text/event-stream',
      ...(await authHeaders()),
    },
    body: JSON.stringify({
      message,
      history: history || [],
      dataset_version_id: datasetVersionId || null,
      step_hint: stepHint || null,
      ui_context: uiContext || null,
      think: Boolean(think),
    })
  })
  
  if (!response.ok) {
    const errText = await response.text()
    let detail = errText
    try {
      detail = JSON.parse(errText)?.detail ?? errText
    } catch {
      // not JSON: keep the raw text
    }
    throw new Error(typeof detail === 'string' ? detail : errText)
  }

  const reader = response.body.getReader()
  const decoder = new TextDecoder('utf-8')
  let buffer = ''

  while (true) {
    const { done, value } = await reader.read()
    if (done) break
    
    buffer += decoder.decode(value, { stream: true })
    const parts = buffer.split('\n\n')
    
    // The last part might be incomplete, keep it in the buffer
    buffer = parts.pop() || ''
    
    for (const part of parts) {
      if (part.startsWith('data: ')) {
        const dataStr = part.slice(6)
        if (dataStr.trim()) {
          try {
            const parsed = JSON.parse(dataStr)
            onEvent?.(parsed.event, parsed.data)
          } catch (e) {
            console.error('Failed to parse SSE data', e, dataStr)
          }
        }
      }
    }
  }
}

/** `keepVersionId`: the open dataset, which must survive the per-user dataset limit (uploading a file to combine with it). */
export async function uploadDataset(file, { keepVersionId, onProgress } = {}) {
  const form = new FormData()
  form.append('file', file)
  const { data } = await api.post('/datasets/upload', form, {
    headers: { 'Content-Type': 'multipart/form-data' },
    // Large plant exports can take longer than the default request timeout.
    timeout: 0,
    params: keepVersionId ? { keep_version_id: keepVersionId } : undefined,
    onUploadProgress: onProgress
      ? (e) => onProgress(e.total ? e.loaded / e.total : null, e.loaded)
      : undefined,
  })
  return data
}

export async function fetchDatasetHistory(datasetId) {
  const { data } = await api.get(`/datasets/${datasetId}/history`)
  return data
}

/** A page of rows (`offset`, `limit`) or a random `sample`, plus `row_count` and `row_numbers`. */
export async function fetchVersionPreview(datasetVersionId, { limit = 25, offset = 0, sample = false, seed = 0 } = {}) {
  const { data } = await api.get(`/datasets/versions/${datasetVersionId}/preview`, {
    params: { limit, offset, sample, seed },
  })
  return data
}

/** Quality profile of a whole version: per-column checks, time sampling and gaps, target coverage. */
export async function fetchVersionProfile(datasetVersionId, { target, timeColumn } = {}) {
  const { data } = await api.get(`/datasets/versions/${datasetVersionId}/profile`, {
    params: { target: target || undefined, time_column: timeColumn || undefined },
    timeout: 0,
  })
  return data
}

/** What a version changed relative to its parent: rows before / after and added, removed, changed columns. */
export async function fetchVersionDiff(datasetVersionId) {
  const { data } = await api.get(`/datasets/versions/${datasetVersionId}/diff`)
  return data
}

/** Models trained on this version or the versions it came from, newest first. */
export async function fetchVersionModels(datasetVersionId) {
  const { data } = await api.get(`/datasets/versions/${datasetVersionId}/models`)
  return data.models
}

/** One model's full report (as training returns it), to make it the active model. */
export async function fetchModelReport(datasetVersionId, modelId) {
  const { data } = await api.get(`/datasets/versions/${datasetVersionId}/models/${modelId}`)
  return data
}

/**
 * Download an export (the request needs the auth header, so it is fetched as a blob and saved).
 * `onProgress(fraction|null, loadedBytes)`; the file name comes from the server.
 */
export async function downloadExport(path, { params, onProgress, fallbackName = 'export' } = {}) {
  const res = await api.get(path, {
    params,
    responseType: 'blob',
    timeout: 0,
    onDownloadProgress: onProgress
      ? (e) => onProgress(e.total ? e.loaded / e.total : null, e.loaded)
      : undefined,
  })
  const match = /filename="([^"]+)"/.exec(res.headers['content-disposition'] || '')
  const url = URL.createObjectURL(res.data)
  const a = document.createElement('a')
  a.href = url
  a.download = match ? match[1] : fallbackName
  a.click()
  setTimeout(() => URL.revokeObjectURL(url), 0)
}

export async function rollbackVersion(datasetVersionId) {
  const { data } = await api.post(`/datasets/versions/${datasetVersionId}/rollback`)
  return data
}

export async function fetchLlmStatus() {
  const { data } = await api.get('/api/llm/status', { timeout: 8_000 })
  return data
}

export async function fetchTools() {
  const { data } = await api.get('/api/tools')
  return data
}

/** `{ steps, tools }`: every tool's form, generated from the schemas the assistant uses. */
export async function fetchManualTools() {
  const { data } = await api.get('/api/tools/manual', { timeout: 20_000 })
  return data
}

/** Run one tool by hand; resolves to the same UI payload as a chat turn's final state. */
export async function runManualTool(toolName, datasetVersionId, args) {
  const { data } = await api.post(`/api/tools/${encodeURIComponent(toolName)}/run`, {
    dataset_version_id: datasetVersionId,
    arguments: args,
  })
  return data
}

/** `{ columns: [{ name, dtype, numeric }], row_count, models }` of a dataset version. */
export async function fetchVersionSchema(datasetVersionId) {
  const { data } = await api.get(`/datasets/versions/${datasetVersionId}/schema`, { timeout: 30_000 })
  return data
}

/** `{ user_id, state }`; state is null until the first save. */
export async function fetchProjectState({ timeout = 20_000 } = {}) {
  const { data } = await api.get('/project/state', { timeout })
  return data
}

export async function saveProjectState(state) {
  const { data } = await api.put('/project/state', { state }, { timeout: 30_000 })
  return data
}

/** How a model would run on live rows: inputs, each step's live mode, checks, warnings, blockers. */
export async function fetchDeploymentPlan(datasetVersionId, modelId) {
  const { data } = await api.get(`/datasets/versions/${datasetVersionId}/deployment-plan`, {
    params: { model_id: modelId || undefined },
    timeout: 0,
  })
  return data
}

/** `{ deployment, api_key }` — the key is only ever returned here and by rotateDeploymentKey. */
export async function createDeployment({ datasetVersionId, modelId, name, biasGain }) {
  const { data } = await api.post('/deployments', {
    dataset_version_id: datasetVersionId,
    model_id: modelId,
    name: name || '',
    bias_gain: biasGain,
  }, { timeout: 0 })
  return data
}

export async function fetchDeployments() {
  const { data } = await api.get('/deployments')
  return data
}

export async function fetchDeployment(deploymentId) {
  const { data } = await api.get(`/deployments/${deploymentId}`)
  return data
}

export async function updateDeployment(deploymentId, changes) {
  const { data } = await api.patch(`/deployments/${deploymentId}`, changes)
  return data
}

export async function deleteDeployment(deploymentId) {
  const { data } = await api.delete(`/deployments/${deploymentId}`)
  return data
}

export async function rotateDeploymentKey(deploymentId) {
  const { data } = await api.post(`/deployments/${deploymentId}/rotate-key`)
  return data
}

/** The latest records and lab results, and the lab accuracy. */
export async function fetchDeploymentMonitor(deploymentId, limit = 500) {
  const { data } = await api.get(`/deployments/${deploymentId}/monitor`, { params: { limit } })
  return data
}

/** Rows of the deployment's own data in the live format, with the lab value of each row (or null). */
export async function fetchSimulationRows(deploymentId, start, count) {
  const { data } = await api.get(`/deployments/${deploymentId}/simulation`, { params: { start, count }, timeout: 0 })
  return data
}

/** Send live rows as the owner (the in-app simulator); a plant connector sends its API key instead. */
export async function sendLiveRows(deploymentId, rows) {
  const { data } = await api.post(`/live/${deploymentId}/predict`, { rows })
  return data.predictions
}

export async function sendLabResults(deploymentId, samples) {
  const { data } = await api.post(`/live/${deploymentId}/lab`, { samples })
  return data
}

/** The URL a plant connector posts to (the API's origin, which differs from the page's in production). */
export function liveApiBase() {
  return API_BASE_URL || window.location.origin
}

/** Delete the caller's datasets, models, files and saved state. */
export async function deleteProject() {
  const { data } = await api.delete('/project', { timeout: 60_000 })
  return data
}

export default api

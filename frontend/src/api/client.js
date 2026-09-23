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

export async function uploadDataset(file) {
  const form = new FormData()
  form.append('file', file)
  const { data } = await api.post('/datasets/upload', form, {
    headers: { 'Content-Type': 'multipart/form-data' },
  })
  return data
}

export async function fetchDatasetHistory(datasetId) {
  const { data } = await api.get(`/datasets/${datasetId}/history`)
  return data
}

export async function fetchVersionPreview(datasetVersionId, limit = 25) {
  const { data } = await api.get(`/datasets/versions/${datasetVersionId}/preview?limit=${limit}`)
  return data
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

/** `{ user_id, state }`; state is null until the first save. */
export async function fetchProjectState({ timeout = 20_000 } = {}) {
  const { data } = await api.get('/project/state', { timeout })
  return data
}

export async function saveProjectState(state) {
  const { data } = await api.put('/project/state', { state }, { timeout: 30_000 })
  return data
}

/** Delete the caller's datasets, models, files and saved state. */
export async function deleteProject() {
  const { data } = await api.delete('/project', { timeout: 60_000 })
  return data
}

export default api

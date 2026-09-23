import { useCallback, useMemo, useState } from 'react'
import { useDropzone } from 'react-dropzone'
import { uploadDataset } from '../../api/client'
import ActionChips from '../ActionChips'

/**
 * Step 1 — Data Ingestion
 */
export default function DataIngestionStep({
  fileName, tablePreview, targetVariable,
  onUploadSuccess, onTargetSelect,
  actions, tips, onAction,
}) {
  const [uploading, setUploading] = useState(false)
  const [uploadError, setUploadError] = useState(null)

  const onDrop = useCallback(async (acceptedFiles) => {
    const file = acceptedFiles?.[0]
    if (!file) return
    setUploadError(null)
    setUploading(true)
    try {
      const result = await uploadDataset(file)
      onUploadSuccess?.(result)
    } catch (err) {
      const detail = err?.response?.data?.detail || err.message || 'Upload failed'
      setUploadError(typeof detail === 'string' ? detail : JSON.stringify(detail))
    } finally {
      setUploading(false)
    }
  }, [onUploadSuccess])

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop, multiple: false,
    accept: {
      'text/csv': ['.csv'],
      'text/plain': ['.txt'],
      'application/octet-stream': ['.parquet', '.pq', '.csv'],
    },
    disabled: uploading,
  })

  const columns = useMemo(() => {
    if (!tablePreview?.length) return []
    return Object.keys(tablePreview[0])
  }, [tablePreview])

  return (
    <div className="animate-rise" style={{ display: 'flex', flexDirection: 'column', gap: '16px' }}>
      {/* Drop zone */}
      <div
        {...getRootProps()}
        className={`dropzone${isDragActive ? ' active' : ''}`}
        style={{ opacity: uploading ? 0.6 : 1 }}
      >
        <input {...getInputProps()} />
        <div style={{ marginBottom: '8px', fontSize: '26px' }}>
          {uploading ? '⏳' : isDragActive ? '📂' : '📁'}
        </div>
        <p style={{ fontWeight: 600, fontSize: '0.9375rem', color: 'var(--color-foam)', margin: '0 0 4px' }}>
          {uploading ? 'Ingesting dataset…' : isDragActive ? 'Drop to upload' : 'Drop CSV / TXT / Parquet here'}
        </p>
        <p style={{ fontSize: '0.8125rem', color: 'var(--color-mist)', margin: 0 }}>
          Creates a root dataset version in <code style={{ fontFamily: 'var(--font-mono)', fontSize: '11px', color: 'var(--color-signal)' }}>/data_storage</code>
        </p>
        {fileName && (
          <p style={{ marginTop: '8px', fontFamily: 'var(--font-mono)', fontSize: '12px', color: 'var(--color-signal)' }}>
            ✓ {fileName}
          </p>
        )}
      </div>

      {uploadError && (
        <div className="badge badge-alert" style={{ display: 'block', padding: '10px 14px', borderRadius: '8px', fontSize: '12px', lineHeight: 1.5 }}>
          {uploadError}
        </div>
      )}

      {/* Dynamic tips & actions */}
      <ActionChips actions={actions} tips={tips} onAction={onAction} />

      {/* Column / tag table */}
      {columns.length > 0 && (
        <div className="panel" style={{ borderRadius: '14px', overflow: 'hidden' }}>
          <div style={{ padding: '12px 16px', borderBottom: '1px solid var(--color-line)', display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
            <div>
              <p style={{ fontFamily: 'var(--font-mono)', fontSize: '10px', textTransform: 'uppercase', letterSpacing: '0.1em', color: 'var(--color-mist)', margin: 0 }}>
                Tag Columns
              </p>
              <p style={{ fontSize: '0.875rem', color: 'var(--color-foam)', margin: '2px 0 0', fontWeight: 500 }}>
                {columns.length} tags · {tablePreview.length} rows
              </p>
            </div>
            {targetVariable && (
              <span className="badge badge-signal">{targetVariable} (target)</span>
            )}
          </div>
          <div style={{ overflowY: 'auto', maxHeight: '300px' }}>
            <table className="data-table">
              <thead>
                <tr>
                  <th>Tag name</th>
                  <th>Sample value</th>
                  <th>Target?</th>
                </tr>
              </thead>
              <tbody>
                {columns.map((col) => {
                  const sample = tablePreview[0]?.[col]
                  const isTarget = targetVariable === col
                  return (
                    <tr key={col} style={isTarget ? { background: 'rgba(46,196,182,0.06)' } : {}}>
                      <td style={{ color: 'var(--color-foam)', fontWeight: isTarget ? 600 : 400 }}>{col}</td>
                      <td>{sample === null || sample === undefined ? '—' : String(sample)}</td>
                      <td>
                        {isTarget ? (
                          <span className="badge badge-signal" style={{ fontSize: '10px' }}>Target ✓</span>
                        ) : (
                          <button
                            type="button"
                            onClick={() => onTargetSelect?.(col)}
                            style={{
                              background: 'none', border: '1px solid var(--color-line-2)',
                              borderRadius: '6px', padding: '2px 8px', fontSize: '10px',
                              color: 'var(--color-mist)', cursor: 'pointer', fontFamily: 'var(--font-mono)',
                            }}
                          >
                            Set target
                          </button>
                        )}
                      </td>
                    </tr>
                  )
                })}
              </tbody>
            </table>
          </div>
        </div>
      )}

      {!columns.length && !tips?.length && (
        <div style={{ textAlign: 'center', padding: '30px 20px', color: 'var(--color-mist)' }}>
          <p style={{ fontSize: '1.5rem', marginBottom: '8px' }}>📊</p>
          <p style={{ fontSize: '0.875rem' }}>Upload a CSV, TXT, or Parquet file to see tag columns here.</p>
        </div>
      )}
    </div>
  )
}

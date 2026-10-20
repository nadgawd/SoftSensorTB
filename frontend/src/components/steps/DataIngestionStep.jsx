import { useCallback, useState } from 'react'
import { useDropzone } from 'react-dropzone'
import { FileSpreadsheet, Upload } from 'lucide-react'
import { errorDetail, uploadDataset } from '../../api/client'
import { formatBytes, formatCount } from '../../utils/format'
import VariablesTable from '../VariablesTable'

const ACCEPT = {
  'text/csv': ['.csv'],
  'text/plain': ['.txt'],
  'application/octet-stream': ['.parquet', '.pq', '.csv'],
}

function UploadProgress({ upload }) {
  const { name, size, fraction, loaded } = upload
  const sent = fraction != null && fraction >= 1
  return (
    <div className="upload-progress" role="status">
      <div className="upload-progress-text">
        <span>{name}</span>
        <span>
          {sent ? 'Reading the file on the server…'
            : fraction != null ? `${Math.round(fraction * 100)}% · ${formatBytes(loaded)} of ${formatBytes(size)}`
            : `${formatBytes(loaded)} sent`}
        </span>
      </div>
      <div className="upload-progress-bar">
        <div className={sent ? 'indeterminate' : ''} style={{ width: `${Math.round((fraction ?? 0) * 100)}%` }} />
      </div>
    </div>
  )
}

/**
 * Step 1 — Data ingestion: upload a file, then give each column its role.
 */
export default function DataIngestionStep({
  fileName, tablePreview, rowCount, profile, timeColumn, targetVariable, selectedFeatures,
  onUploadSuccess, onTargetSelect, onFeaturesChange, onTimeColumnChange,
}) {
  const [upload, setUpload] = useState(null)
  const [uploadError, setUploadError] = useState(null)

  const onDrop = useCallback(async (acceptedFiles) => {
    const file = acceptedFiles?.[0]
    if (!file) return
    setUploadError(null)
    setUpload({ name: file.name, size: file.size, fraction: 0, loaded: 0 })
    try {
      const result = await uploadDataset(file, {
        onProgress: (fraction, loaded) => setUpload((u) => u && { ...u, fraction, loaded }),
      })
      onUploadSuccess?.(result)
    } catch (err) {
      setUploadError(errorDetail(err, 'Upload failed.'))
    } finally {
      setUpload(null)
    }
  }, [onUploadSuccess])

  const { getRootProps, getInputProps, isDragActive, open } = useDropzone({
    onDrop, multiple: false, accept: ACCEPT, disabled: Boolean(upload), noClick: Boolean(fileName),
  })

  const previewColumns = tablePreview?.length ? Object.keys(tablePreview[0]) : []

  return (
    <div className="animate-rise ingest">
      {fileName ? (
        <div {...getRootProps()} className={`ingest-file panel${isDragActive ? ' active' : ''}`}>
          <input {...getInputProps()} />
          {upload ? <UploadProgress upload={upload} /> : (
            <>
              <FileSpreadsheet size={18} aria-hidden="true" className="ingest-file-icon" />
              <div className="ingest-file-text">
                <span className="ingest-file-name">{fileName}</span>
                <span className="ingest-file-meta">
                  {rowCount != null ? `${formatCount(rowCount)} rows × ${previewColumns.length} tags` : ''}
                  {isDragActive ? ' · drop to replace' : ''}
                </span>
              </div>
              <button type="button" className="btn-ghost ingest-replace" onClick={open}>
                <Upload size={13} aria-hidden="true" /> Replace file
              </button>
            </>
          )}
        </div>
      ) : (
        <div {...getRootProps()} className={`dropzone${isDragActive ? ' active' : ''}`}>
          <input {...getInputProps()} />
          {upload ? <UploadProgress upload={upload} /> : (
            <>
              <Upload size={26} aria-hidden="true" className="dropzone-icon" />
              <p className="dropzone-title">{isDragActive ? 'Drop to upload' : 'Drop a CSV, TXT or Parquet file here'}</p>
              <p className="dropzone-sub">or click to choose one. Large historian exports are fine.</p>
            </>
          )}
        </div>
      )}

      {uploadError && <p className="manual-error">{uploadError}</p>}

      {fileName && previewColumns.length > 0 && (
        <VariablesTable
          profile={profile}
          previewColumns={previewColumns}
          timeColumn={timeColumn}
          targetVariable={targetVariable}
          selectedFeatures={selectedFeatures}
          onTimeColumnChange={onTimeColumnChange}
          onTargetSelect={onTargetSelect}
          onFeaturesChange={onFeaturesChange}
        />
      )}
    </div>
  )
}

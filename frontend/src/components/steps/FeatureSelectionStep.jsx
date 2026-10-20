import VariablesTable from '../VariablesTable'

/**
 * Step 4 — Feature selection: the same column roles as Step 1, next to the selection tools.
 */
export default function FeatureSelectionStep({
  tablePreview, profile, timeColumn, targetVariable, selectedFeatures,
  onTargetSelect, onFeaturesChange, onTimeColumnChange,
}) {
  if (!tablePreview?.length) {
    return <p className="manual-hint">Upload a dataset in Step 1 to choose the model inputs.</p>
  }
  return (
    <div className="animate-rise">
      <VariablesTable
        profile={profile}
        previewColumns={Object.keys(tablePreview[0])}
        timeColumn={timeColumn}
        targetVariable={targetVariable}
        selectedFeatures={selectedFeatures}
        onTimeColumnChange={onTimeColumnChange}
        onTargetSelect={onTargetSelect}
        onFeaturesChange={onFeaturesChange}
      />
    </div>
  )
}

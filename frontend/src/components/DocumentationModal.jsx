import { useEffect, useMemo, useState } from 'react';
import { fetchTools as fetchToolsApi } from '../api/client';
import { STEPS, uiPlaceOf } from '../constants/steps';

const STEP_BLURBS = {
  ingestion: 'Look at the data (head, info, describe, profile), column statistics, group by and pivot, sort, combine datasets, convert types and rename columns.',
  quality: 'Missing values (fill, KNN, MICE, interpolation), duplicates, outliers, anomalies, dropping or filtering rows, masking and recoding values, text and regex, and statistical tests.',
  preprocessing: 'Scaling and transforms, encoding, lags and rolling windows, resampling to a regular time grid, formula columns, feature engineering, reshaping and class balancing.',
  'feature-selection': 'Choose the inputs (SelectKBest, RFE, model importance, permutation, SHAP), drop collinear or unwanted columns, and reduce dimensions (PCA, SVD, t-SNE…).',
  modeling: 'Train & test tab: split the rows into training, validation and test sets, train a model on the training rows, then test it on the held-out test rows. Diagnostics tab: parity, residual, importance and coefficient plots, ROC and precision-recall curves, learning curves. Compare models tab: every model side by side with its train, validation, CV and test scores; make one active or plot its parity.',
  export: 'Download the prepared dataset (CSV or Parquet), a trained model, the pipeline recipe (every step with its parameters) and an HTML report. Also SQL queries on the active version and the list of trained models.',
};

const EXTRA_NODES = [
  { id: 'graph', label: 'Graph tab', blurb: 'At the bottom of every step, next to the Data preview: plot any columns of the active version.' },
  { id: 'assistant', label: 'Assistant only', blurb: 'Tools the Assistant uses that have no form in the Tools panel.' },
];

function summary(text = '') {
  const first = text.split(/(?<=\.)\s/)[0];
  return first.length > 160 ? `${first.slice(0, 157)}…` : first;
}

/** Inline `$ref` / single-entry `allOf` pointers into the schema's `$defs`. */
function resolve(schema, defs) {
  if (!schema) return schema;
  if (schema.$ref) return { ...defs?.[schema.$ref.split('/').pop()], ...schema, $ref: undefined };
  if (schema.allOf?.length === 1) return { ...resolve(schema.allOf[0], defs), ...schema, allOf: undefined };
  return schema;
}

function argType(arg) {
  if (arg.enum) return 'choice';
  if (arg.type === 'array') return arg.items?.enum ? 'list of choices' : `list of ${arg.items?.type || 'values'}`;
  if (arg.type) return arg.type;
  if (arg.anyOf) return [...new Set(arg.anyOf.filter((t) => t.type !== 'null').map(argType))].join(' | ');
  return '';
}

function argChoices(arg) {
  const values = arg.enum
    || arg.items?.enum
    || arg.anyOf?.find((t) => t.enum)?.enum
    || arg.anyOf?.find((t) => t.items?.enum)?.items.enum;
  return values?.length ? values : null;
}

function resolveArg(arg, defs) {
  const top = resolve(arg, defs);
  return {
    ...top,
    items: resolve(top.items, defs),
    anyOf: top.anyOf?.map((t) => {
      const r = resolve(t, defs);
      return { ...r, items: resolve(r.items, defs) };
    }),
  };
}

function matchesQuery(tool, query) {
  if (!query) return true;
  const fn = tool.function || {};
  return [fn.name, tool.label, tool.summary, fn.description, ...Object.keys(fn.parameters?.properties || {})]
    .join(' ')
    .toLowerCase()
    .includes(query);
}

/** One node per pipeline step (in the order of the Tools dropdowns), then the Graph tab and Assistant-only tools. */
function buildTree(tools, manualSpecs) {
  if (!manualSpecs) {
    const items = tools.map((t) => ({ ...t, label: t.function?.name }));
    return [{ id: 'all', number: '•', label: 'All tools', blurb: 'The step of each tool could not be loaded from the server.', items }];
  }
  const specs = new Map((manualSpecs || []).map((s) => [s.name, s]));
  const byName = new Map(tools.map((t) => [t.function?.name, t]));
  const withLabel = (spec) => ({ ...byName.get(spec.name), label: spec.label, summary: spec.summary });
  const known = (manualSpecs || []).filter((s) => byName.has(s.name));

  const nodes = STEPS.map((step, i) => ({
    id: step.id,
    number: i + 1,
    label: step.label,
    blurb: STEP_BLURBS[step.id] || step.description,
    items: known.filter((s) => s.steps.some((id) => uiPlaceOf(id).step === step.id)).map(withLabel),
  }));
  nodes.push({ ...EXTRA_NODES[0], number: '◫', items: known.filter((s) => !s.steps.length && s.panel).map(withLabel) });
  nodes.push({
    ...EXTRA_NODES[1],
    number: '✦',
    items: tools.filter((t) => !specs.has(t.function?.name)).map((t) => ({ ...t, label: t.function?.name })),
  });
  return nodes.filter((n) => n.items.length);
}

export default function DocumentationModal({ onClose, manualSpecs }) {
  const [tools, setTools] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [query, setQuery] = useState('');
  const [openSteps, setOpenSteps] = useState(() => new Set());
  const [openTools, setOpenTools] = useState(() => new Set());

  useEffect(() => {
    fetchToolsApi()
      .then(setTools)
      .catch((err) => setError(err.message))
      .finally(() => setLoading(false));
  }, []);

  useEffect(() => {
    const onKey = (e) => e.key === 'Escape' && onClose();
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, [onClose]);

  const tree = useMemo(() => buildTree(tools, manualSpecs), [tools, manualSpecs]);
  const q = query.trim().toLowerCase();
  const visible = useMemo(
    () => tree.map((n) => ({ ...n, items: n.items.filter((t) => matchesQuery(t, q)) })).filter((n) => n.items.length),
    [tree, q],
  );
  const matchCount = visible.reduce((n, s) => n + s.items.length, 0);
  const allOpen = visible.length > 0 && visible.every((n) => openSteps.has(n.id));

  const toggle = (setter, id) => setter((prev) => {
    const next = new Set(prev);
    if (next.has(id)) next.delete(id);
    else next.add(id);
    return next;
  });

  return (
    <div className="doc-backdrop" onClick={onClose}>
      <div className="doc-modal" role="dialog" aria-modal="true" aria-labelledby="doc-title" onClick={(e) => e.stopPropagation()}>
        <div className="doc-header">
          <div>
            <h2 id="doc-title">Toolbox documentation</h2>
            <p className="manual-hint">
              {tools.length ? `${tools.length} tools, shared by the Tools panel and the Assistant, listed by pipeline step.` : ' '}
            </p>
          </div>
          <button type="button" className="doc-close" onClick={onClose} aria-label="Close documentation">✕</button>
        </div>

        {!loading && !error && (
          <div className="doc-toolbar">
            <input
              type="search"
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              placeholder="Search tools by name, description or argument…"
              aria-label="Search tools"
            />
            {!q && (
              <button
                type="button"
                className="btn-ghost doc-expand-all"
                onClick={() => setOpenSteps(allOpen ? new Set() : new Set(visible.map((n) => n.id)))}
              >
                {allOpen ? 'Collapse all' : 'Expand all'}
              </button>
            )}
          </div>
        )}

        <div className="doc-body">
          {loading ? (
            <p className="doc-message">Loading documentation…</p>
          ) : error ? (
            <p className="doc-message manual-error">Error: {error}</p>
          ) : (
            <>
              {q && (
                <p className="manual-hint doc-match-count">
                  {matchCount ? `${matchCount} tool${matchCount === 1 ? '' : 's'} match “${query.trim()}”.` : `No tools match “${query.trim()}”.`}
                </p>
              )}
              <ol className="doc-tree">
                {visible.map((node) => {
                  const open = Boolean(q) || openSteps.has(node.id);
                  return (
                    <li key={node.id} className={`doc-step${open ? ' open' : ''}`}>
                      <button
                        type="button"
                        className="doc-step-head"
                        aria-expanded={open}
                        onClick={() => toggle(setOpenSteps, node.id)}
                      >
                        <span className="doc-step-num" aria-hidden="true">{node.number}</span>
                        <span className="doc-step-text">
                          <span className="doc-step-label">
                            {node.label}
                            <span className="doc-step-count">{node.items.length} tool{node.items.length === 1 ? '' : 's'}</span>
                          </span>
                          <span className="doc-step-blurb">{node.blurb}</span>
                        </span>
                        <span className="doc-chevron" aria-hidden="true">{open ? '▾' : '▸'}</span>
                      </button>
                      {open && (
                        <ul className="doc-tools">
                          {node.items.map((tool) => (
                            <ToolNode
                              key={tool.function?.name}
                              tool={tool}
                              open={openTools.has(tool.function?.name)}
                              onToggle={() => toggle(setOpenTools, tool.function?.name)}
                            />
                          ))}
                        </ul>
                      )}
                    </li>
                  );
                })}
              </ol>
            </>
          )}
        </div>
      </div>
    </div>
  );
}

function ToolNode({ tool, open, onToggle }) {
  const fn = tool.function || {};
  const defs = fn.parameters?.$defs;
  const properties = Object.entries(fn.parameters?.properties || {})
    .filter(([name]) => name !== 'dataset_version_id')
    .map(([name, arg]) => [name, resolveArg(arg, defs)]);
  const required = new Set(fn.parameters?.required || []);

  return (
    <li className={`doc-tool${open ? ' open' : ''}`}>
      <button type="button" className="doc-tool-head" aria-expanded={open} onClick={onToggle}>
        <span className="doc-chevron" aria-hidden="true">{open ? '▾' : '▸'}</span>
        <span className="doc-tool-text">
          <span className="doc-tool-label">
            {tool.label}
            {tool.label !== fn.name && <code>{fn.name}</code>}
          </span>
          {!open && <span className="doc-tool-summary">{tool.summary || summary(fn.description)}</span>}
        </span>
      </button>
      {open && (
        <div className="doc-tool-body">
          {tool.summary && <p className="doc-tool-desc"><strong>{tool.summary}</strong></p>}
          <p className="doc-tool-desc">{fn.description}</p>
          {properties.length > 0 && (
            <table className="doc-args">
              <thead>
                <tr><th>Argument</th><th>Type</th><th>Description</th></tr>
              </thead>
              <tbody>
                {properties.map(([name, arg]) => {
                  const choices = argChoices(arg);
                  return (
                    <tr key={name}>
                      <td>
                        <code>{name}</code>
                        {required.has(name) && <span className="doc-required" title="Required">*</span>}
                      </td>
                      <td>{argType(arg)}</td>
                      <td>
                        {arg.description}
                        {arg.default !== undefined && arg.default !== null && (
                          <span className="doc-default">Default: <code>{JSON.stringify(arg.default)}</code></span>
                        )}
                        {choices && (
                          <span className="doc-choices">
                            {choices.map((c) => <code key={String(c)}>{String(c)}</code>)}
                          </span>
                        )}
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          )}
        </div>
      )}
    </li>
  );
}

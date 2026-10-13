import React, { useState, useEffect, useMemo } from 'react';
import { fetchTools as fetchToolsApi } from '../api/client';

const CATEGORIES = [
  {
    id: 'exploration',
    label: 'Data Exploration',
    blurb: 'Inspect the dataset: summary statistics and ad-hoc SQL queries.',
    tools: ['get_column_statistics', 'run_duckdb_query'],
  },
  {
    id: 'cleaning',
    label: 'Data Cleaning',
    blurb: 'Fix missing values, outliers and anomalous rows.',
    tools: ['remove_missing_data', 'remove_outliers', 'detect_anomalies'],
  },
  {
    id: 'scaling',
    label: 'Data Scaling & Transformation',
    blurb: 'Rescale columns and apply non-linear transforms or binning.',
    tools: ['normalize_data', 'transform_features'],
  },
  {
    id: 'manipulation',
    label: 'Data Manipulation',
    blurb: 'Add, drop and rename columns.',
    tools: ['drop_columns', 'rename_columns', 'execute_formula'],
  },
  {
    id: 'time-series',
    label: 'Time-Series Features',
    blurb: 'Capture process dynamics with lags and rolling windows.',
    tools: ['create_lag_features', 'rolling_aggregate'],
  },
  {
    id: 'engineering',
    label: 'Feature Engineering & Encoding',
    blurb: 'Derive new features and encode categorical columns.',
    tools: ['feature_engineer', 'encode_categorical'],
  },
  {
    id: 'selection',
    label: 'Feature Selection & Reduction',
    blurb: 'Choose the most useful inputs and remove redundancy.',
    tools: ['select_features_rfe', 'drop_collinear_features', 'reduce_dimensions'],
  },
  {
    id: 'sampling',
    label: 'Sampling & Balancing',
    blurb: 'Resample rows to balance the dataset.',
    tools: ['balance_data'],
  },
  {
    id: 'modeling',
    label: 'Modeling',
    blurb: 'Train soft-sensor regression models and list trained ones.',
    tools: ['train_soft_sensor', 'list_trained_models'],
  },
  {
    id: 'plotting',
    label: 'Plotting & Visualization',
    blurb: 'Build charts of the data: scatter, line, histogram, box, heatmap and more.',
    tools: ['generate_custom_plot'],
  },
  {
    id: 'diagnostics',
    label: 'Model Diagnostics',
    blurb: 'Evaluate a trained model with parity, residual, importance and coefficient plots.',
    tools: ['generate_parity_plot', 'generate_residuals_plot', 'generate_importance_plot', 'generate_coefficients_plot'],
  },
];

const OTHER = { id: 'other', label: 'Other', blurb: 'Additional tools.' };

function groupTools(tools) {
  const byName = new Map(tools.map((t) => [t.function?.name, t]));
  const used = new Set();
  const groups = CATEGORIES.map((cat) => {
    const items = cat.tools.filter((name) => byName.has(name)).map((name) => byName.get(name));
    items.forEach((t) => used.add(t.function.name));
    return { ...cat, items };
  });
  const rest = tools.filter((t) => !used.has(t.function?.name));
  if (rest.length) groups.push({ ...OTHER, items: rest });
  return groups.filter((g) => g.items.length);
}

function matchesQuery(tool, query) {
  if (!query) return true;
  const fn = tool.function || {};
  const haystack = [fn.name, fn.description, ...Object.keys(fn.parameters?.properties || {})]
    .join(' ')
    .toLowerCase();
  return haystack.includes(query);
}

export default function DocumentationModal({ onClose }) {
  const [tools, setTools] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [query, setQuery] = useState('');
  const [activeCategory, setActiveCategory] = useState('all');

  useEffect(() => {
    async function fetchTools() {
      try {
        setTools(await fetchToolsApi());
      } catch (err) {
        setError(err.message);
      } finally {
        setLoading(false);
      }
    }
    fetchTools();
  }, []);

  const groups = useMemo(() => groupTools(tools), [tools]);

  const visibleGroups = useMemo(() => {
    const q = query.trim().toLowerCase();
    return groups
      .filter((g) => activeCategory === 'all' || g.id === activeCategory)
      .map((g) => ({ ...g, items: g.items.filter((t) => matchesQuery(t, q)) }))
      .filter((g) => g.items.length);
  }, [groups, query, activeCategory]);

  const visibleCount = visibleGroups.reduce((n, g) => n + g.items.length, 0);

  return (
    <div
      style={{
        position: 'fixed',
        top: 0,
        left: 0,
        width: '100vw',
        height: '100vh',
        background: 'var(--modal-backdrop)',
        backdropFilter: 'blur(4px)',
        zIndex: 9999,
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
      }}
      onClick={onClose}
    >
      <div
        style={{
          background: 'var(--color-depth)',
          border: '1px solid var(--color-line-2)',
          borderRadius: '12px',
          width: '80%',
          maxWidth: '860px',
          maxHeight: '85vh',
          display: 'flex',
          flexDirection: 'column',
          boxShadow: 'var(--modal-shadow)',
        }}
        onClick={(e) => e.stopPropagation()} // Prevent closing when clicking inside
      >
        {/* Header */}
        <div
          style={{
            padding: '20px',
            borderBottom: '1px solid var(--color-line)',
            display: 'flex',
            justifyContent: 'space-between',
            alignItems: 'center',
          }}
        >
          <h2 style={{ margin: 0, color: 'var(--color-foam)', fontFamily: 'var(--font-heading)' }}>
            AI Assistant Toolbox Documentation
          </h2>
          <button
            onClick={onClose}
            style={{
              background: 'transparent',
              border: 'none',
              color: 'var(--color-mist)',
              cursor: 'pointer',
              fontSize: '1.2rem',
            }}
          >
            ✕
          </button>
        </div>

        {/* Search + category filter */}
        {!loading && !error && (
          <div style={{ padding: '14px 20px', borderBottom: '1px solid var(--color-line)', display: 'flex', flexDirection: 'column', gap: '10px' }}>
            <input
              type="search"
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              placeholder="Search tools by name, description or argument…"
              aria-label="Search tools"
              style={{
                width: '100%',
                boxSizing: 'border-box',
                padding: '8px 12px',
                borderRadius: '8px',
                border: '1px solid var(--color-line-2)',
                background: 'var(--color-panel)',
                color: 'var(--color-foam)',
                fontSize: '0.875rem',
                outline: 'none',
              }}
            />
            <div role="group" aria-label="Tool categories" style={{ display: 'flex', gap: '6px', overflowX: 'auto', paddingBottom: '2px' }}>
              {[{ id: 'all', label: 'All', items: tools }, ...groups].map((g) => {
                const selected = activeCategory === g.id;
                return (
                  <button
                    key={g.id}
                    type="button"
                    aria-pressed={selected}
                    onClick={() => setActiveCategory(g.id)}
                    style={{
                      padding: '4px 10px',
                      borderRadius: '999px',
                      border: `1px solid ${selected ? 'var(--color-signal)' : 'var(--color-line-2)'}`,
                      background: selected ? 'var(--history-card-active-bg)' : 'transparent',
                      color: selected ? 'var(--color-signal)' : 'var(--color-mist)',
                      fontSize: '12px',
                      cursor: 'pointer',
                      whiteSpace: 'nowrap',
                      flexShrink: 0,
                    }}
                  >
                    {g.label} <span style={{ opacity: 0.7 }}>({g.items.length})</span>
                  </button>
                );
              })}
            </div>
          </div>
        )}

        {/* Content */}
        <div style={{ padding: '20px', overflowY: 'auto', flex: 1 }}>
          {loading ? (
            <div style={{ color: 'var(--color-mist)', textAlign: 'center', padding: '40px' }}>
              Loading documentation...
            </div>
          ) : error ? (
            <div style={{ color: 'var(--color-alert)', textAlign: 'center', padding: '40px' }}>
              Error: {error}
            </div>
          ) : (
            <div style={{ display: 'flex', flexDirection: 'column', gap: '28px' }}>
              <p style={{ color: 'var(--color-mist)', margin: 0 }}>
                The AI Assistant and the Tools panel share {tools.length} tools, grouped by what they do.
                {(query || activeCategory !== 'all') && ` Showing ${visibleCount}.`}
              </p>

              {visibleGroups.length === 0 && (
                <div style={{ color: 'var(--color-mist)', textAlign: 'center', padding: '30px' }}>
                  No tools match “{query}”.
                </div>
              )}

              {visibleGroups.map((group) => (
                <section key={group.id} style={{ display: 'flex', flexDirection: 'column', gap: '12px' }}>
                  <div>
                    <h3 style={{ margin: 0, color: 'var(--color-foam)', fontSize: '1rem', fontFamily: 'var(--font-heading)' }}>
                      {group.label} <span style={{ color: 'var(--color-mist)', fontWeight: 400, fontSize: '0.85rem' }}>· {group.items.length}</span>
                    </h3>
                    <p style={{ margin: '2px 0 0', color: 'var(--color-mist)', fontSize: '0.8125rem' }}>{group.blurb}</p>
                  </div>
                  {group.items.map((tool) => (
                    <ToolCard key={tool.function?.name} tool={tool} />
                  ))}
                </section>
              ))}
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

function ToolCard({ tool }) {
  const fn = tool.function || {};
  const properties = fn.parameters?.properties || {};

  return (
    <div
      style={{
        background: 'var(--color-panel)',
        border: '1px solid var(--color-line)',
        borderRadius: '8px',
        padding: '16px',
      }}
    >
      <div style={{ display: 'flex', alignItems: 'baseline', gap: '10px', marginBottom: '8px' }}>
        <code
          style={{
            color: 'var(--chip-ember-fg)',
            background: 'var(--chip-ember-bg)',
            padding: '2px 6px',
            borderRadius: '4px',
            fontSize: '0.9rem',
            fontWeight: 600,
          }}
        >
          {fn.name}
        </code>
      </div>
      <p style={{ color: 'var(--color-foam-2)', margin: '0 0 12px 0', fontSize: '0.9rem', lineHeight: '1.5' }}>
        {fn.description}
      </p>

      {Object.keys(properties).length > 0 && (
        <div style={{ marginTop: '12px' }}>
          <div style={{ fontSize: '0.75rem', textTransform: 'uppercase', color: 'var(--color-mist)', marginBottom: '4px', letterSpacing: '0.5px' }}>
            Arguments:
          </div>
          <div style={{ display: 'flex', flexDirection: 'column', gap: '4px' }}>
            {Object.entries(properties).map(([argName, argData]) => (
              <div key={argName} style={{ display: 'flex', gap: '8px', fontSize: '0.8rem' }}>
                <span style={{ color: 'var(--color-foam)', minWidth: '140px' }}>{argName}</span>
                <span style={{ color: 'var(--color-mist)' }}>
                  {argData.type}
                  {fn.parameters.required?.includes(argName) ? ' (required)' : ' (optional)'}
                </span>
                {argData.description && (
                  <span style={{ color: 'var(--color-foam-2)', marginLeft: '10px' }}>- {argData.description}</span>
                )}
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}

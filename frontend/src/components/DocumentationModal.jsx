import React, { useState, useEffect } from 'react';
import { fetchTools as fetchToolsApi } from '../api/client';

export default function DocumentationModal({ onClose }) {
  const [tools, setTools] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

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
          maxWidth: '800px',
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
            <div style={{ display: 'flex', flexDirection: 'column', gap: '24px' }}>
              <p style={{ color: 'var(--color-mist)', marginTop: 0 }}>
                The AI Assistant has access to the following {tools.length} specific tools for data manipulation, EDA, modeling, and plotting:
              </p>
              
              {tools.map((tool, idx) => (
                <div
                  key={idx}
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
                      {tool.function?.name}
                    </code>
                  </div>
                  <p style={{ color: 'var(--color-foam-2)', margin: '0 0 12px 0', fontSize: '0.9rem', lineHeight: '1.5' }}>
                    {tool.function?.description}
                  </p>
                  
                  {/* Arguments (if any) */}
                  {tool.function?.parameters?.properties && Object.keys(tool.function.parameters.properties).length > 0 && (
                    <div style={{ marginTop: '12px' }}>
                      <div style={{ fontSize: '0.75rem', textTransform: 'uppercase', color: 'var(--color-mist)', marginBottom: '4px', letterSpacing: '0.5px' }}>
                        Arguments:
                      </div>
                      <div style={{ display: 'flex', flexDirection: 'column', gap: '4px' }}>
                        {Object.entries(tool.function.parameters.properties).map(([argName, argData]) => (
                          <div key={argName} style={{ display: 'flex', gap: '8px', fontSize: '0.8rem' }}>
                            <span style={{ color: 'var(--color-foam)', minWidth: '140px' }}>{argName}</span>
                            <span style={{ color: 'var(--color-mist)' }}>
                              {argData.type}
                              {tool.function.parameters.required?.includes(argName) ? ' (required)' : ' (optional)'}
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
              ))}
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

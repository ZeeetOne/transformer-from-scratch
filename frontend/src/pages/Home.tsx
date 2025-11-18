import { useState } from 'react';
import { Link } from 'react-router-dom';
import ControlPanel from '../components/ControlPanel';
import AttentionVisualizer from '../components/AttentionVisualizer';
import EmbeddingVisualizer from '../components/EmbeddingVisualizer';
import ArchitectureDiagram from '../components/ArchitectureDiagram';
import { InferenceResponse } from '../services/api';

type ViewMode = 'architecture' | 'embeddings' | 'attention' | 'complete';

export default function Home() {
  const [visualizationData, setVisualizationData] = useState<InferenceResponse | null>(null);
  const [viewMode, setViewMode] = useState<ViewMode>('architecture');
  const [loading, setLoading] = useState(false);

  const handleInferenceComplete = (data: InferenceResponse) => {
    setVisualizationData(data);
    setLoading(false);
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-gray-900 via-blue-900 to-gray-900 text-white">
      {/* Header */}
      <header className="bg-black/30 backdrop-blur-sm border-b border-white/10">
        <div className="container mx-auto px-6 py-4">
          <div className="flex items-center justify-between">
            <div>
              <h1 className="text-3xl font-bold bg-clip-text text-transparent bg-gradient-to-r from-blue-400 to-purple-400">
                Transformer Interactive Visualization
              </h1>
              <p className="text-gray-300 mt-2">
                Learn how transformers work through interactive, step-by-step visualizations
              </p>
            </div>
            <Link
              to="/applications"
              className="px-6 py-3 bg-gradient-to-r from-purple-600 to-pink-600 hover:from-purple-700 hover:to-pink-700 rounded-lg transition-all duration-200 font-semibold shadow-lg hover:shadow-xl"
            >
              Applications →
            </Link>
          </div>
        </div>
      </header>

      {/* Main Content */}
      <div className="container mx-auto px-6 py-8">
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          {/* Control Panel - Left Sidebar */}
          <div className="lg:col-span-1">
            <ControlPanel
              onInferenceComplete={handleInferenceComplete}
              onLoadingChange={setLoading}
            />

            {/* View Mode Selector */}
            {visualizationData && (
              <div className="card mt-6">
                <h3 className="text-lg font-semibold mb-4 text-gray-800 dark:text-white">
                  View Mode
                </h3>
                <div className="grid grid-cols-2 gap-2">
                  {[
                    { id: 'architecture', label: 'Architecture', icon: '🏗️' },
                    { id: 'embeddings', label: 'Embeddings', icon: '📊' },
                    { id: 'attention', label: 'Attention', icon: '🎯' },
                    { id: 'complete', label: 'Complete', icon: '🔍' },
                  ].map((mode) => (
                    <button
                      key={mode.id}
                      onClick={() => setViewMode(mode.id as ViewMode)}
                      className={`
                        py-3 px-4 rounded-lg font-medium transition-all duration-200
                        ${
                          viewMode === mode.id
                            ? 'bg-primary-600 text-white shadow-lg'
                            : 'bg-gray-200 dark:bg-gray-700 text-gray-700 dark:text-gray-300 hover:bg-gray-300 dark:hover:bg-gray-600'
                        }
                      `}
                    >
                      <div className="text-2xl mb-1">{mode.icon}</div>
                      <div className="text-sm">{mode.label}</div>
                    </button>
                  ))}
                </div>
              </div>
            )}

            {/* Model Info */}
            {visualizationData && (
              <div className="card mt-6 bg-blue-50 dark:bg-blue-900/20 border border-blue-200 dark:border-blue-800">
                <h3 className="text-lg font-semibold mb-3 text-blue-900 dark:text-blue-100">
                  Model Info
                </h3>
                <div className="space-y-2 text-sm">
                  <div className="flex justify-between">
                    <span className="text-blue-700 dark:text-blue-300">Mode:</span>
                    <span className="font-medium text-blue-900 dark:text-blue-100">
                      {visualizationData.mode}
                    </span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-blue-700 dark:text-blue-300">Input Tokens:</span>
                    <span className="font-medium text-blue-900 dark:text-blue-100">
                      {visualizationData.source_tokens.length}
                    </span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-blue-700 dark:text-blue-300">Output Tokens:</span>
                    <span className="font-medium text-blue-900 dark:text-blue-100">
                      {visualizationData.target_tokens.length}
                    </span>
                  </div>
                </div>
              </div>
            )}
          </div>

          {/* Visualization Area - Main Content */}
          <div className="lg:col-span-2">
            {loading && (
              <div className="card flex items-center justify-center h-96">
                <div className="text-center">
                  <div className="animate-spin rounded-full h-16 w-16 border-b-2 border-primary-500 mx-auto mb-4"></div>
                  <p className="text-gray-600 dark:text-gray-400">
                    Running transformer inference...
                  </p>
                </div>
              </div>
            )}

            {!loading && !visualizationData && (
              <div className="card h-96 flex items-center justify-center">
                <div className="text-center">
                  <div className="text-6xl mb-4">🤖</div>
                  <h2 className="text-2xl font-bold text-gray-800 dark:text-white mb-2">
                    Welcome to Transformer Visualization
                  </h2>
                  <p className="text-gray-600 dark:text-gray-400 max-w-md">
                    Enter some text in the control panel and click "Run Inference" to see
                    how transformers process your input step-by-step.
                  </p>
                </div>
              </div>
            )}

            {!loading && visualizationData && (
              <>
                {viewMode === 'architecture' && (
                  <ArchitectureDiagram data={visualizationData} />
                )}
                {viewMode === 'embeddings' && (
                  <EmbeddingVisualizer data={visualizationData} />
                )}
                {viewMode === 'attention' && (
                  <AttentionVisualizer data={visualizationData} />
                )}
                {viewMode === 'complete' && (
                  <div className="space-y-6">
                    <ArchitectureDiagram data={visualizationData} />
                    <EmbeddingVisualizer data={visualizationData} />
                    <AttentionVisualizer data={visualizationData} />
                  </div>
                )}
              </>
            )}
          </div>
        </div>
      </div>

      {/* Footer */}
      <footer className="bg-black/30 backdrop-blur-sm border-t border-white/10 mt-12">
        <div className="container mx-auto px-6 py-4 text-center text-gray-400 text-sm">
          <p>
            Built for education • Learn about{' '}
            <a
              href="https://arxiv.org/abs/1706.03762"
              target="_blank"
              rel="noopener noreferrer"
              className="text-blue-400 hover:text-blue-300"
            >
              Attention is All You Need
            </a>
          </p>
        </div>
      </footer>
    </div>
  );
}

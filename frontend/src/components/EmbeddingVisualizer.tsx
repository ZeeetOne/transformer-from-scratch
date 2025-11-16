import { useMemo } from 'react';
import Plot from 'react-plotly.js';
import { InferenceResponse } from '../services/api';

interface EmbeddingVisualizerProps {
  data: InferenceResponse;
}

export default function EmbeddingVisualizer({ data }: EmbeddingVisualizerProps) {
  // Extract embedding data
  const embeddingData = useMemo(() => {
    const encoder = data.visualization_data?.encoder;
    const embedding = encoder?.embedding;

    if (!embedding) return null;

    const tokenEmb = embedding.step_1_token_embedding;
    const posEnc = embedding.step_2_positional_encoding;

    return {
      tokenEmbedding: tokenEmb?.embeddings?.[0] || [], // First batch
      positionalEncoding: posEnc?.positional_encoding?.[0] || [],
      embeddingsWithPosition: posEnc?.embeddings_with_position?.[0] || [],
      pattern: posEnc?.encoding_pattern,
    };
  }, [data]);

  if (!embeddingData) {
    return (
      <div className="card">
        <p className="text-gray-600 dark:text-gray-400">No embedding data available</p>
      </div>
    );
  }

  const tokens = data.source_tokens.map((t, i) => `T${i}`);

  // Extract first few dimensions for visualization
  const tokenEmbFirst4Dims = embeddingData.tokenEmbedding.map(
    (emb: number[]) => emb.slice(0, 4)
  );

  const posEncFirst4Dims = embeddingData.positionalEncoding.map(
    (enc: number[]) => enc.slice(0, 4)
  );

  return (
    <div className="card">
      <h2 className="text-xl font-bold mb-4 text-gray-800 dark:text-white">
        Embeddings & Positional Encoding
      </h2>

      {/* Token Embeddings */}
      <div className="mb-6">
        <h3 className="text-lg font-semibold text-gray-700 dark:text-gray-300 mb-3">
          1. Token Embeddings
        </h3>
        <p className="text-sm text-gray-600 dark:text-gray-400 mb-3">
          Each token is converted to a dense vector. Here are the first 4 dimensions:
        </p>
        <div className="bg-white dark:bg-gray-800 p-4 rounded-lg">
          <Plot
            data={[
              {
                z: tokenEmbFirst4Dims,
                x: ['Dim 0', 'Dim 1', 'Dim 2', 'Dim 3'],
                y: tokens,
                type: 'heatmap',
                colorscale: 'Blues',
                showscale: true,
                hovertemplate: 'Token: %{y}<br>Dim: %{x}<br>Value: %{z:.3f}<extra></extra>',
              },
            ]}
            layout={{
              title: {
                text: 'Token Embeddings (First 4 Dimensions)',
                font: { size: 14 },
              },
              xaxis: { title: 'Embedding Dimensions' },
              yaxis: { title: 'Tokens', autorange: 'reversed' },
              height: 300,
              margin: { l: 80, r: 50, t: 50, b: 60 },
            }}
            config={{ responsive: true, displayModeBar: false }}
            className="w-full"
          />
        </div>
      </div>

      {/* Positional Encoding */}
      <div className="mb-6">
        <h3 className="text-lg font-semibold text-gray-700 dark:text-gray-300 mb-3">
          2. Positional Encoding
        </h3>
        <p className="text-sm text-gray-600 dark:text-gray-400 mb-3">
          Sinusoidal patterns encode position information. Different frequencies for each dimension:
        </p>
        <div className="bg-white dark:bg-gray-800 p-4 rounded-lg">
          <Plot
            data={[
              {
                z: posEncFirst4Dims,
                x: ['Dim 0', 'Dim 1', 'Dim 2', 'Dim 3'],
                y: tokens,
                type: 'heatmap',
                colorscale: 'RdBu',
                showscale: true,
                hovertemplate: 'Position: %{y}<br>Dim: %{x}<br>Value: %{z:.3f}<extra></extra>',
              },
            ]}
            layout={{
              title: {
                text: 'Positional Encoding (First 4 Dimensions)',
                font: { size: 14 },
              },
              xaxis: { title: 'Encoding Dimensions' },
              yaxis: { title: 'Position', autorange: 'reversed' },
              height: 300,
              margin: { l: 80, r: 50, t: 50, b: 60 },
            }}
            config={{ responsive: true, displayModeBar: false }}
            className="w-full"
          />
        </div>
      </div>

      {/* Positional Encoding Pattern */}
      {embeddingData.pattern && (
        <div className="mb-6">
          <h3 className="text-lg font-semibold text-gray-700 dark:text-gray-300 mb-3">
            3. Sinusoidal Pattern Across Positions
          </h3>
          <p className="text-sm text-gray-600 dark:text-gray-400 mb-3">
            Different dimensions use different frequencies (low to high):
          </p>
          <div className="bg-white dark:bg-gray-800 p-4 rounded-lg">
            <Plot
              data={[
                {
                  y: embeddingData.pattern.first_dim,
                  name: 'First Dim (Low Freq)',
                  type: 'scatter',
                  mode: 'lines+markers',
                  line: { color: '#3b82f6' },
                },
                {
                  y: embeddingData.pattern.middle_dim,
                  name: 'Middle Dim',
                  type: 'scatter',
                  mode: 'lines+markers',
                  line: { color: '#8b5cf6' },
                },
                {
                  y: embeddingData.pattern.last_dim,
                  name: 'Last Dim (High Freq)',
                  type: 'scatter',
                  mode: 'lines+markers',
                  line: { color: '#ec4899' },
                },
              ]}
              layout={{
                title: {
                  text: 'Positional Encoding Across Sequence',
                  font: { size: 14 },
                },
                xaxis: { title: 'Position' },
                yaxis: { title: 'Encoding Value' },
                height: 300,
                showlegend: true,
                legend: { x: 1, y: 1, xanchor: 'right' },
                margin: { l: 60, r: 120, t: 50, b: 60 },
              }}
              config={{ responsive: true, displayModeBar: false }}
              className="w-full"
            />
          </div>
        </div>
      )}

      {/* Educational Explanation */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        <div className="p-4 bg-blue-50 dark:bg-blue-900/20 rounded-lg border border-blue-200 dark:border-blue-800">
          <h3 className="text-sm font-semibold text-blue-900 dark:text-blue-100 mb-2">
            📊 Token Embeddings
          </h3>
          <p className="text-xs text-blue-700 dark:text-blue-300 leading-relaxed">
            Each token gets a learned vector representation. Similar tokens should have
            similar embeddings. The model learns these during training to capture semantic
            meaning.
          </p>
        </div>
        <div className="p-4 bg-purple-50 dark:bg-purple-900/20 rounded-lg border border-purple-200 dark:border-purple-800">
          <h3 className="text-sm font-semibold text-purple-900 dark:text-purple-100 mb-2">
            🌊 Positional Encoding
          </h3>
          <p className="text-xs text-purple-700 dark:text-purple-300 leading-relaxed">
            Sinusoidal patterns inject position information. Different frequencies allow
            the model to learn relative positions. Low frequencies = coarse position,
            high frequencies = fine-grained position.
          </p>
        </div>
      </div>
    </div>
  );
}

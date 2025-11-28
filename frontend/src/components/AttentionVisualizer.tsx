import { useState, useMemo } from 'react';
import Plot from 'react-plotly.js';
import { InferenceResponse } from '../shared/api/types';

interface AttentionVisualizerProps {
  data: InferenceResponse;
}

export default function AttentionVisualizer({ data }: AttentionVisualizerProps) {
  const [selectedLayer, setSelectedLayer] = useState(0);
  const [selectedHead, setSelectedHead] = useState(0);

  // Extract attention data from visualization
  const attentionData = useMemo(() => {
    const encoder = data.visualization_data?.encoder;
    const layers = encoder?.layer_wise_details || [];

    if (layers.length === 0) return null;

    const layer = layers[selectedLayer];
    if (!layer) return null;

    const selfAttention = layer.sublayer_1_self_attention;
    const headsData = selfAttention?.attention_per_head || [];

    return {
      nLayers: layers.length,
      nHeads: headsData.length,
      currentHead: headsData[selectedHead],
      allHeads: headsData,
    };
  }, [data, selectedLayer, selectedHead]);

  if (!attentionData || !attentionData.currentHead) {
    return (
      <div className="card">
        <p className="text-gray-600 dark:text-gray-400">No attention data available</p>
      </div>
    );
  }

  // Get attention weights for current head
  const weights = attentionData.currentHead.attention_weights[0]; // First batch item
  const tokens = data.source_tokens.map((_t, i) => `T${i}`); // Token labels

  return (
    <div className="card">
      <h2 className="text-xl font-bold mb-4 text-gray-800 dark:text-white">
        Attention Heatmap
      </h2>

      {/* Layer and Head Selection */}
      <div className="grid grid-cols-2 gap-4 mb-6">
        <div>
          <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
            Layer: {selectedLayer}
          </label>
          <input
            type="range"
            min="0"
            max={attentionData.nLayers - 1}
            value={selectedLayer}
            onChange={(e) => setSelectedLayer(parseInt(e.target.value))}
            className="w-full"
          />
          <div className="flex justify-between text-xs text-gray-500 dark:text-gray-400 mt-1">
            <span>0</span>
            <span>{attentionData.nLayers - 1}</span>
          </div>
        </div>

        <div>
          <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
            Head: {selectedHead}
          </label>
          <input
            type="range"
            min="0"
            max={attentionData.nHeads - 1}
            value={selectedHead}
            onChange={(e) => setSelectedHead(parseInt(e.target.value))}
            className="w-full"
          />
          <div className="flex justify-between text-xs text-gray-500 dark:text-gray-400 mt-1">
            <span>0</span>
            <span>{attentionData.nHeads - 1}</span>
          </div>
        </div>
      </div>

      {/* Attention Statistics */}
      <div className="grid grid-cols-3 gap-4 mb-6">
        <div className="bg-blue-50 dark:bg-blue-900/20 p-3 rounded-lg border border-blue-200 dark:border-blue-800">
          <div className="text-xs text-blue-700 dark:text-blue-300 mb-1">Head Index</div>
          <div className="text-lg font-bold text-blue-900 dark:text-blue-100">
            {attentionData.currentHead.head_index}
          </div>
        </div>
        <div className="bg-purple-50 dark:bg-purple-900/20 p-3 rounded-lg border border-purple-200 dark:border-purple-800">
          <div className="text-xs text-purple-700 dark:text-purple-300 mb-1">Entropy</div>
          <div className="text-lg font-bold text-purple-900 dark:text-purple-100">
            {attentionData.currentHead.avg_attention_entropy.toFixed(3)}
          </div>
        </div>
        <div className="bg-green-50 dark:bg-green-900/20 p-3 rounded-lg border border-green-200 dark:border-green-800">
          <div className="text-xs text-green-700 dark:text-green-300 mb-1">Focus</div>
          <div className="text-lg font-bold text-green-900 dark:text-green-100">
            {attentionData.currentHead.avg_attention_entropy > 2 ? 'Spread' : 'Focused'}
          </div>
        </div>
      </div>

      {/* Heatmap */}
      <div className="bg-white dark:bg-gray-800 p-4 rounded-lg">
        <Plot
          data={[
            {
              z: weights,
              x: tokens,
              y: tokens,
              type: 'heatmap',
              colorscale: 'Viridis',
              showscale: true,
              hovertemplate: 'From: %{y}<br>To: %{x}<br>Weight: %{z:.3f}<extra></extra>',
            },
          ]}
          layout={{
            title: {
              text: `Layer ${selectedLayer}, Head ${selectedHead} - Attention Weights`,
              font: { size: 14 },
            },
            xaxis: {
              title: 'Key (To)',
              side: 'bottom',
            },
            yaxis: {
              title: 'Query (From)',
              autorange: 'reversed',
            },
            width: undefined,
            height: 500,
            autosize: true,
            margin: { l: 80, r: 50, t: 50, b: 80 },
          }}
          config={{
            responsive: true,
            displayModeBar: false,
          }}
          className="w-full"
        />
      </div>

      {/* Educational Explanation */}
      <div className="mt-6 p-4 bg-yellow-50 dark:bg-yellow-900/20 rounded-lg border border-yellow-200 dark:border-yellow-800">
        <h3 className="text-sm font-semibold text-yellow-900 dark:text-yellow-100 mb-2">
          🎯 Understanding Attention
        </h3>
        <p className="text-xs text-yellow-700 dark:text-yellow-300 leading-relaxed">
          Each cell shows how much token i (row) attends to token j (column).
          Brighter colors mean stronger attention. High entropy = attention is spread out across many tokens.
          Low entropy = attention is focused on few tokens.
        </p>
      </div>

      {/* All Heads Overview */}
      <div className="mt-6">
        <h3 className="text-sm font-semibold text-gray-800 dark:text-white mb-3">
          All Heads in Layer {selectedLayer}
        </h3>
        <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
          {attentionData.allHeads.map((head: any) => (
            <button
              key={head.head_index}
              onClick={() => setSelectedHead(head.head_index)}
              className={`
                p-3 rounded-lg border-2 transition-all duration-200
                ${
                  selectedHead === head.head_index
                    ? 'border-primary-500 bg-primary-50 dark:bg-primary-900/20'
                    : 'border-gray-200 dark:border-gray-700 hover:border-gray-300 dark:hover:border-gray-600'
                }
              `}
            >
              <div className="text-lg font-bold text-gray-800 dark:text-white">
                Head {head.head_index}
              </div>
              <div className="text-xs text-gray-600 dark:text-gray-400 mt-1">
                Entropy: {head.avg_attention_entropy.toFixed(2)}
              </div>
            </button>
          ))}
        </div>
      </div>
    </div>
  );
}

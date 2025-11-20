import { motion } from 'framer-motion';
import { useMemo } from 'react';

interface EmbeddingVisualizerProps {
  shape: number[];
  sampleValues: number[][];
  tokens: string[];
}

export default function EmbeddingVisualizer({ shape, sampleValues, tokens }: EmbeddingVisualizerProps) {
  const [numTokens, embeddingDim] = shape;

  // Normalize values for visualization (0-1 range)
  const normalizedValues = useMemo(() => {
    if (!sampleValues || sampleValues.length === 0) return [];

    const flat = sampleValues.flat();
    const max = Math.max(...flat.map(Math.abs));
    return sampleValues.map(row =>
      row.map(val => Math.abs(val) / (max || 1))
    );
  }, [sampleValues]);

  return (
    <div className="space-y-6">
      {/* Educational Description */}
      <div className="bg-purple-500/10 border border-purple-500/30 rounded-lg p-4">
        <h4 className="font-semibold text-purple-300 mb-2">🎯 What's Happening?</h4>
        <p className="text-sm text-gray-300 mb-2">
          Token IDs are just arbitrary numbers. <strong>Embeddings</strong> transform each token into a rich
          {' '}<strong>{embeddingDim}-dimensional vector</strong> that captures its meaning. Think of it like
          GPS coordinates on a map - similar words are close together!
        </p>
        <p className="text-sm text-gray-300">
          Plus, we add <strong>positional encoding</strong> - a unique pattern for each position so the model
          knows word order ("I eat pizza" ≠ "pizza eat I").
        </p>
      </div>

      {/* Shape Information */}
      <div className="grid grid-cols-3 gap-4">
        <div className="bg-white/5 border border-white/10 rounded-lg p-3">
          <div className="text-xs text-gray-400 mb-1">Tokens</div>
          <div className="text-2xl font-bold text-blue-400">{numTokens}</div>
        </div>
        <div className="bg-white/5 border border-white/10 rounded-lg p-3">
          <div className="text-xs text-gray-400 mb-1">Dimensions</div>
          <div className="text-2xl font-bold text-purple-400">{embeddingDim}</div>
        </div>
        <div className="bg-white/5 border border-white/10 rounded-lg p-3">
          <div className="text-xs text-gray-400 mb-1">Shape</div>
          <div className="text-lg font-bold text-green-400 font-mono">
            {numTokens} × {embeddingDim}
          </div>
        </div>
      </div>

      {/* Embedding Heatmap Visualization */}
      <div>
        <h4 className="font-semibold mb-3 flex items-center gap-2">
          <span className="text-2xl">🔥</span>
          Embedding Vectors (Each token as {sampleValues[0]?.length || 16} dimensions)
        </h4>

        <div className="space-y-3">
          {tokens.map((token, tokenIdx) => (
            <motion.div
              key={tokenIdx}
              initial={{ opacity: 0, x: -50 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ delay: tokenIdx * 0.15 }}
              className="bg-white/5 border border-white/10 rounded-lg p-3"
            >
              {/* Token Label */}
              <div className="flex items-center gap-3 mb-2">
                <div className="px-3 py-1 bg-blue-500/30 border border-blue-500 rounded text-sm font-mono">
                  {token.replace('<SOS>', '▶').replace('<EOS>', '◀').replace('<PAD>', '⊗')}
                </div>
                <div className="text-xs text-gray-400">
                  {sampleValues[tokenIdx]?.length || 0} dimensions shown
                </div>
              </div>

              {/* Heatmap Bar */}
              <div className="grid grid-cols-16 gap-1">
                {normalizedValues[tokenIdx]?.map((value, dimIdx) => {
                  const intensity = value;
                  const hue = sampleValues[tokenIdx][dimIdx] >= 0 ? 200 : 340; // Blue for positive, pink for negative

                  return (
                    <motion.div
                      key={dimIdx}
                      initial={{ scaleY: 0, opacity: 0 }}
                      animate={{ scaleY: 1, opacity: 1 }}
                      transition={{ delay: tokenIdx * 0.15 + dimIdx * 0.01 }}
                      className="h-12 rounded group relative"
                      style={{
                        backgroundColor: `hsla(${hue}, 70%, 60%, ${intensity * 0.8 + 0.2})`
                      }}
                      title={`Dim ${dimIdx}: ${sampleValues[tokenIdx][dimIdx].toFixed(4)}`}
                    >
                      {/* Tooltip on hover */}
                      <div className="absolute bottom-full left-1/2 transform -translate-x-1/2 mb-1 px-2 py-1 bg-black text-white text-xs rounded opacity-0 group-hover:opacity-100 transition-opacity whitespace-nowrap pointer-events-none z-10">
                        {sampleValues[tokenIdx][dimIdx].toFixed(3)}
                      </div>
                    </motion.div>
                  );
                })}
              </div>
            </motion.div>
          ))}
        </div>
      </div>

      {/* Analogy */}
      <div className="bg-indigo-500/10 border border-indigo-500/30 rounded-lg p-4">
        <h4 className="font-semibold text-indigo-300 mb-2">🗺️ Analogy: Words as Points on a Map</h4>
        <p className="text-sm text-gray-300">
          Each word is placed at coordinates in a {embeddingDim}-dimensional space. Words with similar meanings
          (like "eat" and "consume") are close neighbors. Words with different meanings are far apart.
          The colors show the values: <span className="text-blue-400">blue = positive</span>,{' '}
          <span className="text-pink-400">pink = negative</span>.
        </p>
      </div>

      {/* Why This Matters */}
      <div className="bg-green-500/10 border border-green-500/30 rounded-lg p-4">
        <h4 className="font-semibold text-green-300 mb-2">💡 Why This Matters</h4>
        <p className="text-sm text-gray-300">
          Raw token IDs have no meaning - token 42 isn't "bigger" than token 5. Embeddings capture
          relationships and context. The model learns these vectors during training, placing similar words
          close together in this high-dimensional space!
        </p>
      </div>
    </div>
  );
}

import { motion } from 'framer-motion';
import { useState, useMemo } from 'react';

interface AttentionVisualizerProps {
  numHeads: number;
  numLayers: number;
  tokens: string[];
  attentionData?: any; // Will contain actual attention weights when available
}

export default function AttentionVisualizer({ numHeads, numLayers, tokens }: AttentionVisualizerProps) {
  const [selectedToken, setSelectedToken] = useState<number | null>(null);

  // Generate mock attention weights for visualization (since backend doesn't return full weights yet)
  const mockAttentionWeights = useMemo(() => {
    const weights: number[][] = [];
    for (let i = 0; i < tokens.length; i++) {
      const row: number[] = [];
      for (let j = 0; j <= i; j++) { // Causal mask: can only attend to previous tokens
        if (i === j) {
          row.push(0.4 + Math.random() * 0.3); // Self-attention
        } else if (j === i - 1) {
          row.push(0.3 + Math.random() * 0.2); // Previous token
        } else {
          row.push(Math.random() * 0.15); // Earlier tokens
        }
      }
      // Normalize to sum to 1
      const sum = row.reduce((a, b) => a + b, 0);
      weights.push(row.map(w => w / sum));
    }
    return weights;
  }, [tokens]);

  return (
    <div className="space-y-6">
      {/* Educational Description */}
      <div className="bg-blue-500/10 border border-blue-500/30 rounded-lg p-4">
        <h4 className="font-semibold text-blue-300 mb-2">🎯 What's Happening?</h4>
        <p className="text-sm text-gray-300 mb-2">
          <strong>Self-Attention</strong> lets each token "look at" all previous tokens and decide which
          ones are most relevant. It's like a spotlight - each token focuses on what matters most for
          understanding its context.
        </p>
        <p className="text-sm text-gray-300">
          For example, when processing "eat", the model pays attention to "I" (the subject who's eating).
        </p>
      </div>

      {/* Model Info */}
      <div className="grid grid-cols-2 gap-4">
        <div className="bg-white/5 border border-white/10 rounded-lg p-3">
          <div className="text-xs text-gray-400 mb-1">Attention Heads</div>
          <div className="text-2xl font-bold text-blue-400">{numHeads}</div>
          <div className="text-xs text-gray-400 mt-1">Different perspectives</div>
        </div>
        <div className="bg-white/5 border border-white/10 rounded-lg p-3">
          <div className="text-xs text-gray-400 mb-1">Layers</div>
          <div className="text-2xl font-bold text-purple-400">{numLayers}</div>
          <div className="text-xs text-gray-400 mt-1">Processing depth</div>
        </div>
      </div>

      {/* Connection Graph Visualization */}
      <div>
        <h4 className="font-semibold mb-3 flex items-center gap-2">
          <span className="text-2xl">🎯</span>
          Attention Connections
          <span className="text-xs text-gray-400 ml-2">(Click a token to see its attention)</span>
        </h4>

        {/* Token Nodes */}
        <div className="bg-black/20 rounded-lg p-6 relative" style={{ minHeight: '300px' }}>
          <svg width="100%" height="280" className="absolute top-0 left-0">
            {/* Draw attention connections */}
            {selectedToken !== null && mockAttentionWeights[selectedToken]?.map((weight, targetIdx) => {
              const sourceX = 50 + (selectedToken * (100 / tokens.length)) + '%';
              const targetX = 50 + (targetIdx * (100 / tokens.length)) + '%';
              const strokeWidth = Math.max(1, weight * 10);
              const opacity = 0.2 + weight * 0.8;

              return (
                <motion.line
                  key={targetIdx}
                  x1={sourceX}
                  y1="80%"
                  x2={targetX}
                  y2="20%"
                  stroke="#3b82f6"
                  strokeWidth={strokeWidth}
                  initial={{ pathLength: 0, opacity: 0 }}
                  animate={{ pathLength: 1, opacity }}
                  transition={{ duration: 0.5, delay: targetIdx * 0.05 }}
                  style={{ filter: 'drop-shadow(0 0 2px #3b82f6)' }}
                />
              );
            })}
          </svg>

          {/* Source Tokens (top row) */}
          <div className="flex justify-around mb-32 relative z-10">
            {tokens.map((token, idx) => (
              <motion.div
                key={`source-${idx}`}
                className={`text-center ${selectedToken !== null && mockAttentionWeights[selectedToken]?.[idx] !== undefined ? 'opacity-100' : 'opacity-40'}`}
              >
                <div className="px-3 py-2 bg-gradient-to-br from-blue-500/40 to-cyan-500/40 border-2 border-cyan-500 rounded-lg text-sm font-mono">
                  {token.replace('<SOS>', '▶').replace('<EOS>', '◀').replace('<PAD>', '⊗')}
                </div>
                {selectedToken !== null && mockAttentionWeights[selectedToken]?.[idx] !== undefined && (
                  <motion.div
                    initial={{ opacity: 0, y: -5 }}
                    animate={{ opacity: 1, y: 0 }}
                    className="text-xs text-cyan-400 mt-1 font-bold"
                  >
                    {(mockAttentionWeights[selectedToken][idx] * 100).toFixed(1)}%
                  </motion.div>
                )}
              </motion.div>
            ))}
          </div>

          {/* Query Tokens (bottom row - clickable) */}
          <div className="flex justify-around relative z-10">
            {tokens.map((token, idx) => (
              <motion.button
                key={`query-${idx}`}
                onClick={() => setSelectedToken(idx === selectedToken ? null : idx)}
                whileHover={{ scale: 1.1 }}
                whileTap={{ scale: 0.95 }}
                className={`px-3 py-2 rounded-lg text-sm font-mono transition-all ${
                  selectedToken === idx
                    ? 'bg-gradient-to-br from-blue-600 to-purple-600 border-2 border-white shadow-lg shadow-blue-500/50'
                    : 'bg-gradient-to-br from-purple-500/40 to-pink-500/40 border-2 border-purple-500 hover:border-white'
                }`}
              >
                {token.replace('<SOS>', '▶').replace('<EOS>', '◀').replace('<PAD>', '⊗')}
              </motion.button>
            ))}
          </div>

          {selectedToken === null && (
            <div className="absolute inset-0 flex items-center justify-center pointer-events-none">
              <p className="text-gray-400 text-sm">👆 Click a token below to see its attention pattern</p>
            </div>
          )}
        </div>
      </div>

      {/* Attention Heatmap (simplified) */}
      {selectedToken !== null && (
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          className="bg-white/5 border border-white/10 rounded-lg p-4"
        >
          <h4 className="font-semibold mb-3">
            Attention Weights for "{tokens[selectedToken].replace('<SOS>', '▶').replace('<EOS>', '◀')}"
          </h4>
          <div className="space-y-2">
            {mockAttentionWeights[selectedToken]?.map((weight, targetIdx) => (
              <div key={targetIdx} className="flex items-center gap-2">
                <div className="w-24 text-sm font-mono text-gray-300">
                  {tokens[targetIdx].replace('<SOS>', '▶').replace('<EOS>', '◀').replace('<PAD>', '⊗')}
                </div>
                <div className="flex-1 bg-white/10 rounded-full h-6 overflow-hidden">
                  <motion.div
                    initial={{ width: 0 }}
                    animate={{ width: `${weight * 100}%` }}
                    transition={{ duration: 0.5, delay: targetIdx * 0.05 }}
                    className="h-full bg-gradient-to-r from-blue-500 to-cyan-400"
                  />
                </div>
                <div className="w-16 text-right text-sm font-semibold text-blue-400">
                  {(weight * 100).toFixed(1)}%
                </div>
              </div>
            ))}
          </div>
        </motion.div>
      )}

      {/* Analogy */}
      <div className="bg-indigo-500/10 border border-indigo-500/30 rounded-lg p-4">
        <h4 className="font-semibold text-indigo-300 mb-2">💡 Analogy: Conversation at a Party</h4>
        <p className="text-sm text-gray-300">
          Imagine you're at a party with friends. When trying to understand a conversation, you naturally
          pay more attention to relevant speakers and less to background chatter. Self-attention does the same -
          each token "listens" more to relevant context and ignores irrelevant parts!
        </p>
      </div>

      {/* Why This Matters */}
      <div className="bg-green-500/10 border border-green-500/30 rounded-lg p-4">
        <h4 className="font-semibold text-green-300 mb-2">🌟 Why This Matters</h4>
        <p className="text-sm text-gray-300">
          Without attention, each token is isolated. Attention creates relationships: "eat" learns it's
          connected to "I" (the subject), making the representation context-aware. This is the breakthrough
          that makes Transformers so powerful!
        </p>
      </div>
    </div>
  );
}

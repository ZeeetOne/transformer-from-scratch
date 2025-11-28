import { motion } from 'framer-motion';
import { useState } from 'react';

interface FeedforwardVisualizerProps {
  hiddenDim: number;
  outputShape: number[];
  tokens: string[];
}

export default function FeedforwardVisualizer({ hiddenDim, outputShape, tokens }: FeedforwardVisualizerProps) {
  const [isExpanded, setIsExpanded] = useState(false);
  const [numTokens, modelDim] = outputShape;

  return (
    <div className="space-y-6">
      {/* Educational Description */}
      <div className="bg-green-500/10 border border-green-500/30 rounded-lg p-4">
        <h4 className="font-semibold text-green-300 mb-2">🎯 What's Happening?</h4>
        <p className="text-sm text-gray-300 mb-2">
          After attention mixes information between tokens, each token needs individual processing.
          The <strong>Feedforward Network</strong> (FFN) processes each token independently:
        </p>
        <ol className="text-sm text-gray-300 list-decimal list-inside space-y-1 ml-2">
          <li><strong>Expand</strong>: {modelDim} → {hiddenDim} dimensions (more "thinking space")</li>
          <li><strong>Transform</strong>: Apply ReLU activation to learn complex patterns</li>
          <li><strong>Compress</strong>: {hiddenDim} → {modelDim} dimensions (back to model size)</li>
        </ol>
      </div>

      {/* Dimension Flow */}
      <div className="grid grid-cols-3 gap-4">
        <div className="bg-white/5 border border-white/10 rounded-lg p-3">
          <div className="text-xs text-gray-400 mb-1">Input</div>
          <div className="text-2xl font-bold text-blue-400 font-mono">{modelDim}</div>
          <div className="text-xs text-gray-400 mt-1">Model dimension</div>
        </div>
        <div className="bg-white/5 border border-green-500/20 rounded-lg p-3 border-2">
          <div className="text-xs text-gray-400 mb-1">Hidden</div>
          <div className="text-2xl font-bold text-green-400 font-mono">{hiddenDim}</div>
          <div className="text-xs text-gray-400 mt-1">4× expansion!</div>
        </div>
        <div className="bg-white/5 border border-white/10 rounded-lg p-3">
          <div className="text-xs text-gray-400 mb-1">Output</div>
          <div className="text-2xl font-bold text-purple-400 font-mono">{modelDim}</div>
          <div className="text-xs text-gray-400 mt-1">Back to original</div>
        </div>
      </div>

      {/* Interactive Expansion Animation */}
      <div>
        <div className="flex justify-between items-center mb-3">
          <h4 className="font-semibold flex items-center gap-2">
            <span className="text-2xl">🧠</span>
            Feedforward Processing
          </h4>
          <button
            onClick={() => setIsExpanded(!isExpanded)}
            className="px-4 py-2 bg-green-600 hover:bg-green-700 rounded-lg text-sm font-semibold transition-colors"
          >
            {isExpanded ? '🔄 Reset' : '▶️ Animate Expansion'}
          </button>
        </div>

        <div className="bg-black/20 rounded-lg p-6 space-y-6">
          {tokens.slice(0, 2).map((token, tokenIdx) => (
            <div key={tokenIdx} className="space-y-3">
              {/* Token Label */}
              <div className="flex items-center gap-2">
                <div className="px-3 py-1 bg-blue-500/30 border border-blue-500 rounded text-sm font-mono">
                  Token: {token.replace('<SOS>', '▶').replace('<EOS>', '◀').replace('<PAD>', '⊗')}
                </div>
                <div className="text-xs text-gray-400">Processing independently</div>
              </div>

              {/* Visualization */}
              <div className="flex items-center gap-4">
                {/* Input */}
                <motion.div
                  className="bg-gradient-to-br from-blue-500/30 to-blue-600/30 border-2 border-blue-500 rounded-lg p-3 flex items-center justify-center"
                  style={{ width: '80px', height: '60px' }}
                >
                  <div className="text-center">
                    <div className="text-xs text-gray-400">Input</div>
                    <div className="text-sm font-bold">{modelDim}</div>
                  </div>
                </motion.div>

                {/* Arrow */}
                <div className="text-2xl text-gray-400">→</div>

                {/* Hidden (Expanded) */}
                <motion.div
                  className="bg-gradient-to-br from-green-500/30 to-green-600/30 border-2 border-green-500 rounded-lg p-3 flex items-center justify-center relative overflow-hidden"
                  animate={{
                    width: isExpanded ? '200px' : '80px',
                    height: isExpanded ? '100px' : '60px'
                  }}
                  transition={{
                    duration: 0.8,
                    delay: tokenIdx * 0.2,
                    type: 'spring',
                    stiffness: 100
                  }}
                >
                  <div className="text-center z-10 relative">
                    <div className="text-xs text-gray-400">Hidden</div>
                    <div className="text-sm font-bold">{hiddenDim}</div>
                  </div>

                  {/* Neurons firing animation */}
                  {isExpanded && (
                    <>
                      {[...Array(8)].map((_, i) => (
                        <motion.div
                          key={i}
                          className="absolute rounded-full bg-green-400"
                          initial={{ opacity: 0, scale: 0 }}
                          animate={{
                            opacity: [0, 0.8, 0],
                            scale: [0, 1, 0],
                            x: Math.random() * 180 - 90,
                            y: Math.random() * 80 - 40
                          }}
                          transition={{
                            duration: 1.5,
                            delay: tokenIdx * 0.2 + i * 0.15,
                            repeat: Infinity,
                            repeatDelay: 2
                          }}
                          style={{ width: '8px', height: '8px' }}
                        />
                      ))}
                    </>
                  )}
                </motion.div>

                {/* Arrow */}
                <div className="text-2xl text-gray-400">→</div>

                {/* Output */}
                <motion.div
                  className="bg-gradient-to-br from-purple-500/30 to-purple-600/30 border-2 border-purple-500 rounded-lg p-3 flex items-center justify-center"
                  style={{ width: '80px', height: '60px' }}
                  animate={{
                    scale: isExpanded ? [1, 1.1, 1] : 1
                  }}
                  transition={{
                    duration: 0.5,
                    delay: 1 + tokenIdx * 0.2
                  }}
                >
                  <div className="text-center">
                    <div className="text-xs text-gray-400">Output</div>
                    <div className="text-sm font-bold">{modelDim}</div>
                  </div>
                </motion.div>
              </div>

              {/* ReLU Activation Explanation */}
              {isExpanded && (
                <motion.div
                  initial={{ opacity: 0, y: -10 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ delay: 0.5 + tokenIdx * 0.2 }}
                  className="bg-yellow-500/10 border border-yellow-500/30 rounded p-2 text-xs text-gray-300"
                >
                  <strong className="text-yellow-400">ReLU Activation:</strong> Clips negative values to zero,
                  keeping only positive activations. This adds non-linearity for complex learning!
                </motion.div>
              )}
            </div>
          ))}

          {tokens.length > 2 && (
            <div className="text-center text-sm text-gray-400 pt-2">
              + {tokens.length - 2} more tokens (all processed in parallel)
            </div>
          )}
        </div>
      </div>

      {/* Analogy */}
      <div className="bg-indigo-500/10 border border-indigo-500/30 rounded-lg p-4">
        <h4 className="font-semibold text-indigo-300 mb-2">🛠️ Analogy: Workshop Processing</h4>
        <p className="text-sm text-gray-300">
          Think of each token entering a workshop. First, you spread out all your tools on a large table
          (expansion). Then you use those tools to build/modify something complex (transformation).
          Finally, you pack up the finished product into a compact box (compression). The larger workspace
          allows more sophisticated processing!
        </p>
      </div>

      {/* Why This Matters */}
      <div className="bg-green-500/10 border border-green-500/30 rounded-lg p-4">
        <h4 className="font-semibold text-green-300 mb-2">💡 Why This Matters</h4>
        <p className="text-sm text-gray-300">
          Attention mixes information <em>between</em> tokens, but feedforward processes each token{' '}
          <em>individually</em>. This combination is crucial! The expansion to {hiddenDim} dimensions
          gives the network more "thinking space" to learn complex transformations that wouldn't be
          possible in the smaller {modelDim}-dimensional space.
        </p>
      </div>

      {/* Output Shape */}
      <div className="bg-white/5 border border-white/10 rounded-lg p-3">
        <div className="text-xs text-gray-400 mb-1">Final Output Shape</div>
        <div className="text-xl font-bold text-green-400 font-mono">
          {numTokens} × {modelDim}
        </div>
        <div className="text-xs text-gray-400 mt-1">
          {numTokens} tokens, each with {modelDim} enriched dimensions
        </div>
      </div>
    </div>
  );
}

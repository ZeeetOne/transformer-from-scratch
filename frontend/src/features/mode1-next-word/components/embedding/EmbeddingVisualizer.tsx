import { motion } from 'framer-motion';
import { useMemo, useState } from 'react';

interface EmbeddingVisualizerV2Props {
  shape: number[];
  sampleValues: number[][];
  tokens: string[];
}

export default function EmbeddingVisualizerV2({ shape, sampleValues, tokens }: EmbeddingVisualizerV2Props) {
  const [numTokens, embeddingDim] = shape;
  const [showComparison, setShowComparison] = useState(false);

  // Color palette for tokens (distinct, pastel colors for educational clarity)
  const tokenColors = [
    '#FF6B6B', // Red
    '#4ECDC4', // Teal
    '#45B7D1', // Blue
    '#FFA07A', // Salmon
    '#98D8C8', // Mint
    '#F7DC6F', // Yellow
    '#BB8FCE', // Purple
    '#85C1E2', // Sky blue
  ];

  // Clean tokens (remove special tokens for display)
  const displayTokens = useMemo(() =>
    tokens.filter(t => !['<SOS>', '<EOS>', '<PAD>'].includes(t)),
    [tokens]
  );

  // Generate mock word embeddings (simplified for visualization - showing 8 dims instead of full 256)
  const wordEmbeddings = useMemo(() => {
    return displayTokens.map(() =>
      Array(8).fill(0).map(() => (Math.random() * 2 - 1).toFixed(2))
    );
  }, [displayTokens]);

  // Generate positional encodings (sinusoidal)
  const positionalEncodings = useMemo(() => {
    return displayTokens.map((_, pos) =>
      Array(8).fill(0).map((_, i) => {
        const value = i % 2 === 0
          ? Math.sin(pos / Math.pow(10000, i / 8))
          : Math.cos(pos / Math.pow(10000, (i - 1) / 8));
        return value.toFixed(2);
      })
    );
  }, [displayTokens]);

  // Final embeddings (WE + PE)
  const finalEmbeddings = useMemo(() => {
    return displayTokens.map((_, tokenIdx) =>
      Array(8).fill(0).map((_, dimIdx) =>
        (parseFloat(wordEmbeddings[tokenIdx][dimIdx]) +
         parseFloat(positionalEncodings[tokenIdx][dimIdx])).toFixed(2)
      )
    );
  }, [displayTokens, wordEmbeddings, positionalEncodings]);

  // Render embedding grid with color coding
  const EmbeddingGrid = ({
    embeddings,
    tokenIdx,
    label
  }: {
    embeddings: string[][];
    tokenIdx: number;
    label: string;
  }) => (
    <div className="flex flex-col items-center">
      <div className="text-xs text-gray-400 mb-1">{label}</div>
      <div className="grid grid-cols-8 gap-0.5">
        {embeddings[tokenIdx].map((val, dimIdx) => {
          const numVal = parseFloat(val);
          const intensity = Math.min(Math.abs(numVal), 1);
          const isPositive = numVal >= 0;

          return (
            <motion.div
              key={dimIdx}
              initial={{ opacity: 0, scale: 0 }}
              animate={{ opacity: 1, scale: 1 }}
              transition={{ delay: dimIdx * 0.02 }}
              className="relative group w-8 h-8 rounded flex items-center justify-center text-[8px] font-mono"
              style={{
                backgroundColor: isPositive
                  ? `rgba(59, 130, 246, ${intensity * 0.8})`
                  : `rgba(239, 68, 68, ${intensity * 0.8})`,
                border: '1px solid rgba(255,255,255,0.1)'
              }}
              title={`Dim ${dimIdx}: ${val}`}
            >
              <span className="text-white drop-shadow">{val}</span>
            </motion.div>
          );
        })}
      </div>
    </div>
  );

  // Sinusoidal wave visualization for PE
  const SinusoidalWave = ({ position, color }: { position: number; color: string }) => {
    const points = Array(100).fill(0).map((_, i) => {
      const x = (i / 100) * 200;
      const y = 20 + Math.sin((position + i) / 10) * 15;
      return `${x},${y}`;
    }).join(' ');

    return (
      <svg width="200" height="40" className="opacity-60">
        <polyline
          points={points}
          fill="none"
          stroke={color}
          strokeWidth="2"
        />
      </svg>
    );
  };

  return (
    <div className="space-y-6">
      {/* Educational Header */}
      <div className="bg-gradient-to-r from-purple-500/10 to-blue-500/10 border border-purple-500/30 rounded-lg p-4">
        <h4 className="font-semibold text-purple-300 mb-2 flex items-center gap-2">
          <span className="text-2xl">📐</span>
          Word Embeddings + Positional Encoding
        </h4>
        <p className="text-sm text-gray-300 mb-2">
          Every token needs two pieces of information:
        </p>
        <div className="grid md:grid-cols-2 gap-3 text-sm">
          <div className="bg-blue-500/10 border border-blue-500/30 rounded p-2">
            <strong className="text-blue-400">Word Embedding (WE)</strong>
            <p className="text-gray-300 text-xs mt-1">
              Learned semantic meaning - what the word represents
            </p>
          </div>
          <div className="bg-green-500/10 border border-green-500/30 rounded p-2">
            <strong className="text-green-400">Positional Encoding (PE)</strong>
            <p className="text-gray-300 text-xs mt-1">
              Sinusoidal position pattern - where the word appears
            </p>
          </div>
        </div>
        <div className="mt-3 bg-purple-500/10 border border-purple-500/30 rounded p-2 text-center">
          <strong className="text-purple-400">Final Embedding = WE + PE</strong>
          <p className="text-gray-300 text-xs mt-1">Element-wise addition: X<sub>i</sub> = E<sub>i</sub> + P<sub>i</sub></p>
        </div>
      </div>

      {/* Comparison Toggle */}
      {displayTokens.length >= 3 && (
        <div className="flex justify-center">
          <button
            onClick={() => setShowComparison(!showComparison)}
            className="px-4 py-2 bg-purple-600 hover:bg-purple-700 rounded-lg text-sm font-semibold transition-colors"
          >
            {showComparison ? '📊 Hide Position Comparison' : '🔄 Show How Position Changes Meaning'}
          </button>
        </div>
      )}

      {/* Main Visualization Flow */}
      <div className="bg-gradient-to-br from-gray-900/50 to-gray-800/50 rounded-xl border border-white/10 p-6">

        {/* Step 1: Input Tokens */}
        <div className="mb-8">
          <div className="text-center mb-4">
            <h5 className="text-sm font-semibold text-gray-400 mb-2">Step 1: Input Tokens</h5>
            <div className="flex justify-center gap-3 flex-wrap">
              {displayTokens.map((token, idx) => (
                <motion.div
                  key={idx}
                  initial={{ opacity: 0, y: -20 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ delay: idx * 0.1 }}
                  className="px-4 py-2 rounded-lg font-mono text-lg font-bold"
                  style={{
                    backgroundColor: tokenColors[idx % tokenColors.length],
                    color: 'white'
                  }}
                >
                  {token}
                </motion.div>
              ))}
            </div>
          </div>

          {/* Arrow Down */}
          <div className="flex justify-center my-4">
            <motion.div
              animate={{ y: [0, 8, 0] }}
              transition={{ duration: 1.5, repeat: Infinity }}
              className="text-3xl text-blue-400"
            >
              ↓
            </motion.div>
          </div>
        </div>

        {/* Step 2: Word Embeddings Table */}
        <div className="mb-8">
          <h5 className="text-sm font-semibold text-gray-400 mb-3 text-center">
            Step 2: Word Embeddings (Semantic Meaning)
          </h5>
          <div className="bg-blue-500/5 border border-blue-500/20 rounded-lg p-4">
            <div className="flex justify-around gap-4 flex-wrap">
              {displayTokens.map((token, tokenIdx) => (
                <div key={tokenIdx} className="flex flex-col items-center">
                  {/* Token Label */}
                  <div
                    className="px-3 py-1 rounded-lg mb-2 font-mono text-sm font-bold"
                    style={{
                      backgroundColor: tokenColors[tokenIdx % tokenColors.length],
                      color: 'white'
                    }}
                  >
                    {token}
                  </div>
                  {/* Embedding Grid */}
                  <EmbeddingGrid
                    embeddings={wordEmbeddings}
                    tokenIdx={tokenIdx}
                    label="WE"
                  />
                </div>
              ))}
            </div>
            <p className="text-xs text-center text-gray-400 mt-3">
              Each word has a learned {embeddingDim}-dimensional vector (showing 8 dims)
            </p>
          </div>
        </div>

        {/* Plus Sign */}
        <div className="flex justify-center my-4">
          <div className="text-4xl font-bold text-green-400">+</div>
        </div>

        {/* Step 3: Positional Encodings */}
        <div className="mb-8">
          <h5 className="text-sm font-semibold text-gray-400 mb-3 text-center">
            Step 3: Positional Encoding (Position Pattern)
          </h5>
          <div className="bg-green-500/5 border border-green-500/20 rounded-lg p-4">
            <div className="flex justify-around gap-4 flex-wrap">
              {displayTokens.map((token, tokenIdx) => (
                <div key={tokenIdx} className="flex flex-col items-center">
                  {/* Position Label */}
                  <div className="text-xs text-gray-400 mb-1">
                    Position {tokenIdx}
                  </div>
                  {/* Sinusoidal Wave */}
                  <SinusoidalWave
                    position={tokenIdx * 20}
                    color={tokenColors[tokenIdx % tokenColors.length]}
                  />
                  {/* Embedding Grid */}
                  <EmbeddingGrid
                    embeddings={positionalEncodings}
                    tokenIdx={tokenIdx}
                    label="PE"
                  />
                </div>
              ))}
            </div>
            <p className="text-xs text-center text-gray-400 mt-3">
              Sinusoidal functions encode position: sin/cos patterns unique to each position
            </p>
          </div>
        </div>

        {/* Equals Sign */}
        <div className="flex justify-center my-4">
          <div className="text-4xl font-bold text-purple-400">=</div>
        </div>

        {/* Step 4: Final Embeddings (WE + PE) */}
        <div>
          <h5 className="text-sm font-semibold text-gray-400 mb-3 text-center">
            Step 4: Final Input Embedding (WE + PE)
          </h5>
          <div className="bg-purple-500/5 border border-purple-500/20 rounded-lg p-4">
            <div className="flex justify-around gap-4 flex-wrap">
              {displayTokens.map((token, tokenIdx) => (
                <motion.div
                  key={tokenIdx}
                  className="flex flex-col items-center"
                  initial={{ opacity: 0, scale: 0.8 }}
                  animate={{ opacity: 1, scale: 1 }}
                  transition={{ delay: 0.5 + tokenIdx * 0.1 }}
                >
                  {/* Token Label */}
                  <div
                    className="px-3 py-1 rounded-lg mb-2 font-mono text-sm font-bold shadow-lg"
                    style={{
                      backgroundColor: tokenColors[tokenIdx % tokenColors.length],
                      color: 'white',
                      boxShadow: `0 0 20px ${tokenColors[tokenIdx % tokenColors.length]}40`
                    }}
                  >
                    {token}
                  </div>
                  {/* Final Embedding Grid */}
                  <EmbeddingGrid
                    embeddings={finalEmbeddings}
                    tokenIdx={tokenIdx}
                    label={`X${tokenIdx}`}
                  />
                </motion.div>
              ))}
            </div>
            <p className="text-xs text-center text-gray-400 mt-3">
              Final embeddings contain both semantic meaning AND positional information
            </p>
          </div>
        </div>
      </div>

      {/* Position Comparison (if toggled) */}
      {showComparison && displayTokens.length >= 2 && (
        <motion.div
          initial={{ opacity: 0, height: 0 }}
          animate={{ opacity: 1, height: 'auto' }}
          className="bg-yellow-500/10 border-2 border-yellow-500/50 rounded-xl p-6"
        >
          <h5 className="text-lg font-bold text-yellow-400 mb-4 text-center">
            🔄 Why Position Matters: Same Words, Different Positions
          </h5>

          <div className="grid md:grid-cols-2 gap-6">
            {/* Original Order */}
            <div className="bg-black/20 rounded-lg p-4">
              <h6 className="text-sm font-semibold text-blue-400 mb-3 text-center">
                Original: "{displayTokens.join(' ')}"
              </h6>
              {displayTokens.slice(0, 2).map((token, idx) => (
                <div key={idx} className="mb-3">
                  <div className="flex items-center gap-2 mb-2">
                    <div
                      className="px-2 py-1 rounded text-xs font-mono"
                      style={{ backgroundColor: tokenColors[idx % tokenColors.length] }}
                    >
                      {token}
                    </div>
                    <span className="text-xs text-gray-400">at position {idx}</span>
                  </div>
                  <div className="text-xs font-mono text-gray-300">
                    PE = [{positionalEncodings[idx].slice(0, 4).join(', ')}...]
                  </div>
                </div>
              ))}
            </div>

            {/* Reversed Order */}
            <div className="bg-black/20 rounded-lg p-4">
              <h6 className="text-sm font-semibold text-purple-400 mb-3 text-center">
                Swapped: "{displayTokens.slice(0, 2).reverse().join(' ')} {displayTokens.slice(2).join(' ')}"
              </h6>
              {displayTokens.slice(0, 2).reverse().map((token, idx) => {
                const originalIdx = displayTokens.indexOf(token);
                return (
                  <div key={idx} className="mb-3">
                    <div className="flex items-center gap-2 mb-2">
                      <div
                        className="px-2 py-1 rounded text-xs font-mono"
                        style={{ backgroundColor: tokenColors[originalIdx % tokenColors.length] }}
                      >
                        {token}
                      </div>
                      <span className="text-xs text-gray-400">at position {idx}</span>
                    </div>
                    <div className="text-xs font-mono text-purple-300">
                      PE = [{positionalEncodings[idx].slice(0, 4).join(', ')}...]
                    </div>
                  </div>
                );
              })}
            </div>
          </div>

          <div className="mt-4 bg-yellow-500/20 border border-yellow-500/50 rounded p-3 text-center">
            <p className="text-sm text-gray-300">
              💡 <strong>Key Insight:</strong> The same word gets <em>different positional encodings</em> at different positions.
              This is how the Transformer knows word order matters!
            </p>
          </div>
        </motion.div>
      )}

      {/* Educational Notes */}
      <div className="grid md:grid-cols-2 gap-4">
        {/* Formula */}
        <div className="bg-indigo-500/10 border border-indigo-500/30 rounded-lg p-4">
          <h4 className="font-semibold text-indigo-300 mb-2">📐 Mathematical Formula</h4>
          <div className="text-sm text-gray-300 space-y-2">
            <p>For token at position <code className="text-blue-400">i</code>:</p>
            <div className="bg-black/30 rounded p-2 font-mono text-xs">
              X<sub>i</sub> = E<sub>i</sub> + P<sub>i</sub>
            </div>
            <p className="text-xs">Where:</p>
            <ul className="text-xs space-y-1 ml-4">
              <li>• X<sub>i</sub> = Final embedding vector</li>
              <li>• E<sub>i</sub> = Word embedding (learned)</li>
              <li>• P<sub>i</sub> = Positional encoding (fixed)</li>
            </ul>
          </div>
        </div>

        {/* Why It Matters */}
        <div className="bg-green-500/10 border border-green-500/30 rounded-lg p-4">
          <h4 className="font-semibold text-green-300 mb-2">💡 Why This Matters</h4>
          <p className="text-sm text-gray-300">
            Transformers process all tokens in parallel (unlike RNNs). Without positional encoding,
            the model would treat "I eat pizza" identically to "pizza eat I".
            PE ensures the model knows word order!
          </p>
        </div>
      </div>

      {/* Dimension Info */}
      <div className="bg-white/5 border border-white/10 rounded-lg p-4">
        <div className="grid grid-cols-3 gap-4 text-center">
          <div>
            <div className="text-xs text-gray-400 mb-1">Tokens</div>
            <div className="text-2xl font-bold text-blue-400">{displayTokens.length}</div>
          </div>
          <div>
            <div className="text-xs text-gray-400 mb-1">Embedding Dim</div>
            <div className="text-2xl font-bold text-purple-400">{embeddingDim}</div>
          </div>
          <div>
            <div className="text-xs text-gray-400 mb-1">Final Shape</div>
            <div className="text-xl font-bold text-green-400 font-mono">{numTokens} × {embeddingDim}</div>
          </div>
        </div>
      </div>
    </div>
  );
}

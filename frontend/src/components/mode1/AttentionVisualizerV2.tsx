import { motion } from 'framer-motion';
import { useState, useMemo } from 'react';

interface AttentionVisualizerV2Props {
  numHeads: number;
  numLayers: number;
  tokens: string[];
}

export default function AttentionVisualizerV2({ numHeads, numLayers, tokens }: AttentionVisualizerV2Props) {
  const [selectedHead, setSelectedHead] = useState(0);

  // Color palette for tokens (consistent across all panels)
  const tokenColors = [
    '#FF6B6B', // Red
    '#4ECDC4', // Teal
    '#45B7D1', // Blue
    '#FFA07A', // Salmon
    '#98D8C8', // Mint
    '#F7DC6F', // Yellow
  ];

  // Head colors for multi-head visualization
  const headColors = [
    '#3B82F6', // Blue
    '#8B5CF6', // Purple
    '#EC4899', // Pink
    '#F59E0B', // Amber
    '#10B981', // Green
    '#06B6D4', // Cyan
    '#EF4444', // Red
    '#6366F1', // Indigo
  ];

  // Clean tokens
  const displayTokens = useMemo(() =>
    tokens.filter(t => !['<SOS>', '<EOS>', '<PAD>'].includes(t)),
    [tokens]
  );

  // Generate mock attention weights (causal mask)
  const attentionWeights = useMemo(() => {
    return Array(numHeads).fill(0).map(() =>
      displayTokens.map((_, i) => {
        const weights = displayTokens.map((_, j) => {
          if (j > i) return 0; // Causal mask
          if (i === j) return 0.4 + Math.random() * 0.3;
          if (j === i - 1) return 0.3 + Math.random() * 0.2;
          return Math.random() * 0.15;
        });
        const sum = weights.reduce((a, b) => a + b, 0);
        return weights.map(w => w / sum);
      })
    );
  }, [displayTokens, numHeads]);

  // Mock Q, K, V vectors (simplified - showing 4 dimensions)
  const qkvDims = 4;
  const mockQKV = useMemo(() => {
    return displayTokens.map(() => ({
      Q: Array(qkvDims).fill(0).map(() => (Math.random() * 2 - 1).toFixed(2)),
      K: Array(qkvDims).fill(0).map(() => (Math.random() * 2 - 1).toFixed(2)),
      V: Array(qkvDims).fill(0).map(() => (Math.random() * 2 - 1).toFixed(2)),
    }));
  }, [displayTokens]);

  // Vector visualization component
  const VectorBox = ({ values, color, label }: { values: string[]; color: string; label: string }) => (
    <div className="flex flex-col items-center">
      <div className="text-[10px] text-gray-400 mb-1">{label}</div>
      <div className="flex gap-0.5">
        {values.map((val, idx) => (
          <div
            key={idx}
            className="w-5 h-5 rounded flex items-center justify-center text-[8px] font-mono"
            style={{
              backgroundColor: color,
              opacity: 0.3 + Math.abs(parseFloat(val)) * 0.4
            }}
            title={val}
          >
            {parseFloat(val).toFixed(1)}
          </div>
        ))}
      </div>
    </div>
  );

  return (
    <div className="space-y-6">
      {/* Educational Header */}
      <div className="bg-gradient-to-r from-blue-500/10 to-purple-500/10 border border-blue-500/30 rounded-lg p-4">
        <h4 className="font-semibold text-blue-300 mb-2 flex items-center gap-2">
          <span className="text-2xl">🎯</span>
          Self-Attention: Mechanism + Behavior
        </h4>
        <p className="text-sm text-gray-300">
          Self-attention lets every token "look at" all other tokens and decide which ones are most relevant.
          This visualization shows both <strong>how it works</strong> (mechanism) and{' '}
          <strong>what it produces</strong> (attention patterns).
        </p>
      </div>

      {/* Three-Panel Layout */}
      <div className="bg-gradient-to-br from-gray-900/50 to-gray-800/50 rounded-xl border border-white/10 p-6">

        {/* Educational Flow Diagram */}
        <div className="grid lg:grid-cols-3 gap-6">

          {/* ============================================ */}
          {/* PANEL A: Input → Q/K/V Projections */}
          {/* ============================================ */}
          <div className="bg-blue-500/5 border border-blue-500/20 rounded-lg p-4">
            <h5 className="text-sm font-bold text-blue-400 mb-3 text-center">
              Panel A: Embeddings → Q, K, V
            </h5>
            <p className="text-xs text-gray-400 mb-4 text-center italic">
              "Step 1: Linear Projections"
            </p>

            <div className="space-y-6">
              {displayTokens.slice(0, 3).map((token, tokenIdx) => (
                <motion.div
                  key={tokenIdx}
                  initial={{ opacity: 0, x: -20 }}
                  animate={{ opacity: 1, x: 0 }}
                  transition={{ delay: tokenIdx * 0.1 }}
                  className="space-y-2"
                >
                  {/* Token */}
                  <div className="flex items-center gap-2">
                    <div
                      className="px-3 py-1 rounded text-xs font-bold text-white"
                      style={{ backgroundColor: tokenColors[tokenIdx % tokenColors.length] }}
                    >
                      {token}
                    </div>
                    <div className="text-[10px] text-gray-500">Token {tokenIdx + 1}</div>
                  </div>

                  {/* Embedding (simplified) */}
                  <div className="flex items-center gap-2 ml-4">
                    <div className="text-[10px] text-gray-400">Embedding:</div>
                    <div className="flex gap-0.5">
                      {[...Array(4)].map((_, i) => (
                        <div
                          key={i}
                          className="w-3 h-3 rounded"
                          style={{
                            backgroundColor: tokenColors[tokenIdx % tokenColors.length],
                            opacity: 0.3 + Math.random() * 0.4
                          }}
                        />
                      ))}
                    </div>
                  </div>

                  {/* Arrows + Q/K/V */}
                  <div className="ml-8 space-y-1">
                    <div className="flex items-center gap-2">
                      <svg width="20" height="12" className="opacity-60">
                        <path d="M 0 6 L 16 6 M 12 2 L 16 6 L 12 10" stroke="#3B82F6" strokeWidth="1.5" fill="none"/>
                      </svg>
                      <VectorBox values={mockQKV[tokenIdx].Q} color="#3B82F6" label="Q" />
                    </div>
                    <div className="flex items-center gap-2">
                      <svg width="20" height="12" className="opacity-60">
                        <path d="M 0 6 L 16 6 M 12 2 L 16 6 L 12 10" stroke="#10B981" strokeWidth="1.5" fill="none"/>
                      </svg>
                      <VectorBox values={mockQKV[tokenIdx].K} color="#10B981" label="K" />
                    </div>
                    <div className="flex items-center gap-2">
                      <svg width="20" height="12" className="opacity-60">
                        <path d="M 0 6 L 16 6 M 12 2 L 16 6 L 12 10" stroke="#F59E0B" strokeWidth="1.5" fill="none"/>
                      </svg>
                      <VectorBox values={mockQKV[tokenIdx].V} color="#F59E0B" label="V" />
                    </div>
                  </div>
                </motion.div>
              ))}
            </div>

            <div className="mt-4 text-xs text-gray-400 bg-blue-500/10 rounded p-2">
              <strong className="text-blue-400">Q</strong> = Query (What am I looking for?)<br/>
              <strong className="text-green-400">K</strong> = Key (What do I contain?)<br/>
              <strong className="text-yellow-400">V</strong> = Value (What information do I have?)
            </div>
          </div>

          {/* ============================================ */}
          {/* PANEL B: Scaled Dot-Product Attention */}
          {/* ============================================ */}
          <div className="bg-purple-500/5 border border-purple-500/20 rounded-lg p-4">
            <h5 className="text-sm font-bold text-purple-400 mb-3 text-center">
              Panel B: Attention Calculation
            </h5>
            <p className="text-xs text-gray-400 mb-4 text-center italic">
              "Step 2: Q · K<sup>T</sup> → Softmax → ∑ V"
            </p>

            <div className="space-y-4">
              {/* Step 1: Q · K^T */}
              <div className="space-y-2">
                <div className="text-xs font-semibold text-blue-300">1. Similarity Scores (Q · K<sup>T</sup>)</div>
                <div className="bg-black/20 rounded p-2">
                  <div className="grid grid-cols-3 gap-1">
                    {displayTokens.slice(0, 3).map((_, qIdx) => (
                      <div key={qIdx} className="text-center">
                        {displayTokens.slice(0, 3).map((_, kIdx) => (
                          <div
                            key={kIdx}
                            className="h-6 rounded mb-1 flex items-center justify-center text-[10px]"
                            style={{
                              backgroundColor: kIdx <= qIdx ? '#8B5CF6' : '#1F2937',
                              opacity: kIdx <= qIdx ? 0.3 + Math.random() * 0.5 : 0.1
                            }}
                          >
                            {kIdx <= qIdx ? (Math.random() * 5).toFixed(1) : '-'}
                          </div>
                        ))}
                      </div>
                    ))}
                  </div>
                  <p className="text-[10px] text-gray-500 mt-1 text-center">
                    How similar is each Q to each K?
                  </p>
                </div>
              </div>

              {/* Arrow */}
              <div className="flex justify-center">
                <svg width="20" height="30">
                  <path d="M 10 0 L 10 26 M 6 22 L 10 26 L 14 22" stroke="#A78BFA" strokeWidth="2" fill="none"/>
                </svg>
              </div>

              {/* Step 2: Softmax */}
              <div className="space-y-2">
                <div className="text-xs font-semibold text-green-300">2. Softmax (Normalize to Probabilities)</div>
                <div className="bg-black/20 rounded p-2">
                  <div className="space-y-1">
                    {displayTokens.slice(0, 3).map((token, idx) => (
                      <div key={idx}>
                        <div className="text-[10px] text-gray-400 mb-0.5">
                          {token}
                        </div>
                        <div className="flex gap-1">
                          {attentionWeights[selectedHead][idx]?.slice(0, 3).map((weight, wIdx) => (
                            <div
                              key={wIdx}
                              className="flex-1 h-4 rounded flex items-center justify-center text-[9px]"
                              style={{
                                backgroundColor: '#10B981',
                                opacity: 0.3 + weight * 0.7
                              }}
                            >
                              {(weight * 100).toFixed(0)}%
                            </div>
                          ))}
                        </div>
                      </div>
                    ))}
                  </div>
                  <p className="text-[10px] text-gray-500 mt-1 text-center">
                    Attention weights (sum to 100%)
                  </p>
                </div>
              </div>

              {/* Arrow */}
              <div className="flex justify-center">
                <svg width="20" height="30">
                  <path d="M 10 0 L 10 26 M 6 22 L 10 26 L 14 22" stroke="#A78BFA" strokeWidth="2" fill="none"/>
                </svg>
              </div>

              {/* Step 3: Weighted Sum */}
              <div className="space-y-2">
                <div className="text-xs font-semibold text-yellow-300">3. Weighted Sum of Values</div>
                <div className="bg-black/20 rounded p-2 text-center">
                  <div className="flex justify-center gap-1">
                    {[...Array(4)].map((_, i) => (
                      <div
                        key={i}
                        className="w-6 h-6 rounded"
                        style={{
                          backgroundColor: '#F59E0B',
                          opacity: 0.4 + Math.random() * 0.4
                        }}
                      />
                    ))}
                  </div>
                  <p className="text-[10px] text-gray-500 mt-1">
                    Attended representation
                  </p>
                </div>
              </div>
            </div>

            <div className="mt-4 text-xs text-gray-400 bg-purple-500/10 rounded p-2">
              <strong>Formula:</strong> Attention(Q, K, V) = softmax(Q·K<sup>T</sup>/√d<sub>k</sub>) · V
            </div>
          </div>

          {/* ============================================ */}
          {/* PANEL C: Multi-Head + Attention Patterns */}
          {/* ============================================ */}
          <div className="bg-pink-500/5 border border-pink-500/20 rounded-lg p-4">
            <h5 className="text-sm font-bold text-pink-400 mb-3 text-center">
              Panel C: Multi-Head Output
            </h5>
            <p className="text-xs text-gray-400 mb-4 text-center italic">
              "Step 3: Concatenate Heads → Final Output"
            </p>

            {/* Multi-Head Mechanism */}
            <div className="space-y-4">
              <div>
                <div className="text-xs font-semibold text-blue-300 mb-2">Multiple Attention Heads:</div>
                <div className="grid grid-cols-4 gap-2">
                  {[...Array(Math.min(4, numHeads))].map((_, headIdx) => (
                    <motion.button
                      key={headIdx}
                      onClick={() => setSelectedHead(headIdx)}
                      whileHover={{ scale: 1.05 }}
                      className={`p-2 rounded border-2 transition-all ${
                        selectedHead === headIdx
                          ? 'border-white shadow-lg'
                          : 'border-white/20'
                      }`}
                      style={{ backgroundColor: headColors[headIdx] + '40' }}
                    >
                      <div className="text-[10px] font-bold" style={{ color: headColors[headIdx] }}>
                        Head {headIdx + 1}
                      </div>
                      <div className="flex gap-0.5 mt-1 justify-center">
                        {[...Array(3)].map((_, i) => (
                          <div
                            key={i}
                            className="w-2 h-2 rounded"
                            style={{ backgroundColor: headColors[headIdx] }}
                          />
                        ))}
                      </div>
                    </motion.button>
                  ))}
                </div>
              </div>

              {/* Concatenate + Linear */}
              <div className="flex flex-col items-center gap-2">
                <svg width="100" height="30">
                  <path d="M 50 0 L 50 26 M 46 22 L 50 26 L 54 22" stroke="#EC4899" strokeWidth="2" fill="none"/>
                </svg>
                <div className="bg-pink-500/20 border border-pink-500/40 rounded px-3 py-1 text-xs font-semibold">
                  Concatenate
                </div>
                <svg width="100" height="30">
                  <path d="M 50 0 L 50 26 M 46 22 L 50 26 L 54 22" stroke="#EC4899" strokeWidth="2" fill="none"/>
                </svg>
                <div className="bg-purple-500/20 border border-purple-500/40 rounded px-3 py-1 text-xs font-semibold">
                  Linear Projection
                </div>
              </div>

              {/* Attention Pattern Visualization */}
              <div className="mt-4">
                <div className="text-xs font-semibold text-yellow-300 mb-2 text-center">
                  Attention Behavior (Head {selectedHead + 1})
                </div>
                <div className="bg-black/30 rounded p-3">
                  {/* Attention Graph */}
                  <svg width="100%" height="180" viewBox="0 0 250 180">
                    {/* Source tokens (left) */}
                    {displayTokens.map((token, idx) => (
                      <g key={`source-${idx}`}>
                        <rect
                          x="10"
                          y={20 + idx * 40}
                          width="60"
                          height="25"
                          rx="4"
                          fill={tokenColors[idx % tokenColors.length]}
                          opacity="0.8"
                        />
                        <text
                          x="40"
                          y={33 + idx * 40}
                          textAnchor="middle"
                          fill="white"
                          fontSize="11"
                          fontWeight="bold"
                        >
                          {token}
                        </text>
                      </g>
                    ))}

                    {/* Target tokens (right) */}
                    {displayTokens.map((token, idx) => (
                      <g key={`target-${idx}`}>
                        <rect
                          x="180"
                          y={20 + idx * 40}
                          width="60"
                          height="25"
                          rx="4"
                          fill={tokenColors[idx % tokenColors.length]}
                          opacity="0.8"
                        />
                        <text
                          x="210"
                          y={33 + idx * 40}
                          textAnchor="middle"
                          fill="white"
                          fontSize="11"
                          fontWeight="bold"
                        >
                          {token}
                        </text>
                      </g>
                    ))}

                    {/* Attention lines */}
                    {displayTokens.map((_, fromIdx) => (
                      attentionWeights[selectedHead][fromIdx]?.map((weight, toIdx) => {
                        if (weight < 0.01) return null;
                        const y1 = 32 + fromIdx * 40;
                        const y2 = 32 + toIdx * 40;
                        return (
                          <motion.line
                            key={`${fromIdx}-${toIdx}`}
                            x1="70"
                            y1={y1}
                            x2="180"
                            y2={y2}
                            stroke={headColors[selectedHead]}
                            strokeWidth={weight * 8}
                            opacity={0.3 + weight * 0.6}
                            initial={{ pathLength: 0 }}
                            animate={{ pathLength: 1 }}
                            transition={{ duration: 0.5, delay: toIdx * 0.05 }}
                          />
                        );
                      })
                    ))}
                  </svg>

                  <div className="text-[10px] text-gray-400 text-center mt-2">
                    <strong>Line thickness</strong> = attention strength<br/>
                    <span style={{ color: headColors[selectedHead] }}>● Head {selectedHead + 1}</span> attention pattern
                  </div>
                </div>
              </div>
            </div>
          </div>
        </div>

        {/* Educational Notes Below */}
        <div className="mt-6 grid md:grid-cols-3 gap-4 text-xs">
          <div className="bg-blue-500/10 rounded p-3 border-l-4 border-blue-500">
            <strong className="text-blue-400">💡 Key Insight:</strong>
            <p className="text-gray-300 mt-1">
              Self-attention lets every token "look at" all other tokens. The mechanism (Q·K·V) determines
              which tokens are relevant.
            </p>
          </div>
          <div className="bg-purple-500/10 rounded p-3 border-l-4 border-purple-500">
            <strong className="text-purple-400">🎯 Multi-Head:</strong>
            <p className="text-gray-300 mt-1">
              Multiple heads learn different relational patterns. One might focus on syntax, another on semantics.
            </p>
          </div>
          <div className="bg-pink-500/10 rounded p-3 border-l-4 border-pink-500">
            <strong className="text-pink-400">📊 Behavior:</strong>
            <p className="text-gray-300 mt-1">
              The attention pattern (right panel) shows which tokens the model considers important for each position.
            </p>
          </div>
        </div>
      </div>

      {/* Model Info */}
      <div className="grid grid-cols-2 gap-4">
        <div className="bg-white/5 border border-white/10 rounded-lg p-3">
          <div className="text-xs text-gray-400 mb-1">Number of Heads</div>
          <div className="text-2xl font-bold text-blue-400">{numHeads}</div>
          <p className="text-xs text-gray-400 mt-1">Different attention perspectives</p>
        </div>
        <div className="bg-white/5 border border-white/10 rounded-lg p-3">
          <div className="text-xs text-gray-400 mb-1">Number of Layers</div>
          <div className="text-2xl font-bold text-purple-400">{numLayers}</div>
          <p className="text-xs text-gray-400 mt-1">Stacked processing depth</p>
        </div>
      </div>

      {/* Why This Matters */}
      <div className="bg-gradient-to-r from-green-500/10 to-emerald-500/10 border border-green-500/30 rounded-lg p-4">
        <h4 className="font-semibold text-green-300 mb-2">🌟 Why This Matters</h4>
        <p className="text-sm text-gray-300">
          Self-attention is the breakthrough that makes Transformers powerful. Unlike RNNs that process
          sequentially, attention allows <strong>parallel processing</strong> while still capturing
          relationships between distant tokens. Multi-head attention lets the model learn{' '}
          <strong>multiple types of relationships</strong> simultaneously (syntax, semantics, coreference, etc.).
        </p>
      </div>
    </div>
  );
}

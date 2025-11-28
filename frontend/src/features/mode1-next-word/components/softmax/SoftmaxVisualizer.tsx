import { motion } from 'framer-motion';
import { useState } from 'react';

interface Prediction {
  token: string;
  token_id?: number;
  probability: number;
}

interface SoftmaxVisualizerProps {
  topPredictions: Prediction[];
  vocabSize?: number;
}

export default function SoftmaxVisualizer({ topPredictions, vocabSize = 1000 }: SoftmaxVisualizerProps) {
  const [showAll, setShowAll] = useState(false);
  const displayCount = showAll ? topPredictions.length : Math.min(10, topPredictions.length);

  // Generate mock logits for before/after comparison
  const mockLogits = topPredictions.slice(0, 5).map((pred, idx) => ({
    token: pred.token,
    logit: 2.5 + (5 - idx) * 0.8 + Math.random() * 0.5
  }));

  return (
    <div className="space-y-6">
      {/* Educational Description */}
      <div className="bg-yellow-500/10 border border-yellow-500/30 rounded-lg p-4">
        <h4 className="font-semibold text-yellow-300 mb-2">🎯 What's Happening?</h4>
        <p className="text-sm text-gray-300 mb-2">
          After all the processing, we have raw scores (called <strong>logits</strong>) for every word in the
          vocabulary. These can be any number - positive, negative, large, or small.
        </p>
        <p className="text-sm text-gray-300">
          <strong>Softmax</strong> converts these raw scores into probabilities between 0 and 1 that sum to
          100%. Now we can interpret: "There's a 35% chance the next word is 'pizza'".
        </p>
      </div>

      {/* Vocabulary Info */}
      <div className="grid grid-cols-2 gap-4">
        <div className="bg-white/5 border border-white/10 rounded-lg p-3">
          <div className="text-xs text-gray-400 mb-1">Vocabulary Size</div>
          <div className="text-2xl font-bold text-blue-400">{vocabSize.toLocaleString()}</div>
          <div className="text-xs text-gray-400 mt-1">Total possible words</div>
        </div>
        <div className="bg-white/5 border border-white/10 rounded-lg p-3">
          <div className="text-xs text-gray-400 mb-1">Showing Top</div>
          <div className="text-2xl font-bold text-green-400">{displayCount}</div>
          <div className="text-xs text-gray-400 mt-1">Most likely predictions</div>
        </div>
      </div>

      {/* Before/After Softmax Comparison */}
      <div className="bg-black/20 rounded-lg p-4">
        <h4 className="font-semibold mb-3 flex items-center gap-2">
          <span className="text-2xl">🔄</span>
          Softmax Transformation
        </h4>

        <div className="grid md:grid-cols-2 gap-4">
          {/* Before (Logits) */}
          <div>
            <div className="text-sm text-gray-400 mb-2 font-semibold">Before: Raw Scores (Logits)</div>
            <div className="space-y-2">
              {mockLogits.map((item, idx) => (
                <motion.div
                  key={idx}
                  initial={{ opacity: 0, x: -20 }}
                  animate={{ opacity: 1, x: 0 }}
                  transition={{ delay: idx * 0.1 }}
                  className="flex items-center gap-2"
                >
                  <div className="w-20 text-xs font-mono text-gray-300 truncate">
                    {item.token.replace('<SOS>', '▶').replace('<EOS>', '◀').replace('<PAD>', '⊗')}
                  </div>
                  <div className="flex-1 bg-white/10 rounded h-6 flex items-center px-2">
                    <div className="text-xs font-mono text-yellow-400">{item.logit.toFixed(2)}</div>
                  </div>
                </motion.div>
              ))}
            </div>
            <div className="text-xs text-gray-500 mt-2">Can be any value (negative to positive)</div>
          </div>

          {/* After (Probabilities) */}
          <div>
            <div className="text-sm text-gray-400 mb-2 font-semibold">After: Probabilities (%)</div>
            <div className="space-y-2">
              {topPredictions.slice(0, 5).map((pred, idx) => (
                <motion.div
                  key={idx}
                  initial={{ opacity: 0, x: 20 }}
                  animate={{ opacity: 1, x: 0 }}
                  transition={{ delay: idx * 0.1 }}
                  className="flex items-center gap-2"
                >
                  <div className="w-20 text-xs font-mono text-gray-300 truncate">
                    {pred.token.replace('<SOS>', '▶').replace('<EOS>', '◀').replace('<PAD>', '⊗')}
                  </div>
                  <div className="flex-1 bg-white/10 rounded-full h-6 overflow-hidden">
                    <motion.div
                      initial={{ width: 0 }}
                      animate={{ width: `${pred.probability * 100}%` }}
                      transition={{ duration: 0.8, delay: 0.5 + idx * 0.1, ease: 'easeOut' }}
                      className="h-full bg-gradient-to-r from-green-500 to-emerald-400"
                    />
                  </div>
                  <div className="w-16 text-right text-xs font-semibold text-green-400">
                    {(pred.probability * 100).toFixed(1)}%
                  </div>
                </motion.div>
              ))}
            </div>
            <div className="text-xs text-green-500 mt-2">✓ Always between 0-100%, sum to 100%</div>
          </div>
        </div>
      </div>

      {/* Top Predictions with Animated Bars */}
      <div>
        <div className="flex justify-between items-center mb-3">
          <h4 className="font-semibold flex items-center gap-2">
            <span className="text-2xl">📊</span>
            Top Predictions
          </h4>
          {topPredictions.length > 10 && (
            <button
              onClick={() => setShowAll(!showAll)}
              className="text-sm text-blue-400 hover:text-blue-300 underline"
            >
              {showAll ? 'Show Less' : `Show All ${topPredictions.length}`}
            </button>
          )}
        </div>

        <div className="space-y-2">
          {topPredictions.slice(0, displayCount).map((pred, idx) => (
            <motion.div
              key={idx}
              initial={{ opacity: 0, y: 10 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: idx * 0.05 }}
              className="flex items-center gap-3 group"
            >
              {/* Rank */}
              <div className="w-8 text-center">
                <div
                  className={`text-sm font-bold ${
                    idx === 0
                      ? 'text-yellow-400'
                      : idx === 1
                      ? 'text-gray-300'
                      : idx === 2
                      ? 'text-orange-400'
                      : 'text-gray-500'
                  }`}
                >
                  #{idx + 1}
                </div>
              </div>

              {/* Token */}
              <div className="w-32 font-mono text-sm text-gray-200 truncate">
                {pred.token.replace('<SOS>', '▶').replace('<EOS>', '◀').replace('<PAD>', '⊗')}
              </div>

              {/* Probability Bar */}
              <div className="flex-1 bg-white/10 rounded-full h-8 overflow-hidden relative">
                <motion.div
                  initial={{ width: 0 }}
                  animate={{ width: `${pred.probability * 100}%` }}
                  transition={{ duration: 1, delay: idx * 0.05, ease: 'easeOut' }}
                  className={`h-full ${
                    idx === 0
                      ? 'bg-gradient-to-r from-yellow-500 via-orange-500 to-red-500'
                      : idx < 3
                      ? 'bg-gradient-to-r from-blue-500 to-purple-500'
                      : 'bg-gradient-to-r from-blue-600 to-blue-400'
                  }`}
                  style={{
                    boxShadow: idx === 0 ? '0 0 20px rgba(234, 179, 8, 0.5)' : 'none'
                  }}
                />

                {/* Percentage Label Inside Bar */}
                {pred.probability > 0.15 && (
                  <div className="absolute inset-0 flex items-center px-3">
                    <span className="text-white font-semibold text-sm drop-shadow">
                      {(pred.probability * 100).toFixed(1)}%
                    </span>
                  </div>
                )}
              </div>

              {/* Percentage */}
              <div className="w-20 text-right">
                <span
                  className={`font-bold ${
                    idx === 0 ? 'text-yellow-400 text-lg' : 'text-blue-400'
                  }`}
                >
                  {(pred.probability * 100).toFixed(1)}%
                </span>
              </div>

              {/* Winner Badge */}
              {idx === 0 && (
                <motion.div
                  initial={{ scale: 0, rotate: -180 }}
                  animate={{ scale: 1, rotate: 0 }}
                  transition={{ delay: 1, type: 'spring', stiffness: 200 }}
                  className="text-2xl"
                >
                  🏆
                </motion.div>
              )}
            </motion.div>
          ))}
        </div>
      </div>

      {/* Analogy */}
      <div className="bg-indigo-500/10 border border-indigo-500/30 rounded-lg p-4">
        <h4 className="font-semibold text-indigo-300 mb-2">🎲 Analogy: Converting Votes to Percentages</h4>
        <p className="text-sm text-gray-300">
          Imagine a contest where candidates get raw votes: Alice (420), Bob (380), Carol (210).
          To get percentages, divide each by the total (1010): Alice 41.6%, Bob 37.6%, Carol 20.8%.
          Softmax does the same thing but with exponentials (exp) to ensure all values are positive
          and to amplify differences!
        </p>
      </div>

      {/* Why This Matters */}
      <div className="bg-green-500/10 border border-green-500/30 rounded-lg p-4">
        <h4 className="font-semibold text-green-300 mb-2">💡 Why This Matters</h4>
        <p className="text-sm text-gray-300">
          Raw logits aren't meaningful - is 4.2 good? Better than 3.8? By how much? Softmax gives us
          interpretable probabilities: "31% chance of 'pizza'". It transforms arbitrary numbers into
          a valid probability distribution we can use for prediction!
        </p>
      </div>
    </div>
  );
}

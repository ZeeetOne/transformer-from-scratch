import { motion } from 'framer-motion';
import { useState } from 'react';

interface Prediction {
  token: string;
  token_id?: number;
  probability: number;
}

interface PredictionVisualizerProps {
  inputText: string;
  predictedWord: string;
  confidence: number;
  topPredictions: Prediction[];
}

export default function PredictionVisualizer({
  inputText,
  predictedWord,
  confidence,
  topPredictions
}: PredictionVisualizerProps) {
  const [showWheel, setShowWheel] = useState(false);

  // Get top 5 for wheel visualization
  const top5 = topPredictions.slice(0, 5);

  return (
    <div className="space-y-6">
      {/* Educational Description */}
      <div className="bg-purple-500/10 border border-purple-500/30 rounded-lg p-4">
        <h4 className="font-semibold text-purple-300 mb-2">🎯 What's Happening?</h4>
        <p className="text-sm text-gray-300 mb-2">
          Now we have probabilities for all possible next words. Time to make a choice!
          Two strategies:
        </p>
        <ul className="text-sm text-gray-300 list-disc list-inside space-y-1 ml-2">
          <li>
            <strong>Greedy Decoding:</strong> Pick the highest probability word (deterministic)
          </li>
          <li>
            <strong>Sampling:</strong> Randomly sample based on probabilities (creative, varied)
          </li>
        </ul>
      </div>

      {/* Prediction Result with Animation */}
      <div className="bg-gradient-to-br from-green-500/20 to-emerald-500/20 border-2 border-green-500 rounded-xl p-6">
        <h4 className="font-semibold mb-4 text-green-300 flex items-center gap-2">
          <span className="text-2xl">🎯</span>
          Final Prediction
        </h4>

        <div className="space-y-4">
          {/* Input + Prediction */}
          <div className="flex flex-wrap items-center gap-2 text-2xl font-mono">
            <span className="text-gray-300">{inputText}</span>
            <motion.span
              initial={{ opacity: 0, scale: 0.5, y: -20 }}
              animate={{ opacity: 1, scale: 1, y: 0 }}
              transition={{ delay: 0.3, type: 'spring', stiffness: 200 }}
              className="px-4 py-2 bg-gradient-to-r from-green-500 to-emerald-500 rounded-lg shadow-lg text-white font-bold"
            >
              {predictedWord.replace('<SOS>', '▶').replace('<EOS>', '◀').replace('<PAD>', '⊗')}
            </motion.span>
            <motion.span
              initial={{ opacity: 0, scale: 0 }}
              animate={{ opacity: 1, scale: 1 }}
              transition={{ delay: 0.6, type: 'spring', stiffness: 300 }}
              className="text-4xl"
            >
              ✨
            </motion.span>
          </div>

          {/* Confidence */}
          <div className="bg-black/20 rounded-lg p-3">
            <div className="text-sm text-gray-400 mb-2">Confidence Score</div>
            <div className="flex items-center gap-3">
              <div className="flex-1 bg-white/10 rounded-full h-8 overflow-hidden">
                <motion.div
                  initial={{ width: 0 }}
                  animate={{ width: `${confidence * 100}%` }}
                  transition={{ duration: 1, delay: 0.5, ease: 'easeOut' }}
                  className="h-full bg-gradient-to-r from-green-500 via-emerald-500 to-green-400"
                />
              </div>
              <div className="text-2xl font-bold text-green-400 min-w-[80px] text-right">
                {(confidence * 100).toFixed(1)}%
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Selection Strategy Visualization */}
      <div>
        <div className="flex justify-between items-center mb-3">
          <h4 className="font-semibold flex items-center gap-2">
            <span className="text-2xl">🎲</span>
            Selection Strategy
          </h4>
          <button
            onClick={() => setShowWheel(!showWheel)}
            className="px-4 py-2 bg-purple-600 hover:bg-purple-700 rounded-lg text-sm font-semibold transition-colors"
          >
            {showWheel ? '📊 Show Greedy' : '🎡 Show Sampling Wheel'}
          </button>
        </div>

        {!showWheel ? (
          /* Greedy Selection */
          <div className="bg-black/20 rounded-lg p-6">
            <div className="text-center space-y-4">
              <div className="text-sm text-gray-400">
                <strong className="text-blue-400">Greedy Decoding:</strong> Always pick the highest
                probability
              </div>

              <div className="flex justify-center items-center gap-4">
                {top5.map((pred, idx) => (
                  <motion.div
                    key={idx}
                    initial={{ opacity: 0.3, scale: 0.8 }}
                    animate={{
                      opacity: idx === 0 ? 1 : 0.3,
                      scale: idx === 0 ? 1.2 : 0.8,
                      y: idx === 0 ? -10 : 0
                    }}
                    transition={{ delay: idx * 0.1 }}
                    className={`px-4 py-3 rounded-lg border-2 ${
                      idx === 0
                        ? 'bg-gradient-to-br from-yellow-500 to-orange-500 border-yellow-400 shadow-lg shadow-yellow-500/50'
                        : 'bg-white/5 border-white/20'
                    }`}
                  >
                    <div className="font-mono text-sm">
                      {pred.token.replace('<SOS>', '▶').replace('<EOS>', '◀').replace('<PAD>', '⊗')}
                    </div>
                    <div className="text-xs mt-1">
                      {(pred.probability * 100).toFixed(1)}%
                    </div>
                  </motion.div>
                ))}
              </div>

              <motion.div
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: 0.5 }}
                className="text-yellow-400 font-semibold"
              >
                → Selected: <span className="text-white">{top5[0]?.token}</span> (highest probability)
              </motion.div>
            </div>
          </div>
        ) : (
          /* Sampling Wheel */
          <div className="bg-black/20 rounded-lg p-6">
            <div className="text-center space-y-4">
              <div className="text-sm text-gray-400 mb-4">
                <strong className="text-purple-400">Sampling:</strong> Random selection weighted by
                probability
              </div>

              {/* Pie Chart Style Wheel */}
              <div className="relative w-64 h-64 mx-auto">
                <svg viewBox="0 0 200 200" className="transform -rotate-90">
                  {(() => {
                    let cumulativePercent = 0;
                    return top5.map((pred, idx) => {
                      const percent = pred.probability;
                      const startAngle = (cumulativePercent * 360);
                      const endAngle = ((cumulativePercent + percent) * 360);
                      cumulativePercent += percent;

                      // Calculate arc path
                      const startRad = (startAngle * Math.PI) / 180;
                      const endRad = (endAngle * Math.PI) / 180;
                      const x1 = 100 + 90 * Math.cos(startRad);
                      const y1 = 100 + 90 * Math.sin(startRad);
                      const x2 = 100 + 90 * Math.cos(endRad);
                      const y2 = 100 + 90 * Math.sin(endRad);
                      const largeArc = endAngle - startAngle > 180 ? 1 : 0;

                      const colors = [
                        '#3b82f6',
                        '#8b5cf6',
                        '#ec4899',
                        '#f59e0b',
                        '#10b981'
                      ];

                      return (
                        <motion.path
                          key={idx}
                          initial={{ opacity: 0 }}
                          animate={{ opacity: 1 }}
                          transition={{ delay: idx * 0.1 }}
                          d={`M 100 100 L ${x1} ${y1} A 90 90 0 ${largeArc} 1 ${x2} ${y2} Z`}
                          fill={colors[idx]}
                          stroke="#1f2937"
                          strokeWidth="2"
                        />
                      );
                    });
                  })()}

                  {/* Center circle */}
                  <circle cx="100" cy="100" r="30" fill="#1f2937" />
                </svg>

                {/* Spinning pointer */}
                <motion.div
                  className="absolute top-1/2 left-1/2 transform -translate-x-1/2 -translate-y-1/2"
                  animate={{ rotate: 360 }}
                  transition={{ duration: 2, repeat: Infinity, ease: 'linear' }}
                >
                  <div className="w-2 h-24 bg-white rounded-full shadow-lg" />
                </motion.div>
              </div>

              {/* Legend */}
              <div className="grid grid-cols-2 gap-2 text-xs">
                {top5.map((pred, idx) => {
                  const colors = ['bg-blue-500', 'bg-purple-500', 'bg-pink-500', 'bg-yellow-500', 'bg-green-500'];
                  return (
                    <div key={idx} className="flex items-center gap-2">
                      <div className={`w-3 h-3 rounded ${colors[idx]}`} />
                      <span className="font-mono">
                        {pred.token.replace('<SOS>', '▶').replace('<EOS>', '◀').replace('<PAD>', '⊗')}
                      </span>
                      <span className="text-gray-400">({(pred.probability * 100).toFixed(1)}%)</span>
                    </div>
                  );
                })}
              </div>

              <div className="text-purple-400 text-sm">
                🎲 Each spin has a different outcome based on probabilities!
              </div>
            </div>
          </div>
        )}
      </div>

      {/* Next Steps */}
      <div className="bg-blue-500/10 border border-blue-500/30 rounded-lg p-4">
        <h4 className="font-semibold text-blue-300 mb-2">🔄 What Happens Next?</h4>
        <p className="text-sm text-gray-300">
          The predicted word "<strong>{predictedWord}</strong>" is added to your sequence:
          <strong className="text-blue-400"> {inputText} {predictedWord}</strong>.
          We can feed this back into the model to predict the <em>next</em> word, and keep going
          to generate complete sentences! This is called <strong>autoregressive generation</strong>.
        </p>
      </div>

      {/* Analogy */}
      <div className="bg-indigo-500/10 border border-indigo-500/30 rounded-lg p-4">
        <h4 className="font-semibold text-indigo-300 mb-2">🎡 Analogy: Weighted Lottery Wheel</h4>
        <p className="text-sm text-gray-300">
          Imagine a lottery wheel where each slice's size matches the probability. Greedy = always
          picking the biggest slice. Sampling = spinning the wheel and taking whatever it lands on.
          Greedy is safe but repetitive; sampling is creative but unpredictable!
        </p>
      </div>

      {/* Why This Matters */}
      <div className="bg-green-500/10 border border-green-500/30 rounded-lg p-4">
        <h4 className="font-semibold text-green-300 mb-2">💡 Why This Matters</h4>
        <p className="text-sm text-gray-300">
          This is where all the computation pays off - the actual prediction! The choice between greedy
          vs. sampling dramatically affects output: greedy is deterministic (same input = same output),
          while sampling adds creativity and variety. This is how GPT and other language models generate
          text, one token at a time!
        </p>
      </div>
    </div>
  );
}

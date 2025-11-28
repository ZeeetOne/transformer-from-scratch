import { motion } from 'framer-motion';

interface TokenizationVisualizerProps {
  tokens: string[];
  tokenIds: number[];
  inputText: string;
}

export default function TokenizationVisualizer({ tokens, tokenIds, inputText }: TokenizationVisualizerProps) {
  return (
    <div className="space-y-6">
      {/* Educational Description */}
      <div className="bg-blue-500/10 border border-blue-500/30 rounded-lg p-4">
        <h4 className="font-semibold text-blue-300 mb-2">🎯 What's Happening?</h4>
        <p className="text-sm text-gray-300">
          The computer can't read text like humans do. <strong>Tokenization</strong> converts your sentence "
          <span className="text-blue-400">{inputText}</span>" into a numbered list. Each token (word/character)
          gets a unique ID number from our vocabulary dictionary.
        </p>
      </div>

      {/* Original Text */}
      <div>
        <h4 className="font-semibold mb-3 flex items-center gap-2">
          <span className="text-2xl">📝</span>
          Original Text
        </h4>
        <div className="bg-white/5 border border-white/20 rounded-lg p-4">
          <p className="text-xl font-mono">{inputText}</p>
        </div>
      </div>

      {/* Token Visualization */}
      <div>
        <h4 className="font-semibold mb-3 flex items-center gap-2">
          <span className="text-2xl">🔤</span>
          Tokens (Words/Characters)
        </h4>
        <div className="flex flex-wrap gap-3">
          {tokens.map((token, idx) => (
            <motion.div
              key={idx}
              initial={{ opacity: 0, scale: 0.5, y: -20 }}
              animate={{ opacity: 1, scale: 1, y: 0 }}
              transition={{ delay: idx * 0.1, type: 'spring', stiffness: 200 }}
              className="relative group"
            >
              <div className="px-4 py-3 bg-gradient-to-br from-blue-500/30 to-purple-500/30 border-2 border-blue-500 rounded-lg shadow-lg hover:shadow-xl transition-shadow">
                <div className="text-center">
                  <div className="text-lg font-mono font-semibold">
                    {token.replace('<SOS>', '▶').replace('<EOS>', '◀').replace('<PAD>', '⊗')}
                  </div>
                  <div className="text-xs text-gray-400 mt-1">Token {idx + 1}</div>
                </div>
              </div>

              {/* Tooltip */}
              <div className="absolute bottom-full left-1/2 transform -translate-x-1/2 mb-2 px-2 py-1 bg-black text-white text-xs rounded opacity-0 group-hover:opacity-100 transition-opacity whitespace-nowrap pointer-events-none">
                {token}
              </div>
            </motion.div>
          ))}
        </div>
      </div>

      {/* Animated Arrow */}
      <div className="flex justify-center">
        <motion.div
          animate={{ y: [0, 10, 0] }}
          transition={{ duration: 1.5, repeat: Infinity }}
          className="text-4xl text-blue-400"
        >
          ↓
        </motion.div>
      </div>

      {/* Token IDs */}
      <div>
        <h4 className="font-semibold mb-3 flex items-center gap-2">
          <span className="text-2xl">🔢</span>
          Token IDs (Computer-Readable Numbers)
        </h4>
        <div className="flex flex-wrap gap-3">
          {tokenIds.map((id, idx) => (
            <motion.div
              key={idx}
              initial={{ opacity: 0, scale: 0.5, y: 20 }}
              animate={{ opacity: 1, scale: 1, y: 0 }}
              transition={{ delay: 0.5 + idx * 0.1, type: 'spring', stiffness: 200 }}
              className="relative group"
            >
              <div className="px-4 py-3 bg-gradient-to-br from-purple-500/30 to-pink-500/30 border-2 border-purple-500 rounded-lg shadow-lg hover:shadow-xl transition-shadow">
                <div className="text-center">
                  <div className="text-lg font-mono font-bold text-purple-300">{id}</div>
                  <div className="text-xs text-gray-400 mt-1">ID {idx + 1}</div>
                </div>
              </div>

              {/* Tooltip showing mapping */}
              <div className="absolute top-full left-1/2 transform -translate-x-1/2 mt-2 px-2 py-1 bg-black text-white text-xs rounded opacity-0 group-hover:opacity-100 transition-opacity whitespace-nowrap pointer-events-none z-10">
                "{tokens[idx]}" → {id}
              </div>
            </motion.div>
          ))}
        </div>
      </div>

      {/* Why This Matters */}
      <div className="bg-green-500/10 border border-green-500/30 rounded-lg p-4">
        <h4 className="font-semibold text-green-300 mb-2">💡 Why This Matters</h4>
        <p className="text-sm text-gray-300">
          These numbers are the "language" the Transformer speaks. Just like Morse code converts letters to
          dots and dashes, tokenization converts text to numbers. The model will now process these numbers
          through its neural network layers!
        </p>
      </div>

      {/* Shape Information */}
      <div className="grid grid-cols-2 gap-4">
        <div className="bg-white/5 border border-white/10 rounded-lg p-3">
          <div className="text-xs text-gray-400 mb-1">Number of Tokens</div>
          <div className="text-2xl font-bold text-blue-400">{tokens.length}</div>
        </div>
        <div className="bg-white/5 border border-white/10 rounded-lg p-3">
          <div className="text-xs text-gray-400 mb-1">Shape</div>
          <div className="text-2xl font-bold text-purple-400 font-mono">({tokens.length},)</div>
        </div>
      </div>
    </div>
  );
}

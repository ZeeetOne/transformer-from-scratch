import { useMemo } from 'react';
import { InferenceResponse } from '../services/api';

interface ArchitectureDiagramProps {
  data: InferenceResponse;
}

export default function ArchitectureDiagram({ data }: ArchitectureDiagramProps) {
  const archInfo = useMemo(() => {
    const vizData = data.visualization_data;
    const encoder = vizData?.encoder;
    const decoder = vizData?.decoder;

    return {
      dModel: vizData?.architecture_info?.d_model || 256,
      nEncoderLayers: encoder?.n_layers || 0,
      nDecoderLayers: decoder?.n_layers || 0,
      sourceLen: data.source_tokens?.length || 0,
      targetLen: data.target_tokens?.length || 0,
      sourceText: data.source_text || '',
      decodedOutput: data.decoded_output || '',
    };
  }, [data]);

  return (
    <div className="card">
      <h2 className="text-xl font-bold mb-4 text-gray-800 dark:text-white">
        Transformer Architecture
      </h2>

      {/* Input/Output Display */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mb-6">
        <div className="p-4 bg-green-50 dark:bg-green-900/20 rounded-lg border border-green-200 dark:border-green-800">
          <h3 className="text-sm font-semibold text-green-900 dark:text-green-100 mb-2">
            📥 Input
          </h3>
          <p className="text-green-700 dark:text-green-300 font-mono text-sm break-all">
            "{archInfo.sourceText}"
          </p>
          <p className="text-xs text-green-600 dark:text-green-400 mt-2">
            {archInfo.sourceLen} tokens
          </p>
        </div>
        <div className="p-4 bg-blue-50 dark:bg-blue-900/20 rounded-lg border border-blue-200 dark:border-blue-800">
          <h3 className="text-sm font-semibold text-blue-900 dark:text-blue-100 mb-2">
            📤 Output
          </h3>
          <p className="text-blue-700 dark:text-blue-300 font-mono text-sm break-all">
            "{archInfo.decodedOutput}"
          </p>
          <p className="text-xs text-blue-600 dark:text-blue-400 mt-2">
            {archInfo.targetLen} tokens
          </p>
        </div>
      </div>

      {/* Architecture Diagram */}
      <div className="relative">
        {/* Encoder Side */}
        <div className="flex flex-col items-center mb-8">
          <div className="text-lg font-semibold text-gray-800 dark:text-white mb-4">
            ENCODER
          </div>

          {/* Input Embedding */}
          <div className="w-full max-w-md mb-3">
            <div className="bg-gradient-to-r from-green-400 to-green-500 text-white p-4 rounded-lg shadow-lg text-center">
              <div className="font-semibold">Input Tokens</div>
              <div className="text-sm opacity-90 mt-1">
                {archInfo.sourceLen} × {archInfo.dModel}
              </div>
            </div>
          </div>

          {/* Positional Encoding */}
          <div className="text-2xl mb-3">⬇️</div>
          <div className="w-full max-w-md mb-3">
            <div className="bg-gradient-to-r from-purple-400 to-purple-500 text-white p-4 rounded-lg shadow-lg text-center">
              <div className="font-semibold">+ Positional Encoding</div>
              <div className="text-sm opacity-90 mt-1">Sinusoidal patterns</div>
            </div>
          </div>

          {/* Encoder Layers */}
          <div className="text-2xl mb-3">⬇️</div>
          <div className="w-full max-w-md space-y-3">
            {Array.from({ length: archInfo.nEncoderLayers }).map((_, idx) => (
              <div key={idx}>
                <div className="bg-gradient-to-r from-blue-400 to-blue-500 text-white p-4 rounded-lg shadow-lg">
                  <div className="font-semibold text-center mb-2">
                    Encoder Layer {idx}
                  </div>
                  <div className="space-y-2 text-sm">
                    <div className="bg-white/20 rounded p-2">
                      🎯 Multi-Head Self-Attention
                    </div>
                    <div className="bg-white/20 rounded p-2">
                      ⚡ Feed-Forward Network
                    </div>
                  </div>
                </div>
                {idx < archInfo.nEncoderLayers - 1 && (
                  <div className="text-2xl text-center my-2">⬇️</div>
                )}
              </div>
            ))}
          </div>

          {/* Encoder Output */}
          <div className="text-2xl my-3">⬇️</div>
          <div className="w-full max-w-md">
            <div className="bg-gradient-to-r from-cyan-400 to-cyan-500 text-white p-4 rounded-lg shadow-lg text-center">
              <div className="font-semibold">Encoder Output</div>
              <div className="text-sm opacity-90 mt-1">Contextualized representations</div>
            </div>
          </div>
        </div>

        {/* Decoder Side (if present) */}
        {archInfo.nDecoderLayers > 0 && (
          <div className="flex flex-col items-center border-t-4 border-gray-300 dark:border-gray-600 pt-8">
            <div className="text-lg font-semibold text-gray-800 dark:text-white mb-4">
              DECODER
            </div>

            {/* Target Embedding */}
            <div className="w-full max-w-md mb-3">
              <div className="bg-gradient-to-r from-orange-400 to-orange-500 text-white p-4 rounded-lg shadow-lg text-center">
                <div className="font-semibold">Target Tokens (shifted)</div>
                <div className="text-sm opacity-90 mt-1">
                  {archInfo.targetLen} × {archInfo.dModel}
                </div>
              </div>
            </div>

            {/* Decoder Layers */}
            <div className="text-2xl mb-3">⬇️</div>
            <div className="w-full max-w-md space-y-3">
              {Array.from({ length: archInfo.nDecoderLayers }).map((_, idx) => (
                <div key={idx}>
                  <div className="bg-gradient-to-r from-pink-400 to-pink-500 text-white p-4 rounded-lg shadow-lg">
                    <div className="font-semibold text-center mb-2">
                      Decoder Layer {idx}
                    </div>
                    <div className="space-y-2 text-sm">
                      <div className="bg-white/20 rounded p-2">
                        🔒 Masked Self-Attention
                      </div>
                      <div className="bg-white/20 rounded p-2">
                        🔗 Cross-Attention (to Encoder)
                      </div>
                      <div className="bg-white/20 rounded p-2">
                        ⚡ Feed-Forward Network
                      </div>
                    </div>
                  </div>
                  {idx < archInfo.nDecoderLayers - 1 && (
                    <div className="text-2xl text-center my-2">⬇️</div>
                  )}
                </div>
              ))}
            </div>

            {/* Output Projection */}
            <div className="text-2xl my-3">⬇️</div>
            <div className="w-full max-w-md">
              <div className="bg-gradient-to-r from-red-400 to-red-500 text-white p-4 rounded-lg shadow-lg text-center">
                <div className="font-semibold">Output Projection</div>
                <div className="text-sm opacity-90 mt-1">Softmax over vocabulary</div>
              </div>
            </div>
          </div>
        )}
      </div>

      {/* Model Statistics */}
      <div className="mt-8 grid grid-cols-2 md:grid-cols-4 gap-4">
        <div className="bg-gray-100 dark:bg-gray-700 p-4 rounded-lg text-center">
          <div className="text-2xl font-bold text-gray-800 dark:text-white">
            {archInfo.dModel}
          </div>
          <div className="text-xs text-gray-600 dark:text-gray-400 mt-1">Model Dim</div>
        </div>
        <div className="bg-gray-100 dark:bg-gray-700 p-4 rounded-lg text-center">
          <div className="text-2xl font-bold text-gray-800 dark:text-white">
            {archInfo.nEncoderLayers}
          </div>
          <div className="text-xs text-gray-600 dark:text-gray-400 mt-1">Enc Layers</div>
        </div>
        <div className="bg-gray-100 dark:bg-gray-700 p-4 rounded-lg text-center">
          <div className="text-2xl font-bold text-gray-800 dark:text-white">
            {archInfo.nDecoderLayers}
          </div>
          <div className="text-xs text-gray-600 dark:text-gray-400 mt-1">Dec Layers</div>
        </div>
        <div className="bg-gray-100 dark:bg-gray-700 p-4 rounded-lg text-center">
          <div className="text-2xl font-bold text-gray-800 dark:text-white">
            {data.mode === 'generation' ? '🤖' : '👨‍🏫'}
          </div>
          <div className="text-xs text-gray-600 dark:text-gray-400 mt-1">
            {data.mode === 'generation' ? 'Generate' : 'Teacher'}
          </div>
        </div>
      </div>

      {/* Educational Explanation */}
      <div className="mt-6 p-4 bg-indigo-50 dark:bg-indigo-900/20 rounded-lg border border-indigo-200 dark:border-indigo-800">
        <h3 className="text-sm font-semibold text-indigo-900 dark:text-indigo-100 mb-2">
          🏗️ Architecture Overview
        </h3>
        <p className="text-xs text-indigo-700 dark:text-indigo-300 leading-relaxed">
          The transformer has two main components: <strong>Encoder</strong> (processes input)
          and <strong>Decoder</strong> (generates output). Each layer refines the representations
          using attention mechanisms. The encoder creates contextual embeddings, while the decoder
          generates tokens autoregressively, attending to both its own history and the encoder output.
        </p>
      </div>
    </div>
  );
}

import { useState } from 'react';
import { Link } from 'react-router-dom';
import { predictNextWord } from '../services/api';

interface PredictionStep {
  step: number;
  title: string;
  description: string;
  icon: string;
  data?: any;
}

interface PredictionResult {
  input_text: string;
  predicted_token: string;
  predicted_word: string;
  confidence: number;
  top_predictions: Array<{ token: string; probability: number }>;
  steps: {
    tokenization: { tokens: string[]; token_ids: number[] };
    embeddings: { shape: number[]; sample_values: number[][] };
    attention: { num_heads: number; num_layers: number; attention_shape: number[] };
    feedforward: { hidden_dim: number; output_shape: number[] };
    output: { logits_shape: number[]; softmax_shape: number[] };
  };
}

export default function Mode1() {
  const [inputText, setInputText] = useState('I eat');
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<PredictionResult | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [currentStep, setCurrentStep] = useState(0);

  const handlePredict = async () => {
    if (!inputText.trim()) {
      setError('Please enter some text');
      return;
    }

    setLoading(true);
    setError(null);

    try {
      const response = await predictNextWord(inputText);
      setResult(response);
      setCurrentStep(0);
    } catch (err: any) {
      setError(err.message || 'Failed to predict next word');
      setResult(null);
    } finally {
      setLoading(false);
    }
  };

  const steps: PredictionStep[] = result
    ? [
        {
          step: 1,
          title: 'Tokenization',
          description: 'Split input text into tokens (words or characters)',
          icon: '📝',
          data: result.steps.tokenization
        },
        {
          step: 2,
          title: 'Embedding + Positional Encoding',
          description: 'Convert tokens to vectors and add position information',
          icon: '🔢',
          data: result.steps.embeddings
        },
        {
          step: 3,
          title: 'Self-Attention & Multi-Head Attention',
          description: 'Allow tokens to attend to previous tokens',
          icon: '🎯',
          data: result.steps.attention
        },
        {
          step: 4,
          title: 'Feedforward Network',
          description: 'Process attended representations through neural network',
          icon: '🧠',
          data: result.steps.feedforward
        },
        {
          step: 5,
          title: 'Output Layer (Softmax)',
          description: 'Convert to probability distribution over vocabulary',
          icon: '📊',
          data: result.steps.output
        },
        {
          step: 6,
          title: 'Next Token Prediction',
          description: 'Select the most likely next token',
          icon: '🎲',
          data: {
            predicted: result.predicted_word,
            confidence: result.confidence,
            alternatives: result.top_predictions
          }
        }
      ]
    : [];

  return (
    <div className="min-h-screen bg-gradient-to-br from-gray-900 via-blue-900 to-gray-900 text-white">
      {/* Header */}
      <header className="bg-black/30 backdrop-blur-sm border-b border-white/10">
        <div className="container mx-auto px-6 py-4">
          <div className="flex items-center justify-between">
            <div>
              <Link
                to="/applications"
                className="text-sm text-blue-400 hover:text-blue-300 mb-2 inline-block"
              >
                ← Back to Applications
              </Link>
              <h1 className="text-3xl font-bold bg-clip-text text-transparent bg-gradient-to-r from-blue-400 to-purple-400">
                Mode 1: Next Word Prediction
              </h1>
              <p className="text-gray-300 mt-2">
                Like Mini-GPT: Predict the next word based on context
              </p>
            </div>
          </div>
        </div>
      </header>

      {/* Main Content */}
      <div className="container mx-auto px-6 py-8">
        <div className="max-w-6xl mx-auto">
          <div className="grid lg:grid-cols-3 gap-6">
            {/* Left Panel - Input */}
            <div className="lg:col-span-1">
              <div className="bg-white/10 backdrop-blur-md rounded-xl border border-white/20 p-6">
                <h2 className="text-xl font-bold mb-4">Input</h2>

                <div className="mb-4">
                  <label className="block text-sm font-medium mb-2 text-gray-300">
                    Enter initial text:
                  </label>
                  <input
                    type="text"
                    value={inputText}
                    onChange={(e) => setInputText(e.target.value)}
                    className="w-full px-4 py-3 bg-white/5 border border-white/20 rounded-lg text-white placeholder-gray-400 focus:outline-none focus:ring-2 focus:ring-blue-500"
                    placeholder="e.g., I eat"
                    onKeyPress={(e) => e.key === 'Enter' && handlePredict()}
                  />
                </div>

                <button
                  onClick={handlePredict}
                  disabled={loading}
                  className="w-full px-6 py-3 bg-gradient-to-r from-blue-600 to-purple-600 hover:from-blue-700 hover:to-purple-700 disabled:from-gray-600 disabled:to-gray-700 text-white font-semibold rounded-lg transition-all duration-200 shadow-lg hover:shadow-xl disabled:cursor-not-allowed"
                >
                  {loading ? (
                    <span className="flex items-center justify-center">
                      <svg className="animate-spin h-5 w-5 mr-2" viewBox="0 0 24 24">
                        <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" fill="none" />
                        <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z" />
                      </svg>
                      Predicting...
                    </span>
                  ) : (
                    'Predict Next Word'
                  )}
                </button>

                {error && (
                  <div className="mt-4 p-3 bg-red-500/20 border border-red-500/50 rounded-lg text-red-200 text-sm">
                    {error}
                  </div>
                )}

                {/* Example Inputs */}
                <div className="mt-6">
                  <h3 className="text-sm font-medium mb-2 text-gray-300">Try these examples:</h3>
                  <div className="space-y-2">
                    {['I eat', 'The cat', 'Hello', 'Machine learning'].map((example) => (
                      <button
                        key={example}
                        onClick={() => setInputText(example)}
                        className="w-full text-left px-3 py-2 bg-white/5 hover:bg-white/10 border border-white/10 rounded text-sm transition-colors"
                      >
                        {example}
                      </button>
                    ))}
                  </div>
                </div>

                {/* Result Summary */}
                {result && (
                  <div className="mt-6 p-4 bg-green-500/20 border border-green-500/50 rounded-lg">
                    <h3 className="font-semibold mb-2 text-green-200">Prediction Result:</h3>
                    <div className="text-2xl font-bold text-white mb-2">
                      {result.input_text} <span className="text-green-400">{result.predicted_word}</span>
                    </div>
                    <div className="text-sm text-gray-300">
                      Confidence: {(result.confidence * 100).toFixed(1)}%
                    </div>
                  </div>
                )}
              </div>
            </div>

            {/* Right Panel - Visualization */}
            <div className="lg:col-span-2">
              {!result && !loading && (
                <div className="bg-white/10 backdrop-blur-md rounded-xl border border-white/20 p-12 text-center">
                  <div className="text-6xl mb-4">🔮</div>
                  <h2 className="text-2xl font-bold mb-2">Ready to Predict</h2>
                  <p className="text-gray-300">
                    Enter some text and click "Predict Next Word" to see the step-by-step process
                  </p>
                </div>
              )}

              {loading && (
                <div className="bg-white/10 backdrop-blur-md rounded-xl border border-white/20 p-12 text-center">
                  <div className="animate-spin rounded-full h-16 w-16 border-b-2 border-blue-500 mx-auto mb-4"></div>
                  <p className="text-gray-300">Processing prediction...</p>
                </div>
              )}

              {result && (
                <div className="space-y-4">
                  {/* Step Navigator */}
                  <div className="bg-white/10 backdrop-blur-md rounded-xl border border-white/20 p-4">
                    <div className="flex items-center justify-between mb-3">
                      <h3 className="font-semibold">Processing Steps</h3>
                      <div className="text-sm text-gray-300">
                        Step {currentStep + 1} of {steps.length}
                      </div>
                    </div>
                    <div className="grid grid-cols-6 gap-2">
                      {steps.map((step, idx) => (
                        <button
                          key={idx}
                          onClick={() => setCurrentStep(idx)}
                          className={`p-3 rounded-lg transition-all ${
                            currentStep === idx
                              ? 'bg-blue-600 text-white scale-110 shadow-lg'
                              : 'bg-white/5 hover:bg-white/10'
                          }`}
                          title={step.title}
                        >
                          <div className="text-2xl">{step.icon}</div>
                          <div className="text-xs mt-1">{step.step}</div>
                        </button>
                      ))}
                    </div>
                  </div>

                  {/* Current Step Details */}
                  <div className="bg-white/10 backdrop-blur-md rounded-xl border border-white/20 p-6">
                    <div className="flex items-start mb-4">
                      <div className="text-5xl mr-4">{steps[currentStep].icon}</div>
                      <div>
                        <h2 className="text-2xl font-bold mb-2">
                          Step {steps[currentStep].step}: {steps[currentStep].title}
                        </h2>
                        <p className="text-gray-300">{steps[currentStep].description}</p>
                      </div>
                    </div>

                    {/* Step-specific visualization */}
                    <div className="mt-6 p-4 bg-black/20 rounded-lg">
                      {currentStep === 0 && (
                        <div>
                          <h4 className="font-semibold mb-3">Tokens:</h4>
                          <div className="flex flex-wrap gap-2 mb-4">
                            {steps[0].data.tokens.map((token: string, idx: number) => (
                              <span key={idx} className="px-3 py-1 bg-blue-500/30 border border-blue-500 rounded text-sm">
                                {token}
                              </span>
                            ))}
                          </div>
                          <h4 className="font-semibold mb-3">Token IDs:</h4>
                          <div className="flex flex-wrap gap-2">
                            {steps[0].data.token_ids.map((id: number, idx: number) => (
                              <span key={idx} className="px-3 py-1 bg-purple-500/30 border border-purple-500 rounded text-sm font-mono">
                                {id}
                              </span>
                            ))}
                          </div>
                        </div>
                      )}

                      {currentStep === 1 && (
                        <div>
                          <h4 className="font-semibold mb-3">Embedding Shape:</h4>
                          <p className="text-gray-300 mb-4">
                            {steps[1].data.shape.join(' × ')}
                            <span className="text-sm text-gray-400 ml-2">
                              (sequence_length × embedding_dimension)
                            </span>
                          </p>
                          <h4 className="font-semibold mb-3">Sample Embedding Values:</h4>
                          <div className="grid grid-cols-8 gap-1 max-h-40 overflow-auto">
                            {steps[1].data.sample_values.flat().slice(0, 64).map((val: number, idx: number) => (
                              <div
                                key={idx}
                                className="h-8 rounded"
                                style={{
                                  backgroundColor: `rgba(59, 130, 246, ${Math.abs(val)})`
                                }}
                                title={val.toFixed(4)}
                              />
                            ))}
                          </div>
                        </div>
                      )}

                      {currentStep === 2 && (
                        <div>
                          <div className="grid grid-cols-2 gap-4 mb-4">
                            <div>
                              <h4 className="font-semibold mb-2">Number of Heads:</h4>
                              <p className="text-3xl font-bold text-blue-400">{steps[2].data.num_heads}</p>
                            </div>
                            <div>
                              <h4 className="font-semibold mb-2">Number of Layers:</h4>
                              <p className="text-3xl font-bold text-purple-400">{steps[2].data.num_layers}</p>
                            </div>
                          </div>
                          <h4 className="font-semibold mb-3">Attention Output Shape:</h4>
                          <p className="text-gray-300">
                            {steps[2].data.attention_shape.join(' × ')}
                          </p>
                        </div>
                      )}

                      {currentStep === 3 && (
                        <div>
                          <h4 className="font-semibold mb-3">Hidden Dimension:</h4>
                          <p className="text-3xl font-bold text-green-400 mb-4">
                            {steps[3].data.hidden_dim}
                          </p>
                          <h4 className="font-semibold mb-3">Output Shape:</h4>
                          <p className="text-gray-300">
                            {steps[3].data.output_shape.join(' × ')}
                          </p>
                        </div>
                      )}

                      {currentStep === 4 && (
                        <div>
                          <h4 className="font-semibold mb-3">Logits Shape:</h4>
                          <p className="text-gray-300 mb-4">
                            {steps[4].data.logits_shape.join(' × ')}
                            <span className="text-sm text-gray-400 ml-2">
                              (sequence_length × vocabulary_size)
                            </span>
                          </p>
                          <h4 className="font-semibold mb-3">After Softmax:</h4>
                          <p className="text-gray-300">
                            {steps[4].data.softmax_shape.join(' × ')}
                            <span className="text-sm text-gray-400 ml-2">
                              (probabilities sum to 1.0)
                            </span>
                          </p>
                        </div>
                      )}

                      {currentStep === 5 && (
                        <div>
                          <h4 className="font-semibold mb-3">Top Predictions:</h4>
                          <div className="space-y-2">
                            {steps[5].data.alternatives.slice(0, 5).map((pred: any, idx: number) => (
                              <div key={idx} className="flex items-center">
                                <div className="w-32 font-mono text-sm">
                                  {pred.token}
                                </div>
                                <div className="flex-1">
                                  <div className="h-6 bg-white/10 rounded-full overflow-hidden">
                                    <div
                                      className={`h-full ${idx === 0 ? 'bg-green-500' : 'bg-blue-500'} transition-all`}
                                      style={{ width: `${pred.probability * 100}%` }}
                                    />
                                  </div>
                                </div>
                                <div className="w-20 text-right text-sm">
                                  {(pred.probability * 100).toFixed(1)}%
                                </div>
                              </div>
                            ))}
                          </div>
                        </div>
                      )}
                    </div>

                    {/* Navigation Buttons */}
                    <div className="flex justify-between mt-6">
                      <button
                        onClick={() => setCurrentStep(Math.max(0, currentStep - 1))}
                        disabled={currentStep === 0}
                        className="px-4 py-2 bg-white/10 hover:bg-white/20 disabled:bg-white/5 disabled:text-gray-500 rounded-lg transition-colors"
                      >
                        ← Previous
                      </button>
                      <button
                        onClick={() => setCurrentStep(Math.min(steps.length - 1, currentStep + 1))}
                        disabled={currentStep === steps.length - 1}
                        className="px-4 py-2 bg-white/10 hover:bg-white/20 disabled:bg-white/5 disabled:text-gray-500 rounded-lg transition-colors"
                      >
                        Next →
                      </button>
                    </div>
                  </div>
                </div>
              )}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

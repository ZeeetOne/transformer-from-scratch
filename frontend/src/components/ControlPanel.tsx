import { useState } from 'react';
import api, { InferenceRequest, InferenceResponse } from '../services/api';

interface ControlPanelProps {
  onInferenceComplete: (data: InferenceResponse) => void;
  onLoadingChange: (loading: boolean) => void;
}

const EXAMPLE_PROMPTS = [
  'Hello world',
  'The quick brown fox',
  'Transformers are amazing',
  'I love machine learning',
];

export default function ControlPanel({ onInferenceComplete, onLoadingChange }: ControlPanelProps) {
  const [sourceText, setSourceText] = useState('Hello world');
  const [targetText, setTargetText] = useState('');
  const [generate, setGenerate] = useState(true);
  const [maxGenLen, setMaxGenLen] = useState(20);
  const [error, setError] = useState<string | null>(null);

  const handleRunInference = async () => {
    if (!sourceText.trim()) {
      setError('Please enter some text');
      return;
    }

    setError(null);
    onLoadingChange(true);

    const request: InferenceRequest = {
      source_text: sourceText,
      target_text: targetText || undefined,
      generate,
      max_gen_len: maxGenLen,
    };

    try {
      const result = await api.runInference(request);
      onInferenceComplete(result);
    } catch (err: any) {
      setError(err.response?.data?.detail || err.message || 'Inference failed');
      onLoadingChange(false);
    }
  };

  const loadExample = (example: string) => {
    setSourceText(example);
    setError(null);
  };

  return (
    <div className="card">
      <h2 className="text-xl font-bold mb-4 text-gray-800 dark:text-white">
        Control Panel
      </h2>

      {/* Source Text Input */}
      <div className="mb-4">
        <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
          Source Text
        </label>
        <textarea
          value={sourceText}
          onChange={(e) => setSourceText(e.target.value)}
          className="w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-lg focus:ring-2 focus:ring-primary-500 focus:border-transparent bg-white dark:bg-gray-700 text-gray-900 dark:text-white"
          rows={3}
          placeholder="Enter text to process..."
        />
      </div>

      {/* Example Prompts */}
      <div className="mb-4">
        <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
          Quick Examples
        </label>
        <div className="flex flex-wrap gap-2">
          {EXAMPLE_PROMPTS.map((prompt) => (
            <button
              key={prompt}
              onClick={() => loadExample(prompt)}
              className="text-xs px-3 py-1 bg-gray-100 dark:bg-gray-700 hover:bg-gray-200 dark:hover:bg-gray-600 rounded-full transition-colors duration-200 text-gray-700 dark:text-gray-300"
            >
              {prompt}
            </button>
          ))}
        </div>
      </div>

      {/* Generation Mode */}
      <div className="mb-4">
        <label className="flex items-center space-x-2 cursor-pointer">
          <input
            type="checkbox"
            checked={generate}
            onChange={(e) => setGenerate(e.target.checked)}
            className="w-4 h-4 text-primary-600 rounded focus:ring-primary-500"
          />
          <span className="text-sm font-medium text-gray-700 dark:text-gray-300">
            Generate Output (Autoregressive)
          </span>
        </label>
        <p className="text-xs text-gray-500 dark:text-gray-400 mt-1 ml-6">
          {generate
            ? 'Model will generate output token-by-token'
            : 'Use target text for teacher forcing'}
        </p>
      </div>

      {/* Target Text (if not generating) */}
      {!generate && (
        <div className="mb-4">
          <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
            Target Text (Optional)
          </label>
          <textarea
            value={targetText}
            onChange={(e) => setTargetText(e.target.value)}
            className="w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-lg focus:ring-2 focus:ring-primary-500 focus:border-transparent bg-white dark:bg-gray-700 text-gray-900 dark:text-white"
            rows={2}
            placeholder="Leave empty to echo source text..."
          />
        </div>
      )}

      {/* Max Generation Length */}
      {generate && (
        <div className="mb-4">
          <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
            Max Generation Length: {maxGenLen}
          </label>
          <input
            type="range"
            min="5"
            max="50"
            value={maxGenLen}
            onChange={(e) => setMaxGenLen(parseInt(e.target.value))}
            className="w-full h-2 bg-gray-200 dark:bg-gray-700 rounded-lg appearance-none cursor-pointer"
          />
          <div className="flex justify-between text-xs text-gray-500 dark:text-gray-400 mt-1">
            <span>5</span>
            <span>50</span>
          </div>
        </div>
      )}

      {/* Error Display */}
      {error && (
        <div className="mb-4 p-3 bg-red-50 dark:bg-red-900/20 border border-red-200 dark:border-red-800 rounded-lg">
          <p className="text-sm text-red-600 dark:text-red-400">{error}</p>
        </div>
      )}

      {/* Run Button */}
      <button
        onClick={handleRunInference}
        className="w-full btn-primary"
      >
        Run Inference
      </button>

      {/* Educational Info */}
      <div className="mt-6 p-4 bg-blue-50 dark:bg-blue-900/20 rounded-lg border border-blue-200 dark:border-blue-800">
        <h3 className="text-sm font-semibold text-blue-900 dark:text-blue-100 mb-2">
          💡 What's happening?
        </h3>
        <p className="text-xs text-blue-700 dark:text-blue-300 leading-relaxed">
          The transformer will process your input through multiple stages:
          embeddings, positional encoding, multi-head attention, and feed-forward
          networks. Each step transforms the representation to capture meaning.
        </p>
      </div>
    </div>
  );
}

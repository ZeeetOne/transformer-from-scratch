import { Link } from 'react-router-dom';

interface ApplicationMode {
  id: string;
  title: string;
  description: string;
  icon: string;
  status: 'available' | 'coming-soon';
  path: string;
}

const applicationModes: ApplicationMode[] = [
  {
    id: 'mode1',
    title: 'Mode 1: Next Word Prediction',
    description: 'Like Mini-GPT: Predict the next word/token based on previous tokens using a decoder-only architecture.',
    icon: '🔮',
    status: 'available',
    path: '/applications/mode1'
  },
  {
    id: 'mode2',
    title: 'Mode 2: Sequence-to-Sequence',
    description: 'Mini Machine Translation: Translate text from one sequence to another using encoder-decoder architecture.',
    icon: '🌐',
    status: 'coming-soon',
    path: '/applications/mode2'
  },
  {
    id: 'mode3',
    title: 'Mode 3: Encoder Output Visualization',
    description: 'Visualize how the encoder processes and represents input sequences.',
    icon: '🔍',
    status: 'coming-soon',
    path: '/applications/mode3'
  },
  {
    id: 'mode4',
    title: 'Mode 4: Mini Text Generator',
    description: 'Generate creative text continuations using autoregressive decoding.',
    icon: '✍️',
    status: 'coming-soon',
    path: '/applications/mode4'
  }
];

export default function Applications() {
  return (
    <div className="min-h-screen bg-gradient-to-br from-gray-900 via-blue-900 to-gray-900 text-white">
      {/* Header */}
      <header className="bg-black/30 backdrop-blur-sm border-b border-white/10">
        <div className="container mx-auto px-6 py-4">
          <div className="flex items-center justify-between">
            <div>
              <Link to="/" className="text-3xl font-bold bg-clip-text text-transparent bg-gradient-to-r from-blue-400 to-purple-400 hover:from-blue-300 hover:to-purple-300 transition-all">
                Transformer Interactive Visualization
              </Link>
              <p className="text-gray-300 mt-2">
                Explore practical applications of transformers
              </p>
            </div>
            <Link
              to="/"
              className="px-4 py-2 bg-blue-600 hover:bg-blue-700 rounded-lg transition-colors font-medium"
            >
              Back to Main
            </Link>
          </div>
        </div>
      </header>

      {/* Main Content */}
      <div className="container mx-auto px-6 py-12">
        <div className="max-w-5xl mx-auto">
          <div className="text-center mb-12">
            <h1 className="text-4xl md:text-5xl font-bold mb-4 bg-clip-text text-transparent bg-gradient-to-r from-blue-400 via-purple-400 to-pink-400">
              Transformer Applications
            </h1>
            <p className="text-xl text-gray-300 max-w-3xl mx-auto">
              Discover how transformers power different types of AI applications.
              Each mode demonstrates a unique use case with interactive visualizations.
            </p>
          </div>

          {/* Application Cards Grid */}
          <div className="grid md:grid-cols-2 gap-6">
            {applicationModes.map((mode) => (
              <div
                key={mode.id}
                className="bg-white/10 backdrop-blur-md rounded-xl border border-white/20 overflow-hidden hover:shadow-2xl hover:shadow-blue-500/20 transition-all duration-300 hover:scale-105"
              >
                <div className="p-6">
                  <div className="flex items-start justify-between mb-4">
                    <div className="text-5xl">{mode.icon}</div>
                    {mode.status === 'coming-soon' && (
                      <span className="px-3 py-1 bg-yellow-500/20 text-yellow-300 text-xs font-semibold rounded-full border border-yellow-500/30">
                        Coming Soon
                      </span>
                    )}
                    {mode.status === 'available' && (
                      <span className="px-3 py-1 bg-green-500/20 text-green-300 text-xs font-semibold rounded-full border border-green-500/30">
                        Available
                      </span>
                    )}
                  </div>

                  <h3 className="text-2xl font-bold mb-3 text-white">
                    {mode.title}
                  </h3>

                  <p className="text-gray-300 mb-6 leading-relaxed">
                    {mode.description}
                  </p>

                  {mode.status === 'available' ? (
                    <Link
                      to={mode.path}
                      className="inline-block w-full text-center px-6 py-3 bg-gradient-to-r from-blue-600 to-purple-600 hover:from-blue-700 hover:to-purple-700 text-white font-semibold rounded-lg transition-all duration-200 shadow-lg hover:shadow-xl"
                    >
                      Launch Application →
                    </Link>
                  ) : (
                    <button
                      disabled
                      className="w-full px-6 py-3 bg-gray-600/50 text-gray-400 font-semibold rounded-lg cursor-not-allowed"
                    >
                      Coming Soon
                    </button>
                  )}
                </div>
              </div>
            ))}
          </div>

          {/* Info Section */}
          <div className="mt-12 p-6 bg-blue-500/10 border border-blue-500/20 rounded-xl backdrop-blur-sm">
            <h2 className="text-2xl font-bold mb-3 text-blue-300">
              About These Applications
            </h2>
            <p className="text-gray-300 leading-relaxed mb-4">
              Each application mode demonstrates a different capability of transformer models:
            </p>
            <ul className="space-y-2 text-gray-300">
              <li className="flex items-start">
                <span className="text-blue-400 mr-2">•</span>
                <span><strong>Mode 1</strong> shows decoder-only architecture (like GPT) for next token prediction</span>
              </li>
              <li className="flex items-start">
                <span className="text-blue-400 mr-2">•</span>
                <span><strong>Mode 2</strong> demonstrates encoder-decoder architecture for sequence transformation</span>
              </li>
              <li className="flex items-start">
                <span className="text-blue-400 mr-2">•</span>
                <span><strong>Mode 3</strong> focuses on understanding how encoders process information</span>
              </li>
              <li className="flex items-start">
                <span className="text-blue-400 mr-2">•</span>
                <span><strong>Mode 4</strong> explores creative text generation with sampling strategies</span>
              </li>
            </ul>
          </div>
        </div>
      </div>
    </div>
  );
}

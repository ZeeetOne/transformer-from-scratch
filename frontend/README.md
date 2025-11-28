# Transformer Visualization - Frontend

Interactive React frontend for exploring transformer architecture through visualizations. Built with TypeScript, React, Plotly.js, D3.js, and Tailwind CSS.

![React](https://img.shields.io/badge/React-18.2-blue)
![TypeScript](https://img.shields.io/badge/TypeScript-5.0-blue)
![Vite](https://img.shields.io/badge/Vite-5.0-purple)
![Tailwind](https://img.shields.io/badge/Tailwind-3.4-teal)

---

## 📋 Overview

The frontend provides an intuitive, educational interface for interacting with transformer models. Users can input text, run predictions, and explore step-by-step visualizations of the transformer's internal operations.

### Key Features

- **Interactive Visualizations** - Attention heatmaps, embedding plots, architecture diagrams
- **Step-by-Step Pipeline** - 6-step visualization walkthrough for Mode 1
- **Real-Time Updates** - Instant visualization of model computations
- **Educational UI** - Clear explanations, tooltips, and mathematical foundations
- **Responsive Design** - Works seamlessly on desktop and tablet devices
- **Modern Stack** - React 18, TypeScript, Vite, Tailwind CSS

---

## 🚀 Quick Start

### Installation

```bash
# Navigate to frontend directory
cd frontend

# Install dependencies
npm install

# Or using yarn
yarn install
```

### Development

```bash
# Start development server
npm run dev

# Server will start at http://localhost:3000
```

**Development Features:**
- ⚡ Hot Module Replacement (HMR)
- 🔍 TypeScript type checking
- 🔄 Auto-reload on file changes
- 🌐 Proxy to backend API (port 8000)

### Building

```bash
# Build for production
npm run build

# Output: dist/ directory

# Preview production build
npm run preview
```

### Linting

```bash
# Run ESLint
npm run lint

# Fix auto-fixable issues
npm run lint -- --fix
```

---

## 📁 Project Structure

```
frontend/
├── src/
│   ├── pages/                         # Page components (routes)
│   │   ├── Home.tsx                   # Landing page
│   │   ├── Applications.tsx           # Mode selection hub
│   │   └── Mode1.tsx                  # Mode 1: Next Word Prediction
│   │
│   ├── components/                    # Reusable components
│   │   ├── mode1/                     # Mode 1 visualizers
│   │   │   ├── TokenizationVisualizer.tsx
│   │   │   ├── EmbeddingVisualizerV2.tsx       # Step 2 (Educational redesign)
│   │   │   ├── AttentionVisualizerV2.tsx       # Step 3 (Three-panel layout)
│   │   │   ├── FeedforwardVisualizer.tsx
│   │   │   ├── SoftmaxVisualizer.tsx
│   │   │   └── PredictionVisualizer.tsx
│   │   └── ...                        # Shared components
│   │
│   ├── services/
│   │   └── api.ts                     # Backend API client
│   │
│   ├── App.tsx                        # Main app with routing
│   ├── main.tsx                       # Entry point
│   ├── index.css                      # Global styles
│   └── vite-env.d.ts                  # Vite type definitions
│
├── public/                            # Static assets
├── index.html                         # HTML template
├── vite.config.ts                     # Vite configuration
├── tailwind.config.js                 # Tailwind CSS config
├── tsconfig.json                      # TypeScript config
├── package.json                       # Dependencies
└── README.md                          # This file
```

---

## 🎯 Application Pages

### Home Page (`src/pages/Home.tsx`)

**Route:** `/`

**Description:** Landing page with project overview, features, and call-to-action.

**Key Sections:**
- Hero section with project title
- Feature highlights
- Quick start guide
- Navigation to Applications

### Applications Page (`src/pages/Applications.tsx`)

**Route:** `/applications`

**Description:** Mode selection hub where users choose which transformer mode to explore.

**Current Modes:**
- **Mode 1:** Next Word Prediction (Mini-GPT) ✅ Available

**Planned Modes:**
- Mode 2: Translation (Seq2Seq)
- Mode 3: Masked Language Modeling (BERT-style)
- Mode 4: Custom Model Loading

### Mode 1 Page (`src/pages/Mode1.tsx`)

**Route:** `/applications/mode1`

**Description:** Complete interactive interface for next-word prediction with 6-step visualization pipeline.

**Features:**
1. **Input Section**
   - Text input field
   - Example prompts
   - Predict button

2. **6-Step Visualization Pipeline**
   - Step 1: Tokenization
   - Step 2: Embeddings + Positional Encoding
   - Step 3: Self-Attention & Multi-Head Attention
   - Step 4: Feedforward Network
   - Step 5: Output Layer (Softmax)
   - Step 6: Prediction Result

3. **Navigation**
   - Step selector
   - Previous/Next buttons
   - Progress indicator

---

## 🎨 Visualization Components

### Mode 1 Visualizers

#### 1. TokenizationVisualizer (`components/mode1/TokenizationVisualizer.tsx`)

**Step 1: Tokenization**

**Features:**
- Token breakdown visualization
- Token ID display
- Special tokens explanation
- Color-coded tokens

**Data:**
```typescript
{
  tokens: string[],          // ["I", "eat"]
  token_ids: number[],       // [245, 89]
  special_tokens: object     // {<PAD>: 0, <SOS>: 1, <EOS>: 2}
}
```

#### 2. EmbeddingVisualizerV2 (`components/mode1/EmbeddingVisualizerV2.tsx`)

**Step 2: Embeddings + Positional Encoding**

**Features:**
- **Three-section educational layout:**
  1. Word Embeddings (WE) - Learned semantic vectors
  2. Positional Encoding (PE) - Sinusoidal position patterns
  3. Final Embedding - WE + PE combination
- Color-coded grid visualizations
- Sinusoidal wave plots
- Interactive position comparison
- Hover tooltips with exact values

**Educational Value:**
- Shows how transformers handle **meaning** (word embeddings)
- Shows how transformers handle **position** (positional encoding)
- Demonstrates **element-wise addition** of WE + PE

#### 3. AttentionVisualizerV2 (`components/mode1/AttentionVisualizerV2.tsx`)

**Step 3: Self-Attention & Multi-Head Attention**

**Features:**
- **Three-panel integrated layout:**
  - **Panel A:** Input → Q/K/V projections (mechanism)
  - **Panel B:** Scaled dot-product attention calculation
  - **Panel C:** Multi-head attention + attention graph
- Interactive head selector (4 heads, color-coded)
- Attention graph with token-to-token connections
- Line thickness represents attention weight
- Hover details on vectors
- Animated transitions

**Educational Value:**
- Shows **HOW** attention works (mechanism panels A & B)
- Shows **WHAT** attention produces (behavior panel C)
- Allows comparison between different attention heads
- Visualizes which tokens attend to which

**Data:**
```typescript
{
  layer_0: {
    attention_weights: number[][][],  // [n_heads, seq_len, seq_len]
    attention_output: number[][]      // [seq_len, d_model]
  }
}
```

#### 4. FeedforwardVisualizer (`components/mode1/FeedforwardVisualizer.tsx`)

**Step 4: Feedforward Network**

**Features:**
- Input/output comparison
- Dimension expansion visualization (d_model → d_ff → d_model)
- Activation function display (ReLU)
- Value distributions

#### 5. SoftmaxVisualizer (`components/mode1/SoftmaxVisualizer.tsx`)

**Step 5: Output Layer (Softmax)**

**Features:**
- Logits visualization
- Softmax probability distribution
- Top-k predictions bar chart
- Temperature parameter explanation

**Data:**
```typescript
{
  logits: number[],            // [vocab_size]
  probabilities: number[],     // softmax(logits)
  predicted_token_id: number,
  predicted_token: string
}
```

#### 6. PredictionVisualizer (`components/mode1/PredictionVisualizer.tsx`)

**Step 6: Prediction Result**

**Features:**
- Final predicted word display
- Confidence score
- Top-5 alternative predictions
- Probability breakdown
- Copy result button

---

## 🔧 API Integration

### API Client (`src/services/api.ts`)

Centralized API client for backend communication.

**Base URL:** `http://localhost:8000/api/v1`

**Functions:**

```typescript
// Mode 1: Predict next word
export async function predictNextWord(text: string): Promise<PredictionResult> {
  const response = await fetch(`${API_BASE_URL}/predict-next-word`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ text })
  });
  return response.json();
}

// Health check
export async function healthCheck(): Promise<HealthStatus> {
  const response = await fetch(`${API_BASE_URL}/health`);
  return response.json();
}
```

**Type Definitions:**

```typescript
export interface PredictionResult {
  input_text: string;
  predicted_token: string;
  predicted_word: string;
  confidence: number;
  top_predictions: Array<{
    word: string;
    probability: number;
    log_prob: number;
  }>;
  steps: {
    tokenization: TokenizationStep;
    embeddings: EmbeddingsStep;
    attention: AttentionStep;
    feedforward: FeedforwardStep;
    output: OutputStep;
  };
}
```

**Usage Example:**

```typescript
import { predictNextWord } from './services/api';

const result = await predictNextWord("I eat");
console.log(result.predicted_word);  // "vegetables"
console.log(result.confidence);      // 0.529
```

---

## 🎨 Styling & Design

### Tailwind CSS

Custom theme configuration in `tailwind.config.js`:

```javascript
theme: {
  extend: {
    colors: {
      primary: {
        500: '#0ea5e9',
        600: '#0284c7',
        // ... more shades
      }
    }
  }
}
```

**Utility Classes:**

```tsx
// Card component
<div className="bg-white rounded-lg shadow-md p-6">
  <h2 className="text-2xl font-bold text-gray-800">Title</h2>
  <p className="text-gray-600 mt-2">Description</p>
</div>

// Button
<button className="px-4 py-2 bg-blue-500 text-white rounded hover:bg-blue-600">
  Click Me
</button>
```

### Design System

**Color Palette:**

```typescript
// Token colors (consistent across visualizations)
const tokenColors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A', ...];

// Attention head colors
const headColors = ['#3B82F6', '#8B5CF6', '#EC4899', '#F59E0B'];

// Component colors
const COLORS = {
  Q: '#3B82F6',  // Blue
  K: '#10B981',  // Green
  V: '#F59E0B',  // Yellow
  WE: '#3B82F6', // Blue (Word Embeddings)
  PE: '#F59E0B', // Yellow (Positional Encoding)
};
```

**Typography:**

```css
/* Headings */
.heading-1 { @apply text-4xl font-bold; }
.heading-2 { @apply text-2xl font-semibold; }
.heading-3 { @apply text-xl font-medium; }

/* Body */
.body-text { @apply text-base text-gray-700; }
.body-small { @apply text-sm text-gray-600; }
```

---

## ⚛️ React Patterns

### State Management

**Using React hooks (no external state library):**

```typescript
// Component state
const [inputText, setInputText] = useState('');
const [result, setResult] = useState<PredictionResult | null>(null);
const [currentStep, setCurrentStep] = useState(0);
const [loading, setLoading] = useState(false);

// Derived state with useMemo
const attentionData = useMemo(() => {
  if (!result) return null;
  return extractAttentionData(result.steps.attention);
}, [result]);
```

### Performance Optimization

**Memoization for expensive computations:**

```typescript
const visualizationData = useMemo(() => {
  if (!result) return null;
  return processVisualizationData(result);
}, [result]);
```

**React.memo for component optimization:**

```typescript
const ExpensiveComponent = React.memo(({ data }) => {
  // Component implementation
});
```

### Routing

**React Router DOM:**

```typescript
// In App.tsx
import { BrowserRouter, Routes, Route } from 'react-router-dom';

<BrowserRouter>
  <Routes>
    <Route path="/" element={<Home />} />
    <Route path="/applications" element={<Applications />} />
    <Route path="/applications/mode1" element={<Mode1 />} />
  </Routes>
</BrowserRouter>
```

---

## 📊 Visualization Libraries

### Plotly.js

**Interactive plots with Plotly:**

```typescript
import Plot from 'react-plotly.js';

<Plot
  data={[{
    z: attentionWeights,     // 2D array
    type: 'heatmap',
    colorscale: 'Viridis',
    hovertemplate: 'Attention: %{z:.3f}<extra></extra>'
  }]}
  layout={{
    title: 'Attention Heatmap',
    xaxis: { title: 'Key Position' },
    yaxis: { title: 'Query Position' },
    width: 600,
    height: 500
  }}
  config={{
    displayModeBar: false,  // Hide toolbar
    responsive: true
  }}
/>
```

### D3.js

**Custom SVG visualizations:**

```typescript
import * as d3 from 'd3';
import { useEffect, useRef } from 'react';

const AttentionGraph = ({ data }) => {
  const svgRef = useRef<SVGSVGElement>(null);

  useEffect(() => {
    const svg = d3.select(svgRef.current);
    // D3 visualization code
  }, [data]);

  return <svg ref={svgRef} width={800} height={600}></svg>;
};
```

---

## 🔧 Configuration

### Environment Variables

Create `.env.local`:

```env
# Backend API URL
VITE_API_URL=http://localhost:8000/api/v1

# Enable debug mode
VITE_DEBUG=true
```

**Usage in code:**

```typescript
const API_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000/api/v1';
```

### Vite Configuration (`vite.config.ts`)

```typescript
import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';

export default defineConfig({
  plugins: [react()],
  server: {
    port: 3000,
    proxy: {
      '/api': {
        target: 'http://localhost:8000',
        changeOrigin: true,
      }
    }
  },
  build: {
    outDir: 'dist',
    sourcemap: true
  }
});
```

---

## 🧪 Testing

### Setup (Future)

```bash
# Install testing libraries
npm install --save-dev @testing-library/react @testing-library/jest-dom vitest

# Run tests
npm test
```

### Example Test

```typescript
import { render, screen } from '@testing-library/react';
import { describe, it, expect } from 'vitest';
import Home from './pages/Home';

describe('Home Page', () => {
  it('renders hero section', () => {
    render(<Home />);
    expect(screen.getByText(/Transformer/i)).toBeInTheDocument();
  });
});
```

---

## 🐛 Troubleshooting

### Common Issues

#### API Connection Errors

**Problem:** Frontend can't connect to backend

**Solution:**
1. Ensure backend is running on port 8000
2. Check CORS settings in backend `app/main.py`
3. Verify proxy configuration in `vite.config.ts`
4. Check browser console for errors

```bash
# Verify backend is running
curl http://localhost:8000/health
```

#### Build Errors

**Problem:** Build fails with dependency errors

**Solution:**
```bash
# Clear node_modules and reinstall
rm -rf node_modules package-lock.json
npm install

# Clear Vite cache
rm -rf node_modules/.vite
```

#### Type Errors

**Problem:** TypeScript compilation errors

**Solution:**
```bash
# Check TypeScript configuration
npx tsc --noEmit

# Reinstall type definitions
npm install --save-dev @types/react @types/react-dom
```

#### Slow Development Server

**Problem:** HMR slow or not working

**Solution:**
```bash
# Clear Vite cache
rm -rf node_modules/.vite

# Restart dev server
npm run dev
```

---

## 🚀 Deployment

### Build for Production

```bash
# Build optimized bundle
npm run build

# Output: dist/ directory
```

### Static Hosting

Deploy `dist/` folder to:

**Vercel:**
```bash
npm install -g vercel
vercel --prod
```

**Netlify:**
```bash
npm install -g netlify-cli
netlify deploy --prod --dir=dist
```

**GitHub Pages:**
```bash
# Install gh-pages
npm install --save-dev gh-pages

# Add to package.json
"scripts": {
  "deploy": "gh-pages -d dist"
}

# Deploy
npm run deploy
```

### Environment Variables (Production)

Set production API URL:

```env
# .env.production
VITE_API_URL=https://api.your-domain.com/api/v1
```

### Production Checklist

- [ ] Build optimized bundle (`npm run build`)
- [ ] Set production API URL
- [ ] Test production build locally (`npm run preview`)
- [ ] Enable HTTPS
- [ ] Configure CDN (Cloudflare, AWS CloudFront)
- [ ] Set up error tracking (Sentry)
- [ ] Enable analytics (Google Analytics, Plausible)
- [ ] Add security headers
- [ ] Optimize assets (images, fonts)
- [ ] Set up monitoring

---

## ♿ Accessibility

**Features:**
- Semantic HTML elements (`<header>`, `<nav>`, `<main>`, `<section>`)
- ARIA labels for interactive elements
- Keyboard navigation support
- Color contrast compliance (WCAG AA)
- Screen reader friendly

**Example:**

```tsx
<button
  aria-label="Predict next word"
  onClick={handlePredict}
  className="btn-primary"
>
  Predict
</button>
```

---

## 📱 Browser Support

**Tested Browsers:**
- Chrome/Edge: Latest 2 versions ✅
- Firefox: Latest 2 versions ✅
- Safari: Latest 2 versions ✅

**Required Features:**
- ES2020+
- CSS Grid
- Flexbox
- SVG

---

## 📚 Documentation

### Additional Resources

- **Developer Guide:** [../CLAUDE.md](../CLAUDE.md)
- **Backend API:** [../backend/README.md](../backend/README.md)
- **Mode 1 Backend:** [../backend/app/features/mode1_next_word/README.md](../backend/app/features/mode1_next_word/README.md)
- **Project Plan:** [../PROJECT_PLAN.md](../PROJECT_PLAN.md)

---

## 🤝 Contributing

### Development Workflow

1. **Create feature branch:**
   ```bash
   git checkout -b feature/new-visualization
   ```

2. **Develop feature:**
   - Create component in appropriate folder
   - Add TypeScript types
   - Include educational comments

3. **Quality checks:**
   ```bash
   npm run lint
   npm run build  # Includes type checking
   ```

4. **Test manually:**
   - Start dev server
   - Test all interactions
   - Check responsive design

5. **Update documentation:**
   - Component README (if applicable)
   - Main frontend README
   - CLAUDE.md

6. **Submit pull request**

### Code Style

**TypeScript:**
- **Formatter:** Prettier (via ESLint)
- **Linter:** ESLint with TypeScript rules
- **Naming:** PascalCase for components, camelCase for functions
- **Components:** Functional components with hooks

**Educational Philosophy:**
- Add comments explaining **what** visualizations show
- Include tooltips with mathematical formulas
- Provide clear step-by-step explanations
- Use consistent color schemes

---

## 📜 License

MIT License - See [LICENSE](../LICENSE) file for details.

---

## 📞 Support

For issues, questions, or contributions:

- **Issues:** Check troubleshooting section above
- **Developer Guide:** [CLAUDE.md](../CLAUDE.md)
- **Backend Docs:** [backend/README.md](../backend/README.md)
- **Mode 1 Docs:** [backend/app/features/mode1_next_word/README.md](../backend/app/features/mode1_next_word/README.md)

---

**Last Updated:** 2025-11-29
**Status:** ✅ Production Ready
**Current Features:** Mode 1 (Next Word Prediction) with 6-step visualization pipeline

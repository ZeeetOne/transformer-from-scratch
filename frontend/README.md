# Transformer Visualization - Frontend

Interactive React frontend for exploring transformer architecture through visualizations. Built with TypeScript, Plotly, and Tailwind CSS.

## Features

- **Interactive Visualizations**: Attention heatmaps, embedding plots, architecture diagrams
- **Real-time Updates**: Instant visualization of transformer computations
- **Educational UI**: Clear explanations and tooltips throughout
- **Responsive Design**: Works on desktop and mobile devices
- **Dark Mode**: Comfortable viewing in any lighting

## Installation

```bash
# Install dependencies
npm install

# Or using yarn
yarn install
```

## Development

```bash
# Start development server
npm run dev

# Server will start at http://localhost:3000
```

The dev server includes:
- Hot module replacement
- TypeScript type checking
- Proxy to backend API

## Building

```bash
# Build for production
npm run build

# Preview production build
npm run preview
```

## Project Structure

```
frontend/
├── src/
│   ├── components/           # React components
│   │   ├── ControlPanel.tsx          # Input controls
│   │   ├── AttentionVisualizer.tsx   # Attention heatmaps
│   │   ├── EmbeddingVisualizer.tsx   # Embedding plots
│   │   └── ArchitectureDiagram.tsx   # Architecture view
│   ├── services/
│   │   └── api.ts                    # Backend API client
│   ├── App.tsx                       # Main app component
│   ├── main.tsx                      # Entry point
│   └── index.css                     # Global styles
├── public/                   # Static assets
├── index.html               # HTML template
├── vite.config.ts           # Vite configuration
├── tailwind.config.js       # Tailwind CSS config
└── package.json
```

## Components

### ControlPanel

Input controls for running transformer inference:
- Text input (source/target)
- Generation mode toggle
- Example prompts
- Configuration options

### AttentionVisualizer

Interactive attention visualization:
- Layer/head selection sliders
- Attention weight heatmaps
- Entropy statistics
- Per-head comparison grid

### EmbeddingVisualizer

Embedding and positional encoding plots:
- Token embedding heatmaps
- Positional encoding patterns
- Sinusoidal wave visualization
- First N dimensions display

### ArchitectureDiagram

Visual representation of transformer architecture:
- Encoder/decoder flow diagram
- Layer-by-layer breakdown
- Input/output display
- Model statistics

## API Integration

The frontend communicates with the backend via REST API:

```typescript
import { api } from './services/api';

// Run inference
const result = await api.runInference({
  source_text: "Hello world",
  generate: true,
  max_gen_len: 20
});

// Get attention visualization
const attention = await api.getAttentionVisualization(
  "Hello world",
  layerIdx=0,
  headIdx=0
);
```

## Configuration

### Environment Variables

Create `.env.local`:

```env
VITE_API_URL=http://localhost:8000/api/v1
```

### Vite Config

Proxy configuration for development (`vite.config.ts`):

```typescript
export default defineConfig({
  server: {
    port: 3000,
    proxy: {
      '/api': {
        target: 'http://localhost:8000',
        changeOrigin: true,
      }
    }
  }
})
```

## Styling

### Tailwind CSS

Custom theme in `tailwind.config.js`:

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

### Component Classes

Use utility classes and custom component classes:

```tsx
<div className="card">
  <button className="btn-primary">Click me</button>
</div>
```

## Visualizations

### Plotly.js

Interactive plots with Plotly:

```tsx
import Plot from 'react-plotly.js';

<Plot
  data={[{
    z: attentionWeights,
    type: 'heatmap',
    colorscale: 'Viridis'
  }]}
  layout={{
    title: 'Attention Heatmap',
    width: 600,
    height: 500
  }}
/>
```

### Customization

Modify plot configurations in component files:
- `AttentionVisualizer.tsx` - Attention heatmaps
- `EmbeddingVisualizer.tsx` - Embedding plots

## State Management

Using Zustand for global state (if needed):

```typescript
import create from 'zustand';

const useStore = create((set) => ({
  visualizationData: null,
  setVisualizationData: (data) => set({ visualizationData: data }),
}));
```

## TypeScript

### Type Definitions

API types in `src/services/api.ts`:

```typescript
export interface InferenceRequest {
  source_text: string;
  target_text?: string;
  generate: boolean;
  max_gen_len?: number;
}

export interface InferenceResponse {
  source_text: string;
  decoded_output: string;
  visualization_data: any;
}
```

### Type Checking

```bash
# Check types
npm run build  # Includes type checking

# Or using TypeScript directly
tsc --noEmit
```

## Performance

### Optimization Tips

1. **Lazy Load Components**: Use React.lazy for heavy components
2. **Memoize Calculations**: Use useMemo for expensive computations
3. **Debounce Inputs**: Prevent excessive API calls
4. **Virtual Scrolling**: For large token lists

Example memoization:

```typescript
const attentionData = useMemo(() => {
  return extractAttentionData(visualizationData);
}, [visualizationData]);
```

## Troubleshooting

### API Connection Issues

1. Check backend is running on port 8000
2. Verify CORS settings in backend
3. Check proxy configuration in `vite.config.ts`

### Build Errors

```bash
# Clear cache
rm -rf node_modules
npm install

# Clear Vite cache
rm -rf node_modules/.vite
```

### Type Errors

```bash
# Reinstall type definitions
npm install --save-dev @types/react @types/react-dom
```

## Deployment

### Build

```bash
npm run build
# Output in dist/ directory
```

### Static Hosting

Deploy `dist/` folder to:
- Vercel
- Netlify
- AWS S3 + CloudFront
- GitHub Pages

### Environment Variables

Set production API URL:

```env
VITE_API_URL=https://api.your-domain.com/api/v1
```

## Testing

### Component Tests (Future)

```bash
# Install testing libraries
npm install --save-dev @testing-library/react vitest

# Run tests
npm test
```

## Accessibility

- Semantic HTML elements
- ARIA labels for interactive elements
- Keyboard navigation support
- Color contrast compliance

## Browser Support

- Chrome/Edge: Latest 2 versions
- Firefox: Latest 2 versions
- Safari: Latest 2 versions

## License

MIT

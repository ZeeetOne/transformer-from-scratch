# Website Features & User Interface

This document describes the website features, user interface components, and interactive elements of the Transformer Interactive Visualization platform.

## Website Structure

### Pages

#### 1. Home Page (`/`)
**Purpose**: Landing page introducing the platform and its educational goals.

**Features**:
- Project overview and introduction
- Quick start guide
- Navigation to applications
- Educational objectives
- Links to resources

**Design**: Clean, welcoming interface with clear call-to-action.

#### 2. Applications Page (`/applications`)
**Purpose**: Hub for all transformer visualization modes.

**Features**:
- List of available modes
- Mode descriptions and use cases
- Quick navigation cards
- Status indicators (available/coming soon)

**Current Available Modes**:
- Mode 1: Next Word Prediction (Mini-GPT) ✅

**Planned Modes**:
- Mode 2: Translation (Seq2Seq) 🔜
- Mode 3: Masked Language Modeling (BERT) 🔜
- Mode 4: Custom Model Loading 🔜

#### 3. Mode 1 Page (`/applications/mode1`)
**Purpose**: Interactive next-word prediction with step-by-step visualizations.

**Layout**:
```
┌─────────────────────────────────────┐
│         Header & Navigation         │
├─────────────────────────────────────┤
│         Input Panel                 │
│  [Text Input] [Predict Button]     │
├─────────────────────────────────────┤
│         Step Navigator              │
│  [Step 1] [Step 2] ... [Step 6]    │
├─────────────────────────────────────┤
│         Visualization Panel         │
│  (Dynamic based on selected step)   │
└─────────────────────────────────────┘
```

## UI Components

### Navigation Components

#### Header Navigation
- **Home Link**: Returns to landing page
- **Applications Link**: Goes to mode selection
- **Breadcrumbs**: Shows current location
- **Responsive**: Collapses to hamburger menu on mobile

#### Step Navigator
- **Visual Progress**: Shows current step in pipeline
- **Click Navigation**: Jump to any completed step
- **Status Indicators**: Completed/current/upcoming steps
- **Tooltips**: Hover to see step descriptions

### Input Components

#### Text Input Panel
**Features**:
- **Multi-line Text Area**: For entering input sequences
- **Character Counter**: Shows current input length
- **Example Suggestions**: Quick-fill common examples
- **Clear Button**: Reset input
- **Validation**: Real-time input validation

**Example Inputs Provided**:
- "I eat" (simple)
- "The cat sat" (common phrase)
- "Hello world" (greeting)
- "Transformer models" (technical)

#### Predict Button
- **Loading State**: Shows spinner during processing
- **Disabled State**: When input is empty or processing
- **Keyboard Shortcut**: Enter to submit (with modifier)
- **Error Handling**: Displays error messages

### Visualization Components

#### 1. Tokenization Visualizer
**Location**: Step 1

**Visual Elements**:
- Token breakdown animation
- Color-coded tokens with consistent colors
- Token IDs displayed below each token
- Special token highlighting (`<SOS>`, `<EOS>`, etc.)

**Interactive Features**:
- Hover over tokens to see details
- Click to highlight token flow in later steps

**Design**:
- Clean, card-based layout
- Smooth animations (stagger effect)
- Accessibility: ARIA labels for screen readers

#### 2. Embedding Visualizer V2
**Location**: Step 2

**Visual Elements**:
- **Three-section layout**:
  - Word Embeddings (WE): Learned semantic vectors
  - Positional Encoding (PE): Sinusoidal position patterns
  - Final Embedding: WE + PE result
- **Grid Visualization**: 8×1 grids showing first 8 dimensions
- **Color Coding**: Blue/red gradient for positive/negative values
- **Sinusoidal Wave**: SVG representation of PE patterns
- **Mathematical Formulas**: Clear equations at each step

**Interactive Features**:
- **Hover Tooltips**: Show exact values for each dimension
- **Position Comparison Toggle**: Button to show how position changes meaning
- **Side-by-side Comparison**: Same words at different positions
- **Animated Transitions**: Smooth fade-in effects

**Design Style**:
- Textbook-quality layout
- Pastel color palette
- Clear labels and annotations
- Grid-based organization

#### 3. Attention Visualizer V2
**Location**: Step 3

**Visual Elements**:
- **Three-panel integrated layout**:
  - **Panel A** (Left): Input → Q/K/V Projections
    - Shows token embeddings
    - Linear transformations to Q, K, V
    - Color-coded vectors
  - **Panel B** (Center): Attention Calculation
    - Step 1: Q·K^T similarity matrix
    - Step 2: Softmax normalization
    - Step 3: Weighted sum of values
  - **Panel C** (Right): Multi-Head Attention + Patterns
    - Head selector buttons (4 heads)
    - Attention pattern graph
    - Token-to-token connection lines

**Interactive Features**:
- **Head Selection**: Click any head button to switch views
- **Attention Graph**: Lines showing attention flow
  - Line thickness = attention weight
  - Line opacity = attention strength
  - Color = selected head color
- **Hover Details**: Tooltips on Q/K/V vectors
- **Animated Lines**: Draw-in effect for attention connections

**Design Features**:
- Color-coded components (Q=Blue, K=Green, V=Yellow)
- Consistent token colors from previous steps
- Causal masking visualization (grayed future tokens)
- Head-specific color schemes
- Educational annotations throughout

**Attention Graph**:
```
Token 1 ═══════════> Token 1  (thick: high attention)
        ─────────> Token 2  (medium)
        ········> Token 3  (thin: low attention)

Token 2 ════════> Token 1  (attending to previous)
        ─────────> Token 2  (self-attention)
        (no line to Token 3: causal mask)
```

#### 4. Feedforward Visualizer
**Location**: Step 4

**Visual Elements**:
- Input representation heatmap
- Hidden layer activations (expanded dimension)
- Output representation after projection
- ReLU activation pattern visualization

**Interactive Features**:
- Hover to see activation values
- Toggle between input/hidden/output views
- Dimension highlighting

**Design**:
- Heatmap-based visualization
- Color intensity = activation strength
- Clear layer boundaries

#### 5. Softmax Visualizer
**Location**: Step 5

**Visual Elements**:
- Vocabulary distribution heatmap
- Top-k predictions with probability bars
- Confidence score display
- Probability distribution curve

**Interactive Features**:
- Hover to see exact probabilities
- Click predictions to see details
- Adjustable top-k value

**Design**:
- Bar chart for top predictions
- Percentage displays
- Color gradient by probability

#### 6. Prediction Result Visualizer
**Location**: Step 6

**Visual Elements**:
- Final predicted word/token (large, highlighted)
- Confidence score with visual indicator
- Top 5 alternative predictions
- Full prediction statistics

**Interactive Features**:
- Copy prediction to clipboard
- Export results as JSON
- Try again with prediction appended

**Design**:
- Card-based layout
- Success/warning colors by confidence
- Clear typography hierarchy

## Visual Design System

### Color Palette

#### Token Colors (Consistent Throughout)
```css
Token 1: #FF6B6B  /* Red */
Token 2: #4ECDC4  /* Teal */
Token 3: #45B7D1  /* Blue */
Token 4: #FFA07A  /* Salmon */
Token 5: #98D8C8  /* Mint */
Token 6: #F7DC6F  /* Yellow */
```

#### Attention Head Colors
```css
Head 1: #3B82F6  /* Blue */
Head 2: #8B5CF6  /* Purple */
Head 3: #EC4899  /* Pink */
Head 4: #F59E0B  /* Amber */
```

#### Component Colors
```css
Query (Q):  #3B82F6  /* Blue */
Key (K):    #10B981  /* Green */
Value (V):  #F59E0B  /* Yellow */
```

#### UI Colors
```css
Background: #FFFFFF
Text Primary: #1F2937
Text Secondary: #6B7280
Border: #E5E7EB
Accent: #3B82F6
Success: #10B981
Warning: #F59E0B
Error: #EF4444
```

### Typography

**Fonts**:
- Primary: Inter, system-ui, sans-serif
- Monospace: 'Fira Code', 'Courier New', monospace

**Hierarchy**:
- **Headings**:
  - H1: 32px, bold
  - H2: 24px, semi-bold
  - H3: 18px, semi-bold
- **Body**: 16px, regular
- **Small**: 14px, regular
- **Captions**: 12px, regular
- **Code/Values**: 14px, monospace

### Spacing & Layout

**Grid System**: 8px base unit
- **xs**: 4px
- **sm**: 8px
- **md**: 16px
- **lg**: 24px
- **xl**: 32px
- **2xl**: 48px

**Container Widths**:
- Mobile: 100% (with padding)
- Tablet: 768px
- Desktop: 1024px
- Wide: 1280px

**Border Radius**:
- Small: 4px
- Medium: 8px
- Large: 12px
- Full: 9999px (pills)

### Animations

**Transitions**:
- Default: 200ms ease-in-out
- Slow: 300ms ease-in-out
- Fast: 150ms ease-in-out

**Effects**:
- **Fade In**: Opacity 0 → 1
- **Slide In**: Transform + opacity
- **Stagger**: Sequential with 100ms delay
- **Pulse**: Scale 1 → 1.05 → 1
- **Draw**: SVG path animation

## Responsive Design

### Breakpoints

```css
Mobile:  < 640px
Tablet:  640px - 1024px
Desktop: > 1024px
```

### Mobile Optimizations

- **Single Column Layout**: Stacked visualizations
- **Collapsible Panels**: Expand/collapse sections
- **Touch-Friendly**: Larger tap targets (44px minimum)
- **Simplified Visualizations**: Reduced complexity on small screens
- **Horizontal Scroll**: For wide visualizations
- **Bottom Navigation**: Fixed navigation bar

### Tablet Optimizations

- **Two Column Layout**: Where appropriate
- **Adaptive Grid**: Flexible grid layouts
- **Touch + Mouse**: Support both interaction modes
- **Medium Detail**: Balance between mobile and desktop

### Desktop Optimizations

- **Three Column Layout**: Maximum information density
- **Keyboard Shortcuts**: Power user features
- **Hover States**: Rich hover interactions
- **Full Detail**: All visualization details shown

## Accessibility

### WCAG Compliance

**Target**: WCAG 2.1 Level AA

**Features**:
- **Color Contrast**: Minimum 4.5:1 for text
- **Keyboard Navigation**: Full keyboard support
- **Screen Readers**: ARIA labels and descriptions
- **Focus Indicators**: Clear focus states
- **Skip Links**: Skip to main content
- **Alt Text**: Descriptive alternatives for visuals

### Keyboard Shortcuts

```
Enter:        Submit prediction (when focused)
Tab:          Navigate between elements
Shift + Tab:  Navigate backward
Arrow Keys:   Navigate steps
Esc:          Close modals/overlays
```

## Performance Features

### Optimization Strategies

1. **Code Splitting**: Lazy load mode pages
2. **Memoization**: `useMemo` for expensive calculations
3. **Virtualization**: For large lists (if needed)
4. **Debouncing**: Input validation debounced
5. **Lazy Loading**: Images and components loaded on demand
6. **Caching**: API responses cached (planned)

### Loading States

- **Skeleton Screens**: During initial load
- **Spinners**: During API calls
- **Progress Bars**: For multi-step processes
- **Optimistic Updates**: Immediate UI feedback

### Error Handling

**Error Types**:
- **Network Errors**: Connection issues
- **Validation Errors**: Invalid input
- **Server Errors**: Backend failures
- **Timeout Errors**: Request timeouts

**Error Display**:
- **Toast Notifications**: Non-blocking messages
- **Inline Errors**: Field-level validation
- **Error Pages**: For critical failures
- **Retry Buttons**: Allow user recovery

## Future UI Enhancements

### Planned Features

1. **Dark Mode**: Toggle between light/dark themes
2. **Customizable Layout**: User preferences for visualization layout
3. **Export Visualizations**: Download as PNG/SVG
4. **Comparison Mode**: Compare different inputs side-by-side
5. **Animation Playback**: Step-by-step animation through pipeline
6. **Tutorial Tooltips**: Interactive onboarding
7. **Shareable Links**: URL state for sharing configurations
8. **Notebook Integration**: Embed visualizations in Jupyter
9. **Presentation Mode**: Full-screen for teaching
10. **Accessibility Improvements**: Enhanced screen reader support

### Experimental Features

- **3D Visualizations**: Three.js for embedding spaces
- **Audio Explanations**: Text-to-speech for descriptions
- **Multi-language**: Internationalization (i18n)
- **Collaborative Mode**: Real-time shared sessions
- **Custom Theming**: User-created color schemes

## Browser Support

**Supported Browsers**:
- Chrome/Edge: Last 2 versions
- Firefox: Last 2 versions
- Safari: Last 2 versions

**Required Features**:
- ES6+ JavaScript
- CSS Grid & Flexbox
- SVG rendering
- WebSocket (for future features)

**Not Supported**:
- Internet Explorer
- Very old browsers (pre-2020)

## User Experience Principles

### Design Philosophy

1. **Clarity**: Every element has a clear purpose
2. **Consistency**: Similar patterns throughout
3. **Feedback**: Immediate response to actions
4. **Forgiveness**: Easy to undo/reset
5. **Efficiency**: Minimize clicks to goals
6. **Learnability**: Intuitive for first-time users
7. **Memorability**: Easy to remember how to use

### Interaction Patterns

- **Progressive Disclosure**: Show details on demand
- **Contextual Help**: Tooltips and hints where needed
- **Visual Hierarchy**: Important elements stand out
- **Affordances**: Interactive elements look clickable
- **Feedback**: Loading states and confirmations
- **Error Prevention**: Validation before submission

---

*Last Updated: 2025-11-27*

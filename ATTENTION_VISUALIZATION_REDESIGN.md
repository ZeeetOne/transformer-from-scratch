# Self-Attention & Multi-Head Attention Visualization Redesign

## Overview

I've created a **premium, textbook-quality visualization** that integrates both the **internal mechanism** (how attention works) and the **resulting behavior** (attention patterns) in a single, comprehensive three-panel layout.

This visualization matches the educational style of high-end ML tutorials like *The Illustrated Transformer*, BERTViz, and academic research papers.

## ✨ Three-Panel Integrated Design

### 🎨 Layout: Left → Center → Right

```
┌─────────────┬─────────────────┬──────────────────┐
│  PANEL A    │    PANEL B      │     PANEL C      │
│  Input→Q/K/V│  Attention Calc │  Multi-Head      │
│             │                 │  + Patterns      │
│ Mechanism 1 │  Mechanism 2    │  Mechanism 3 +   │
│             │                 │  Behavior        │
└─────────────┴─────────────────┴──────────────────┘
```

---

## 📊 Panel A: Input → Q/K/V Projections

### **Purpose**: Show how embeddings are transformed into Queries, Keys, and Values

### Visual Elements:

1. **Token Display**
   - Each token shown with consistent color (e.g., "Ayam" = Red, "makan" = Teal)
   - Color persists throughout entire diagram

2. **Embedding Representation**
   - Small colored squares showing simplified embedding (4 dimensions)
   - Color opacity varies with values

3. **Linear Projections**
   - Three arrows branching from embedding:
     - **Blue arrow** → Q (Query)
     - **Green arrow** → K (Key)
     - **Yellow arrow** → V (Value)

4. **Q/K/V Vectors**
   - Each shown as a row of colored boxes
   - Values displayed in boxes (e.g., "0.23", "-0.45")
   - Color intensity = magnitude

### Educational Annotations:

```
Q = Query (What am I looking for?)
K = Key (What do I contain?)
V = Value (What information do I have?)
```

### Visual Style:
- Pastel blue background tint
- Thin arrows (1.5px)
- Rounded boxes with soft shadows
- Clean labels in small font

---

## 📊 Panel B: Scaled Dot-Product Attention Calculation

### **Purpose**: Show the step-by-step attention computation

### Three-Step Flow:

#### **Step 1: Q · K^T (Similarity Scores)**
```
Visual: 3×3 grid showing similarity matrix
Color: Purple gradient
Shows: How similar each Query is to each Key
Annotation: "How similar is each Q to each K?"
```

- Diagonal and lower triangle filled (causal masking)
- Upper triangle grayed out (can't attend to future)
- Values displayed in each cell

#### **Step 2: Softmax (Normalize to Probabilities)**
```
Visual: Row of percentage bars
Color: Green gradient
Shows: Attention weights (sum to 100%)
Display: "45%", "30%", "25%", etc.
```

- Each token gets a row of attention weights
- Bar width = probability
- Labels show percentage values

#### **Step 3: Weighted Sum of Values**
```
Visual: Final attended representation
Color: Yellow/amber gradient
Shows: Result of attention-weighted V vectors
Display: 4 colored boxes representing output
```

### Educational Annotations:

```
Formula: Attention(Q, K, V) = softmax(Q·K^T/√d_k) · V
```

### Visual Style:
- Purple background tint
- Vertical flow with downward arrows
- Grid layouts for matrices
- Clear step numbering (1, 2, 3)

---

## 📊 Panel C: Multi-Head Attention + Behavior Patterns

### **Purpose**: Show both mechanism (multi-head) AND behavior (attention graph)

### Top Section: Multi-Head Mechanism

#### **Head Selector**
```
Visual: 4 clickable head buttons (or 8, depending on model)
Layout: 2×2 or 2×4 grid
Colors: Different color per head
  - Head 1: Blue
  - Head 2: Purple
  - Head 3: Pink
  - Head 4: Amber
```

**Interactive:**
- Click any head to see its attention pattern
- Selected head has white border + shadow
- Each head shows mini output vector (3 colored boxes)

#### **Concatenate + Linear Projection**
```
Visual: Vertical flow diagram
Steps:
  1. All heads shown
     ↓
  2. "Concatenate" box (pink border)
     ↓
  3. "Linear Projection" box (purple border)
     ↓
  4. Final multi-head output
```

### Bottom Section: **ATTENTION BEHAVIOR GRAPH** 🔥

This is the **key innovation** - integrating mechanism and behavior!

#### **Attention Pattern Visualization**

```
Layout:
  Left Side              Right Side
  ┌─────┐               ┌─────┐
  │Token│───────────────│Token│
  │  1  │───────────────│  1  │
  └─────┘    ╱─────╲    └─────┘
  ┌─────┐───╱───────╲───┌─────┐
  │Token│──────────────│Token│
  │  2  │              │  2  │
  └─────┘              └─────┘
  ┌─────┐              ┌─────┐
  │Token│              │Token│
  │  3  │              │  3  │
  └─────┘              └─────┘
```

**Visual Elements:**

1. **Source Tokens (Left)**
   - Colored rectangles with token text
   - Same colors from Panel A

2. **Target Tokens (Right)**
   - Identical layout to source tokens

3. **Attention Lines**
   - Connect source → target
   - **Line thickness** = attention weight
   - **Line opacity** = attention strength (0.3 + weight × 0.6)
   - **Line color** = selected head color
   - Animated appearance (stagger effect)

4. **Legend**
   - "Line thickness = attention strength"
   - "● Head X attention pattern" (in head color)

### **Example Visualization:**

For "Ayam makan Taufik" with Head 1 selected:

```
Ayam (Red)  ═══════════════════════> Ayam (Red)      [Self-attention: thick line]
            ────────────────────────> makan (Teal)    [Moderate line]
            ·······················> Taufik (Blue)   [Thin line]

makan (Teal) ════════════════════> Ayam (Red)      [Thick: attending to subject]
             ─────────────────────> makan (Teal)    [Moderate: self]
             (no line to Taufik - causal mask)

Taufik (Blue) ══════════════════> Ayam (Red)       [Attending to subject]
              ══════════════════> makan (Teal)     [Attending to verb]
              ─────────────────> Taufik (Blue)    [Self-attention]
```

### Educational Annotations:

```
💡 Key Insights:
- Line darkness = attention strength
- Different colors = different heads
- This is the result of the mechanism shown in Panel A + B
```

### Visual Style:
- Pink/magenta background tint
- SVG-based attention graph
- Interactive head selection
- Smooth animations

---

## 🎨 Global Design Principles

### Color Palette

#### **Token Colors (Consistent Throughout):**
```javascript
'#FF6B6B'  // Red - Token 1
'#4ECDC4'  // Teal - Token 2
'#45B7D1'  // Blue - Token 3
'#FFA07A'  // Salmon - Token 4
'#98D8C8'  // Mint - Token 5
'#F7DC6F'  // Yellow - Token 6
```

#### **Head Colors:**
```javascript
'#3B82F6'  // Blue - Head 1
'#8B5CF6'  // Purple - Head 2
'#EC4899'  // Pink - Head 3
'#F59E0B'  // Amber - Head 4
'#10B981'  // Green - Head 5
'#06B6D4'  // Cyan - Head 6
'#EF4444'  // Red - Head 7
'#6366F1'  // Indigo - Head 8
```

#### **Component Colors:**
```javascript
Q (Query):  '#3B82F6'  // Blue
K (Key):    '#10B981'  // Green
V (Value):  '#F59E0B'  // Yellow/Amber
```

### Typography

- **Panel Headers**: 14px, bold, colored (blue/purple/pink)
- **Step Labels**: 12px, semi-bold
- **Body Text**: 11-12px, regular
- **Annotations**: 10px, italic, gray-400
- **Values in Grids**: 8-10px, monospace

### Spacing & Layout

- **Panel Padding**: 16px
- **Gap Between Panels**: 24px
- **Element Spacing**: 8-12px
- **Border Radius**: 6-8px (rounded corners)
- **Border Width**: 1-2px

### Animations

1. **Token Appearance**: Stagger by 0.1s, slide from left
2. **Arrow Flow**: Pulse effect on hover
3. **Attention Lines**: Draw-in effect (pathLength animation)
4. **Head Selection**: Scale + shadow transition
5. **Panel Transitions**: Fade + slide

---

## 📚 Educational Content

### Marginal Notes (Subtle, Textbook-Style):

```
Top of Diagram:
"Self-attention lets every token look at every other token."

Left Side:
"Mechanism (left and center) → Behavior (right)."

Right Side:
"Multi-head attention allows different relational patterns to be learned."

Bottom:
"Each head can specialize: syntax, semantics, coreference, etc."
```

### Key Insights Panel (Below Main Diagram):

```
┌──────────────────────────────────────────────────────┐
│ 💡 Key Insight                                       │
│ Self-attention lets every token "look at" all other  │
│ tokens. The mechanism (Q·K·V) determines which       │
│ tokens are relevant.                                  │
├──────────────────────────────────────────────────────┤
│ 🎯 Multi-Head                                        │
│ Multiple heads learn different relational patterns.  │
│ One might focus on syntax, another on semantics.     │
├──────────────────────────────────────────────────────┤
│ 📊 Behavior                                          │
│ The attention pattern (right panel) shows which      │
│ tokens the model considers important.                │
└──────────────────────────────────────────────────────┘
```

---

## 🎯 Learning Objectives

After viewing this visualization, students should understand:

### ✅ **Mechanism (How it Works):**

1. **Q/K/V Projections**
   - Input embeddings are transformed into three representations
   - Each serves a different purpose in attention

2. **Attention Computation**
   - Q·K^T calculates similarity scores
   - Softmax normalizes to probabilities
   - Weighted sum of V produces output

3. **Multi-Head Architecture**
   - Multiple parallel attention mechanisms
   - Each head learns different patterns
   - Outputs concatenated and projected

### ✅ **Behavior (What it Does):**

1. **Attention Patterns**
   - See which tokens attend to which
   - Line thickness = relevance
   - Different heads = different perspectives

2. **Causal Masking**
   - Tokens can only attend to previous tokens
   - Prevents "looking into the future"
   - Visualized as missing connections

3. **Head Specialization**
   - Each head learns unique patterns
   - Compare different heads interactively
   - Understand why multiple perspectives matter

---

## 🔧 Technical Implementation

### Component: `AttentionVisualizerV2.tsx`

#### **Key Features:**

1. **Mock Attention Weight Generation**
   ```typescript
   // Generates causal-masked attention weights
   // Normalized to sum to 1.0 per row
   attentionWeights[head][from][to]
   ```

2. **Interactive Head Selection**
   ```typescript
   const [selectedHead, setSelectedHead] = useState(0);
   // Click head button → update attention graph
   ```

3. **SVG Attention Graph**
   ```typescript
   // Dynamic line rendering based on weights
   strokeWidth={weight * 8}
   opacity={0.3 + weight * 0.6}
   ```

4. **Animated Rendering**
   ```typescript
   // Lines draw in with pathLength animation
   initial={{ pathLength: 0 }}
   animate={{ pathLength: 1 }}
   ```

#### **Data Flow:**

```
Input Props:
├── tokens: ["<SOS>", "Ayam", "makan", "Taufik"]
├── numHeads: 4
└── numLayers: 4

↓ Generate Mock Data

Q/K/V Vectors (4 dims each):
├── Token 0: Q=[0.23, -0.45, ...], K=[...], V=[...]
├── Token 1: Q=[...], K=[...], V=[...]
└── ...

Attention Weights (per head):
├── Head 0: [[0.5, 0.3, 0.2], [0.4, 0.4, 0.2], ...]
├── Head 1: [[0.6, 0.2, 0.2], [0.3, 0.5, 0.2], ...]
└── ...

↓ Render Visualization

Three Panels:
├── Panel A: Q/K/V projections
├── Panel B: Attention calculation
└── Panel C: Multi-head + graph
```

---

## 🚀 Usage

### Access the Visualization:

1. Navigate to: http://localhost:3001
2. Go to **Applications** → **Mode 1: Next Word Prediction**
3. Enter text (e.g., "Ayam makan Taufik")
4. Click **"Predict Next Word"**
5. Navigate to **Step 3: Self-Attention & Multi-Head Attention**

### Interactive Elements:

1. **Click Head Buttons** (Panel C)
   - Switch between different attention heads
   - See how patterns differ per head

2. **Observe Attention Lines**
   - Thickness = attention strength
   - Color = current head
   - Animated drawing effect

3. **Hover Over Elements**
   - Q/K/V vectors show tooltips with values
   - Attention cells show exact scores

---

## 📸 What You'll See

### **Visual Flow:**

```
PANEL A                  PANEL B                  PANEL C
┌──────────┐            ┌──────────┐            ┌──────────┐
│ Token    │            │ Q · K^T  │            │ Head 1   │
│ "Ayam"   │──┬─→ Q    │ Matrix   │            │ [****]   │
│  [emb]   │  ├─→ K    │   ↓      │            │ Head 2   │
│          │  └─→ V    │ Softmax  │            │ [****]   │
└──────────┘            │   ↓      │            │  ...     │
                        │ Weighted │            ├──────────┤
                        │   Sum    │            │ Attention│
                        └──────────┘            │  Graph   │
                                                │  [Lines] │
                                                └──────────┘
```

### **Attention Graph Example:**

```
Ayam  ═══════════> Ayam     (thick blue line)
      ─────────> makan    (medium blue line)
      ········> Taufik   (thin blue line)

makan ═════════> Ayam     (thick line - attending to subject)
      ─────────> makan
      (no line to Taufik - future)

Taufik ════════> Ayam
       ════════> makan
       ─────────> Taufik
```

---

## 🎓 Educational Impact

This integrated visualization achieves:

### ✅ **Mechanism Understanding**
- See exact Q/K/V transformations
- Follow computation step-by-step
- Understand multi-head architecture

### ✅ **Behavior Insight**
- Observe which tokens attend to which
- Compare different head patterns
- Understand why attention is powerful

### ✅ **Integration**
- **Left panels** explain HOW it works
- **Right panel** shows WHAT it does
- Single coherent narrative from mechanism → behavior

### ✅ **Interactive Learning**
- Click heads to explore
- See real-time pattern changes
- Engage with the material actively

---

## 📊 Comparison: Old vs New

### Old Design:
```
❌ Generic connection graph only
❌ No mechanism explanation
❌ Limited educational context
❌ Hard to understand Q/K/V roles
❌ No multi-head visualization
```

### New Design:
```
✅ Three-panel integrated layout
✅ Complete mechanism explanation (Q/K/V → Attention)
✅ Interactive head selection
✅ Attention behavior graph with animations
✅ Educational annotations throughout
✅ Textbook-quality visual style
✅ Color-coded tokens consistent across panels
✅ Step-by-step attention calculation
```

---

## 🎨 Visual Style Matches:

- **The Illustrated Transformer** (Jay Alammar)
- **BERTViz** attention visualizations
- **Attention is All You Need** (Vaswani et al.) paper diagrams
- **3Blue1Brown** educational videos
- **ML textbooks** (Bishop, Goodfellow)

---

## ✅ Status

- ✅ **Component Created**: `AttentionVisualizerV2.tsx`
- ✅ **Integrated into Mode1.tsx**
- ✅ **Hot-Reloaded Successfully**
- ✅ **Live on**: http://localhost:3001

---

## 🔮 Future Enhancements

1. **Layer-by-Layer Playback**
   - Animate attention through all layers
   - Show how patterns evolve

2. **Attention Flow Visualization**
   - Sankey diagram showing information flow
   - Token-to-token paths

3. **Real Attention Weights**
   - Backend returns actual attention matrices
   - Show real model behavior

4. **Attention Rollout**
   - Aggregate attention across layers
   - Show final effective attention

5. **Export Diagrams**
   - Download as SVG/PNG
   - For presentations and papers

---

## 🌟 The Result

This visualization transforms abstract attention math into a **concrete, visual journey** that shows both:

1. **The Internal Math** (Mechanism)
2. **The External Behavior** (Patterns)

Students can now **SEE** how attention works AND what it produces, making this critical concept intuitive and memorable! 🚀

**Component Location**: `frontend/src/components/mode1/AttentionVisualizerV2.tsx`

**Try it now**: Navigate to Mode 1, enter text, and explore Step 3! ✨

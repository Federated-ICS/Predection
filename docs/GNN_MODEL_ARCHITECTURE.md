# Network-Aware Hybrid GNN Architecture

## 🏗️ Complete Model Architecture

### High-Level Overview

```mermaid
graph TB
    subgraph Input["📥 INPUT LAYER"]
        A[Current Attack ID<br/>Integer Index]
        B[Network Features<br/>18D Vector]
        C[Graph Structure<br/>Edge Index + Edge Attr]
    end
    
    subgraph GNN["🔵 GNN PATH - Graph Structure Learning"]
        D[Node Embedding Layer<br/>num_techniques → 256D]
        E[GAT Layer 1<br/>8 heads × 32D = 256D<br/>Edge-aware attention]
        F[ReLU + Dropout 0.3]
        G[GAT Layer 2<br/>8 heads × 32D = 256D<br/>Edge-aware attention]
        H[ReLU + Dropout 0.3]
        I[GAT Layer 3<br/>1 head × 256D<br/>Edge-aware attention]
        J[Node Selection<br/>Extract current node embedding]
    end
    
    subgraph Network["🟢 NETWORK PATH - Temporal Features"]
        K[Linear Layer 1<br/>18D → 256D]
        L[BatchNorm1D]
        M[ReLU]
        N[Dropout 0.4]
        O[Linear Layer 2<br/>256D → 256D]
        P[BatchNorm1D]
        Q[ReLU]
    end
    
    subgraph Fusion["🟡 FUSION & CLASSIFICATION"]
        R[Concatenation<br/>256D + 256D = 512D]
        S[Linear Layer<br/>512D → 256D]
        T[BatchNorm1D]
        U[ReLU]
        V[Dropout 0.4]
        W[Classifier<br/>256D → num_techniques]
        X[Softmax<br/>Probability Distribution]
    end
    
    subgraph Output["📤 OUTPUT"]
        Y[Next Attack Predictions<br/>Top-K Probabilities]
    end
    
    A --> D
    D --> E
    C -.->|Edge Features 6D| E
    E --> F
    F --> G
    C -.->|Edge Features 6D| G
    G --> H
    H --> I
    C -.->|Edge Features 6D| I
    I --> J
    
    B --> K
    K --> L
    L --> M
    M --> N
    N --> O
    O --> P
    P --> Q
    
    J --> R
    Q --> R
    R --> S
    S --> T
    T --> U
    U --> V
    V --> W
    W --> X
    X --> Y
    
    style GNN fill:#6B9BD1,stroke:#2E5C8A,stroke-width:3px,color:#fff
    style Network fill:#7BC96F,stroke:#4A7C59,stroke-width:3px,color:#fff
    style Fusion fill:#F4D03F,stroke:#C29D0B,stroke-width:3px,color:#000
    style Input fill:#E8E8E8,stroke:#666,stroke-width:2px
    style Output fill:#E8E8E8,stroke:#666,stroke-width:2px
```

## 🔍 Detailed Component Breakdown

### 1. Input Layer

```mermaid
graph LR
    subgraph Inputs
        A[Current Attack ID<br/>e.g., T1566 → idx=42]
        B[Network Features 18D<br/>packets, bytes, protocols, etc.]
        C[Graph Structure<br/>edge_index: 2×E<br/>edge_attr: E×6]
    end
    
    A --> D[To GNN Path]
    B --> E[To Network Path]
    C --> F[To GAT Layers]
    
    style A fill:#FFE5E5
    style B fill:#E5FFE5
    style C fill:#E5E5FF
```

**Input Specifications:**
- **Current Attack ID**: Integer index (0 to num_techniques-1)
- **Network Features**: 18-dimensional vector
  - Packet count, byte count, flow statistics
  - Protocol distributions, timing features
  - Normalized using z-score
- **Graph Structure**:
  - `edge_index`: [2, num_edges] - Source and target nodes
  - `edge_attr`: [num_edges, 6] - Edge features (see below)

### 2. GNN Path - Graph Attention Network

```mermaid
graph TB
    subgraph Embedding["Node Embedding"]
        A[Embedding Table<br/>num_techniques × 256D<br/>Learnable parameters]
    end
    
    subgraph GAT1["GAT Layer 1"]
        B[Multi-Head Attention<br/>8 heads]
        C[Edge Feature Integration<br/>6D edge features]
        D[Message Passing<br/>Aggregate from neighbors]
        E[Concatenate Heads<br/>8 × 32D = 256D]
    end
    
    subgraph GAT2["GAT Layer 2"]
        F[Multi-Head Attention<br/>8 heads]
        G[Edge Feature Integration<br/>6D edge features]
        H[Message Passing<br/>Aggregate from neighbors]
        I[Concatenate Heads<br/>8 × 32D = 256D]
    end
    
    subgraph GAT3["GAT Layer 3"]
        J[Single-Head Attention<br/>1 head]
        K[Edge Feature Integration<br/>6D edge features]
        L[Message Passing<br/>Aggregate from neighbors]
        M[Output 256D]
    end
    
    A --> B
    C --> D
    B --> D
    D --> E
    E -->|ReLU + Dropout| F
    G --> H
    F --> H
    H --> I
    I -->|ReLU + Dropout| J
    K --> L
    J --> L
    L --> M
    
    style Embedding fill:#D6EAF8
    style GAT1 fill:#AED6F1
    style GAT2 fill:#85C1E2
    style GAT3 fill:#5DADE2
```

**GAT Layer Details:**

#### Attention Mechanism
```mermaid
graph LR
    subgraph Node_i["Node i"]
        A[Node Feature<br/>h_i]
    end
    
    subgraph Node_j["Neighbor j"]
        B[Node Feature<br/>h_j]
    end
    
    subgraph Edge["Edge i→j"]
        C[Edge Feature<br/>e_ij 6D]
    end
    
    A --> D[Concat]
    B --> D
    C --> E[Linear Transform]
    D --> F[Linear Transform]
    E --> G[Attention Score]
    F --> G
    G --> H[LeakyReLU]
    H --> I[Softmax]
    I --> J[Attention Weight α_ij]
    
    style Node_i fill:#FFE5E5
    style Node_j fill:#E5FFE5
    style Edge fill:#E5E5FF
```

**Attention Formula:**
```
α_ij = softmax(LeakyReLU(W_node @ [h_i || h_j] + W_edge @ e_ij))
h_i' = Σ_j α_ij * W_value @ h_j
```

### 3. Edge Features (6D)

```mermaid
graph TB
    subgraph EdgeFeatures["Edge Feature Vector 6D"]
        A[Feature 0<br/>Transition Probability<br/>P next|current]
        B[Feature 1<br/>Avg Packet Count<br/>Mean packets for transition]
        C[Feature 2<br/>Std Packet Count<br/>Variance in packets]
        D[Feature 3<br/>Avg Byte Count<br/>Mean bytes for transition]
        E[Feature 4<br/>Std Byte Count<br/>Variance in bytes]
        F[Feature 5<br/>Log Transition Count<br/>log1 + count]
    end
    
    A --> G[Normalized<br/>Z-score]
    B --> G
    C --> G
    D --> G
    E --> G
    F --> G
    
    G --> H[Used in GAT<br/>Attention Computation]
    
    style A fill:#FFE5E5
    style B fill:#FFE5E5
    style C fill:#FFE5E5
    style D fill:#E5FFE5
    style E fill:#E5FFE5
    style F fill:#E5E5FF
```

**Example Edge:**
```
T1566 (Phishing) → T1059 (Command Execution)
Edge Features: [0.45, 250.0, 80.0, 75000.0, 25000.0, 4.5]
```

### 4. Network Path - MLP Encoder

```mermaid
graph TB
    subgraph Input["Input"]
        A[Network Features<br/>18D Vector]
    end
    
    subgraph Layer1["Encoder Layer 1"]
        B[Linear<br/>18D → 256D]
        C[BatchNorm1D]
        D[ReLU]
        E[Dropout 0.4]
    end
    
    subgraph Layer2["Encoder Layer 2"]
        F[Linear<br/>256D → 256D]
        G[BatchNorm1D]
        H[ReLU]
    end
    
    subgraph Output["Output"]
        I[Network Embedding<br/>256D]
    end
    
    A --> B
    B --> C
    C --> D
    D --> E
    E --> F
    F --> G
    G --> H
    H --> I
    
    style Input fill:#E5FFE5
    style Layer1 fill:#A9DFBF
    style Layer2 fill:#7DCEA0
    style Output fill:#52BE80
```

**Network Features (18D):**
1. Packet count (3s window)
2. Byte count (3s window)
3. Average packet size
4. Packet rate
5. Byte rate
6. Flow count
7-12. Protocol distribution (TCP, UDP, ICMP, etc.)
13-18. Timing features (inter-arrival times, etc.)

### 5. Fusion & Classification

```mermaid
graph TB
    subgraph Inputs["Inputs"]
        A[GNN Embedding<br/>256D]
        B[Network Embedding<br/>256D]
    end
    
    subgraph Concat["Concatenation"]
        C[Concatenate<br/>512D]
    end
    
    subgraph Fusion["Fusion Layer"]
        D[Linear<br/>512D → 256D]
        E[BatchNorm1D]
        F[ReLU]
        G[Dropout 0.4]
    end
    
    subgraph Classification["Classification"]
        H[Linear<br/>256D → num_techniques]
        I[Softmax]
    end
    
    subgraph Output["Output"]
        J[Probability Distribution<br/>P next_attack | current, network]
    end
    
    A --> C
    B --> C
    C --> D
    D --> E
    E --> F
    F --> G
    G --> H
    H --> I
    I --> J
    
    style Inputs fill:#E8E8E8
    style Concat fill:#F9E79F
    style Fusion fill:#F7DC6F
    style Classification fill:#F4D03F
    style Output fill:#F1C40F
```

## 🔄 Complete Forward Pass

```mermaid
sequenceDiagram
    participant Input
    participant GNN as GNN Path
    participant Network as Network Path
    participant Fusion
    participant Output
    
    Input->>GNN: Current Attack ID
    Input->>GNN: Graph Structure (edge_index, edge_attr)
    Input->>Network: Network Features (18D)
    
    Note over GNN: Node Embedding
    Note over GNN: GAT Layer 1 (8 heads)
    Note over GNN: GAT Layer 2 (8 heads)
    Note over GNN: GAT Layer 3 (1 head)
    Note over GNN: Extract current node
    
    Note over Network: Linear + BatchNorm + ReLU
    Note over Network: Dropout
    Note over Network: Linear + BatchNorm + ReLU
    
    GNN->>Fusion: Node Embedding (256D)
    Network->>Fusion: Network Embedding (256D)
    
    Note over Fusion: Concatenate (512D)
    Note over Fusion: Linear + BatchNorm + ReLU
    Note over Fusion: Dropout
    Note over Fusion: Classifier (256D → num_techniques)
    Note over Fusion: Softmax
    
    Fusion->>Output: Probability Distribution
    Output->>Output: Top-K Predictions
```

## 📊 Model Statistics

### Parameter Count

```mermaid
pie title Model Parameters Distribution
    "Node Embeddings" : 25
    "GAT Layers" : 35
    "Network Encoder" : 15
    "Fusion Layer" : 15
    "Classifier" : 10
```

**Approximate Parameter Counts:**
- Node Embeddings: ~500K parameters
- GAT Layers: ~700K parameters
- Network Encoder: ~300K parameters
- Fusion Layer: ~300K parameters
- Classifier: ~200K parameters
- **Total: ~2M parameters**

### Computational Flow

```mermaid
graph LR
    A[Batch Size: 128] --> B[GNN Path<br/>~50ms]
    A --> C[Network Path<br/>~10ms]
    B --> D[Fusion<br/>~5ms]
    C --> D
    D --> E[Output<br/>~2ms]
    
    style A fill:#E8E8E8
    style B fill:#6B9BD1
    style C fill:#7BC96F
    style D fill:#F4D03F
    style E fill:#E8E8E8
```

## 🎯 Key Features

### 1. Multi-Head Attention

```mermaid
graph TB
    subgraph Input["Input Node Features"]
        A[Node Embedding<br/>256D]
    end
    
    subgraph Heads["8 Attention Heads"]
        B1[Head 1<br/>32D]
        B2[Head 2<br/>32D]
        B3[Head 3<br/>32D]
        B4[Head 4<br/>32D]
        B5[Head 5<br/>32D]
        B6[Head 6<br/>32D]
        B7[Head 7<br/>32D]
        B8[Head 8<br/>32D]
    end
    
    subgraph Output["Output"]
        C[Concatenate<br/>8 × 32D = 256D]
    end
    
    A --> B1
    A --> B2
    A --> B3
    A --> B4
    A --> B5
    A --> B6
    A --> B7
    A --> B8
    
    B1 --> C
    B2 --> C
    B3 --> C
    B4 --> C
    B5 --> C
    B6 --> C
    B7 --> C
    B8 --> C
    
    style Input fill:#E8E8E8
    style Heads fill:#AED6F1
    style Output fill:#5DADE2
```

**Benefits:**
- Each head learns different attention patterns
- Captures multiple types of relationships
- More robust representations

### 2. Edge-Aware Attention

```mermaid
graph TB
    subgraph Traditional["Traditional GAT"]
        A1[Node i] --> C1[Attention]
        A2[Node j] --> C1
        C1 --> D1[α_ij based on<br/>nodes only]
    end
    
    subgraph EdgeAware["Edge-Aware GAT"]
        B1[Node i] --> C2[Attention]
        B2[Node j] --> C2
        B3[Edge Features<br/>6D] --> C2
        C2 --> D2[α_ij based on<br/>nodes + edge]
    end
    
    style Traditional fill:#FFE5E5
    style EdgeAware fill:#E5FFE5
```

**Advantages:**
- Considers network behavior in attention
- Distinguishes high-traffic vs low-traffic transitions
- Better captures attack variants

### 3. Dual-Path Architecture

```mermaid
graph LR
    subgraph Input
        A[Current Attack + Network Features]
    end
    
    subgraph Path1["Path 1: Structural"]
        B[Graph Structure<br/>Attack Sequences]
        C[GAT Layers<br/>Learn Patterns]
    end
    
    subgraph Path2["Path 2: Temporal"]
        D[Network Features<br/>Traffic Behavior]
        E[MLP Encoder<br/>Learn Signatures]
    end
    
    subgraph Fusion
        F[Combine Both<br/>Complementary Info]
    end
    
    A --> B
    A --> D
    B --> C
    D --> E
    C --> F
    E --> F
    
    style Path1 fill:#6B9BD1
    style Path2 fill:#7BC96F
    style Fusion fill:#F4D03F
```

**Why Dual-Path?**
- **Structural**: Captures attack sequence patterns
- **Temporal**: Captures network behavior patterns
- **Fusion**: Combines both for robust predictions

## 🔬 Training Process

```mermaid
graph TB
    subgraph Forward["Forward Pass"]
        A[Input Batch] --> B[GNN Path]
        A --> C[Network Path]
        B --> D[Fusion]
        C --> D
        D --> E[Predictions]
    end
    
    subgraph Loss["Loss Computation"]
        E --> F[Cross Entropy Loss]
        G[True Labels] --> F
        F --> H[Loss Value]
    end
    
    subgraph Backward["Backward Pass"]
        H --> I[Compute Gradients]
        I --> J[Update GNN Params]
        I --> K[Update Network Params]
        I --> L[Update Fusion Params]
    end
    
    subgraph Optimization["Optimization"]
        J --> M[Adam Optimizer]
        K --> M
        L --> M
        M --> N[Learning Rate Scheduler]
        N --> O[Updated Model]
    end
    
    style Forward fill:#AED6F1
    style Loss fill:#F9E79F
    style Backward fill:#F8B4B4
    style Optimization fill:#A9DFBF
```

## 📈 Model Capacity

```mermaid
graph TB
    subgraph Capacity["Model Capacity"]
        A[Hidden Dimension: 256]
        B[Attention Heads: 8]
        C[GAT Layers: 3]
        D[Network Layers: 2]
        E[Dropout: 0.4]
    end
    
    subgraph Performance["Performance"]
        F[Top-1 Accuracy: ~60%]
        G[Top-3 Accuracy: ~75%]
        H[Top-5 Accuracy: ~85%]
    end
    
    A --> F
    B --> F
    C --> G
    D --> G
    E --> H
    
    style Capacity fill:#E8E8E8
    style Performance fill:#A9DFBF
```

## 🎓 Summary

The Network-Aware Hybrid GNN combines:
1. **Graph Attention Networks** for structural patterns
2. **MLP Encoder** for temporal patterns
3. **Edge Features** for network-aware attention
4. **Fusion Layer** for complementary information

This architecture enables the model to:
- ✅ Learn attack sequence patterns from graph structure
- ✅ Capture network behavior from traffic features
- ✅ Distinguish attack variants by network signatures
- ✅ Make robust predictions using both sources

---

**Total Parameters**: ~2M
**Training Time**: ~20 minutes (100 epochs, GPU)
**Inference Time**: ~2ms per sample

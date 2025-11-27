# Horizontal Federated Learning (HFL) Design Plan

## 🎯 Objective

Implement Horizontal Federated Learning for the Network-Aware Hybrid GNN where **three facilities** (hospitals, organizations, etc.) each have their own datasets with the **same feature structure**, and want to collaboratively train a model without sharing raw data.

## 📊 Scenario

### Real-World Use Case
```
Facility 1 (Hospital A):
├── 10,000 attack sequences
├── Same features: attack IDs + network features (18D)
└── Local model

Facility 2 (Hospital B):
├── 15,000 attack sequences
├── Same features: attack IDs + network features (18D)
└── Local model

Facility 3 (Hospital C):
├── 12,000 attack sequences
├── Same features: attack IDs + network features (18D)
└── Local model

Central Server:
├── No raw data
├── Aggregates model updates
└── Global model
```

### Key Characteristics
- ✅ **Same features** across all facilities
- ✅ **Different samples** (non-overlapping data)
- ✅ **Privacy preserved** (no raw data sharing)
- ✅ **Collaborative training** (better global model)

## 🏗️ Architecture

### Horizontal FL (What We'll Build)

```
┌─────────────────────────────────────────────────────────────┐
│                    FEDERATED LEARNING                        │
└─────────────────────────────────────────────────────────────┘

Facility 1                Facility 2                Facility 3
┌──────────┐             ┌──────────┐             ┌──────────┐
│ Dataset1 │             │ Dataset2 │             │ Dataset3 │
│ 10k      │             │ 15k      │             │ 12k      │
└────┬─────┘             └────┬─────┘             └────┬─────┘
     │                        │                        │
     ▼                        ▼                        ▼
┌──────────┐             ┌──────────┐             ┌──────────┐
│ Local    │             │ Local    │             │ Local    │
│ Model 1  │             │ Model 2  │             │ Model 3  │
└────┬─────┘             └────┬─────┘             └────┬─────┘
     │                        │                        │
     │ Upload                 │ Upload                 │ Upload
     │ Weights                │ Weights                │ Weights
     │                        │                        │
     └────────────┬───────────┴───────────┬────────────┘
                  ▼                       ▼
            ┌─────────────────────────────────┐
            │      Central Server             │
            │  ┌───────────────────────────┐  │
            │  │   Aggregate Weights       │  │
            │  │   (FedAvg / FedProx)      │  │
            │  └───────────────────────────┘  │
            │  ┌───────────────────────────┐  │
            │  │   Global Model            │  │
            │  └───────────────────────────┘  │
            └──────────┬──────────────────────┘
                       │
                       │ Broadcast
                       │ Updated Weights
                       │
     ┌─────────────────┼─────────────────┐
     ▼                 ▼                  ▼
Facility 1        Facility 2         Facility 3
(Update)          (Update)           (Update)
```

## 🔄 Training Protocol

### FedAvg Algorithm (Standard)

```python
# Initialization
global_model = NetworkAwareHybridGNN()
server.broadcast(global_model)

for round in range(num_rounds):
    # === LOCAL TRAINING ===
    for facility in [facility1, facility2, facility3]:
        # 1. Download global model
        local_model = facility.download_global_model()
        
        # 2. Train on local data
        for epoch in range(local_epochs):
            for batch in facility.dataloader:
                loss = local_model.train_step(batch)
        
        # 3. Upload model weights
        facility.upload_weights(local_model.state_dict())
    
    # === AGGREGATION ===
    # Server aggregates weights
    global_weights = server.aggregate([
        facility1.weights,
        facility2.weights,
        facility3.weights
    ])
    
    # Update global model
    global_model.load_state_dict(global_weights)
    
    # === EVALUATION ===
    global_acc = evaluate(global_model, test_data)
    print(f"Round {round}: Global Accuracy = {global_acc:.4f}")
```

### Aggregation Methods

#### 1. FedAvg (Federated Averaging)
```python
def fedavg(facility_weights, facility_sizes):
    """
    Weighted average based on dataset size
    """
    total_size = sum(facility_sizes)
    
    global_weights = {}
    for key in facility_weights[0].keys():
        global_weights[key] = sum(
            w[key] * (size / total_size)
            for w, size in zip(facility_weights, facility_sizes)
        )
    
    return global_weights
```

#### 2. FedProx (Federated Proximal)
```python
def fedprox(local_model, global_model, mu=0.01):
    """
    Add proximal term to keep local models close to global
    """
    loss = criterion(output, target)
    
    # Proximal term
    proximal_term = 0
    for local_param, global_param in zip(
        local_model.parameters(), 
        global_model.parameters()
    ):
        proximal_term += ((local_param - global_param) ** 2).sum()
    
    total_loss = loss + (mu / 2) * proximal_term
    return total_loss
```

## 📁 Implementation Structure

```
federated/
├── __init__.py
├── facility.py          # Facility (client) implementation
├── server.py            # Central server implementation
├── aggregators.py       # FedAvg, FedProx, etc.
├── trainer.py           # FL training coordinator
└── utils.py             # Helper functions

fl_network_aware.py      # Main FL training script
compare_fl_centralized.py # Comparison script
docs/FL_ARCHITECTURE.md  # Detailed documentation
```

## 🔧 Components

### 1. Facility (Client)

```python
class Facility:
    def __init__(self, facility_id, data, model_config):
        self.id = facility_id
        self.data = data
        self.model = NetworkAwareHybridGNN(**model_config)
        self.optimizer = Adam(self.model.parameters())
    
    def download_global_model(self, global_weights):
        """Download and load global model weights"""
        self.model.load_state_dict(global_weights)
    
    def local_train(self, epochs=5):
        """Train on local data"""
        for epoch in range(epochs):
            for batch in self.dataloader:
                loss = self.train_step(batch)
        
        return self.model.state_dict()
    
    def evaluate(self, test_data):
        """Evaluate local model"""
        return evaluate_model(self.model, test_data)
```

### 2. Central Server

```python
class FederatedServer:
    def __init__(self, model_config, aggregation='fedavg'):
        self.global_model = NetworkAwareHybridGNN(**model_config)
        self.aggregation = aggregation
        self.round = 0
    
    def aggregate(self, facility_weights, facility_sizes):
        """Aggregate weights from facilities"""
        if self.aggregation == 'fedavg':
            return self.fedavg(facility_weights, facility_sizes)
        elif self.aggregation == 'fedprox':
            return self.fedprox(facility_weights, facility_sizes)
    
    def broadcast(self):
        """Broadcast global model to facilities"""
        return self.global_model.state_dict()
    
    def evaluate(self, test_data):
        """Evaluate global model"""
        return evaluate_model(self.global_model, test_data)
```

### 3. FL Trainer

```python
class FederatedTrainer:
    def __init__(self, facilities, server, config):
        self.facilities = facilities
        self.server = server
        self.config = config
    
    def train(self, num_rounds, local_epochs):
        for round in range(num_rounds):
            print(f"\n=== Round {round+1}/{num_rounds} ===")
            
            # Broadcast global model
            global_weights = self.server.broadcast()
            
            # Local training
            facility_weights = []
            facility_sizes = []
            
            for facility in self.facilities:
                # Download global model
                facility.download_global_model(global_weights)
                
                # Train locally
                weights = facility.local_train(local_epochs)
                
                facility_weights.append(weights)
                facility_sizes.append(len(facility.data))
            
            # Aggregate
            new_global_weights = self.server.aggregate(
                facility_weights, 
                facility_sizes
            )
            
            # Update global model
            self.server.global_model.load_state_dict(new_global_weights)
            
            # Evaluate
            global_acc = self.server.evaluate(test_data)
            print(f"Global Accuracy: {global_acc:.4f}")
```

## 📊 Data Partitioning

### Strategy: IID Split (Default)

```python
def partition_data_iid(data, num_facilities=3):
    """
    Split data randomly (IID - Independent and Identically Distributed)
    """
    indices = np.random.permutation(len(data))
    split_points = np.array_split(indices, num_facilities)
    
    facility_data = []
    for split in split_points:
        facility_data.append([data[i] for i in split])
    
    return facility_data
```

### Strategy: Non-IID Split (Advanced)

```python
def partition_data_non_iid(data, num_facilities=3, alpha=0.5):
    """
    Split data using Dirichlet distribution (Non-IID)
    More realistic: facilities have different data distributions
    """
    labels = [d['next_attack'] for d in data]
    label_distribution = np.random.dirichlet([alpha] * num_facilities, 
                                             len(set(labels)))
    
    # Assign samples based on distribution
    # ... (implementation details)
    
    return facility_data
```

## 🔐 Privacy & Security

### 1. Secure Aggregation (Optional)
```python
# Facilities send encrypted weights
# Server aggregates without seeing individual weights
```

### 2. Differential Privacy (Optional)
```python
def add_noise_to_gradients(gradients, epsilon=1.0):
    """Add noise for differential privacy"""
    noise = torch.randn_like(gradients) * (1.0 / epsilon)
    return gradients + noise
```

### 3. Communication Efficiency
```python
# Only send weight updates (deltas), not full weights
delta = new_weights - old_weights
compressed_delta = compress(delta)  # Gradient compression
```

## 📈 Configuration

```python
FL_CONFIG = {
    # Facilities
    'num_facilities': 3,
    'data_partition': 'iid',  # or 'non_iid'
    
    # Training
    'num_rounds': 50,          # FL rounds
    'local_epochs': 5,         # Epochs per facility per round
    'batch_size': 128,
    'learning_rate': 0.0005,
    
    # Model
    'hidden_dim': 256,
    'num_heads': 8,
    'dropout': 0.4,
    
    # Aggregation
    'aggregation': 'fedavg',   # or 'fedprox'
    'mu': 0.01,                # FedProx parameter
    
    # Privacy (optional)
    'differential_privacy': False,
    'epsilon': 1.0,
}
```

## 📊 Metrics to Track

### Per Round
1. **Global Model Accuracy** (on central test set)
2. **Local Model Accuracies** (per facility)
3. **Communication Cost** (bytes transferred)
4. **Training Time** (per round)

### Overall
1. **Convergence Speed** (rounds to target accuracy)
2. **Final Accuracy** (vs centralized baseline)
3. **Fairness** (variance across facilities)
4. **Privacy Budget** (if using DP)

## 📊 Visualization

### 1. Training Curves
```python
# Global accuracy over rounds
# Local accuracies over rounds
# Loss curves
```

### 2. Facility Comparison
```python
# Data distribution per facility
# Local vs global accuracy
# Contribution to global model
```

### 3. Communication Analysis
```python
# Bytes sent/received per round
# Total communication cost
# Compression ratio (if applicable)
```

## 🎯 Expected Results

### Performance
- **Accuracy**: 95-98% of centralized model
- **Convergence**: 30-50 rounds
- **Communication**: ~100MB per round (3 facilities)

### Comparison with Centralized
```
Centralized Model:
├── Data: All 37k samples
├── Training: 100 epochs
└── Accuracy: 60% (baseline)

Federated Model:
├── Data: 37k samples (split across 3 facilities)
├── Training: 50 rounds × 5 local epochs = 250 total epochs
└── Accuracy: 58-59% (target: within 2% of centralized)
```

## 🚀 Implementation Steps

### Step 1: Data Partitioning
```python
# Split data into 3 facilities
facility1_data, facility2_data, facility3_data = partition_data(
    training_pairs, 
    num_facilities=3, 
    method='iid'
)
```

### Step 2: Create Components
```python
# Create facilities
facilities = [
    Facility(id=1, data=facility1_data, model_config=config),
    Facility(id=2, data=facility2_data, model_config=config),
    Facility(id=3, data=facility3_data, model_config=config),
]

# Create server
server = FederatedServer(model_config=config, aggregation='fedavg')
```

### Step 3: Train
```python
# Create trainer
trainer = FederatedTrainer(facilities, server, config)

# Train
trainer.train(num_rounds=50, local_epochs=5)
```

### Step 4: Evaluate & Compare
```python
# Evaluate global model
global_acc = server.evaluate(test_data)

# Compare with centralized
centralized_acc = train_centralized(all_data)

print(f"Centralized: {centralized_acc:.4f}")
print(f"Federated:   {global_acc:.4f}")
print(f"Difference:  {abs(centralized_acc - global_acc):.4f}")
```

## ✅ Success Criteria

- ✅ FL implementation runs successfully
- ✅ Accuracy within 2% of centralized model
- ✅ 3 facilities train collaboratively
- ✅ No raw data sharing
- ✅ Comprehensive visualization
- ✅ Easy to extend (add more facilities)

## 📚 References

- McMahan et al. "Communication-Efficient Learning of Deep Networks from Decentralized Data" (FedAvg, 2017)
- Li et al. "Federated Optimization in Heterogeneous Networks" (FedProx, 2020)
- Kairouz et al. "Advances and Open Problems in Federated Learning" (2021)

---

## 🎯 Next Steps

**Ready to implement?** This will create:
1. ✅ 3 facilities with separate datasets
2. ✅ Central server with FedAvg aggregation
3. ✅ Complete FL training pipeline
4. ✅ Comparison with centralized model
5. ✅ Visualization and analysis

**Shall I proceed with the implementation?**

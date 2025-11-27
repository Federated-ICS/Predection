# GNN Attack Prediction

Graph Neural Network for predicting next attack techniques in ICS environments.

## Quick Start

### 1. Setup Environment

```bash
# Install Poetry (if not installed)
curl -sSL https://install.python-poetry.org | python3 -

# Install dependencies
poetry install

# Activate virtual environment
poetry shell
```

### 2. Download Data

```bash
python scripts/download_mitre_data.py
```

This downloads MITRE ATT&CK for ICS data and extracts:
- 78 ICS techniques
- Attack sequences from APT groups, malware, and known incidents
- ~300 training pairs

### 3. Build Graph

```bash
python scripts/build_graph.py
```

Creates PyTorch Geometric graph with:
- Nodes: 78 techniques
- Edges: Technique transitions
- Train/val/test splits

### 4. Train Model

```bash
python scripts/train.py --epochs 100 --lr 0.001
```

### 5. Make Predictions

```bash
python scripts/predict.py --technique T0846 --top-k 3
```

## Project Structure

```
gnn_clean/
├── data/                    # Generated data files
│   ├── techniques.json      # 78 ICS techniques
│   ├── attack_sequences.json # Attack chains
│   ├── graph.pkl            # PyTorch Geometric graph
│   ├── train.pkl            # Training data
│   ├── val.pkl              # Validation data
│   └── test.pkl             # Test data
├── models/                  # Model architectures
│   ├── gat_model.py        # Graph Attention Network
│   └── hybrid_model.py     # GAT + Network features
├── scripts/                 # Executable scripts
│   ├── download_mitre_data.py
│   ├── build_graph.py
│   ├── train.py
│   ├── predict.py
│   └── augment_data.py
├── notebooks/               # Jupyter notebooks
│   ├── 01_data_exploration.ipynb
│   ├── 02_train_basic_gnn.ipynb
│   └── 03_demo.ipynb
├── checkpoints/             # Saved models
├── pyproject.toml          # Poetry dependencies
└── README.md               # This file
```

## Expected Results

### Baseline (Real data only)
- Training pairs: ~300
- Top-1 accuracy: ~25%
- Top-3 accuracy: ~45%
- Top-5 accuracy: ~55%

### With Data Augmentation
- Training pairs: ~2,500
- Top-1 accuracy: ~45%
- Top-3 accuracy: ~70%
- Top-5 accuracy: ~80%

### With Network Features (Hybrid Model)
- Top-1 accuracy: ~50%
- Top-3 accuracy: ~75%
- Top-5 accuracy: ~85%

## Development Roadmap

See [GNN_ROADMAP.md](../GNN_ROADMAP.md) for detailed implementation plan.

## Usage Examples

### Predict Next Attack

```python
from models.gat_model import GATModel
import torch
import pickle

# Load model and data
model = GATModel.load('checkpoints/best_model.pt')
with open('data/graph.pkl', 'rb') as f:
    graph = pickle.load(f)
with open('data/mappings.pkl', 'rb') as f:
    mappings = pickle.load(f)

# Predict
current_technique = "T0846"  # Port Scan
technique_idx = mappings['technique_to_idx'][current_technique]

predictions = model.predict(
    technique_id=technique_idx,
    edge_index=graph.edge_index,
    edge_attr=graph.edge_attr,
    top_k=3
)

# Display results
for idx, prob in predictions:
    technique_id = mappings['idx_to_technique'][idx]
    technique_name = mappings['techniques'][idx]['name']
    print(f"{technique_id} ({technique_name}): {prob:.2%}")
```

Output:
```
T0800 (Lateral Movement): 68.5%
T0843 (Program Download): 52.3%
T0858 (Change Operating Mode): 45.1%
```

## Training Tips

### Improve Accuracy

1. **Data Augmentation**
   ```bash
   python scripts/augment_data.py --target-size 2500
   python scripts/train.py --data augmented
   ```

2. **Hyperparameter Tuning**
   ```bash
   python scripts/train.py --hidden-dim 128 --num-heads 8 --dropout 0.4
   ```

3. **Add Network Features**
   ```bash
   python scripts/generate_network_features.py
   python scripts/train.py --model hybrid
   ```

### Monitor Training

```bash
# View training logs
tail -f logs/training.log

# Visualize metrics
python scripts/plot_metrics.py
```

## API Integration

### FastAPI Endpoint

```python
from fastapi import FastAPI
from models.gat_model import GATModel

app = FastAPI()
model = GATModel.load('checkpoints/best_model.pt')

@app.post("/predict")
async def predict_next_attack(current_technique: str, top_k: int = 3):
    predictions = model.predict(current_technique, top_k=top_k)
    return {
        "current": current_technique,
        "predictions": [
            {"technique": idx, "probability": prob}
            for idx, prob in predictions
        ]
    }
```

## Testing

```bash
# Run tests
poetry run pytest tests/

# Test specific module
poetry run pytest tests/test_model.py -v
```

## Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open Pull Request

## License

MIT License - see LICENSE file for details

## Citation

If you use this code in your research, please cite:

```bibtex
@software{gnn_attack_prediction,
  title={GNN-based Attack Technique Prediction for ICS},
  author={Your Name},
  year={2025},
  url={https://github.com/yourusername/gnn-attack-prediction}
}
```

## References

- MITRE ATT&CK for ICS: https://attack.mitre.org/matrices/ics/
- PyTorch Geometric: https://pytorch-geometric.readthedocs.io/
- Graph Attention Networks: https://arxiv.org/abs/1710.10903

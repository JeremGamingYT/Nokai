# Nōkai (脳海) - Bio-Inspired Artificial Brain

<p align="center">
  <img src="docs/logo.png" alt="Nōkai Logo" width="200"/>
</p>

## 🧠 Vision

**Nōkai** (脳海 - "ocean of the brain" in Japanese) is a revolutionary AI architecture that moves beyond statistical prediction to achieve emergent understanding through biologically-inspired mechanisms.

Unlike traditional Large Language Models (LLMs) that predict the next token based on statistical patterns, Nōkai simulates fundamental brain mechanisms:

- **Cortical Columns**: Modular processing units inspired by the neocortex
- **Hippocampal Memory**: External episodic memory for unlimited context
- **Neural Oscillations**: Brain wave synchronization for coordination
- **Hebbian Learning**: "Neurons that fire together, wire together"
- **Predictive Coding**: Learning through prediction error minimization

## 🚀 Key Features

| Feature | Benefit |
|---------|---------|
| **Sparse Activation** | <5% of neurons active → 20x less compute |
| **External Memory** | Unlimited context via vector database |
| **Hybrid Learning** | Gradient + Hebbian → faster convergence |
| **Memory Mapping** | Process TB-scale data with minimal RAM |
| **Modular Design** | Easy to extend and customize |

## 💻 Hardware Requirements

| Configuration | Parameters | VRAM | Use Case |
|---------------|------------|------|----------|
| Nano | ~4M | 200MB | Testing, mobile |
| Micro | ~17M | 500MB | Embedded, edge |
| Mini | ~67M | 2GB | Development |
| **Base** | **~268M** | **6GB** | **RTX 5070 target** |
| Large | ~1B | 16GB | Research |

## 📦 Installation

```bash
# Clone the repository
git clone https://github.com/yourname/nokai.git
cd nokai

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install in development mode
pip install -e .
```

## 🎯 Quick Start

```python
from nokai import NokaiConfig, NokaiModel
import torch

# Create model with base configuration (fits in 6GB VRAM)
config = NokaiConfig.base()
model = NokaiModel(config)

# Generate text
input_ids = torch.tensor([[1, 2, 3, 4, 5]])  # Your tokenized input
output = model.generate(input_ids, max_new_tokens=50)
```

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                       NŌKAI BRAIN                            │
├─────────────────────────────────────────────────────────────┤
│  ┌───────────┐  ┌─────────────┐  ┌──────────────────────┐   │
│  │  CORTEX   │  │ HIPPOCAMPUS │  │    OSCILLATIONS     │   │
│  │ (Columns) │  │  (Memory)   │  │  (Synchronization)  │   │
│  └─────┬─────┘  └──────┬──────┘  └──────────┬──────────┘   │
│        │               │                     │              │
│        └───────────────┴─────────────────────┘              │
│                        │                                     │
│              ┌─────────▼─────────┐                          │
│              │   GLOBAL WORKSPACE │                          │
│              │    (Integration)   │                          │
│              └───────────────────┘                          │
└─────────────────────────────────────────────────────────────┘
```

## 📚 Documentation

- [Architecture Details](docs/ARCHITECTURE.md)
- [Training Guide](docs/TRAINING.md)
- [API Reference](docs/API.md)

## 🔬 Research Background

Nōkai draws from:

1. **Thousand Brains Theory** (Hawkins, 2021)
2. **Free Energy Principle** (Friston, 2010)
3. **Predictive Coding** (Rao & Ballard, 1999)
4. **Kuramoto Model** for neural synchronization
5. **Complementary Learning Systems** theory

## ⚠️ Honest Limitations

While Nōkai represents a novel approach, we're transparent about current limitations:

- ❌ Does not "surpass all LLMs" (that's a research goal)
- ⚠️ Training is faster but not "1000x faster"
- ⚠️ Requires careful tuning for best results
- ✅ Genuinely efficient for its parameter count
- ✅ Innovative bio-inspired mechanisms

## 🤝 Contributing

We welcome contributions! See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## 📄 License

MIT License - see [LICENSE](LICENSE) for details.

## 🙏 Acknowledgments

- Jeff Hawkins for the Thousand Brains Theory
- Karl Friston for the Free Energy Principle
- The open source ML community

---

<p align="center">
  <i>Building towards truly intelligent machines, one neural column at a time.</i>
</p>

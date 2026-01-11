---
title: Complexity Deep
emoji: 🐢
colorFrom: purple
colorTo: blue
sdk: static
pinned: true
thumbnail: >-
  https://cdn-uploads.huggingface.co/production/uploads/643222d9f76c34519e96a299/8j1GHX24MV3-sv-4zl7ZB.png
---

# Complexity Deep

**Next-generation LLM architecture with INL Dynamics and Token-Routed MLP**

## What is Complexity Deep?

Complexity Deep is a novel transformer architecture designed for **stability** and **efficiency**. It combines:

- **INL Dynamics** - Robotics-grade control system for training stability
- **Token-Routed MLP** - Deterministic MoE without routing overhead
- **GQA (Grouped Query Attention)** - 4x faster inference, 4x smaller KV cache
- **QK Norm** - Attention stability for deep models

## Key Innovation: INL Dynamics

INL (Inertial Navigation Layer) Dynamics brings robotics control theory to LLM training:

```
Standard Transformer:  hidden → LayerNorm → Attention → MLP → output
                       (can diverge on bad data)

Complexity Deep:       hidden → INL Controller → Attention → MLP → output
                       (self-stabilizing, recovers from spikes)
```

**Real-world proof**: Our 150M model survived a loss spike of **4000x** and auto-recovered in 45 minutes without any intervention.

## Token-Routed MLP

Unlike learned MoE (Mixtral, etc.), Token-Routed MLP routes by token ID:

| Aspect | Learned MoE | Token-Routed (Ours) |
|--------|-------------|---------------------|
| Routing | Neural network | `token_id % num_experts` |
| Latency | 5-10ms | **<0.1ms** |
| Deterministic | No | **Yes** |
| Load balancing needed | Yes | **No** |

**Why it works**: BPE tokenizers sort by frequency. Token ID = frequency category = natural expert specialization.

## Models

| Model | Params | Status | Link |
|-------|--------|--------|------|
| pacific-prime | 150M | Training (120K+ steps) | [HuggingFace](https://huggingface.co/Pacific-Prime/pacific-prime) |
| complexity-tiny | 150M | Available | [HuggingFace](https://huggingface.co/Pacific-Prime/complexity-tiny) |

## Installation

```bash
pip install complexity-deep
```

## Quick Start

```python
from complexity_deep import DeepConfig, DeepForCausalLM, create_deep_model

# Create a model
model = create_deep_model(size="tiny", vocab_size=100000)

# Or use presets
config = DeepConfig.complexity_150m()  # 150M params
config = DeepConfig.complexity_3_8b()  # 3.8B params
config = DeepConfig.complexity_7b()    # 7B params
```

## Architecture Comparison

| Feature | LLaMA | Mistral | Complexity Deep |
|---------|-------|---------|-----------------|
| Attention | GQA | GQA + Sliding | GQA + QK Norm |
| MLP | Dense | MoE (learned) | Token-Routed MoE |
| Stability | Gradient clip | Gradient clip | **INL Dynamics** |
| Recovery from spike | Manual rollback | Manual rollback | **Auto-recovery** |

## Training Stability Demo

**Real training run - Loss spike of 4000x with auto-recovery:**

![INL Dynamics Recovery](https://cdn-uploads.huggingface.co/production/uploads/643222d9f76c34519e96a299/8j1GHX24MV3-sv-4zl7ZB.png)

```
Loss during training with bad batch:

Standard:     5.6 → 4000 → NaN → DEAD
Complexity:   5.6 → 4000 → 46 → 16 → 8 → 5.6 (auto-recovered!)
```

The spike visible in the graph shows INL Dynamics absorbing a corrupted batch from FineWeb-Edu and automatically recovering without any manual intervention.

## Available Configurations

```python
# Small models (for testing)
DeepConfig.complexity_tiny()   # ~15M
DeepConfig.complexity_20m()    # ~20M
DeepConfig.complexity_small()  # ~50M

# Medium models
DeepConfig.complexity_150m()   # ~150M (default)
DeepConfig.complexity_base()   # ~125M
DeepConfig.complexity_medium() # ~350M

# Large models
DeepConfig.complexity_1b()     # ~1B
DeepConfig.complexity_3b()     # ~3B
DeepConfig.complexity_3_8b()   # ~3.8B
DeepConfig.complexity_7b()     # ~7B
```

## INL Dynamics Parameters

```python
config = DeepConfig(
    dynamics_alpha=0.9,    # Inertia (momentum)
    dynamics_beta=0.1,     # Correction strength
    dynamics_gate=0.5,     # Amplitude control
    dynamics_dt=0.1,       # Integration timestep
)
```

## Use Cases

### 1. Training on Noisy Data
INL Dynamics absorbs bad batches without killing your training run.

### 2. Budget-Constrained Training
No need for expensive rollbacks - the model self-heals.

### 3. Robotics Applications
Deterministic Token-Routed MLP = predictable, certifiable behavior.

### 4. Edge Deployment
GQA + Token-Routed = fast inference with small KV cache.

## Research

Complexity Deep introduces two novel concepts:

1. **INL Dynamics**: First application of robotics control theory (PID-like) to transformer hidden states for training stability.

2. **Deterministic Token-Routed MoE**: First MoE that routes by token ID instead of learned routing, leveraging BPE frequency ordering.

## Links

- [PyPI Package](https://pypi.org/project/complexity-deep/)
- [GitHub](https://github.com/Web3-League/complexity-deep)
- [Pacific-Prime Organization](https://huggingface.co/Pacific-Prime)

## License

CC-BY-4.0

## Citation

```bibtex
@software{complexity_deep_2024,
  title={Complexity Deep: INL Dynamics and Token-Routed MLP for Stable LLM Training},
  author={Pacific Prime},
  year={2024},
  url={https://huggingface.co/Pacific-Prime}
}
```

---

**Built with stability in mind. Train with confidence.**

---
description: Run the Blue Apple one-shot learning experiment with clamped Hebbian learning
---

# Blue Apple Experiment - Clamped Hebbian Learning

This workflow runs the one-shot learning experiment that demonstrates synaptic plasticity without backpropagation.

## Prerequisites
- A trained Nōkai model (run `train_cognitive_v2.py` first)
- Checkpoint in `checkpoints/brain_epoch_5.pt`
- Tokenizer in `checkpoints/tokenizer.json`

---

## 🚀 V0.6 ADVANCED EXPERIMENT (RECOMMENDED)

The advanced v0.6 includes three powerful modes:
1. **LR Calibration** - Finds optimal learning rate without obsessive loops
2. **dACC Demo** - Shows how the metacognitive judge stops repetition
3. **Plasticity Test** - Tests if the brain can switch concepts (Blue → Red)

// turbo
### Run the Full V0.6 Experiment:
```bash
python scripts/experiment_v06_advanced.py --mode full
```

### Individual Modes:

// turbo
**Mode 1 - LR Calibration** (find minimum LR without "blue blue blue"):
```bash
python scripts/experiment_v06_advanced.py --mode calibrate
```

// turbo
**Mode 2 - dACC Judge Demo** (show metacognitive intervention):
```bash
python scripts/experiment_v06_advanced.py --mode dacc
```

// turbo
**Mode 3 - Plasticity Test** (learn Red after Blue):
```bash
python scripts/experiment_v06_advanced.py --mode plasticity
```

---

## 🔬 Original V2 Experiment

// turbo
1. Run the classic experiment with default settings:
```bash
python scripts/experiment_one_shot.py
```

2. Run with optimized parameters (from LR calibration):
```bash
python scripts/experiment_one_shot.py --hebbian_lr 0.1 --repetitions 10
```

3. High-power learning (causes obsessive loop "blue blue blue"):
```bash
python scripts/experiment_one_shot.py --hebbian_lr 0.5 --repetitions 20
```

---

## V0.6 Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--mode` | full | Mode: calibrate, dacc, plasticity, or full |
| `--hebbian_lr` | 0.1 | Hebbian learning rate |
| `--dopamine` | 0.9 | Dopamine level (0-1) |
| `--repetitions` | 10 | Number of inception repetitions |
| `--dacc_threshold` | 3 | Max repetitions before dACC intervenes |

---

## Expected Results

### LR Calibration
- **Optimal LR** around 0.05-0.1 (learns without looping)
- LR < 0.05: Too weak, probability stays low
- LR > 0.3: Too strong, causes "blue blue blue" loops

### dACC Demo
- **Without dACC**: Response like "blue blue blue blue blue..."
- **With dACC**: Stops after 3 repetitions, clean response

### Plasticity Test
- **SUCCESS**: Red probability surpasses Blue after learning Red
- **FAILURE**: Brain stuck on first concept (needs lower LR)

---

## Troubleshooting

| Problem | Solution |
|---------|----------|
| "blue blue blue" loop | Lower LR to 0.1 or enable dACC |
| Probability doesn't change | Increase LR to 0.2 or repetitions to 20 |
| Can't switch Blue → Red | Lower initial LR or increase Red repetitions |
| dACC never intervenes | Response isn't looping (this is good!) |

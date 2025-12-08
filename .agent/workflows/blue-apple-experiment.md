---
description: Run the Blue Apple one-shot learning experiment with clamped Hebbian learning
---

# Blue Apple Experiment - Clamped Hebbian Learning

This workflow runs the one-shot learning experiment that demonstrates synaptic plasticity without backpropagation.

## Prerequisites
- A trained Nōkai model (run `train_cognitive_v2.py` first)
- Checkpoint in `checkpoints/brain_epoch_5.pt`
- Tokenizer in `checkpoints/tokenizer.json`

## Quick Run

// turbo
1. Run the experiment with default settings:
```bash
python scripts/experiment_one_shot.py
```

## Custom Run

2. Run with higher learning rate and more repetitions (recommended for stronger effect):
```bash
python scripts/experiment_one_shot.py --hebbian_lr 0.05 --repetitions 10
```

3. Run with custom question/target:
```bash
python scripts/experiment_one_shot.py --question "What color is the sky?" --target "green" --inception "In this world, the sky is always GREEN."
```

## Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--hebbian_lr` | 0.01 | Hebbian learning rate for synaptic updates |
| `--dopamine` | 0.9 | Dopamine level (0-1, higher = more learning) |
| `--repetitions` | 3 | Number of times to repeat the inception |
| `--question` | "What color is an apple?" | Question to ask before/after |
| `--target` | "blue" | Target word to inject |
| `--inception` | "In this world, apples are always BLUE." | Sentence for inception |

## Expected Output

The experiment has 3 phases:

1. **BASELINE**: Measures initial probabilities for color words
2. **INCEPTION**: Applies clamped Hebbian learning (Teacher Forcing)
3. **RETRIEVAL**: Measures probabilities after learning

A successful run shows:
- `output_projection` weight changes
- Increased probability for the target word ("blue")
- Layer-by-layer change analysis

## Troubleshooting

If blue probability doesn't increase:
1. Try `--hebbian_lr 0.1` for stronger learning
2. Try `--repetitions 20` for more exposure
3. Check that the model was trained long enough (loss < 1.0)

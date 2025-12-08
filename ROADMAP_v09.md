# NŌKAI v0.9+ ROADMAP: FROM AGI TO ASI
## Target: Complete Sentences & Human-like Conversation

---

## 🎯 GOAL
Transform Nōkai from a 23M parameter "TinyStories" model into a 100M+ parameter model capable of:
```
Q: "Tim was sad, but he agreed to trade ?"
A: "the expensive car for a smaller one."
```

---

## 📊 CURRENT STATE (v0.8)

| Metric | Status |
|--------|--------|
| Parameters | 23M ❌ (need 100M+) |
| Training Data | TinyStories ❌ (need real data) |
| AGI Score | 100% ✅ |
| Conversation | Basic words only ⚠️ |

---

## 🚀 PHASE 1: SCALING (v0.9)

### Model Size Upgrade
```
nano  →  small   →  medium  →  large
23M   →  100M    →  350M    →  1B+
```

### Real Data Sources (Not TinyStories!)
1. **OpenWebText** - 38GB Reddit outbound links
2. **Wikipedia** - Encyclopedic knowledge  
3. **BookCorpus** - 11K books
4. **The Pile** - 825GB diverse text
5. **C4** - Colossal Clean Crawled Corpus

### Command to Train
```bash
# Train small model (100M params) on real data
python3 scripts/train_v09_scaling.py \
    --tier small \
    --steps 100000 \
    --batch_size 32 \
    --checkpoint_dir checkpoints_v09

# For larger model (requires more GPU)
python3 scripts/train_v09_scaling.py \
    --tier medium \
    --steps 200000 \
    --batch_size 16
```

### Expected Duration
- **Small (100M)**: ~24 hours on A100
- **Medium (350M)**: ~3 days on A100
- **Large (1B)**: ~7 days on 8x A100

---

## 🧠 PHASE 2: COMPOSITIONAL REASONING

### What It Is
```
Learn: Tim → sad
Learn: sad → agree  
Learn: agree → trade

Infer: Tim → sad → agree → trade
```

### Implementation
1. `CompositionalReasoner` class tracks relation chains
2. Hebbian learning strengthens A→B connections
3. Inference follows chains at runtime

### Test Cases
```python
# Chain reasoning
"If Tim is sad, and sad people agree reluctantly, what does Tim do?"
# Expected: "Tim agrees reluctantly"

# Transitive property
"A is larger than B, B is larger than C. Is A larger than C?"
# Expected: "Yes"
```

---

## 🎯 PHASE 3: GOAL-DIRECTED BEHAVIOR

### What It Is
The model can:
1. Receive a goal: "Write a story about Tim"
2. Plan steps: Intro → Conflict → Resolution
3. Execute each step
4. Self-evaluate output

### Implementation
```python
agent = GoalDirectedAgent(brain)
agent.set_goal("Complete the sentence: Tim was sad, but...")
agent.plan_steps()  # Decomposes goal
agent.execute()     # Generates output
```

---

## 🔄 PHASE 4: SELF-IMPROVEMENT LOOP

### What It Is
The model identifies its weaknesses and improves them **without human intervention**.

### Cycle
```
1. Evaluate → "I got 70% on math questions"
2. Identify → "Weakness: arithmetic"
3. Learn → Targeted Hebbian updates on math examples
4. Re-evaluate → "Now 85% on math questions"
5. Repeat forever
```

### This is KEY for ASI!
Once the model can improve itself, it enters a positive feedback loop.

---

## 💬 PHASE 5: REAL CONVERSATION

### Target Output Quality
```
Input:  "Tim was sad, but he agreed to trade"
Output: "the expensive car for a smaller one."

Input:  "The weather was perfect, so they decided to"
Output: "go on a picnic in the park."

Input:  "She opened the door and saw"
Output: "her old friend standing there with a smile."
```

### Evaluation Metrics
- Perplexity < 20
- BLEU-4 > 0.3
- Human rating > 4/5

---

## 🖥️ HARDWARE REQUIREMENTS

| Tier | Parameters | GPU RAM | Training Time |
|------|------------|---------|---------------|
| nano | 23M | 4GB | 2h |
| small | 100M | 16GB | 24h |
| medium | 350M | 40GB | 3 days |
| large | 1B | 80GB | 7 days |

### Recommended
- **Minimum**: RTX 3090 / A10 (24GB)
- **Optimal**: A100 (40GB or 80GB)
- **Best**: 8x A100 with DeepSpeed

---

## 📋 ACTION CHECKLIST

### Immediate (Today)
- [x] Create v0.9 training pipeline
- [ ] Test data loading on server
- [ ] Start small model training

### This Week
- [ ] Train 100M model for 24h
- [ ] Evaluate conversation quality
- [ ] Implement compositional reasoning tests

### This Month
- [ ] Train 350M model
- [ ] Achieve coherent multi-sentence output
- [ ] Implement self-improvement loop

### Long Term
- [ ] Train 1B+ model
- [ ] Human-level conversation
- [ ] Self-improvement → ASI trajectory

---

## 🎉 SUCCESS CRITERIA

### v0.9 Complete When:
1. ✅ Model can complete: "Tim was sad, but he agreed to trade ___"
2. ✅ Perplexity < 30 on held-out data
3. ✅ Can answer factual questions (capital cities, etc.)
4. ✅ Self-improvement shows measurable gains

### ASI Trajectory When:
1. Model improves faster than humans can improve it
2. Model generates novel insights humans hadn't considered
3. Model achieves superhuman performance on reasoning benchmarks

---

## 🚀 LET'S GO!

Run this on your server:
```bash
cd /workspace/Nokai
python3 scripts/train_v09_scaling.py --tier small --steps 50000
```

This will:
1. Load OpenWebText/Wikipedia
2. Train a 100M parameter Nōkai
3. Save checkpoints every 5000 steps
4. Evaluate conversation quality

**The journey from 23M to 1B+ starts now!** 🧠✨

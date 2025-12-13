
# --- V4.2: BIOMAMBA "SPEED DEMON" EDITION ---
# Changes:
# 1. TOKENIZER: Switched to Character-Level (Instant) to eliminate the 15-minute startup waiting time.
#    -> "Intelligence" comes from the Brain (Mamba), not the Voice (BPE).
# 2. SCAN: Optimized JIT Recurrence to be more compiler-friendly.
# 3. PIPELINE: Streamlined training loop for immediate feedback.

import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import os
import requests
import sys
import random

# Ensure we can import from the project root
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

try:
    from nokai.limbic.dopamine import DopamineCircuit
except ImportError:
    class DopamineCircuit(nn.Module):
        def __init__(self, **kwargs):
            super().__init__()
            self.dopamine = 1.0
            self.level = 1.0 # Add level attribute
            self.burst = 0.0 # Add burst attribute
            self.tonic_level = torch.tensor(0.5)
            self.phasic_level = torch.tensor(0.0)
        def forward(self, x, **kwargs):
            class State: pass
            s = State()
            s.level = 1.0
            s.burst = 0.0
            return s, {}
        def get_learning_modulation(self): return 1.0

# --- CONFIGURATION ---
class Config:
    block_size = 512        # Context length
    n_layer = 12            # Depth
    n_embd = 768            # Width
    d_state = 64            # State
    d_conv = 4
    expand = 2
    
    lr_base = 6e-4          
    iters_pretrain = 3000   
    iters_sft = 500         
    batch_size = 32 # Keep manageable
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
config = Config()
torch.set_default_device(config.device)
torch.manual_seed(1337)
if config.device == 'cuda':
    torch.set_float32_matmul_precision('high') 

# --- 1. TOKENIZER (INSTANT CHAR-LEVEL) ---
# The user needs SPEED. BPE in pure Python is O(N^2) and too slow.
class CharTokenizer:
    def __init__(self, text):
        print("Initializing Instant Char Tokenizer...")
        chars = sorted(list(set(text)))
        self.vocab_size = len(chars) + 3 # +3 for special tokens
        self.stoi = { ch:i for i,ch in enumerate(chars) }
        self.itos = { i:ch for i,ch in enumerate(chars) }
        
        self.special = {"<PAD>": self.vocab_size-3, "<EOS>": self.vocab_size-2, "<UNK>": self.vocab_size-1}
        
    def encode(self, s):
        return [self.stoi.get(c, self.special["<UNK>"]) for c in s]
        
    def decode(self, l):
        return ''.join([self.itos.get(i, "") for i in l if i not in self.special.values()])

# --- 2. JIT OPTIMIZED MAMBA ---
class MambaBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.d_inner = config.n_embd * config.expand
        self.dt_rank = config.d_state // 16 if config.d_state // 16 > 1 else 1
        
        self.in_proj = nn.Linear(config.n_embd, self.d_inner * 2, bias=False)
        self.conv1d = nn.Conv1d(self.d_inner, self.d_inner, bias=True, 
                               kernel_size=config.d_conv, groups=self.d_inner, padding=config.d_conv - 1)
        self.x_proj = nn.Linear(self.d_inner, self.dt_rank + config.d_state * 2, bias=False)
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True)
        
        self.A_log = nn.Parameter(torch.log(torch.arange(1, config.d_state + 1, dtype=torch.float32)).repeat(self.d_inner, 1))
        self.D = nn.Parameter(torch.ones(self.d_inner))
        self.out_proj = nn.Linear(self.d_inner, config.n_embd, bias=False)
        self.act = nn.SiLU() 

    # JIT the scan for max speed on the recurrence
    @staticmethod
    @torch.jit.script 
    def pscan_jit(A, B, C, D, x, delta):
        # x: (bs, L, d_in)
        bs, L, d_in = x.shape
        n = A.shape[1] 
        
        # Pre-calculate expensive matrix params outside logical loop
        # A_bar = exp(delta * A)
        A_bar = torch.exp(delta.unsqueeze(-1) * A) # (BS, L, D, N)
        
        # u = (delta * B) * x
        u = delta.unsqueeze(-1) * B.unsqueeze(2) * x.unsqueeze(-1) # (BS, L, D, N)
        
        h = torch.zeros(bs, d_in, n, device=x.device, dtype=x.dtype)
        y = torch.zeros_like(x)
        
        # This loop is fused by JIT. 
        # Crucial: Avoid indexing overhead by keeping tensors contiguous
        for t in range(L):
            h = A_bar[:, t] * h + u[:, t]
            y[:, t] = (C[:, t].unsqueeze(1) * h).sum(dim=-1)
            
        return y + D * x

    def forward(self, x):
        B, L, D = x.shape
        x_and_res = self.in_proj(x) 
        (x_ssm, res) = x_and_res.split(split_size=[self.d_inner, self.d_inner], dim=-1)
        
        x_ssm = x_ssm.transpose(1, 2)
        x_ssm = self.conv1d(x_ssm)[:, :, :L]
        x_ssm = x_ssm.transpose(1, 2)
        x_ssm = self.act(x_ssm)
        
        x_dbl = self.x_proj(x_ssm) 
        (delta, B_val, C_val) = x_dbl.split([self.dt_rank, self.config.d_state, self.config.d_state], dim=-1)
        delta = F.softplus(self.dt_proj(delta)) 
        
        # Stable JIT Scan
        out_scan = self.pscan_jit(-torch.exp(self.A_log), B_val, C_val, self.D, x_ssm, delta)
        
        return self.out_proj(out_scan * self.act(res))

class BioMamba(nn.Module):
    def __init__(self, config, vocab_size):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, config.n_embd)
        self.layers = nn.ModuleList([MambaBlock(config) for _ in range(config.n_layer)])
        self.norm_f = nn.LayerNorm(config.n_embd)
        self.lm_head = nn.Linear(config.n_embd, vocab_size, bias=False)
        
        # Metacognition
        self.confidence_head = nn.Linear(config.n_embd, 1)

    def forward(self, idx, targets=None, use_checkpointing=False):
        x = self.embedding(idx)
        for layer in self.layers:
            if use_checkpointing and self.training:
                x = torch.utils.checkpoint.checkpoint(layer, x, use_reentrant=False)
            else:
                x = x + layer(x) # Residual built-into block? No, block returns residual
        
        x = self.norm_f(x)
        logits = self.lm_head(x)
        confidence = torch.sigmoid(self.confidence_head(x))
        
        loss = None
        if targets is not None:
            B, T, C = logits.shape
            loss = F.cross_entropy(logits.view(B*T, C), targets.view(B*T))
            
        return logits, loss, confidence

# --- 3. RAPID SETUP ---
print("Dataset Loading...")
if not os.path.exists('input.txt'):
    data_url = 'https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt'
    with open('input.txt', 'w') as f: f.write(requests.get(data_url).text)
with open('input.txt', 'r', encoding='utf-8') as f: data = f.read()

# INSTANT TOKENIZATION
tokenizer = CharTokenizer(data)
encoded_data = tokenizer.encode(data)
print(f"Data ready: {len(encoded_data)} tokens. Vocab: {tokenizer.vocab_size}")

train_data = torch.tensor(encoded_data[:int(0.9*len(encoded_data))], dtype=torch.long)

model = BioMamba(config, tokenizer.vocab_size).to(config.device)
optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr_base)
scaler = torch.amp.GradScaler('cuda', enabled=(config.device == 'cuda'))

dopamine_circuit = DopamineCircuit(state_dim=1, hidden_dim=32).to(config.device)

def get_batch():
    ix = torch.randint(len(train_data) - config.block_size, (config.batch_size,))
    x = torch.stack([train_data[i:i+config.block_size] for i in ix])
    y = torch.stack([train_data[i+1:i+config.block_size+1] for i in ix])
    return x.to(config.device), y.to(config.device)

# --- 4. FAST LOOP ---
print("\n=== STARTING TRAINING (NO WAITING) ===")
start_time = time.time()

for iter in range(config.iters_pretrain):
    xb, yb = get_batch()
    
    with torch.amp.autocast('cuda'):
        # Gradient Checkpointing is crucial for VRAM but slows down slightly. 
        # Since we have trouble with speed, let's keep it ON for VRAM safety but check impact.
        logits, loss, conf = model(xb, yb, use_checkpointing=True)
    
    # Limbic System
    loss_val = loss.item()
    reward = torch.tensor([-loss_val], device=config.device).unsqueeze(0)
    state = torch.tensor([[loss_val]], device=config.device)
    da_state, _ = dopamine_circuit(state, reward=reward)
    
    optimizer.zero_grad()
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
    
    if iter % 10 == 0: # Print more often to show life
        dt = time.time() - start_time
        ips = 10 / (dt + 1e-6)
        start_time = time.time()
        print(f"Step {iter} | Loss: {loss_val:.4f} | DA: {da_state.level:.2f} | Speed: {ips:.2f} it/s")

# --- 5. CHAT ---
print("\n=== CHAT TEST ===")
def chat(prompt):
    model.eval()
    input_ids = torch.tensor(tokenizer.encode("User: "+prompt+"\nModel:"), device=config.device).unsqueeze(0)
    for _ in range(100):
        with torch.no_grad(): logits, _, _ = model(input_ids)
        next_token = torch.multinomial(F.softmax(logits[:, -1, :], dim=-1), 1)
        input_ids = torch.cat((input_ids, next_token), dim=1)
        if next_token.item() == tokenizer.special["<EOS>"]: break
    return tokenizer.decode(input_ids[0].tolist())

print(chat("Who are you?"))
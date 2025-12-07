"""
BPE Tokenizer - Semantic Understanding for Nōkai

Biological Parallel:
    The human brain doesn't process language letter-by-letter. Instead,
    it recognizes morphemes (smallest meaningful units) and whole words.
    
    BPE (Byte Pair Encoding) mimics this by learning subword units from data:
    - Common words → Single token ("the", "and", "brain")
    - Rare words → Subword decomposition ("neuromorphic" → "neuro" + "morphic")
    - Unknown words → Character fallback (graceful degradation)

Mathematical Foundation:
    BPE operates by iteratively merging the most frequent adjacent pairs:
    
    1. Initialize vocabulary V = {all characters}
    2. For each iteration i:
       - Count all adjacent symbol pairs in corpus
       - Find most frequent pair (a, b)
       - Add merge rule: ab → new_symbol
       - Update V = V ∪ {new_symbol}
    3. Repeat until |V| = target_vocab_size
    
    This greedy algorithm achieves O(n log n) compression with 
    near-optimal subword segmentation.

Implementation:
    We use the HuggingFace tokenizers library for production performance:
    - Rust-based core for 10-100x speedup over pure Python
    - Training on 1M+ documents in minutes
    - Serializable JSON format for persistence
    - Pre/post-processing pipelines

Usage:
    tokenizer = NokaiTokenizer.train(texts, vocab_size=32000)
    tokenizer.save("checkpoints/tokenizer.json")
    
    # Later...
    tokenizer = NokaiTokenizer.load("checkpoints/tokenizer.json")
    tokens = tokenizer.encode("Hello world!")
    text = tokenizer.decode(tokens)
"""

import os
import json
from pathlib import Path
from typing import List, Optional, Union, Dict, Iterator
from dataclasses import dataclass

import torch


# ============================================
# CHECK FOR TOKENIZERS LIBRARY
# ============================================

try:
    from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders, processors
    from tokenizers.normalizers import NFD, Lowercase, StripAccents, Sequence as NormalizerSequence
    HAS_TOKENIZERS = True
except ImportError:
    HAS_TOKENIZERS = False


# ============================================
# DATA CLASSES
# ============================================

@dataclass
class TokenizerConfig:
    """
    Configuration for the BPE tokenizer.
    
    Biological Mapping:
        - vocab_size: "Lexicon" size - number of distinct concepts the brain can represent
        - min_frequency: "Familiarity threshold" - minimum exposure to learn a concept
        - special_tokens: "Reserved neural pathways" - fixed meaning symbols
    """
    vocab_size: int = 32000
    min_frequency: int = 2
    special_tokens: List[str] = None
    
    # Special token IDs (fixed positions for predictability)
    pad_token: str = "<pad>"
    unk_token: str = "<unk>"
    bos_token: str = "<s>"
    eos_token: str = "</s>"
    mask_token: str = "<mask>"
    
    # Processing options
    lowercase: bool = False  # Preserve case for better semantics
    unicode_normalizer: str = "NFD"
    
    def __post_init__(self):
        if self.special_tokens is None:
            self.special_tokens = [
                self.pad_token,
                self.unk_token, 
                self.bos_token,
                self.eos_token,
                self.mask_token,
            ]


# ============================================
# MAIN TOKENIZER CLASS
# ============================================

class NokaiTokenizer:
    """
    Production-Grade BPE Tokenizer for Nōkai.
    
    Biological Parallel (Thalamus-Lexicon Interface):
        The human brain's language processing involves:
        1. Phonological processing (sound patterns → basic units)
        2. Morphological decomposition (words → morphemes)  
        3. Lexical access (morphemes → meaning)
        
        Our BPE tokenizer mimics this hierarchy:
        1. Byte-level encoding (UTF-8 → byte sequences)
        2. Subword segmentation (bytes → BPE tokens)
        3. Embedding projection (tokens → semantic vectors)
    
    Efficiency:
        - O(n) encoding with compiled merge rules (hash table lookup)
        - Batch processing for throughput
        - Memory-efficient vocabulary (prefix tree structure)
    """
    
    def __init__(self, config: Optional[TokenizerConfig] = None):
        self.config = config or TokenizerConfig()
        self._tokenizer: Optional[Tokenizer] = None
        
        # Special token IDs (set after training/loading)
        self.pad_token_id: int = 0
        self.unk_token_id: int = 1
        self.bos_token_id: int = 2
        self.eos_token_id: int = 3
        self.mask_token_id: int = 4
    
    @property
    def vocab_size(self) -> int:
        """Get actual vocabulary size."""
        if self._tokenizer is None:
            return self.config.vocab_size
        return self._tokenizer.get_vocab_size()
    
    @property
    def is_trained(self) -> bool:
        """Check if tokenizer has been trained or loaded."""
        return self._tokenizer is not None
    
    # =====================================
    # TRAINING
    # =====================================
    
    @classmethod
    def train(
        cls,
        texts: Union[List[str], Iterator[str]],
        config: Optional[TokenizerConfig] = None,
        show_progress: bool = True,
    ) -> "NokaiTokenizer":
        """
        Train a new BPE tokenizer on the provided texts.
        
        Biological Analogy:
            This is like a child learning language through exposure.
            The more frequently a pattern appears, the stronger the
            neural pathway becomes (encoded as a single token).
        
        Args:
            texts: List or iterator of training texts
            config: Tokenizer configuration
            show_progress: Show training progress
            
        Returns:
            Trained NokaiTokenizer instance
            
        Mathematical Details:
            The BPE algorithm performs O(V × N × log N) operations where:
            - V = target vocabulary size
            - N = total corpus size in characters
            
            Each merge iteration requires:
            1. Pair counting: O(N)
            2. Max-finding: O(current_pairs) ≈ O(V)
            3. Corpus update: O(N)
            
            Modern implementations use incremental updates for O(V × N).
        """
        if not HAS_TOKENIZERS:
            raise ImportError(
                "The 'tokenizers' library is required for BPE training. "
                "Install with: pip install tokenizers"
            )
        
        instance = cls(config)
        config = instance.config
        
        print(f"[NokaiTokenizer] Training BPE with vocab_size={config.vocab_size}...")
        
        # =====================================
        # BUILD TOKENIZER PIPELINE
        # =====================================
        
        # 1. Model: BPE with byte-level fallback
        tokenizer = Tokenizer(models.BPE(unk_token=config.unk_token))
        
        # 2. Normalizer: Unicode normalization + optional lowercase
        normalizers = [NFD()]
        if config.lowercase:
            normalizers.append(Lowercase())
        normalizers.append(StripAccents())
        tokenizer.normalizer = NormalizerSequence(normalizers)
        
        # 3. Pre-tokenizer: Split on whitespace and punctuation
        # This mimics how humans naturally segment speech
        tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=True)
        
        # 4. Decoder: Reconstruct text from tokens
        tokenizer.decoder = decoders.ByteLevel()
        
        # 5. Post-processor: Add special tokens
        tokenizer.post_processor = processors.ByteLevel(trim_offsets=True)
        
        # =====================================
        # TRAIN
        # =====================================
        
        trainer = trainers.BpeTrainer(
            vocab_size=config.vocab_size,
            min_frequency=config.min_frequency,
            special_tokens=config.special_tokens,
            show_progress=show_progress,
            initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        )
        
        # Convert to list if iterator (tokenizers library requires list for in-memory training)
        if not isinstance(texts, list):
            print("[NokaiTokenizer] Collecting texts for training...")
            texts = list(texts)
        
        # Train from iterator
        tokenizer.train_from_iterator(texts, trainer=trainer)
        
        # =====================================
        # FINALIZE
        # =====================================
        
        instance._tokenizer = tokenizer
        instance._update_special_token_ids()
        
        print(f"[NokaiTokenizer] Training complete! Vocab size: {instance.vocab_size}")
        
        return instance
    
    def _update_special_token_ids(self):
        """Update special token ID mappings from tokenizer vocabulary."""
        if self._tokenizer is None:
            return
            
        vocab = self._tokenizer.get_vocab()
        
        # Map special tokens to their IDs
        self.pad_token_id = vocab.get(self.config.pad_token, 0)
        self.unk_token_id = vocab.get(self.config.unk_token, 1)
        self.bos_token_id = vocab.get(self.config.bos_token, 2)
        self.eos_token_id = vocab.get(self.config.eos_token, 3)
        self.mask_token_id = vocab.get(self.config.mask_token, 4)
    
    # =====================================
    # ENCODING / DECODING
    # =====================================
    
    def encode(
        self,
        text: str,
        add_special_tokens: bool = True,
        max_length: Optional[int] = None,
        truncation: bool = True,
        padding: bool = False,
    ) -> List[int]:
        """
        Encode text to token IDs.
        
        Biological Analogy:
            This is like the brain parsing speech into recognized patterns.
            Unknown sequences get decomposed into smaller known units,
            similar to how we sound out unfamiliar words.
        
        Args:
            text: Input text to encode
            add_special_tokens: Add BOS/EOS tokens
            max_length: Maximum sequence length
            truncation: Truncate if exceeds max_length
            padding: Pad to max_length
            
        Returns:
            List of token IDs
        """
        if not self.is_trained:
            raise RuntimeError("Tokenizer not trained or loaded. Call train() or load() first.")
        
        # Encode
        encoding = self._tokenizer.encode(text, add_special_tokens=False)
        ids = encoding.ids
        
        # Add BOS/EOS
        if add_special_tokens:
            ids = [self.bos_token_id] + ids + [self.eos_token_id]
        
        # Truncation
        if max_length is not None and len(ids) > max_length and truncation:
            ids = ids[:max_length]
            # Ensure EOS is present
            if add_special_tokens and ids[-1] != self.eos_token_id:
                ids[-1] = self.eos_token_id
        
        # Padding
        if max_length is not None and padding:
            while len(ids) < max_length:
                ids.append(self.pad_token_id)
        
        return ids
    
    def encode_batch(
        self,
        texts: List[str],
        add_special_tokens: bool = True,
        max_length: Optional[int] = None,
        truncation: bool = True,
        padding: bool = True,
        return_tensors: bool = False,
    ) -> Union[List[List[int]], torch.Tensor]:
        """
        Encode a batch of texts.
        
        Efficiency:
            Uses parallel encoding for ~4x speedup on multi-core systems.
        
        Args:
            texts: List of texts to encode
            add_special_tokens: Add BOS/EOS tokens
            max_length: Maximum sequence length (None = auto from batch)
            truncation: Truncate if exceeds max_length
            padding: Pad all sequences to same length
            return_tensors: Return PyTorch tensor instead of lists
            
        Returns:
            Batch of token IDs (list or tensor)
        """
        if not self.is_trained:
            raise RuntimeError("Tokenizer not trained or loaded.")
        
        # Batch encode
        encodings = self._tokenizer.encode_batch(texts, add_special_tokens=False)
        
        # Process each encoding
        batch_ids = []
        for encoding in encodings:
            ids = list(encoding.ids)
            
            # Add special tokens
            if add_special_tokens:
                ids = [self.bos_token_id] + ids + [self.eos_token_id]
            
            batch_ids.append(ids)
        
        # Determine max length
        if max_length is None:
            max_length = max(len(ids) for ids in batch_ids)
        
        # Truncation and padding
        for i, ids in enumerate(batch_ids):
            if len(ids) > max_length and truncation:
                batch_ids[i] = ids[:max_length]
                if add_special_tokens:
                    batch_ids[i][-1] = self.eos_token_id
            elif padding and len(ids) < max_length:
                batch_ids[i] = ids + [self.pad_token_id] * (max_length - len(ids))
        
        if return_tensors:
            return torch.tensor(batch_ids, dtype=torch.long)
        
        return batch_ids
    
    def decode(
        self,
        ids: Union[List[int], torch.Tensor],
        skip_special_tokens: bool = True,
    ) -> str:
        """
        Decode token IDs back to text.
        
        Args:
            ids: Token IDs to decode
            skip_special_tokens: Remove special tokens from output
            
        Returns:
            Decoded text string
        """
        if not self.is_trained:
            raise RuntimeError("Tokenizer not trained or loaded.")
        
        if isinstance(ids, torch.Tensor):
            ids = ids.tolist()
        
        # Filter special tokens if requested
        if skip_special_tokens:
            special_ids = {
                self.pad_token_id,
                self.bos_token_id,
                self.eos_token_id,
                self.mask_token_id,
            }
            ids = [i for i in ids if i not in special_ids]
        
        return self._tokenizer.decode(ids)
    
    def decode_batch(
        self,
        batch_ids: Union[List[List[int]], torch.Tensor],
        skip_special_tokens: bool = True,
    ) -> List[str]:
        """Decode a batch of token ID sequences."""
        if isinstance(batch_ids, torch.Tensor):
            batch_ids = batch_ids.tolist()
        
        return [self.decode(ids, skip_special_tokens) for ids in batch_ids]
    
    # =====================================
    # PERSISTENCE
    # =====================================
    
    def save(self, path: Union[str, Path]):
        """
        Save tokenizer to disk.
        
        Creates two files:
        - tokenizer.json: The main tokenizer model
        - tokenizer_config.json: Our configuration
        """
        if not self.is_trained:
            raise RuntimeError("Cannot save untrained tokenizer.")
        
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save tokenizer
        self._tokenizer.save(str(path))
        
        # Save config
        config_path = path.parent / "tokenizer_config.json"
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump({
                'vocab_size': self.config.vocab_size,
                'min_frequency': self.config.min_frequency,
                'special_tokens': self.config.special_tokens,
                'pad_token': self.config.pad_token,
                'unk_token': self.config.unk_token,
                'bos_token': self.config.bos_token,
                'eos_token': self.config.eos_token,
                'mask_token': self.config.mask_token,
                'lowercase': self.config.lowercase,
                # Save actual IDs
                'pad_token_id': self.pad_token_id,
                'unk_token_id': self.unk_token_id,
                'bos_token_id': self.bos_token_id,
                'eos_token_id': self.eos_token_id,
                'mask_token_id': self.mask_token_id,
            }, f, indent=2)
        
        print(f"[NokaiTokenizer] Saved to {path}")
    
    @classmethod
    def load(cls, path: Union[str, Path]) -> "NokaiTokenizer":
        """
        Load tokenizer from disk.
        
        Args:
            path: Path to tokenizer.json file
            
        Returns:
            Loaded NokaiTokenizer instance
        """
        if not HAS_TOKENIZERS:
            raise ImportError("The 'tokenizers' library is required. Install with: pip install tokenizers")
        
        path = Path(path)
        
        # Load config if exists
        config_path = path.parent / "tokenizer_config.json"
        if config_path.exists():
            with open(config_path, 'r', encoding='utf-8') as f:
                config_data = json.load(f)
            config = TokenizerConfig(**{k: v for k, v in config_data.items() 
                                       if k in TokenizerConfig.__dataclass_fields__})
        else:
            config = TokenizerConfig()
        
        instance = cls(config)
        instance._tokenizer = Tokenizer.from_file(str(path))
        instance._update_special_token_ids()
        
        # Override with saved IDs if available
        if config_path.exists():
            instance.pad_token_id = config_data.get('pad_token_id', instance.pad_token_id)
            instance.unk_token_id = config_data.get('unk_token_id', instance.unk_token_id)
            instance.bos_token_id = config_data.get('bos_token_id', instance.bos_token_id)
            instance.eos_token_id = config_data.get('eos_token_id', instance.eos_token_id)
            instance.mask_token_id = config_data.get('mask_token_id', instance.mask_token_id)
        
        print(f"[NokaiTokenizer] Loaded from {path} (vocab_size={instance.vocab_size})")
        
        return instance
    
    # =====================================
    # UTILITIES
    # =====================================
    
    def get_vocab(self) -> Dict[str, int]:
        """Get vocabulary as token → id mapping."""
        if not self.is_trained:
            return {}
        return self._tokenizer.get_vocab()
    
    def token_to_id(self, token: str) -> Optional[int]:
        """Convert token string to ID."""
        if not self.is_trained:
            return None
        return self._tokenizer.token_to_id(token)
    
    def id_to_token(self, id: int) -> Optional[str]:
        """Convert ID to token string."""
        if not self.is_trained:
            return None
        return self._tokenizer.id_to_token(id)
    
    def __len__(self) -> int:
        """Return vocabulary size."""
        return self.vocab_size


# ============================================
# FALLBACK SIMPLE TOKENIZER (NO DEPENDENCIES)
# ============================================

class SimpleBPETokenizer:
    """
    Fallback BPE tokenizer without external dependencies.
    
    This is a simplified pure-Python implementation for environments
    where the 'tokenizers' library cannot be installed.
    
    Note: This is ~100x slower than the Rust-based implementation.
    Use NokaiTokenizer when possible.
    
    Biological Parallel:
        Like learning language without formal education - slower,
        but still achieves basic competence through pattern recognition.
    """
    
    def __init__(self, vocab_size: int = 32000):
        self.vocab_size = vocab_size
        self.merges: Dict[tuple, str] = {}
        self.vocab: Dict[str, int] = {}
        self.inverse_vocab: Dict[int, str] = {}
        
        # Special tokens
        self.pad_token_id = 0
        self.unk_token_id = 1
        self.bos_token_id = 2
        self.eos_token_id = 3
        
        # Initialize special tokens
        self._init_special_tokens()
    
    def _init_special_tokens(self):
        """Initialize vocabulary with special tokens."""
        special = ['<pad>', '<unk>', '<s>', '</s>']
        for i, token in enumerate(special):
            self.vocab[token] = i
            self.inverse_vocab[i] = token
    
    def train(self, texts: List[str], num_merges: Optional[int] = None):
        """
        Train BPE on texts.
        
        Algorithm (Sennrich et al., 2016):
        1. Initialize vocabulary with characters
        2. Count all adjacent symbol pairs
        3. Merge most frequent pair
        4. Repeat until vocab_size reached
        """
        if num_merges is None:
            num_merges = self.vocab_size - len(self.vocab)
        
        print(f"[SimpleBPE] Training with {num_merges} merges...")
        
        # Collect all characters
        chars = set()
        for text in texts:
            chars.update(text)
        
        # Add characters to vocab
        for char in sorted(chars):
            if char not in self.vocab:
                idx = len(self.vocab)
                self.vocab[char] = idx
                self.inverse_vocab[idx] = char
        
        # Tokenize corpus
        words = []
        for text in texts:
            # Split into words and convert to character tuples
            for word in text.split():
                if word:
                    words.append(tuple(word) + ('</w>',))
        
        # Add end-of-word marker
        if '</w>' not in self.vocab:
            idx = len(self.vocab)
            self.vocab['</w>'] = idx
            self.inverse_vocab[idx] = '</w>'
        
        # BPE merges
        for merge_idx in range(num_merges):
            if len(self.vocab) >= self.vocab_size:
                break
            
            # Count pairs
            pairs = {}
            for word in words:
                for i in range(len(word) - 1):
                    pair = (word[i], word[i + 1])
                    pairs[pair] = pairs.get(pair, 0) + 1
            
            if not pairs:
                break
            
            # Find most frequent
            best_pair = max(pairs, key=pairs.get)
            
            # Create new symbol
            new_symbol = best_pair[0] + best_pair[1]
            if new_symbol.endswith('</w>'):
                new_symbol = new_symbol[:-4] + '</w>'
            
            # Add to vocab
            if new_symbol not in self.vocab:
                idx = len(self.vocab)
                self.vocab[new_symbol] = idx
                self.inverse_vocab[idx] = new_symbol
            
            # Store merge rule
            self.merges[best_pair] = new_symbol
            
            # Apply merge to all words
            new_words = []
            for word in words:
                new_word = []
                i = 0
                while i < len(word):
                    if i < len(word) - 1 and (word[i], word[i + 1]) == best_pair:
                        new_word.append(new_symbol)
                        i += 2
                    else:
                        new_word.append(word[i])
                        i += 1
                new_words.append(tuple(new_word))
            words = new_words
            
            if (merge_idx + 1) % 1000 == 0:
                print(f"  Merge {merge_idx + 1}/{num_merges}, vocab={len(self.vocab)}")
        
        print(f"[SimpleBPE] Training complete. Vocab size: {len(self.vocab)}")
    
    def encode(self, text: str) -> List[int]:
        """Encode text to token IDs."""
        ids = [self.bos_token_id]
        
        for word in text.split():
            if not word:
                continue
            
            # Convert to characters
            symbols = list(word) + ['</w>']
            
            # Apply merges
            while len(symbols) > 1:
                pairs = [(symbols[i], symbols[i + 1]) for i in range(len(symbols) - 1)]
                
                # Find applicable merge
                merge_found = False
                for pair in pairs:
                    if pair in self.merges:
                        # Apply merge
                        new_symbols = []
                        i = 0
                        while i < len(symbols):
                            if i < len(symbols) - 1 and (symbols[i], symbols[i + 1]) == pair:
                                new_symbols.append(self.merges[pair])
                                i += 2
                            else:
                                new_symbols.append(symbols[i])
                                i += 1
                        symbols = new_symbols
                        merge_found = True
                        break
                
                if not merge_found:
                    break
            
            # Convert to IDs
            for symbol in symbols:
                ids.append(self.vocab.get(symbol, self.unk_token_id))
        
        ids.append(self.eos_token_id)
        return ids
    
    def decode(self, ids: List[int]) -> str:
        """Decode token IDs to text."""
        tokens = []
        for i in ids:
            if i in [self.pad_token_id, self.bos_token_id, self.eos_token_id]:
                continue
            token = self.inverse_vocab.get(i, '<unk>')
            tokens.append(token)
        
        # Reconstruct text
        text = ''.join(tokens)
        text = text.replace('</w>', ' ')
        return text.strip()
    
    def save(self, path: str):
        """Save tokenizer."""
        with open(path, 'w', encoding='utf-8') as f:
            json.dump({
                'vocab': self.vocab,
                'merges': {f"{k[0]}|||{k[1]}": v for k, v in self.merges.items()},
                'vocab_size': self.vocab_size,
            }, f, ensure_ascii=False, indent=2)
    
    @classmethod
    def load(cls, path: str) -> "SimpleBPETokenizer":
        """Load tokenizer."""
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        tokenizer = cls(data['vocab_size'])
        tokenizer.vocab = data['vocab']
        tokenizer.inverse_vocab = {int(v): k for k, v in data['vocab'].items()}
        tokenizer.merges = {
            tuple(k.split('|||')): v for k, v in data['merges'].items()
        }
        
        return tokenizer


# ============================================
# FACTORY FUNCTION
# ============================================

def create_tokenizer(
    vocab_size: int = 32000,
    use_fast: bool = True,
) -> Union[NokaiTokenizer, SimpleBPETokenizer]:
    """
    Create a tokenizer instance.
    
    Args:
        vocab_size: Target vocabulary size
        use_fast: Use fast Rust-based tokenizer if available
        
    Returns:
        Tokenizer instance (NokaiTokenizer if available, else SimpleBPETokenizer)
    """
    if use_fast and HAS_TOKENIZERS:
        return NokaiTokenizer(TokenizerConfig(vocab_size=vocab_size))
    else:
        if use_fast:
            print("[Warning] 'tokenizers' library not found. Using slow SimpleBPETokenizer.")
            print("         Install with: pip install tokenizers")
        return SimpleBPETokenizer(vocab_size)

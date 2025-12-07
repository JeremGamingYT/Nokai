"""
Nōkai Inference / Test Script

Load a trained model and generate text.
"""

import torch
import argparse
from pathlib import Path
from tokenizers import Tokenizer

from nokai import NokaiConfig, NokaiModel


def load_model(checkpoint_path: str, device: str = "cuda"):
    """Load a trained Nōkai model from checkpoint."""
    print(f"📂 Loading checkpoint: {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Reconstruct config
    config_dict = checkpoint['config']
    config = NokaiConfig(**config_dict)
    config.device = device
    
    # Create model
    model = NokaiModel(config)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    print(f"✓ Model loaded (step {checkpoint.get('step', 'unknown')})")
    return model, config


def generate(
    model: NokaiModel,
    tokenizer: Tokenizer,
    prompt: str,
    max_tokens: int = 100,
    temperature: float = 0.8,
    top_k: int = 50,
    top_p: float = 0.9,
    device: str = "cuda",
):
    """Generate text from a prompt."""
    # Tokenize prompt
    tokens = tokenizer.encode(prompt).ids
    input_ids = torch.tensor([tokens], device=device)
    
    print(f"\n📝 Prompt: \"{prompt}\"")
    print(f"   Tokens: {len(tokens)}")
    print(f"\n🧠 Generating...")
    
    # Generate
    with torch.no_grad():
        output_ids = model.generate(
            input_ids,
            max_new_tokens=max_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
        )
    
    # Decode
    generated_tokens = output_ids[0].tolist()
    generated_text = tokenizer.decode(generated_tokens)
    
    return generated_text


def interactive_mode(model, tokenizer, config):
    """Interactive chat mode."""
    print("\n" + "="*60)
    print("🧠 Nōkai Interactive Mode")
    print("="*60)
    print("Type your prompt and press Enter. Type 'quit' to exit.\n")
    
    device = config.device
    
    while True:
        try:
            prompt = input("You: ").strip()
            
            if prompt.lower() in ['quit', 'exit', 'q']:
                print("👋 Goodbye!")
                break
            
            if not prompt:
                continue
            
            # Generate response
            response = generate(
                model,
                tokenizer,
                prompt,
                max_tokens=100,
                temperature=0.8,
                device=device,
            )
            
            print(f"\n🧠 Nōkai: {response}\n")
            
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            break


def main():
    parser = argparse.ArgumentParser(description="Test Nōkai Model")
    
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to model checkpoint (.pt file)")
    parser.add_argument("--tokenizer", type=str, default="./data/tokenizer.json",
                        help="Path to tokenizer.json")
    parser.add_argument("--prompt", type=str, default=None,
                        help="Text prompt for generation")
    parser.add_argument("--max-tokens", type=int, default=100,
                        help="Maximum tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.8,
                        help="Sampling temperature (higher = more random)")
    parser.add_argument("--top-k", type=int, default=50,
                        help="Top-k sampling")
    parser.add_argument("--top-p", type=float, default=0.9,
                        help="Top-p (nucleus) sampling")
    parser.add_argument("--interactive", action="store_true",
                        help="Interactive mode")
    parser.add_argument("--cpu", action="store_true",
                        help="Force CPU")
    
    args = parser.parse_args()
    
    # Device
    if args.cpu or not torch.cuda.is_available():
        device = "cpu"
    else:
        device = "cuda"
        print(f"🚀 Using GPU: {torch.cuda.get_device_name()}")
    
    # Load tokenizer
    tokenizer_path = Path(args.tokenizer)
    if not tokenizer_path.exists():
        print(f"❌ Tokenizer not found: {args.tokenizer}")
        print("   Make sure you trained a model first!")
        return
    
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    print(f"✓ Tokenizer loaded (vocab_size={tokenizer.get_vocab_size()})")
    
    # Load model
    model, config = load_model(args.checkpoint, device)
    
    # Run
    if args.interactive:
        interactive_mode(model, tokenizer, config)
    elif args.prompt:
        response = generate(
            model,
            tokenizer,
            args.prompt,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            device=device,
        )
        print(f"\n📝 Generated:\n{response}")
    else:
        # Default prompts for testing
        test_prompts = [
            "The brain is",
            "Neural networks",
            "Artificial intelligence",
            "Learning occurs when",
        ]
        
        print("\n🧪 Running test prompts...\n")
        
        for prompt in test_prompts:
            response = generate(
                model,
                tokenizer,
                prompt,
                max_tokens=50,
                temperature=0.8,
                device=device,
            )
            print(f"Prompt: \"{prompt}\"")
            print(f"Output: {response}\n")
            print("-" * 40)


if __name__ == "__main__":
    main()

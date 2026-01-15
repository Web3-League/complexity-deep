"""
Train a BPE tokenizer on edu-web dataset using complexity-framework.

Usage:
    python train_tokenizer.py --dataset Pacific-Prime/edu-web --vocab-size 100000
    python train_tokenizer.py --data ./local_data/*.txt --vocab-size 100000

Requires:
    pip install complexity-framework datasets
"""

import argparse
from pathlib import Path

from complexity.tokenizer import Tokenizer, TokenizerConfig


def main():
    parser = argparse.ArgumentParser(description="Train BPE tokenizer on edu-web")

    # Data source
    parser.add_argument("--dataset", type=str, default="Pacific-Prime/edu-web",
                        help="HuggingFace dataset name")
    parser.add_argument("--data", type=str, default=None,
                        help="Local data files (glob pattern, e.g., './data/*.txt')")
    parser.add_argument("--text-field", type=str, default="text",
                        help="Text field in dataset")
    parser.add_argument("--split", type=str, default="train",
                        help="Dataset split")
    parser.add_argument("--max-samples", type=int, default=None,
                        help="Max samples to use (None = all)")

    # Tokenizer config
    parser.add_argument("--vocab-size", type=int, default=100000,
                        help="Vocabulary size")
    parser.add_argument("--min-frequency", type=int, default=2,
                        help="Min frequency for a token")
    parser.add_argument("--method", type=str, default="bpe",
                        choices=["bpe", "unigram", "wordpiece"],
                        help="Tokenizer method")

    # Output
    parser.add_argument("--output", type=str, default="./tokenizer-eduweb",
                        help="Output directory")

    # HuggingFace
    parser.add_argument("--token", type=str, default=None,
                        help="HuggingFace token")

    args = parser.parse_args()

    print("=" * 60)
    print("TOKENIZER TRAINING (complexity-framework)")
    print("=" * 60)
    print(f"Dataset: {args.dataset if not args.data else args.data}")
    print(f"Vocab size: {args.vocab_size}")
    print(f"Method: {args.method}")
    print(f"Output: {args.output}")
    print()

    # Config
    config = TokenizerConfig(
        vocab_size=args.vocab_size,
        min_frequency=args.min_frequency,
        method=args.method,
        format="complexity",
    )

    # Train
    if args.data:
        # Local files
        tokenizer = Tokenizer.train(args.data, config=config)
    else:
        # HuggingFace dataset - use train_from_iterator
        from datasets import load_dataset

        print(f"Loading dataset: {args.dataset}...")
        ds = load_dataset(
            args.dataset,
            split=args.split,
            token=args.token,
            streaming=True,
        )

        # Create text iterator
        def text_iterator():
            count = 0
            for example in ds:
                text = example.get(args.text_field) or example.get("text") or example.get("content")
                if text:
                    yield text
                    count += 1
                    if count % 50000 == 0:
                        print(f"  Processed {count:,} samples...")
                    if args.max_samples and count >= args.max_samples:
                        break

        tokenizer = Tokenizer.train_from_iterator(text_iterator, config=config)

    # Save
    output_path = Path(args.output)
    tokenizer.save(str(output_path))

    # Also save HuggingFace-compatible files
    import json

    hf_config = {
        "added_tokens_decoder": {
            "0": {"content": "<|endoftext|>", "special": True},
            "1": {"content": "<|pad|>", "special": True},
            "2": {"content": "<|startoftext|>", "special": True},
            "3": {"content": "<|unk|>", "special": True},
        },
        "bos_token": "<|startoftext|>",
        "eos_token": "<|endoftext|>",
        "pad_token": "<|pad|>",
        "unk_token": "<|unk|>",
        "model_max_length": 1000000000000000019884624838656,
        "tokenizer_class": "PreTrainedTokenizerFast",
    }
    with open(output_path / "tokenizer_config.json", "w") as f:
        json.dump(hf_config, f, indent=2)

    special_map = {
        "bos_token": "<|startoftext|>",
        "eos_token": "<|endoftext|>",
        "pad_token": "<|pad|>",
        "unk_token": "<|unk|>",
    }
    with open(output_path / "special_tokens_map.json", "w") as f:
        json.dump(special_map, f, indent=2)

    print()
    print("=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)
    print(f"Saved to: {output_path}")
    print(f"Vocab size: {len(tokenizer)}")
    print()

    # Test
    print("Testing tokenizer:")
    test_texts = [
        "The mitochondria is the powerhouse of the cell.",
        "In mathematics, the Pythagorean theorem states that a² + b² = c².",
        "Python is a programming language: def hello(): print('Hello')",
        "La photosynthèse convertit la lumière en énergie.",
    ]

    for text in test_texts:
        tokens = tokenizer.encode(text, add_special_tokens=False)
        ratio = len(tokens) / len(text.split())
        print(f"  '{text[:40]}...'")
        print(f"    -> {len(tokens)} tokens ({ratio:.2f} tok/word)")
        print()


if __name__ == "__main__":
    main()

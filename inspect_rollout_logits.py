#!/usr/bin/env python3

import argparse
import json
import os
from pathlib import Path
from typing import List, Optional
import math

from text_heatmap import render_text_heatmap_terminal

try:
    from transformers import AutoTokenizer  # type: ignore
except Exception:
    AutoTokenizer = None  # transformers may not be available


def find_latest_episode(episodes_dir: Path) -> Optional[int]:
    """Find the highest episode number in the episodes directory."""
    if not episodes_dir.exists():
        return None
    
    episode_files = list(episodes_dir.glob("eps_*.json"))
    if not episode_files:
        return None
    
    # Extract episode numbers and find the maximum
    episode_numbers = []
    for file_path in episode_files:
        try:
            # Extract number from filename like "eps_000123.json"
            episode_num = int(file_path.stem.split("_")[1])
            episode_numbers.append(episode_num)
        except (ValueError, IndexError):
            continue
    
    return max(episode_numbers) if episode_numbers else None


def load_episode_data(run_path: Path, episode_num: int, is_eval: bool) -> List[dict]:
    """Load episode data from JSON file."""
    if is_eval:
        episodes_dir = run_path / "eval_episodes"
    else:
        episodes_dir = run_path / "episodes"
    
    episode_file = episodes_dir / f"eps_{episode_num:06d}.json"
    
    if not episode_file.exists():
        raise FileNotFoundError(f"Episode file not found: {episode_file}")
    
    with open(episode_file, 'r') as f:
        return json.load(f)


def tokenize_text(text: str) -> List[str]:
    """Simple tokenization that splits on whitespace and preserves special tokens."""
    tokens = []
    current_token = ""
    
    for char in text:
        if char.isspace():
            if current_token:
                tokens.append(current_token)
                current_token = ""
        else:
            current_token += char
    
    if current_token:
        tokens.append(current_token)
    
    return tokens


def visualize_logits(tokens: List[str], logits: List[float], logit_type: str):
    """Visualize logits using text coloring for already-tokenized text.

    - tokens: list of token strings for the response only (already tokenized)
    - logits: list of per-token values aligned to the response tokens
    - logit_type: label for the visualization
    """
    # Prepare logits (sanitize NaNs/Infs first)
    cleaned_logits: List[float] = []
    for logit in logits:
        try:
            val = float(logit)
            if math.isnan(val) or math.isinf(val):
                cleaned_logits.append(0.0)
            else:
                cleaned_logits.append(val)
        except Exception:
            cleaned_logits.append(0.0)
    logits = cleaned_logits

    # Map tokenizer whitespace/newline markers for display only
    display_tokens = [
        t.replace("Ġ", "_")  # GPT-2/RoBERTa space marker
        #  .replace("▁", " ")   # SentencePiece space marker
        .replace("Ċ", "\n")  # newline marker
        for t in tokens
    ]

    # Align logits to response tokens defensively
    num_resp_tokens = len(display_tokens)
    if num_resp_tokens == 0:
        print(f"No tokens found for {logit_type} visualization")
        return

    if len(logits) >= num_resp_tokens:
        response_logits = logits[-num_resp_tokens:]
    else:
        print(f"Warning: logits shorter ({len(logits)}) than tokens ({num_resp_tokens}); truncating tokens")
        response_logits = logits[:]
        display_tokens = display_tokens[:len(response_logits)]
        num_resp_tokens = len(display_tokens)

    if all(v == 0.0 for v in response_logits):
        print(f"All {logit_type} values are zero or NaN")
        return

    colour_desc = "green = low, red = high" if "Q_logits" in logit_type else "blue = low, red = high"
    print(f"\n{logit_type} visualization: {colour_desc}")
    print("-" * 64)

    # Color scheme
    if "Q_logits" in logit_type:
        color_start = [0, 255, 0]
        color_end = [255, 0, 0]
    else:
        color_start = [0, 0, 255]
        color_end = [255, 0, 0]

    render_text_heatmap_terminal(
        display_tokens,
        response_logits,
        color_start=color_start,
        color_end=color_end,
        rescale_value=True,
    )


def main():
    parser = argparse.ArgumentParser(description="Inspect rollout logits from training episodes")
    parser.add_argument("--run_name", type=str, required=True, help="Name of the run to inspect")
    parser.add_argument("--episode", type=int, default=-1, help="Episode number to inspect (default: -1 for latest)")
    parser.add_argument("--eval", action="store_true", help="Inspect evaluation episodes instead of training episodes")
    parser.add_argument("--n", type=int, default=3, help="Number of rollouts to inspect (default: 3)")
    parser.add_argument("--tokenizer", type=str, default="Qwen/Qwen2.5-3B-Instruct", help="Hugging Face tokenizer id to tokenize responses (e.g. 'Qwen/Qwen2.5-3B-Instruct')")
    parser.add_argument("--print_logits", action="store_true", help="Print the logits to the terminal")
    
    args = parser.parse_args()
    
    # Construct the run path
    run_path = Path("runs") / args.run_name
    
    if not run_path.exists():
        print(f"Run directory not found: {run_path}")
        return
    
    # Determine episode number
    if args.episode == -1:
        episodes_dir = run_path / ("eval_episodes" if args.eval else "episodes")
        episode_num = find_latest_episode(episodes_dir)
        if episode_num is None:
            print(f"No episodes found in {episodes_dir}")
            return
        print(f"Using latest episode: {episode_num}")
    else:
        episode_num = args.episode
    
    # Load episode data
    try:
        episode_data = load_episode_data(run_path, episode_num, args.eval)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return
    
    # Load tokenizer if requested
    tokenizer = None
    if args.tokenizer:
        if AutoTokenizer is None:
            print("Warning: transformers is not installed; falling back to whitespace tokenization")
        else:
            try:
                tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True)
                print(f"Loaded tokenizer: {args.tokenizer}")
            except Exception as e:
                print(f"Warning: failed to load tokenizer '{args.tokenizer}': {e}. Falling back to whitespace tokenization")
                tokenizer = None
    
    # Check if logits are present
    has_q_logits = any("Q_logits" in episode for episode in episode_data)
    has_format_logits = any("format_reward_logits" in episode for episode in episode_data)
    
    if not has_q_logits and not has_format_logits:
        print(f"No logits present in {args.run_name}")
        return
    
    print(f"Inspecting episode {episode_num} from {args.run_name}")
    print(f"Found {len(episode_data)} rollouts")
    print(f"Q logits present: {has_q_logits}")
    print(f"Format reward logits present: {has_format_logits}")
    print("=" * 80)
    
    # Inspect the first n rollouts
    n_rollouts = min(args.n, len(episode_data))

    # Collect queries and responses to enable batch tokenization
    queries: List[str] = [episode_data[i]["query"] for i in range(n_rollouts)]
    responses: List[str] = [episode_data[i]["response"] for i in range(n_rollouts)]

    # Tokenize queries and responses (batched if HF tokenizer is available)
    if tokenizer is not None:
        try:
            # Tokenize queries and responses
            batch_q = tokenizer(queries, add_special_tokens=False)
            batch_r = tokenizer(responses, add_special_tokens=False)

            # Convert token ids to tokens
            query_tokens_list = [tokenizer.convert_ids_to_tokens(ids) for ids in batch_q["input_ids"]]
            resp_tokens_list = [tokenizer.convert_ids_to_tokens(ids) for ids in batch_r["input_ids"]]
        except Exception:
            print("Warning: failed to batch tokenize; falling back to whitespace tokenization")
            query_tokens_list = [tokenize_text(q) for q in queries]
            resp_tokens_list = [tokenize_text(r) for r in responses]
    else:
        query_tokens_list = [tokenize_text(q) for q in queries]
        resp_tokens_list = [tokenize_text(r) for r in responses]

    for i in range(n_rollouts):
        rollout = episode_data[i]

        print(f"\nRollout {i+1}/{n_rollouts}")
        print(f"Reward: {rollout['reward']}")
        print(f"Query: {rollout['query']}")
        print(f"Response: {rollout['response']}")

        query_num_tokens = len(query_tokens_list[i])
        response_num_tokens = len(resp_tokens_list[i])

        # Visualize Q logits if present
        if "Q_logits" in rollout and rollout["Q_logits"]:
            q_slice = rollout["Q_logits"][query_num_tokens:query_num_tokens+response_num_tokens]
            visualize_logits(resp_tokens_list[i], q_slice, "Q_logits")
            if args.print_logits:
                print(q_slice)

        # Visualize format reward logits if present
        if "format_reward_logits" in rollout and rollout["format_reward_logits"]:
            f_slice = rollout["format_reward_logits"][query_num_tokens:query_num_tokens+response_num_tokens]
            visualize_logits(resp_tokens_list[i], f_slice, "format_reward_logits")
            if args.print_logits:
                print(f_slice)

        print("\n" + "#" * 96)


if __name__ == "__main__":
    main()

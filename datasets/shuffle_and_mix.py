import json
import random
import argparse
from pathlib import Path

def mix_and_shuffle(
    source_path: Path,
    target_path: Path,
    output_path: Path,
    fraction: float = 0.5
):
    # 1. Load JSON data (assumes each file is a JSON array)
    with source_path.open('r', encoding='utf-8') as f:
        data_source = json.load(f)
    with target_path.open('r', encoding='utf-8') as f:
        data_target = json.load(f)

    # 2. Determine sample size and sample without replacement
    sample_size = max(1, int(len(data_source) * fraction))
    sampled = random.sample(data_source, sample_size)

    # 3. Mix with the second dataset
    combined = data_target + sampled

    # 4. Shuffle the combined list in place
    random.shuffle(combined)

    # 5. Write out the combined, shuffled list
    with output_path.open('w', encoding='utf-8') as f:
        json.dump(combined, f, ensure_ascii=False, indent=2)

    print(f"Sampled {sample_size} items from {source_path.name},")
    print(f"combined with {len(data_target)} items from {target_path.name},")
    print(f"and wrote {len(combined)} total items to {output_path.name}.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Sample 20% from one JSON list and mix with another, then shuffle."
    )
    parser.add_argument(
        "--source_json",
        type=Path,
        help="Path to the JSON file to sample from (expects a JSON array)."
    )
    parser.add_argument(
        "--target_json",
        type=Path,
        help="Path to the JSON file to mix into (expects a JSON array)."
    )
    parser.add_argument(
        "--output_json",
        type=Path,
        help="Path where the combined, shuffled JSON array will be saved."
    )
    parser.add_argument(
        "--fraction",
        type=float,
        default=0.5,
        help="Fraction of the first file to sample (default: 0.2 = 20%)."
    )

    args = parser.parse_args()
    mix_and_shuffle(
        args.source_json,
        args.target_json,
        args.output_json,
        args.fraction
    )

# cd datasets && python shuffle_and_mix.py --target_json ../LLaMA-Factory/data/mix-vpt-spatial.json --source_json ../LLaMA-Factory/data/mix-vpt-llava.json --output_json  ../LLaMA-Factory/data/mix-vpt-llava.json --fraction 0.5

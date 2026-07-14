#!/usr/bin/env python3
"""
Builds the TSV files expected by `zipvoice.bin.prepare_dataset` from a raw
directory of paired wav/txt files, such as the InfoRe Vietnamese dataset
(files named "{id}.wav" / "{id}.txt", one utterance per pair).

Any stray *.txt file without a matching *.wav (e.g. InfoRe's
"unaligned.txt" alignment log) is skipped.

Usage:

python3 local/prepare_infore.py \
    --data-dir ../../data \
    --output-dir data/raw \
    --prefix infore \
    --num-dev 100

Produces "data/raw/infore_train.tsv" and "data/raw/infore_dev.tsv".
"""

import argparse
import random
from pathlib import Path


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-dir",
        type=str,
        required=True,
        help="Directory containing paired {id}.wav/{id}.txt files.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/raw",
        help="Where to write the output TSV files.",
    )
    parser.add_argument(
        "--prefix",
        type=str,
        default="infore",
        help="Dataset name prefix used in the output TSV filenames.",
    )
    parser.add_argument(
        "--num-dev",
        type=int,
        default=100,
        help="Number of utterances held out for the dev set.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed used for the train/dev split.",
    )
    return parser.parse_args()


def main():
    args = get_args()
    data_dir = Path(args.data_dir).resolve()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    utterances = []
    skipped_no_wav = 0
    skipped_empty = 0
    for txt_path in sorted(data_dir.glob("*.txt")):
        uniq_id = txt_path.stem
        wav_path = txt_path.with_suffix(".wav")
        if not wav_path.is_file():
            skipped_no_wav += 1
            continue
        text = " ".join(txt_path.read_text(encoding="utf-8").split())
        if not text:
            skipped_empty += 1
            continue
        utterances.append((uniq_id, text, str(wav_path)))

    print(
        f"Found {len(utterances)} valid utterances in {data_dir} "
        f"(skipped {skipped_no_wav} without a wav, {skipped_empty} with empty text)"
    )
    if not utterances:
        raise RuntimeError(f"No valid wav/txt pairs found under {data_dir}")

    rng = random.Random(args.seed)
    rng.shuffle(utterances)

    num_dev = min(args.num_dev, len(utterances) - 1)
    dev_set = utterances[:num_dev]
    train_set = utterances[num_dev:]

    for subset, rows in [("train", train_set), ("dev", dev_set)]:
        out_path = output_dir / f"{args.prefix}_{subset}.tsv"
        with open(out_path, "w", encoding="utf-8") as f:
            for uniq_id, text, wav_path in rows:
                f.write(f"{uniq_id}\t{text}\t{wav_path}\n")
        print(f"Wrote {len(rows)} utterances to {out_path}")


if __name__ == "__main__":
    main()

#!/bin/bash

# Fine-tune ZipVoice on the InfoRe Vietnamese dataset, starting from the
# pre-trained checkpoint already available under ../../checkpoints/zipvoice/
# (no need to re-download it from HuggingFace).

# Add project root to PYTHONPATH
export PYTHONPATH=../../:$PYTHONPATH

set -e
set -u
set -o pipefail

stage=0
stop_stage=6

# Number of jobs for data preparation
nj=20

# InfoRe is Vietnamese, so we use the espeak (IPA) tokenizer.
# See https://github.com/rhasspy/espeak-ng/blob/master/docs/languages.md
lang=vi
tokenizer=espeak

# Maximum length (seconds) of the training utterance, will filter out longer
# utterances. Check with:
# `lhotse cut describe data/fbank/infore-finetune_cuts_train.jsonl.gz`
max_len=20

# Raw InfoRe data: paired {id}.wav/{id}.txt files (see ../../.gitignore,
# extracted from infore_16k_denoised.zip)
raw_data_dir=../../data
num_dev=100

# Pre-trained base checkpoint (already downloaded into the repo)
base_dir=../../checkpoints/zipvoice

### Prepare the training data (0 - 3)

if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ]; then
      echo "Stage 0: Build TSV files from raw InfoRe wav/txt pairs"
      mkdir -p data/raw
      python3 local/prepare_infore.py \
            --data-dir ${raw_data_dir} \
            --output-dir data/raw \
            --prefix infore \
            --num-dev ${num_dev}
      # Produces data/raw/infore_train.tsv and data/raw/infore_dev.tsv
fi

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
      echo "Stage 1: Prepare manifests for InfoRe dataset from tsv files"
      for subset in train dev; do
            python3 -m zipvoice.bin.prepare_dataset \
                  --tsv-path data/raw/infore_${subset}.tsv \
                  --prefix infore-finetune \
                  --subset raw_${subset} \
                  --num-jobs ${nj} \
                  --output-dir data/manifests
      done
      # The output manifest files are "data/manifests/infore-finetune_cuts_raw_train.jsonl.gz"
      # and "data/manifests/infore-finetune_cuts_raw_dev.jsonl.gz".
fi

if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
      echo "Stage 2: Add tokens to manifests"
      for subset in train dev; do
            python3 -m zipvoice.bin.prepare_tokens \
                  --input-file data/manifests/infore-finetune_cuts_raw_${subset}.jsonl.gz \
                  --output-file data/manifests/infore-finetune_cuts_${subset}.jsonl.gz \
                  --tokenizer ${tokenizer} \
                  --lang ${lang}
      done
fi

if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
      echo "Stage 3: Compute Fbank for InfoRe dataset"
      for subset in train dev; do
            python3 -m zipvoice.bin.compute_fbank \
                  --source-dir data/manifests \
                  --dest-dir data/fbank \
                  --dataset infore-finetune \
                  --subset ${subset} \
                  --num-jobs ${nj}
      done
fi

### Training ZipVoice (4 - 5)

if [ ${stage} -le 4 ] && [ ${stop_stage} -ge 4 ]; then
      echo "Stage 4: Fine-tune the ZipVoice model from the base checkpoint"

      [ -z "$max_len" ] && { echo "Error: max_len is not set!" >&2; exit 1; }

      # --world-size assumes a single GPU; raise it if you have more.
      python3 -m zipvoice.bin.train_zipvoice \
            --world-size 1 \
            --use-fp16 1 \
            --finetune 1 \
            --base-lr 0.0001 \
            --num-iters 10000 \
            --save-every-n 1000 \
            --max-duration 500 \
            --max-len ${max_len} \
            --model-config ${base_dir}/model.json \
            --checkpoint ${base_dir}/models.pt \
            --tokenizer ${tokenizer} \
            --lang ${lang} \
            --token-file ${base_dir}/tokens.txt \
            --dataset custom \
            --train-manifest data/fbank/infore-finetune_cuts_train.jsonl.gz \
            --dev-manifest data/fbank/infore-finetune_cuts_dev.jsonl.gz \
            --exp-dir exp/zipvoice_finetune_infore
fi

if [ ${stage} -le 5 ] && [ ${stop_stage} -ge 5 ]; then
      echo "Stage 5: Average the checkpoints for ZipVoice"
      python3 -m zipvoice.bin.generate_averaged_model \
            --iter 10000 \
            --avg 2 \
            --model-name zipvoice \
            --exp-dir exp/zipvoice_finetune_infore
      # The generated model is exp/zipvoice_finetune_infore/iter-10000-avg-2.pt
      # This is the checkpoint to pass as --teacher-model in run_distill_infore.sh
fi

### Inference with PyTorch model (6)

if [ ${stage} -le 6 ] && [ ${stop_stage} -ge 6 ]; then
      echo "Stage 6: Inference of the fine-tuned ZipVoice model"
      python3 -m zipvoice.bin.infer_zipvoice \
            --model-name zipvoice \
            --model-dir exp/zipvoice_finetune_infore/ \
            --checkpoint-name iter-10000-avg-2.pt \
            --tokenizer ${tokenizer} \
            --lang ${lang} \
            --test-list test.tsv \
            --res-dir results/test_finetune_infore \
            --num-step 16
fi

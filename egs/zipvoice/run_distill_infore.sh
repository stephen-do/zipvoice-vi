#!/bin/bash

# Distill the InfoRe-finetuned ZipVoice model (produced by
# run_finetune_infore.sh) into a ZipVoice-Distill model, following the
# two-stage distillation recipe from run_emilia.sh.
#
# Run run_finetune_infore.sh first (through stage 5) so that
# exp/zipvoice_finetune_infore/iter-10000-avg-2.pt exists.

# Add project root to PYTHONPATH
export PYTHONPATH=../../:$PYTHONPATH

set -e
set -u
set -o pipefail

stage=1
stop_stage=4

lang=vi
tokenizer=espeak

# The base checkpoint's model.json / tokens.txt are shared across all stages
base_dir=../../checkpoints/zipvoice

# Fine-tuned ZipVoice model used as the fixed teacher for distillation stage 1
teacher_model_stage1=exp/zipvoice_finetune_infore/iter-10000-avg-2.pt

train_manifest=data/fbank/infore-finetune_cuts_train.jsonl.gz
dev_manifest=data/fbank/infore-finetune_cuts_dev.jsonl.gz

if [ ! -f "$teacher_model_stage1" ]; then
      echo "Error: expect $teacher_model_stage1 !" >&2
      echo "Run run_finetune_infore.sh (stages 0-5) first." >&2
      exit 1
fi

### Training ZipVoice-Distill (1 - 4)

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
      echo "Stage 1: Train the ZipVoice-Distill model (first stage)"
      # --world-size assumes a single GPU; raise it if you have more.
      python3 -m zipvoice.bin.train_zipvoice_distill \
            --world-size 1 \
            --use-fp16 1 \
            --num-iters 60000 \
            --max-duration 500 \
            --base-lr 0.0005 \
            --model-config ${base_dir}/model.json \
            --tokenizer ${tokenizer} \
            --lang ${lang} \
            --token-file ${base_dir}/tokens.txt \
            --dataset custom \
            --train-manifest ${train_manifest} \
            --dev-manifest ${dev_manifest} \
            --teacher-model ${teacher_model_stage1} \
            --distill-stage first \
            --exp-dir exp/zipvoice_distill_infore_1stage
fi

if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
      echo "Stage 2: Average the checkpoints for ZipVoice-Distill (first stage)"
      python3 -m zipvoice.bin.generate_averaged_model \
            --iter 60000 \
            --avg 7 \
            --model-name zipvoice_distill \
            --exp-dir exp/zipvoice_distill_infore_1stage
      # The generated model is exp/zipvoice_distill_infore_1stage/iter-60000-avg-7.pt
fi

if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
      echo "Stage 3: Train the ZipVoice-Distill model (second stage)"
      python3 -m zipvoice.bin.train_zipvoice_distill \
            --world-size 1 \
            --use-fp16 1 \
            --num-iters 2000 \
            --save-every-n 1000 \
            --max-duration 500 \
            --base-lr 0.0001 \
            --model-config ${base_dir}/model.json \
            --tokenizer ${tokenizer} \
            --lang ${lang} \
            --token-file ${base_dir}/tokens.txt \
            --dataset custom \
            --train-manifest ${train_manifest} \
            --dev-manifest ${dev_manifest} \
            --teacher-model exp/zipvoice_distill_infore_1stage/iter-60000-avg-7.pt \
            --distill-stage second \
            --exp-dir exp/zipvoice_distill_infore
fi

if [ ${stage} -le 4 ] && [ ${stop_stage} -ge 4 ]; then
      echo "Stage 4: Average the checkpoints for ZipVoice-Distill (second stage)"
      python3 -m zipvoice.bin.generate_averaged_model \
            --iter 2000 \
            --avg 2 \
            --model-name zipvoice_distill \
            --exp-dir exp/zipvoice_distill_infore
      # The generated model is exp/zipvoice_distill_infore/iter-2000-avg-2.pt
fi

### Inference with PyTorch model (5)

if [ ${stage} -le 5 ] && [ ${stop_stage} -ge 5 ]; then
      echo "Stage 5: Inference of the ZipVoice-Distill model"
      python3 -m zipvoice.bin.infer_zipvoice \
            --model-name zipvoice_distill \
            --model-dir exp/zipvoice_distill_infore/ \
            --checkpoint-name iter-2000-avg-2.pt \
            --tokenizer ${tokenizer} \
            --lang ${lang} \
            --test-list test.tsv \
            --res-dir results/test_distill_infore \
            --num-step 8
fi

stage=${1:-0}
stop_stage=${2:-99}
model_name=${3:-zipvoice}

echo "Start stage: $stage, Stop stage: $stop_stage"
echo "Model name: $model_name"
export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH=$PYTHONPATH:/workspace/zipvoice-vi

MODEL_DIR=/workspace/zipvoice-vi/checkpoints # huggingface model dir (only used by stage 1 download)

# CKPT_DIR: directory containing model.pt, tokens.txt, model.json for $model_name.
# TRT_DIR: directory to write/read the exported TensorRT engine.
# Both default to the huggingface download layout, but can be overridden to
# point at your own fine-tuned/distilled checkpoint, e.g.:
#   CKPT_DIR=/path/to/your/models bash run.sh 2 4 zipvoice_distill
CKPT_DIR=${CKPT_DIR:-$MODEL_DIR/$model_name}
TRT_DIR=${TRT_DIR:-$MODEL_DIR/${model_name}_trt}
MODEL_REPO=./model_repo_${model_name}

# Tokenizer used to train/fine-tune the checkpoint, one of:
# emilia | espeak | libritts | simple. LANG is only used when TOKENIZER=espeak
# (see https://github.com/rhasspy/espeak-ng/blob/master/docs/languages.md).
TOKENIZER=${TOKENIZER:-espeak}
LANG=${LANG:-vi}

if [ "$stage" -le 1 ] && [ "$stop_stage" -ge 1 ]; then
    echo "Stage 1: Download huggingface models"
    hf download k2-fsa/ZipVoice --local-dir $MODEL_DIR || exit 1
fi

if [ "$stage" -le 2 ] && [ "$stop_stage" -ge 2 ]; then
    echo "Stage 2: Export Zipvoice TensorRT model"
    python3 -m zipvoice.bin.tensorrt_export \
        --model-name $model_name \
        --model-dir $CKPT_DIR \
        --checkpoint-name model.pt \
        --max-batch-size 16 \
        --trt-engine-file-name fm_decoder.fp16.plan \
        --tensorrt-model-dir $TRT_DIR || exit 1
fi

if [ "$stage" -le 3 ] && [ "$stop_stage" -ge 3 ]; then
    echo "Building triton server"
    rm -rf $MODEL_REPO
    cp -r ./model_repo $MODEL_REPO
    python3 scripts/fill_template.py -i $MODEL_REPO/zipvoice/config.pbtxt model_dir:$CKPT_DIR,model_name:$model_name,trt_engine_path:$TRT_DIR/fm_decoder.fp16.plan,tokenizer:$TOKENIZER,lang:$LANG
fi

if [ "$stage" -le 4 ] && [ "$stop_stage" -ge 4 ]; then
    echo "Starting triton server"
    tritonserver --model-repository=$MODEL_REPO
fi

if [ "$stage" -le 5 ] && [ "$stop_stage" -ge 5 ]; then
    echo "Testing triton server"
    num_tasks=(1 2 4 8)
    split_name=wenetspeech4tts
    for num_task in ${num_tasks[@]}; do
        log_dir=./log_${model_name}_concurrent_${num_task}_${split_name}
        python3 client_grpc.py  \
                --num-tasks $num_task --huggingface-dataset yuekai/seed_tts_cosy2 \
                --split-name $split_name --log-dir $log_dir
    done
fi

if [ "$stage" -le 6 ] && [ "$stop_stage" -ge 6 ]; then
    echo "Testing http client"
    wget -nc https://raw.githubusercontent.com/SparkAudio/Spark-TTS/main/example/prompt_audio.wav -O prompt.wav
    python3 client_http.py --reference-audio prompt.wav \
        --reference-text "吃燕窝就选燕之屋，本节目由26年专注高品质燕窝的燕之屋冠名播出。豆奶牛奶换着喝，营养更均衡，本节目由豆本豆豆奶特约播出。" \
        --target-text "身临其境，换新体验。塑造开源语音合成新范式，让智能语音更自然。" \
        --output-audio "./test.wav"
fi

if [ "$stage" -le 7 ] && [ "$stop_stage" -ge 7 ]; then
    echo "Starting pytriton server"
    wget -nc https://raw.githubusercontent.com/FunAudioLLM/CosyVoice/main/asset/zero_shot_prompt.wav -O prompt_short.wav
    python3 pytriton_server.py  \
        --model_dir $CKPT_DIR \
        --model_name $model_name \
        --trt_engine_path $TRT_DIR/fm_decoder.fp16.plan \
        --reference_audio_sample_rate 16000 \
        --port 8000 \
        --max_batch_size 4 \
        --use_speaker_cache \
        --prompt_audio prompt_short.wav \
        --prompt_text "希望你以后能够做得比我还好呦。"
fi

if [ "$stage" -le 8 ] && [ "$stop_stage" -ge 8 ]; then
    echo "Testing pytriton server with speaker cache"
    num_tasks=(1 2 4 8)
    split_name=wenetspeech4tts
    for num_task in ${num_tasks[@]}; do
        log_dir=./log_spk_cache_pytriton_${model_name}_concurrent_${num_task}_${split_name}
        python3 client_grpc.py  --num-tasks $num_task --huggingface-dataset yuekai/seed_tts_cosy2 --split-name $split_name --log-dir $log_dir --use-spk2info-cache True
    done
fi
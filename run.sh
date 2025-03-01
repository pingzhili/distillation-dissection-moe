export PYTHONPATH=$PYTHONPATH:src
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
# python debug.py
# python scripts/generate-distillation-data.py
accelerate launch --num_processes=8 scripts/generate-distillation-data.py


CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python3 -m sglang.launch_server \
    --model-path moonshotai/Moonlight-16B-A3B-Instruct --trust-remote-code --disable-radix-cache \
    --max-running-requests 32 --max-prefill-tokens 1024 --chunked-prefill-size 16384 \
    --tp 8 --dp 8 --enable-dp-attention --enable-ep-moe
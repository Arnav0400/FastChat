torchrun --nproc_per_node=2 --master_port=8000 fastchat/train/train_glora.py \
    --model_name_or_path meta-llama/Llama-2-7b-hf  \
    --glora_r 4 \
    --data_path /home/arnav.chavan/NIPS23/FastChat/data/shareGPT_clean.json \
    --bf16 True \
    --output_dir /l/users/arnav.chavan/checkpoints_glora_llama2_BS=32_R=4 \
    --num_train_epochs 15 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 16 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 1200 \
    --save_total_limit 10 \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.01 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 2048 \
    --lazy_preprocess True \

# torchrun --nproc_per_node=2 --master_port=6014 fastchat/train/train_lora.py \
#     --model_name_or_path /home/arnav.chavan/NIPS23/llama-7b-hf  \
#     --lora_r 8 \
#     --data_path /home/arnav.chavan/NIPS23/FastChat/data/shareGPT_clean.json \
#     --bf16 True \
#     --output_dir /l/users/arnav.chavan/temp \
#     --num_train_epochs 3 \
#     --per_device_train_batch_size 1 \
#     --per_device_eval_batch_size 1 \
#     --gradient_accumulation_steps 32 \
#     --evaluation_strategy "no" \
#     --save_strategy "steps" \
#     --save_steps 1200 \
#     --save_total_limit 10 \
#     --learning_rate 2e-5 \
#     --weight_decay 0. \
#     --warmup_ratio 0.03 \
#     --lr_scheduler_type "cosine" \
#     --logging_steps 1 \
#     --tf32 True \
#     --model_max_length 2048 \
#     --lazy_preprocess True \
#     --lora_alpha 16 \
#     --lora_dropout 0.05  \

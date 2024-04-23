MODEL_KEY=deepseek-ai/deepseek-coder-1.3b-base
MODEL_NAME_OR_PATH=deepseek-coder-8x1.3b-top-6-moe-base
OUTPUT_DIR=ds-8x1.3b-top-6-universal-evol-instruct-5e-5_bs_64_epoch_4

CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch --main_process_port 29500 train.py \
    --model_key $MODEL_KEY \
    --model_name_or_path $MODEL_NAME_OR_PATH \
    --use_flash_attention True \
    --use_moe True \
    --datafile_paths \
        ../../data/data-evol_instruct-decontaminated.jsonl \
    --output_dir $OUTPUT_DIR \
    --bf16 True \
    --num_train_epochs 4 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 16 \
    --max_training_seq_length 1024 \
    --group_by_length False \
    --ddp_find_unused_parameters False \
    --logging_steps 1 \
    --log_level info \
    --save_strategy epoch \
    --save_steps 20 \
    --optim adafactor \
    --max_grad_norm -1 \
    --warmup_steps 500 \
    --learning_rate 5e-5 \
    --lr_scheduler_type linear

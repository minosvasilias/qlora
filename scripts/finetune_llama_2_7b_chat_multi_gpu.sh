accelerate launch qlora.py \
    --model_name_or_path meta-llama/Llama-2-7b-chat-hf \
    --use_auth \
    --output_dir ./output/llama-2-chat-qlora-multi-gpu \
    --dataset data/llama_example_data.jsonl \
    --dataset_format llama2 \
    --logging_steps 1 \
    --save_strategy steps \
    --data_seed 42 \
    --save_steps 200 \
    --save_total_limit 40 \
    --evaluation_strategy steps \
    --eval_dataset_size 200 \
    --max_eval_samples 200 \
    --per_device_eval_batch_size 1 \
    --max_new_tokens 32 \
    --dataloader_num_workers 1 \
    --group_by_length \
    --logging_strategy steps \
    --remove_unused_columns False \
    --do_train \
    --do_eval \
    --lora_r 64 \
    --lora_alpha 16 \
    --lora_modules all \
    --double_quant \
    --quant_type nf4 \
    --bf16 \
    --bits 4 \
    --warmup_ratio 0.03 \
    --lr_scheduler_type constant \
    --source_max_len 2596 \
    --target_max_len 1500 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 16 \
    --max_steps 2000 \
    --eval_steps 1000 \
    --learning_rate 0.0002 \
    --adam_beta2 0.999 \
    --max_grad_norm 0.3 \
    --lora_dropout 0.1 \
    --weight_decay 0.0 \
    --seed 0 \
    --ddp_find_unused_parameters False \
    --report_to wandb
    
#!/bin/bash

export CUSTOM_MODEL_DIR="custom-model-path"
export OUTPUT_DIR="output-path"

filename=("backpack" "backpack_dog" "bear_plushie" "berry_bowl" "can" "candle" "cat" \
"cat2" "clock" "colorful_sneaker" "dog" "dog2" "dog3" "dog5" "dog6" "dog7" "dog8" "duck_toy" "fancy_boot" "grey_sloth_plushie" \
"monster_toy" "pink_sunglasses" "poop_emoji" "rc_car" "red_cartoon" "robot_toy" "shiny_sneaker" "teapot" "vase" "wolf_plushie")


for i in "${!filename[@]}"; do
    file_item=${filename[$i]}
    
    echo "Generating images with model #$(($i+1)) $file_item"

    accelerate launch inference.py \
    --custom_model_path= "${CUSTOM_MODEL_PATH}/${file_item}" \
    --num_samples=4 \
    --num_inference_steps=50 \
    --output_path="${OUTPUT_DIR}/${file_item}" \
    --seed=42 \
    --modifier_token="sks" \
    --mask_token="sks" \
    --model_version="sd-1.4" \
    --attn_scale=$scale
    
    if [ $? -ne 0 ]; then
        echo "Error: An error occurred while processing $file_item."
    exit 1
    fi
done
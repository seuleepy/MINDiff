"""
This code performs inference on a Dreambooth dataset.
"""

import os
import argparse
from tqdm.auto import tqdm
import json

import torch
from accelerate import Accelerator

from data.constant import (
    TARGET2CLASS_MAPPING,
    CLASS2OBJECT_MAPPING,
    OBJECT_PROMPT_LIST,
    LIVE_PROMPT_LIST,
)
from utils.set_attn_proc import set_mask_attn
from utils.pipeline_stable_diffusion_mask import StableDiffusionMaskPipeline
from utils.mask_attention_processor import MaskAttnProcessor2_0


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--custom_model_path",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--model_version",
        type=str,
        required=True,
        choices=["sd-1.4", "sd-2.1"],
        help="Model version (e.g., sd-1.4, sd-2.1)",
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=4,
    )
    parser.add_argument(
        "--num_inference_steps",
        type=int,
        default=50,
    )
    parser.add_argument(
        "--output_path",
        type=str,
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )
    parser.add_argument(
        "--modifier_token",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--attn_scale",
        type=float,
        default=0.5,
    )
    parser.add_argument(
        "--use_to_q_pre",
        action="store_true",
    )
    parser.add_argument(
        "--mask_token",
        type=str,
    )

    args = parser.parse_args()
    return args


resolution_map = {
    "sd-1.4": 16,
    "sd-2.1": 24,
}


def append_to_json(output_file, new_data):
    if os.path.exists(output_file):
        with open(output_file, "r") as f:
            existing_data = json.load(f)
    else:
        existing_data = []

    existing_data.append(new_data)

    with open(output_file, "w") as f:
        json.dump(existing_data, f, indent=4)


def main(args):

    # load pipeline
    pipeline = StableDiffusionMaskPipeline.from_pretrained(
        args.custom_model_path,
    )
    pipeline.safety_checker = None
    unet = pipeline.unet
    unet = set_mask_attn(
        unet=unet,
        mask_resolution=resolution_map[args.model_version],
    )
    mask_token_id = pipeline.tokenizer.convert_tokens_to_ids(f"{args.mask_token}</w>")

    # prepare the generation with accelerator
    accelerator = Accelerator()
    pipeline.to(accelerator.device)
    pipeline.set_progress_bar_config(disable=True)
    generator = torch.Generator(device=accelerator.device).manual_seed(args.seed)

    # prepare prompt list
    target_name = os.path.basename(args.custom_model_path)
    class_type = TARGET2CLASS_MAPPING[target_name]
    is_object = CLASS2OBJECT_MAPPING[class_type]
    if is_object:
        prompt_list = OBJECT_PROMPT_LIST
    else:
        prompt_list = LIVE_PROMPT_LIST

    # calculate total images
    progress_bar = tqdm(
        total=len(prompt_list),
        initial=0,
        desc="Generating images",
        disable=not accelerator.is_local_main_process,
    )

    if accelerator.is_main_process:
        os.makedirs(args.output_path, exist_ok=True)
    accelerator.wait_for_everyone()

    # inference
    indexed_prompt_ls = [
        (idx, prompt_template) for idx, prompt_template in enumerate(prompt_list)
    ]
    with accelerator.split_between_processes(indexed_prompt_ls) as split_prompt_ls:
        for global_idx, prompt_template in split_prompt_ls:

            prompt = prompt_template.format(f"{args.modifier_token} {class_type}")
            sub_prompt = f"{args.modifier_token} {class_type}"

            MaskAttnProcessor2_0.mask = None
            result = pipeline(
                prompt=prompt,
                num_images_per_prompt=args.num_samples,
                num_inference_steps=args.num_inference_steps,
                guidance_scale=7.5,
                generator=generator,
                mask_token_id=mask_token_id,
                sub_prompt=sub_prompt,
                attn_scale=args.attn_scale,
            ).images

            for img_idx in range(args.num_samples):
                gen_file_path = (
                    f"{args.output_path}/prompt_{global_idx:02d}_{img_idx:02d}.jpg"
                )

                result[img_idx].save(gen_file_path)

            if global_idx == len(prompt_list) - 1:
                progress_bar.update(1)
            else:
                progress_bar.update(1 * accelerator.num_processes)

    progress_bar.close()


if __name__ == "__main__":
    args = parse_args()
    main(args)

import torch
import argparse
import numpy as np
import PIL.Image as Image
import os
import cv2
from mmdit.flux_pipeline import FluxPipeline
from inversion.flow_direct_correction_inv_flux import Accurate_Inversion_FLUX
from inversion.inv_utils import fix_seed, view_images
from controller import attn_norm_ctrl_flux


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_steps', type=int, default=30)
    parser.add_argument('--skip_steps', type=int, default=0)
    parser.add_argument('--inv_cfg', type=float, default=1.0)
    parser.add_argument('--recov_cfg', type=float, default=2.0)
    parser.add_argument('--ly_ratio', type=float, default=0.0)
    parser.add_argument('--attn_ratio', type=float, default=0.15)
    parser.add_argument('--src_prompt', type=str, default="")
    parser.add_argument('--tar_prompt', type=str, default="")
    parser.add_argument('--src_path', type=str, default=None, required=True)
    parser.add_argument('--saved_path', type=str, default=None, required=True)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--seed', type=int, default=2024)
    parser.add_argument('--mask_path', type=str, default=None, required=False)
    return parser.parse_args()


if __name__ == "__main__":
    args = get_parser()
    fix_seed(args.seed)
    g = torch.Generator(device=args.device).manual_seed(args.seed)

    pipe = FluxPipeline.from_pretrained("black-forest-labs/FLUX.1-dev", torch_dtype=torch.float16)
    pipe = pipe.to("cuda")

    pipe = pipe.to(args.device)
    pipe.transformer.eval()
    pipe.vae.eval()
    invf = Accurate_Inversion_FLUX(pipe, args.num_steps, args.device, args.inv_cfg, args.recov_cfg, args.skip_steps, args.saved_path)

    img_f = args.src_path
    mask_f = args.mask_path
    image = Image.open(img_f).convert("RGB")
    if mask_f is not None:
        mask = Image.open(mask_f).convert("L")
    else:
        mask = None

    src_prompt = args.src_prompt
    tar_prompt = args.tar_prompt
    print(src_prompt)
    print(tar_prompt)
    prompts = [src_prompt, tar_prompt]


    ################## edit ###################
    attn_norm_ctrl_flux.register_attention_control_flux(pipe, None, None)
    all_latents, delta_list = invf.euler_flow_inversion(prompt=src_prompt, image=img_f)

    controller_ada = attn_norm_ctrl_flux.FluxAdalayernorm_replace(prompts, args.num_steps, args.ly_ratio, pipe.tokenizer, pipe.tokenizer_2, device="cuda")
    controller_attn = attn_norm_ctrl_flux.FluxAttentionReplace(prompts, args.num_steps, args.attn_ratio)
    attn_norm_ctrl_flux.register_attention_control_flux(pipe, controller_attn, controller_ada)

    image_list = invf.direct_inversion(prompts, controller=controller_ada, all_latents=all_latents,
                                           delta_list=delta_list, original_size=image.size, mask_image=mask)

    result_path = args.saved_path + '/' + img_f.split('/')[-1][:-4]
    view_images(image_list, result_path)

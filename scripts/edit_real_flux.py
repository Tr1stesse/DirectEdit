import torch
from mmdit.flux_pipeline import FluxPipeline
from inversion.flow_direct_correction_inv_flux import Accurate_Inversion_FLUX
from inversion.inv_utils import fix_seed, view_images, load_PIE_images
from controller import attn_norm_ctrl_flux
import argparse
import numpy as np
import PIL.Image as Image
import os
import traceback 

def get_parser():
    parser = argparse.ArgumentParser()
    # parser.add_argument('--model_path', type=str, default=None, required=True)
    parser.add_argument('--num_steps', type=int, default=30)
    parser.add_argument('--skip_steps', type=int, default=0)
    parser.add_argument('--inv_cfg', type=float, default=1.0)
    parser.add_argument('--recov_cfg', type=float, default=2.0)
    parser.add_argument('--ly_ratio', type=float, default=0)
    parser.add_argument('--attn_ratio', type=float, default=1.0)
    parser.add_argument('--src_prompt', type=str, default="",)
    parser.add_argument('--tar_prompt', type=str, default="",)
    parser.add_argument('--src_path', type=str, default=None, required=True)
    parser.add_argument('--saved_path', type=str, default=None, required=True)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--seed', type=int, default=2024)
    return parser.parse_args()

if __name__ == "__main__":
    args = get_parser()
    fix_seed(args.seed)
    g = torch.Generator(device=args.device).manual_seed(args.seed)

    print("Loading model...")
    pipe = FluxPipeline.from_pretrained("black-forest-labs/FLUX.1-dev",
                                                    torch_dtype=torch.float16)
    pipe = pipe.to(args.device)
    pipe.transformer.eval()
    pipe.vae.eval()

    invf = Accurate_Inversion_FLUX(pipe, args.num_steps, args.device, args.inv_cfg, args.recov_cfg, args.skip_steps, args.saved_path)

    ######## Read images from the PIE
    ori_prp_list, edi_prp_list, img_list, edi_ins_list, bld_list, _ = load_PIE_images(args.src_path, edit_category_list=["0","1","2","3","4","5","6","7","8","9"])

    total_imgs = len(ori_prp_list)
    print(f"Total images to process: {total_imgs}")

    for i in range(total_imgs):
        img_f = img_list[i]

        try:
            print(f"[{i+1}/{total_imgs}] Processing: {img_f}")

            image = Image.open(img_f).convert("RGB")

            file_name_stem = os.path.splitext(os.path.basename(img_f))[0]

            mask_name = f"{file_name_stem}_mask.jpg"
            mask_path = os.path.join(args.src_path, "mask_generated", mask_name)

            if os.path.exists(mask_path):
                print(f"  -> Loading external mask: {mask_path}")
                mask = Image.open(mask_path).convert("L")

                if mask.size != image.size:
                    mask = mask.resize(image.size, Image.NEAREST)
            else:
                print(f"  -> [Warning] Mask not found at {mask_path}. Using blank mask.")
                mask = Image.new("L", image.size, 0)

            # =======================================================

            src_prompt = ori_prp_list[i].replace("[", "").replace("]", "")
            tar_prompt = edi_prp_list[i].replace("[", "").replace("]", "")

            prompts = [src_prompt, tar_prompt]

            ################## edit ###################
            attn_norm_ctrl_flux.register_attention_control_flux(pipe, None, None)

            all_latents, delta_list = invf.euler_flow_inversion(prompt=src_prompt, image=img_f)

            controller_ada = attn_norm_ctrl_flux.FluxAdalayernorm_replace(prompts, args.num_steps, args.ly_ratio, pipe.tokenizer, pipe.tokenizer_2, device="cuda")
            controller_attn = attn_norm_ctrl_flux.FluxAttentionReplace(prompts, args.num_steps, args.attn_ratio)
            attn_norm_ctrl_flux.register_attention_control_flux(pipe, controller_attn, controller_ada)

            image_list = invf.direct_inversion(prompts, controller=controller_ada, all_latents=all_latents,
                                                delta_list=delta_list, original_size=image.size, mask_image=mask)


            if not os.path.exists(args.saved_path):
                os.makedirs(args.saved_path, exist_ok=True)

            file_name = os.path.basename(img_f).split('.')[0]
            result_path = os.path.join(args.saved_path, file_name)
            view_images(image_list, result_path)

            print(f"Successfully saved to: {result_path}")

        except Exception as e:
            print(f"\n{'!'*20} ERROR {'!'*20}")
            print(f"Error processing image index {i}: {img_f}")
            print(f"Error message: {str(e)}")
            print("Traceback details:")
            traceback.print_exc()
            print(f"{'!'*47}\n")

            continue




import os
import sys

os.environ["GRADIO_TEMP_DIR"] = "./gradio_tmp"
if not os.path.exists("./gradio_tmp"):
    os.makedirs("./gradio_tmp")

import gradio as gr
import torch
import numpy as np
import cv2
from PIL import Image
import tempfile
import shutil
import traceback

from transformers import Sam2Model, Sam2Processor

from mmdit.flux_pipeline import FluxPipeline
from inversion.flow_direct_correction_inv_flux import Accurate_Inversion_FLUX
from inversion.inv_utils import fix_seed
from controller import attn_norm_ctrl_flux

device = "cuda" if torch.cuda.is_available() else "cpu"

loaded_pipe = None
sam_model = None
sam_processor = None

SAM2_MODEL_ID = "facebook/sam2.1-hiera-large"

colors = [(255, 0, 0), (0, 255, 0)]
markers = [1, 5]


def load_models(progress=gr.Progress()):
    global loaded_pipe, sam_model, sam_processor
    status = []

    if loaded_pipe is None:
        try:
            progress(0.1, desc="Loading FLUX Pipeline...")
            pipe = FluxPipeline.from_pretrained(
                "black-forest-labs/FLUX.1-dev",
                torch_dtype=torch.float16
            )
            pipe = pipe.to(device)
            pipe.transformer.eval()
            pipe.vae.eval()
            loaded_pipe = pipe
            status.append("✅ FLUX Loaded Successfully")
        except Exception as e:
            status.append(f"❌ FLUX Load Failed: {e}")
    else:
        status.append("✅ FLUX Ready")

    if sam_model is None or sam_processor is None:
        try:
            progress(0.5, desc="Loading SAM2 Model...")
            sam_dtype = torch.float16 if device == "cuda" else torch.float32
            try:
                sam_model = Sam2Model.from_pretrained(SAM2_MODEL_ID, dtype=sam_dtype).to(device)
            except TypeError:
                sam_model = Sam2Model.from_pretrained(SAM2_MODEL_ID, torch_dtype=sam_dtype).to(device)
            sam_model.eval()
            sam_processor = Sam2Processor.from_pretrained(SAM2_MODEL_ID)
            status.append("✅ SAM2 Loaded Successfully")
        except Exception as e:
            status.append(f"❌ SAM2 Load Failed: {e}")
    else:
        status.append("✅ SAM2 Ready")

    return "\n".join(status)


def resize_image(input_image, resolution):
    H, W, C = input_image.shape
    k = float(resolution) / min(H, W)
    H *= k
    W *= k
    H = int(np.round(H / 64.0)) * 64
    W = int(np.round(W / 64.0)) * 64
    img = cv2.resize(input_image, (W, H), interpolation=cv2.INTER_LANCZOS4 if k > 1 else cv2.INTER_AREA)
    return img


def store_img(img):
    if img is None:
        return None, None, None, []

    if min(img.shape[0], img.shape[1]) > 1024:
        img = resize_image(img, 1024)

    return img, img, None, []


def _move_sam_inputs_to_device(inputs):
    sam_dtype = next(sam_model.parameters()).dtype
    moved = {}
    for key, value in inputs.items():
        if not torch.is_tensor(value):
            moved[key] = value
        elif torch.is_floating_point(value):
            moved[key] = value.to(device=device, dtype=sam_dtype)
        else:
            moved[key] = value.to(device=device)
    return moved


def run_sam(img, sel_pix, dilation_amt):
    if sam_model is None or sam_processor is None:
        return img, None

    if len(sel_pix) == 0:
        return img, None

    points = []
    labels = []
    for p, l in sel_pix:
        points.append(p)
        labels.append(l)

    inputs = sam_processor(
        images=Image.fromarray(img).convert("RGB"),
        input_points=[[points]],
        input_labels=[[labels]],
        return_tensors="pt",
    )
    original_sizes = inputs["original_sizes"]
    inputs = _move_sam_inputs_to_device(inputs)

    with torch.no_grad():
        outputs = sam_model(**inputs, multimask_output=False)

    masks = sam_processor.post_process_masks(outputs.pred_masks.cpu(), original_sizes)[0]
    mask_tensor = masks[0, 0] if masks.ndim == 4 else masks[0]
    final_mask = mask_tensor.numpy() > 0

    # === Dilation ===
    if dilation_amt > 0:
        mask_uint8 = (final_mask.astype(np.uint8) * 255)

        kernel_size = int(dilation_amt)
        kernel = np.ones((kernel_size, kernel_size), np.uint8)

        mask_dilated = cv2.dilate(mask_uint8, kernel, iterations=1)

        final_mask = mask_dilated > 127

    mask_visual = np.zeros_like(img, dtype=np.uint8)
    color_mask = np.array([30, 144, 255])

    masked_img = img.copy()
    alpha = 0.6
    masked_img[final_mask] = (masked_img[final_mask] * (1 - alpha) + color_mask * alpha).astype(np.uint8)

    for point, label in sel_pix:
        cv2.drawMarker(masked_img, tuple(point), colors[label], markerType=markers[label], markerSize=15, thickness=3)

    return masked_img, final_mask


def get_point(img, sel_pix, point_type, dilation, evt: gr.SelectData):
    if sam_model is None or sam_processor is None:
        raise gr.Error("Please click [Load Models] first!")

    if point_type == 'foreground':
        sel_pix.append((evt.index, 1))
    else:
        sel_pix.append((evt.index, 0))

    masked_img, final_mask = run_sam(img, sel_pix, dilation)
    return masked_img, final_mask


def undo_points(orig_img, sel_pix, dilation):
    if len(sel_pix) > 0:
        sel_pix.pop()

    if len(sel_pix) == 0:
        return orig_img, None

    masked_img, final_mask = run_sam(orig_img, sel_pix, dilation)
    return masked_img, final_mask


def update_mask_only(orig_img, sel_pix, dilation):
    if len(sel_pix) == 0:
        return orig_img, None
    masked_img, final_mask = run_sam(orig_img, sel_pix, dilation)
    return masked_img, final_mask


def run_direct_edit(
        original_image_np,
        binary_mask,
        src_prompt, tar_prompt,
        num_steps, skip_steps,
        inv_cfg, recov_cfg,
        ly_ratio, attn_ratio,
        seed,
        invert_mask_flag,
        progress=gr.Progress()
):
    global loaded_pipe

    if loaded_pipe is None:
        raise gr.Error("Please load FLUX model first!")
    if original_image_np is None:
        raise gr.Error("Please upload an image first!")

    # Mask Processing
    if binary_mask is None:
        print("No mask detected, performing full image editing")
        mask_pil = None
    else:
        mask_arr = binary_mask.astype(np.uint8) * 255
        if invert_mask_flag:
            mask_arr = 255 - mask_arr
        mask_pil = Image.fromarray(mask_arr).convert("L")
        print("Mask generated")

    input_pil = Image.fromarray(original_image_np).convert("RGB")
    fix_seed(seed)

    temp_dir = tempfile.mkdtemp()
    try:
        src_path = os.path.join(temp_dir, "temp_src.png")
        input_pil.save(src_path)

        saved_path = "gradio_outputs"
        os.makedirs(saved_path, exist_ok=True)

        invf = Accurate_Inversion_FLUX(
            loaded_pipe, num_steps, device, inv_cfg, recov_cfg, skip_steps, saved_path
        )

        prompts = [src_prompt, tar_prompt]

        progress(0.2, desc="Executing Euler Flow Inversion...")

        attn_norm_ctrl_flux.register_attention_control_flux(loaded_pipe, None, None)
        all_latents, delta_list = invf.euler_flow_inversion(prompt=src_prompt, image=src_path)

        progress(0.6, desc="Executing Direct Inversion (Editing)...")

        ly_ratio_tuple = (0.0, ly_ratio)
        attn_ratio_tuple = (0.0, attn_ratio)

        controller_ada = attn_norm_ctrl_flux.FluxAdalayernorm_replace(
            prompts, num_steps, ly_ratio_tuple, loaded_pipe.tokenizer, loaded_pipe.tokenizer_2, device=device
        )
        controller_attn = attn_norm_ctrl_flux.FluxAttentionReplace(
            prompts, num_steps, attn_ratio_tuple
        )
        attn_norm_ctrl_flux.register_attention_control_flux(loaded_pipe, controller_attn, controller_ada)

        image_list = invf.direct_inversion(
            prompts,
            controller=controller_ada,
            all_latents=all_latents,
            delta_list=delta_list,
            original_size=input_pil.size,
            mask_image=mask_pil
        )
        return image_list

    except Exception as e:
        traceback.print_exc()
        raise gr.Error(f"Execution Error: {str(e)}")
    finally:
        shutil.rmtree(temp_dir)


with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🎨 FLUX DirectEdit")

    state_origin_img = gr.State(None)
    state_mask = gr.State(None)
    state_points = gr.State([])

    with gr.Row():
        with gr.Column(scale=3):
            model_status = gr.Textbox(label="Model Status", value="⚪ Not Loaded", interactive=False)
        with gr.Column(scale=1):
            load_btn = gr.Button("⬇️ Load Models (FLUX + SAM2)", variant="primary")

    gr.Markdown("---")

    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### 1. Segmentation & Dilation")
            input_image_ui = gr.Image(label="Click to Select Area", type="numpy", height=600, interactive=True)

            with gr.Row():
                point_type_radio = gr.Radio(['foreground', 'background'], label='Click Mode', value='foreground')
                undo_btn = gr.Button("↩️ Undo")

            dilation_slider = gr.Slider(
                minimum=0, maximum=30, value=5, step=1,
                label="Mask Dilation",
                info="Expands mask to cover edges; larger values expand more."
            )
            invert_chk = gr.Checkbox(label="Invert Mask (Edit Background)", value=False)

        with gr.Column(scale=1):
            gr.Markdown("### 2. Editing Parameters")

            src_prompt = gr.Textbox(label="Source Prompt", placeholder="e.g. a photo of a cat")
            tar_prompt = gr.Textbox(label="Target Prompt", placeholder="e.g. a photo of a dog")

            run_btn = gr.Button("🚀 Run Edit", variant="primary", size="lg")

            with gr.Accordion("⚙️ Advanced Settings", open=True):
                num_steps = gr.Slider(10, 50, value=30, step=1, label="Steps")
                skip_steps = gr.Slider(0, 20, value=0, step=1, label="Skip")
                inv_cfg = gr.Slider(1.0, 10.0, value=1.0, step=0.5, label="Inv CFG")
                recov_cfg = gr.Slider(1.0, 10.0, value=2.0, step=0.5, label="Rec CFG")
                seed = gr.Number(value=2024, precision=0, label="Seed")
                ly_ratio = gr.Slider(0.0, 1.0, value=0.0, step=0.05, label="ly_ratio")
                attn_ratio = gr.Slider(0.0, 1.0, value=0.05, step=0.05, label="attn_ratio")

        with gr.Column(scale=1):
            gr.Markdown("### 3. Results")
            gallery = gr.Gallery(label="Output", columns=1, height=800, object_fit="contain")


    load_btn.click(load_models, outputs=model_status)

    input_image_ui.upload(
        store_img,
        inputs=[input_image_ui],
        outputs=[input_image_ui, state_origin_img, state_mask, state_points]
    )

    input_image_ui.select(
        get_point,
        inputs=[state_origin_img, state_points, point_type_radio, dilation_slider],
        outputs=[input_image_ui, state_mask]
    )

    undo_btn.click(
        undo_points,
        inputs=[state_origin_img, state_points, dilation_slider],
        outputs=[input_image_ui, state_mask]
    )

    dilation_slider.change(
        update_mask_only,
        inputs=[state_origin_img, state_points, dilation_slider],
        outputs=[input_image_ui, state_mask]
    )

    run_btn.click(
        run_direct_edit,
        inputs=[
            state_origin_img, state_mask,
            src_prompt, tar_prompt,
            num_steps, skip_steps,
            inv_cfg, recov_cfg,
            ly_ratio, attn_ratio,
            seed, invert_chk
        ],
        outputs=[gallery]
    )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=8888, share=True)

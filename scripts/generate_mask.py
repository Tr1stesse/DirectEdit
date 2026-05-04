import os
import cv2
import json
import base64
import argparse
import numpy as np
import torch
from PIL import Image
from openai import OpenAI
from transformers import Sam2Model, Sam2Processor

# os.environ["DASHSCOPE_API_KEY"] = "sk-..."

SAM2_MODEL_ID = "facebook/sam2.1-hiera-large"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SUPPORTED_MASK_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}



def encode_image_to_base64(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')


def get_editing_params_from_mllm(client, image_path, source_prompt, target_prompt):

    base64_image = encode_image_to_base64(image_path)

    system_prompt = """You are an intelligent image editing assistant. 
        Analyze the source image, source prompt, and target prompt to determine the editing task and the region of interest.

        Task Definitions:
        1. Local: Editing a specific object or its attributes.
           Examples: Replacing a dog with a cat, removing a person, changing a shirt's color to red, altering facial expression.
        2. Background: Changing the environment or background details while preserving the main subject.
           Examples: Changing the setting from a street to a beach, changing the background color to white, moving the subject to a forest.
        3. Global: Holistic changes affecting the entire image atmosphere or style.
           Examples: Turning a photo into an oil painting, changing summer to winter, converting day to night, cyberpunk style transfer.
        4. Other: Structural changes, additions, or partial modifications in a specific non-object region.
           Examples: Adding a bird in the sky, adding glasses to a face, modifying a specific texture patch.

        Output a JSON object strictly with these fields:
        - "type": One of ["Local", "Background", "Global", "Other"].
        - "bbox": [x1, y1, x2, y2] representing the bounding box of the region of interest using normalized coordinates (0-1000). 

          CRITICAL BBOX RULES based on "type":
          - For "Local": Enclose the SPECIFIC OBJECT that needs to be edited, removed, or altered.
          - For "Background": Enclose the FOREGROUND OBJECT (the subject) that must be PROTECTED/PRESERVED. Do NOT enclose the background itself.
          - For "Other": Enclose the target region where the new content will be added or modified.
          - For "Global": Return [0, 0, 1000, 1000].
        """

    user_content = f"Source Prompt: {source_prompt}\nTarget Prompt: {target_prompt}\nPlease analyze the editing intent and output the JSON."

    try:
        completion = client.chat.completions.create(
            model="qwen3.6-plus",
            messages=[
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}},
                        {"type": "text", "text": user_content},
                    ],
                },
            ],
            temperature=0.1,
        )
        response_text = completion.choices[0].message.content

        if "```json" in response_text:
            response_text = response_text.split("```json")[1].split("```")[0].strip()
        elif "```" in response_text:
            response_text = response_text.replace("```", "").strip()

        result = json.loads(response_text)
        return result
    except Exception as e:
        print(f"\n[Warning] MLLM Error for {image_path}: {e}")
        return {"type": "Global", "bbox": [0, 0, 1000, 1000]}


def move_sam_inputs_to_device(inputs, model):
    model_dtype = next(model.parameters()).dtype
    moved = {}
    for key, value in inputs.items():
        if not torch.is_tensor(value):
            moved[key] = value
        elif torch.is_floating_point(value):
            moved[key] = value.to(device=DEVICE, dtype=model_dtype)
        else:
            moved[key] = value.to(device=DEVICE)
    return moved


def sam2_predict_from_box(image_rgb, input_box, model, processor):
    inputs = processor(
        images=Image.fromarray(image_rgb).convert("RGB"),
        input_boxes=[[input_box.tolist()]],
        return_tensors="pt",
    )
    original_sizes = inputs["original_sizes"]
    inputs = move_sam_inputs_to_device(inputs, model)

    with torch.no_grad():
        outputs = model(**inputs, multimask_output=False)

    masks = processor.post_process_masks(outputs.pred_masks.cpu(), original_sizes)[0]
    mask_tensor = masks[0, 0] if masks.ndim == 4 else masks[0]
    return mask_tensor.numpy() > 0


def dilate_mask(mask, kernel_size):
    if kernel_size <= 0:
        return mask

    kernel = np.ones((int(kernel_size), int(kernel_size)), np.uint8)
    return cv2.dilate(mask, kernel, iterations=1)


def generate_mask_logic(image_rgb, edit_type, bbox_norm, model, processor, dilation_kernel_size=8):
    h, w = image_rgb.shape[:2]

    x1 = int(bbox_norm[0] / 1000 * w)
    y1 = int(bbox_norm[1] / 1000 * h)
    x2 = int(bbox_norm[2] / 1000 * w)
    y2 = int(bbox_norm[3] / 1000 * h)

    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)

    input_box = np.array([x1, y1, x2, y2])

    # Case: Global
    if edit_type == "Global":
        return np.ones((h, w), dtype=np.uint8) * 255

    # Case: Other (bbox only)
    if edit_type == "Other":
        mask = np.zeros((h, w), dtype=np.uint8)
        mask[y1:y2, x1:x2] = 255
        return mask

    # Case: Local / Background (Use SAM2)
    sam_mask = sam2_predict_from_box(image_rgb, input_box, model, processor)

    if edit_type == "Local":
        final_mask = sam_mask.astype(np.uint8) * 255
        final_mask = dilate_mask(final_mask, dilation_kernel_size)
    elif edit_type == "Background":
        final_mask = (1 - sam_mask.astype(np.uint8)) * 255
        final_mask = dilate_mask(final_mask, dilation_kernel_size)
    else:
        final_mask = sam_mask.astype(np.uint8) * 255

    return final_mask


def load_sam2_model():
    print(f"Loading SAM2 ({SAM2_MODEL_ID})...")
    sam_dtype = torch.float16 if DEVICE == "cuda" else torch.float32
    try:
        sam_model = Sam2Model.from_pretrained(SAM2_MODEL_ID, dtype=sam_dtype).to(DEVICE)
    except TypeError:
        sam_model = Sam2Model.from_pretrained(SAM2_MODEL_ID, torch_dtype=sam_dtype).to(DEVICE)
    sam_model.eval()
    sam_processor = Sam2Processor.from_pretrained(SAM2_MODEL_ID)
    return sam_model, sam_processor


def resolve_output_mask_path(image_path, output_mask_path):
    image_stem = os.path.splitext(os.path.basename(image_path))[0]

    if not output_mask_path:
        return os.path.join(os.path.dirname(image_path), f"{image_stem}_mask.png")

    output_mask_path = os.path.expanduser(output_mask_path)
    _, ext = os.path.splitext(output_mask_path)
    looks_like_dir = (
        output_mask_path.endswith("/")
        or output_mask_path.endswith("\\")
        or os.path.isdir(output_mask_path)
        or ext == ""
    )

    if looks_like_dir:
        return os.path.join(output_mask_path, f"{image_stem}_mask.png")

    if ext.lower() not in SUPPORTED_MASK_EXTENSIONS:
        raise ValueError(
            f"Unsupported mask extension '{ext}'. "
            f"Use one of: {', '.join(sorted(SUPPORTED_MASK_EXTENSIONS))}"
        )

    return output_mask_path


def generate_mask_for_image(image_path, output_mask_path, source_prompt, target_prompt, dilation_kernel_size=8):
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")

    image_bgr = cv2.imread(image_path)
    if image_bgr is None:
        raise ValueError(f"Cannot read image: {image_path}")

    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    client = OpenAI(
        api_key=os.getenv("DASHSCOPE_API_KEY"),
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    )

    sam_model, sam_processor = load_sam2_model()

    mllm_res = get_editing_params_from_mllm(client, image_path, source_prompt, target_prompt)
    print(f"MLLM Result: {mllm_res}")

    edit_type = mllm_res.get("type", "Global")
    bbox = mllm_res.get("bbox", [0, 0, 1000, 1000])
    mask = generate_mask_logic(
        image_rgb,
        edit_type,
        bbox,
        sam_model,
        sam_processor,
        dilation_kernel_size=dilation_kernel_size,
    )

    output_mask_path = resolve_output_mask_path(image_path, output_mask_path)
    output_dir = os.path.dirname(output_mask_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    if not cv2.imwrite(output_mask_path, mask):
        raise RuntimeError(f"Failed to write mask: {output_mask_path}")

    return output_mask_path, mllm_res


def main():
    parser = argparse.ArgumentParser(description="Generate one mask for a single image with MLLM + SAM2.")
    parser.add_argument("--image_path", type=str, required=True, help="Path to the input image")
    parser.add_argument(
        "--output_mask_path",
        type=str,
        default="masks",
        help="Mask file path or output directory. Directories save as <image_name>_mask.png.",
    )
    parser.add_argument("--src_prompt", type=str, required=True, help="Source prompt describing the input image")
    parser.add_argument("--tar_prompt", type=str, required=True, help="Target prompt describing the desired edit")
    parser.add_argument(
        "--dilation_kernel_size",
        type=int,
        default=8,
        help="Dilation kernel size for Local/Background masks. Set 0 to disable.",
    )
    args = parser.parse_args()

    output_mask_path, _ = generate_mask_for_image(
        args.image_path,
        args.output_mask_path,
        args.src_prompt,
        args.tar_prompt,
        dilation_kernel_size=args.dilation_kernel_size,
    )
    print(f"Mask saved to: {output_mask_path}")


if __name__ == "__main__":
    main()

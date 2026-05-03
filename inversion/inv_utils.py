import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg
import numpy as np
import PIL.Image as Image
import torch
import json
import os
import random


'''
Inversion utility
'''

def fix_seed(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.enabled=False
    torch.backends.cudnn.deterministic=True


edit_category_list = ["0","1","2","3","4","5","6","7","8","9"]

def view_images(images, name, num_rows=1, offset_ratio=0.02):
    if not isinstance(images, list):
        if hasattr(images, 'ndim') and images.ndim == 4:
            images = [images[i] for i in range(images.shape[0])]
        else:
            images = [images]

    processed_images = []
    for img in images:
        if isinstance(img, Image.Image):
            img = np.array(img)
        processed_images.append(img)
    images = processed_images

    num_empty = 0
    if len(images) % num_rows != 0:
        num_cols = (len(images) + num_rows - 1) // num_rows
        num_empty = num_cols * num_rows - len(images)

    empty_images = np.ones(images[0].shape, dtype=np.uint8) * 255

    images = [image.astype(np.uint8) for image in images] + [empty_images] * num_empty
    num_items = len(images)

    h, w, c = images[0].shape
    offset = int(h * offset_ratio)
    num_cols = num_items // num_rows

    image_ = np.ones((h * num_rows + offset * (num_rows - 1),
                      w * num_cols + offset * (num_cols - 1), 3), dtype=np.uint8) * 255

    for i in range(num_rows):
        for j in range(num_cols):
            y_start = i * (h + offset)
            y_end = y_start + h
            x_start = j * (w + offset)
            x_end = x_start + w
            image_[y_start:y_end, x_start:x_end] = images[i * num_cols + j]

    pil_img = Image.fromarray(image_)
    pil_img.save("{}_edited.png".format(name))


def mask_decode(encoded_mask, image_shape=[512, 512]):
    length = image_shape[0] * image_shape[1]
    mask_array = np.zeros((length,), dtype=np.uint8)

    for i in range(0, len(encoded_mask), 2):
        start_idx = encoded_mask[i]
        run_len = encoded_mask[i + 1]

        splice_len = min(run_len, length - start_idx)

        mask_array[start_idx: start_idx + splice_len] = 1

    mask_array = mask_array.reshape(image_shape[0], image_shape[1])

    mask_array[0, :] = 1
    mask_array[-1, :] = 1
    mask_array[:, 0] = 1
    mask_array[:, -1] = 1

    return mask_array


def load_PIE_images(data_path, edit_category_list):
    '''
    data_path: json file path
    edit_category_list: test edit categories
    '''

    ori_prp_list, edi_prp_list, img_list, edi_ins_list, bld_list, mask_list = [], [], [], [], [], []

    with open(f"{data_path}/mapping_file.json", "r") as f:
        editing_instruction = json.load(f)

    print("#### editing", len(editing_instruction))
    for key, item in editing_instruction.items():

        if item["editing_type_id"] not in edit_category_list:
            continue

        original_prompt = item["original_prompt"].replace("[", "").replace("]", "")
        editing_prompt = item["editing_prompt"].replace("[", "").replace("]", "")
        image_path = os.path.join(f"{data_path}/annotation_images", item["image_path"])
        editing_instruction = item["editing_instruction"]
        blended_word = item["blended_word"].split(" ") if item["blended_word"] != "" else []

        mask_np = mask_decode(item["mask"])
        mask_np = (mask_np * 255).astype(np.uint8)
        mask = Image.fromarray(mask_np, mode='L')

        ori_prp_list.append(original_prompt)
        edi_prp_list.append(editing_prompt)
        img_list.append(image_path)
        edi_ins_list.append(editing_instruction)
        bld_list.append(blended_word)
        mask_list.append(mask)

    return ori_prp_list, edi_prp_list, img_list, edi_ins_list, bld_list, mask_list



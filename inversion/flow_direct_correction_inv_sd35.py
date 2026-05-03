import torch
from mmdit.sd35_pipeline import StableDiffusion3Pipeline, retrieve_timesteps
from diffusers.image_processor import VaeImageProcessor
import numpy as np
from tqdm import tqdm
from PIL import Image
from torchvision import transforms
import torch.nn.functional as F
import os
import json
import cv2


'''
Rectified flow matching inversion with DDPM edit friendly inversion
'''
class Accurate_Inversion_SD3:
    def __init__(self, model, steps, device, inv_cfg, recov_cfg, skip_steps, saved_path):
        self.model = model
        self.num_steps = steps
        self.device = device
        self.inv_cfg = inv_cfg
        self.recov_cfg = recov_cfg
        self.skip_steps = skip_steps
        self.saved_path = saved_path

    def get_embeddings(self, prompt):
        '''
        get the text embeddings for the model
        '''
        (
            prompt_embeds,
            negative_prompt_embeds,
            pooled_prompt_embeds,
            negative_pooled_prompt_embeds,
        ) = self.model.encode_prompt(
            prompt=prompt,
            prompt_2=None,
            prompt_3=None,
            negative_prompt=None,
            negative_prompt_2=None,
            negative_prompt_3=None,
            do_classifier_free_guidance=True,
            prompt_embeds=None,
            negative_prompt_embeds=None,
            pooled_prompt_embeds=None,
            negative_pooled_prompt_embeds=None,
            device=self.device,
            clip_skip=None,
            num_images_per_prompt=1,
            max_sequence_length=256,
        )
        
        return prompt_embeds, negative_prompt_embeds, pooled_prompt_embeds, negative_pooled_prompt_embeds,


    def encode_latent(self, image_path, device="cuda", dtype=torch.float16):
        image = Image.open(image_path).convert("RGB")
        original_width, original_height = image.size

        target_w = original_width if original_width % 16 == 0 else (original_width // 16 + 1) * 16
        target_h = original_height if original_height % 16 == 0 else (original_height // 16 + 1) * 16

        if target_w != original_width or target_h != original_height:
            padded_image = Image.new("RGB", (target_w, target_h), (255, 255, 255))
            padded_image.paste(image, (0, 0))
            image = padded_image

        image_src = self.model.image_processor.preprocess(image)
        image_src = image_src.to(device).to(dtype)

        with torch.autocast(device), torch.inference_mode():
            dist = self.model.vae.encode(image_src).latent_dist
            x0_src_denorm = dist.mode()

        shift_factor = getattr(self.model.vae.config, "shift_factor", 0.0)
        scaling_factor = getattr(self.model.vae.config, "scaling_factor", 0.18215)

        x0_src = (x0_src_denorm - shift_factor) * scaling_factor

        return x0_src


    def latent2image(self, latent, original_size=None):
        shift_factor = getattr(self.model.vae.config, "shift_factor", 0.0)
        scaling_factor = getattr(self.model.vae.config, "scaling_factor", 0.18215)

        x0_denorm = (latent / scaling_factor) + shift_factor

        with torch.autocast("cuda"), torch.inference_mode():
            image_tensor = self.model.vae.decode(x0_denorm, return_dict=False)[0]

        image_pil = self.model.image_processor.postprocess(image_tensor)[0]

        if original_size is not None:
            target_w, target_h = original_size
            image_pil = image_pil.crop((0, 0, target_w, target_h))

        return image_pil


    def prepare_mask(self, mask_image, latent_shape, device, dtype, dilation_pixels=3):
        lat_h, lat_w = latent_shape[-2], latent_shape[-1]
        target_pixel_h = lat_h * 8
        target_pixel_w = lat_w * 8

        mask = mask_image.convert("L")

        padded_mask = Image.new("L", (target_pixel_w, target_pixel_h), 0)

        padded_mask.paste(mask, (0, 0))

        if dilation_pixels > 0:
            mask_np = np.array(padded_mask)
            kernel_size = 2 * int(dilation_pixels) + 1
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
            mask_np = cv2.dilate(mask_np, kernel, iterations=1)
            padded_mask = Image.fromarray(mask_np)

        mask_tensor = transforms.ToTensor()(padded_mask)
        mask_tensor = mask_tensor.to(device).to(dtype)

        mask_latent = F.interpolate(
            mask_tensor.unsqueeze(0),
            size=(lat_h, lat_w),
            mode='bilinear',
            align_corners=False
        )

        return mask_latent


    @torch.no_grad()
    def euler_flow_inversion(self, prompt, image):
        '''
        invert rectified flow with 1st order euler method without correction
        xt = x_{t_1} - (sigma_t) * dx/dt
        '''
        # reverse the time step
        prompt_embeds, negative_prompt_embeds, pooled_prompt_embeds, negative_pooled_prompt_embeds = self.get_embeddings(
            prompt)

        if self.inv_cfg > 0:
            self.prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)
            self.pooled_prompt_embeds = torch.cat([negative_pooled_prompt_embeds, pooled_prompt_embeds], dim=0)

        # prepare timesteps
        timesteps, num_inference_steps = retrieve_timesteps(self.model.scheduler, self.num_steps, self.device, None)
        num_warmup_steps = max(len(timesteps) - num_inference_steps * self.model.scheduler.order, 0)
        self.model._num_timesteps = len(timesteps)

        # encode the image latent
        latent_prev = self.encode_latent(image)

        all_latents = [latent_prev.clone().detach()]
        delta_list = []

        # compute the predicted noise
        for i in tqdm(range(self.num_steps)):
            t = timesteps[len(timesteps) - i - 1]

            ### get estimated z_t-1
            latent, _ = self.flow_step(latent_prev, t, i, self.prompt_embeds, self.pooled_prompt_embeds, is_inverse=True)
            all_latents.append(latent.clone().detach())
            delta_list.append(latent_prev.detach() - latent.detach())
            latent_prev = latent

        return all_latents, delta_list


    @torch.no_grad()
    def flow_step(self, latent, t, cur_step, prompt_embeds, pooled_prompt_embeds, is_inverse=True):
        '''
        standard euler step for rectified flow inversion
        '''
        approximated_z_tp1 = latent.clone()
        
        with torch.no_grad():
            prompt_embeds_in = prompt_embeds
            pooled_prompt_embeds_in = pooled_prompt_embeds

        if self.inv_cfg > 0:
            latent_model_input = torch.cat([approximated_z_tp1] * 2)
        else:
            latent_model_input = approximated_z_tp1

        timestep = t.expand(latent_model_input.shape[0])

        noise_pred = self.model.transformer(
            hidden_states=latent_model_input,
            timestep=timestep,
            encoder_hidden_states=prompt_embeds_in,
            pooled_projections=pooled_prompt_embeds_in,
            joint_attention_kwargs=None,
            return_dict=False,
        )[0]

        if self.inv_cfg > 0:
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + self.inv_cfg * (noise_pred_text - noise_pred_uncond)

        # use Euler to invert current latent to the next latent
        sample = latent.to(torch.float32)
        if is_inverse:
            sigma = self.model.scheduler.sigmas[self.num_steps - cur_step]
            sigma_next = self.model.scheduler.sigmas[self.num_steps - cur_step - 1]
        else:
            sigma = self.model.scheduler.sigmas[cur_step]
            sigma_next = self.model.scheduler.sigmas[cur_step + 1]

        approximated_z_tp1 = sample + (sigma_next - sigma) * noise_pred
        approximated_z_tp1 = approximated_z_tp1.to(noise_pred.dtype)

        return approximated_z_tp1, noise_pred



    @torch.no_grad()
    def direct_inversion(self, prompts, controller, all_latents, delta_list, original_size=None, mask_image=None):
        '''
        direct inversion with euler method, and then edit with corrected latents
        '''
        latent_cur = torch.cat([all_latents[-1].clone().detach()] * 2, dim=0).to(self.device)
        latent_mask = None
        if mask_image is not None:
            latent_mask = self.prepare_mask(mask_image, latent_cur.shape, self.device, latent_cur.dtype)
            print("Mask enabled. Latent mask shape:", latent_mask.shape)

        # reverse the time step
        src_prompt_embeds, src_negative_prompt_embeds, src_pooled_prompt_embeds, src_negative_pooled_prompt_embeds = self.get_embeddings(
            prompts[0])
        tar_prompt_embeds, tar_negative_prompt_embeds, tar_pooled_prompt_embeds, tar_negative_pooled_prompt_embeds = self.get_embeddings(
            prompts[1])

        prompt_embeds = torch.cat([src_prompt_embeds, tar_prompt_embeds], dim=0)
        negative_prompt_embeds = torch.cat([src_negative_prompt_embeds, tar_negative_prompt_embeds], dim=0)
        pooled_prompt_embeds = torch.cat([src_pooled_prompt_embeds, tar_pooled_prompt_embeds], dim=0)
        negative_pooled_prompt_embeds = torch.cat(
            [src_negative_pooled_prompt_embeds, tar_negative_pooled_prompt_embeds], dim=0)

        if self.recov_cfg > 0:
            prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)
            pooled_prompt_embeds = torch.cat([negative_pooled_prompt_embeds, pooled_prompt_embeds], dim=0)
            self.prompt_embeds = torch.cat([tar_negative_prompt_embeds, tar_prompt_embeds], dim=0)
            self.pooled_prompt_embeds = torch.cat([tar_negative_pooled_prompt_embeds, tar_pooled_prompt_embeds], dim=0)

        # prepare timesteps
        timesteps, num_inference_steps = retrieve_timesteps(self.model.scheduler, self.num_steps, self.device, None)
        num_warmup_steps = max(len(timesteps) - num_inference_steps * self.model.scheduler.order, 0)
        self.model._num_timesteps = len(timesteps)

        # compute the predicted noise
        for i in tqdm(range(self.num_steps)):
            if i < self.skip_steps:
                if controller is not None:
                    controller.cur_step += 1
                continue
            t = timesteps[i]

            delta_z_src = delta_list[-1 - i].to(latent_cur.dtype).to(self.device)
            z_src = latent_cur[0].unsqueeze(0) + delta_z_src
            z_tar = latent_cur[1].unsqueeze(0) + delta_z_src

            if self.recov_cfg > 0:
                latent_model_input = torch.cat([z_src, z_tar] * 2, dim=0)
            else:
                latent_model_input = torch.cat([z_src, z_tar], dim=0)

            timestep = t.expand(latent_model_input.shape[0])

            noise_pred = self.model.transformer(
                hidden_states=latent_model_input,
                timestep=timestep,
                encoder_hidden_states=prompt_embeds,
                pooled_projections=pooled_prompt_embeds,
                joint_attention_kwargs=None,
                return_dict=False,
            )[0]


            if self.recov_cfg > 0:
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)

                noise_pred_uncond_src, noise_pred_uncond_tar = noise_pred_uncond.chunk(2)
                noise_pred_text_src, noise_pred_text_tar = noise_pred_text.chunk(2)

                src_guidance_scale = self.inv_cfg
                noise_pred_src = noise_pred_uncond_src + src_guidance_scale * (
                            noise_pred_text_src - noise_pred_uncond_src)

                tar_guidance_scale = self.recov_cfg
                noise_pred_tar = noise_pred_uncond_tar + tar_guidance_scale * (
                            noise_pred_text_tar - noise_pred_uncond_tar)

                noise_pred = torch.cat([noise_pred_src, noise_pred_tar], dim=0)

            # inversion with euler method
            sample = latent_cur.to(torch.float32)
            sigma = self.model.scheduler.sigmas[i]
            sigma_next = self.model.scheduler.sigmas[i + 1]

            # prev_sample_src = sample[0:1] + (sigma_next - sigma) * noise_pred[0:1]
            gt_source_latent = all_latents[-(i + 2)].to(self.device)
            # mse = torch.mean((prev_sample_src - gt_source_latent) ** 2)
            # print(f"Step {i}: Source MSE with GT latent: {mse.item():.6f}")
            
            # Reduce errors caused by minor error sources
            prev_sample_src = gt_source_latent

            prev_sample_tar = sample[1:2] + (sigma_next - sigma) * noise_pred[1:2]

            if latent_mask is not None:
                prev_sample_tar = prev_sample_tar * latent_mask + prev_sample_src * (1 - latent_mask)

            prev_sample = torch.cat([prev_sample_src, prev_sample_tar], dim=0)
            latent_cur = prev_sample.to(noise_pred.dtype)


        rec_img = self.latent2image(latent_cur[0].unsqueeze(0), original_size)
        edited_img = self.latent2image(latent_cur[1].unsqueeze(0), original_size)
        image_list = [rec_img, edited_img]

        self.model.maybe_free_model_hooks()

        return image_list



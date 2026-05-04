import torch
import numpy as np
from tqdm import tqdm
from PIL import Image
import torch.nn.functional as F
from diffusers.pipelines.flux.pipeline_flux import calculate_shift, retrieve_timesteps


class Accurate_Inversion_FLUX:
    def __init__(self, model, steps, device, inv_cfg, recov_cfg, skip_steps, saved_path):
        self.model = model
        self.num_steps = steps
        self.device = device
        self.inv_cfg = inv_cfg
        self.recov_cfg = recov_cfg
        self.skip_steps = skip_steps
        self.saved_path = saved_path

        # Flux VAE scale/shift factors
        self.vae_scale_factor = 2 ** (len(self.model.vae.config.block_out_channels) - 1)
        self.flux_vae_scale_factor = self.vae_scale_factor

    def get_embeddings(self, prompt):
        '''
        Get text embeddings and text_ids for Flux
        '''
        (
            prompt_embeds,
            pooled_prompt_embeds,
            text_ids,
        ) = self.model.encode_prompt(
            prompt=prompt,
            prompt_2=None,
            device=self.device,
            num_images_per_prompt=1,
            max_sequence_length=512,
        )

        return prompt_embeds, pooled_prompt_embeds, text_ids

    def encode_latent(self, image_path, device="cuda", dtype=torch.float16):
        # 1. Image Loading & Preprocessing
        image = Image.open(image_path).convert("RGB")
        w, h = image.size

        target_w = (w + 15) // 16 * 16
        target_h = (h + 15) // 16 * 16

        if target_w != w or target_h != h:
            padded_image = Image.new("RGB", (target_w, target_h), (255, 255, 255))
            padded_image.paste(image, (0, 0))
            image = padded_image

        self.width = target_w
        self.height = target_h

        image_src = self.model.image_processor.preprocess(image)
        image_src = image_src.to(device).to(dtype)

        # 2. VAE Encode
        with torch.no_grad():
            dist = self.model.vae.encode(image_src).latent_dist
            x0_src_denorm = dist.mode()

        # 3. Shift & Scale 
        shift_factor = self.model.vae.config.shift_factor
        scaling_factor = self.model.vae.config.scaling_factor
        x0_src = (x0_src_denorm - shift_factor) * scaling_factor

        latents = self.model._pack_latents(
            x0_src,
            batch_size=x0_src.shape[0],
            num_channels_latents=x0_src.shape[1],
            height=target_h // self.flux_vae_scale_factor,
            width=target_w // self.flux_vae_scale_factor
        )

        return latents

    def latent2image(self, packed_latents, original_size=None):
        # 1. Unpack Latents
        # packed: [B, Seq, Dim] -> [B, C, H, W]
        latents = self.model._unpack_latents(
            packed_latents,
            height=self.height,
            width=self.width,
            vae_scale_factor=self.flux_vae_scale_factor
        )

        # 2. Unscale & Unshift
        shift_factor = self.model.vae.config.shift_factor
        scaling_factor = self.model.vae.config.scaling_factor
        latents = (latents / scaling_factor) + shift_factor

        # 3. VAE Decode
        with torch.no_grad():
            image_tensor = self.model.vae.decode(latents, return_dict=False)[0]

        # 4. Postprocess
        image_pil = self.model.image_processor.postprocess(image_tensor)[0]

        if original_size is not None:
            w, h = original_size
            image_pil = image_pil.crop((0, 0, w, h))

        return image_pil


    def prepare_mask(self, mask_image, device, dtype):
        if self.width is None or self.height is None:
            raise ValueError("Must call encode_latent before prepare_mask to determine padded dimensions.")

        target_h = self.height // 16 
        target_w = self.width // 16  

        mask = mask_image.convert("L")

        padded_mask = Image.new("L", (self.width, self.height), 0)
        padded_mask.paste(mask, (0, 0))

        mask_tensor = torch.from_numpy(np.array(padded_mask)).float() / 255.0
        mask_tensor = mask_tensor.unsqueeze(0).unsqueeze(0)  # [1, 1, H_px, W_px]

        mask_latent = F.interpolate(
            mask_tensor,
            size=(target_h, target_w),
            mode='bilinear',
            align_corners=False
        )

        mask_latent = mask_latent.flatten(2).transpose(1, 2)

        return mask_latent.to(device).to(dtype)


    def _prepare_ids(self, batch_size):
        latent_height = self.height // self.flux_vae_scale_factor
        latent_width = self.width // self.flux_vae_scale_factor

        img_ids = self.model._prepare_latent_image_ids(
            batch_size,
            latent_height // 2,
            latent_width // 2,
            self.device,
            torch.bfloat16
        )
        return img_ids


    @torch.no_grad()
    def euler_flow_inversion(self, prompt, image):
        prompt_embeds, pooled_prompt_embeds, text_ids = self.get_embeddings(prompt)

        latent_prev = self.encode_latent(image, device=self.device, dtype=prompt_embeds.dtype)

        img_ids = self._prepare_ids(batch_size=1)

        num_channels_latents = self.model.transformer.config.in_channels // 4
        latent_seq_len = (self.height // 16) * (self.width // 16)

        mu = calculate_shift(
            image_seq_len=latent_seq_len,
            base_seq_len=self.model.scheduler.config.base_image_seq_len,
            max_seq_len=self.model.scheduler.config.max_image_seq_len,
            base_shift=self.model.scheduler.config.base_shift,
            max_shift=self.model.scheduler.config.max_shift,
        )

        timesteps, num_inference_steps = retrieve_timesteps(
            self.model.scheduler,
            self.num_steps,
            self.device,
            mu=mu
        )
        timesteps = torch.cat([timesteps, torch.tensor([0], device=timesteps.device)])

        all_latents = [latent_prev.clone().detach()]
        delta_list = []

        guidance_val = self.inv_cfg if self.inv_cfg > 1 else 1.0

        for i in tqdm(range(self.num_steps)):
            inv_idx = self.num_steps - i
            t = timesteps[inv_idx]
            t_curr = t
            t_next = timesteps[inv_idx - 1]

            t_val = t_next / 1000.0

            guidance_tensor = torch.full([1], guidance_val, device=self.device, dtype=latent_prev.dtype)

            noise_pred = self.model.transformer(
                hidden_states=latent_prev,
                timestep=t_val.unsqueeze(0),
                guidance=guidance_tensor,
                encoder_hidden_states=prompt_embeds,
                pooled_projections=pooled_prompt_embeds,
                txt_ids=text_ids,
                img_ids=img_ids,
                return_dict=False,
            )[0]

            dt = (t_next - t_curr) / 1000.0
            latent_next = latent_prev + dt * noise_pred

            all_latents.append(latent_next.clone().detach())
            delta_list.append(latent_prev.detach() - latent_next.detach())
            latent_prev = latent_next

        return all_latents, delta_list


    @torch.no_grad()
    def direct_inversion(self, prompts, controller, all_latents, delta_list, original_size=None, mask_image=None):
        '''
        Direct inversion / Editing for Flux
        '''
        latent_cur = torch.cat([all_latents[-1].clone().detach()] * 2, dim=0).to(self.device)

        latent_mask = None
        if mask_image is not None:
            latent_mask = self.prepare_mask(mask_image, self.device, latent_cur.dtype)
            print("Mask enabled. Latent mask shape:", latent_mask.shape)

        src_prompt_embeds, src_pooled, src_ids = self.get_embeddings(prompts[0])
        tar_prompt_embeds, tar_pooled, tar_ids = self.get_embeddings(prompts[1])

        prompt_embeds = torch.cat([src_prompt_embeds, tar_prompt_embeds], dim=0)
        pooled_prompt_embeds = torch.cat([src_pooled, tar_pooled], dim=0)

        text_ids = src_ids

        img_ids = self._prepare_ids(batch_size=2)

        latent_seq_len = (self.height // 16) * (self.width // 16)
        mu = calculate_shift(
            image_seq_len=latent_seq_len,
            base_seq_len=self.model.scheduler.config.base_image_seq_len,
            max_seq_len=self.model.scheduler.config.max_image_seq_len,
            base_shift=self.model.scheduler.config.base_shift,
            max_shift=self.model.scheduler.config.max_shift,
        )
        timesteps, num_inference_steps = retrieve_timesteps(self.model.scheduler, self.num_steps, self.device, mu=mu)
        timesteps = torch.cat([timesteps, torch.tensor([0], device=timesteps.device)])

        guidance_vec = torch.tensor([self.inv_cfg, self.recov_cfg], device=self.device, dtype=latent_cur.dtype)

        for i in tqdm(range(self.num_steps)):
            if i < self.skip_steps:
                if controller is not None:
                    controller.cur_step += 1
                continue

            t = timesteps[i]
            t_val = t / 1000.0

            # Direct Alignment using stored deltas
            delta_z_src = delta_list[-1 - i].to(latent_cur.dtype).to(self.device)
            z_src = latent_cur[0].unsqueeze(0) + delta_z_src
            z_tar = latent_cur[1].unsqueeze(0) + delta_z_src

            latent_model_input = torch.cat([z_src, z_tar], dim=0)

            timestep = t_val.unsqueeze(0).expand(latent_model_input.shape[0])

            noise_pred = self.model.transformer(
                hidden_states=latent_model_input,
                timestep=timestep,
                guidance=guidance_vec,
                encoder_hidden_states=prompt_embeds,
                pooled_projections=pooled_prompt_embeds,
                txt_ids=text_ids,
                img_ids=img_ids,
                return_dict=False,
            )[0]

            t_next = timesteps[i + 1]

            dt = (t_next - t) / 1000.0

            # prev_sample_src = latent_cur[0:1] + dt * noise_pred[0:1]
            gt_source_latent = all_latents[-(i + 2)].to(self.device)
            # mse = torch.mean((prev_sample_src - gt_source_latent) ** 2)
            # print(f"Step {i}: Source MSE with GT latent: {mse.item():.8f}")

            # Reduce errors caused by minor error sources
            prev_sample_src = gt_source_latent

            prev_sample_tar = latent_cur[1:2] + dt * noise_pred[1:2]

            if latent_mask is not None:
                prev_sample_tar = prev_sample_tar * latent_mask + prev_sample_src * (1 - latent_mask)

            latent_cur = torch.cat([prev_sample_src, prev_sample_tar], dim=0)


        rec_img = self.latent2image(latent_cur[0].unsqueeze(0), original_size)
        edited_img = self.latent2image(latent_cur[1].unsqueeze(0), original_size)
        image_list = [rec_img, edited_img]

        self.model.maybe_free_model_hooks()
        return image_list

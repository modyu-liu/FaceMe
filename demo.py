import os
import argparse
import torch
import numpy as np
import cv2
from tqdm import tqdm
from PIL import Image
from diffusers import (
    AutoencoderKL,
    ControlNetModel,
    DDPMScheduler,
    StableDiffusionXLControlNetPipeline,
)
from diffusers import UNet2DConditionModel as OriginalUNet2DConditionModel, StableDiffusionXLControlNetPipeline
from transformers import AutoTokenizer, PretrainedConfig, CLIPImageProcessor
from huggingface_hub import hf_hub_download
from arch.idencoder import PhotoMakerIDEncoder, Mix
from utils.wavelet_color_fix import wavelet_reconstruction
from utils.insightface_package import FaceAnalysis2, analyze_faces
from typing import Any, Callable, Dict, List, Optional, Union, Tuple
from utils.load_photomaker import load_photomaker
import gradio as gr


def import_model_class_from_model_name_or_path(pretrained_model_name_or_path: str, revision: str = None, subfolder: str = "text_encoder"):
    text_encoder_config = PretrainedConfig.from_pretrained(
        pretrained_model_name_or_path,
        subfolder=subfolder,
        revision=revision,
    )
    model_class = text_encoder_config.architectures[0]
    if model_class == "CLIPTextModel":
        from transformers import CLIPTextModel
        return CLIPTextModel
    elif model_class == "CLIPTextModelWithProjection":
        from transformers import CLIPTextModelWithProjection
        return CLIPTextModelWithProjection
    else:
        raise ValueError(f"{model_class} is not supported.")


def encode_prompt(text_encoders, text_input_ids_list=None):
    prompt_embeds_list = []
    for i, text_encoder in enumerate(text_encoders):
        prompt_embeds = text_encoder(
            text_input_ids_list[i].to(text_encoder.device), output_hidden_states=True, return_dict=False
        )
        pooled_prompt_embeds = prompt_embeds[0]
        prompt_embeds = prompt_embeds[-1][-2]
        bs_embed, seq_len, _ = prompt_embeds.shape
        prompt_embeds = prompt_embeds.view(bs_embed, seq_len, -1)
        prompt_embeds_list.append(prompt_embeds)
    prompt_embeds = torch.concat(prompt_embeds_list, dim=-1)
    pooled_prompt_embeds = pooled_prompt_embeds.view(bs_embed, -1)
    return prompt_embeds, pooled_prompt_embeds

def token_prompt(text_tokenizer, prompt):
    text_tokens = []
    for i, tokenizer in enumerate(text_tokenizer):
        tokens = tokenizer(prompt, padding="max_length", max_length=tokenizer.model_max_length,
                                   truncation=True, return_tensors="pt").input_ids
        text_tokens.append(tokens)
    
    return text_tokens

def prepare_text_encoder(args):

    tokenizer_one = AutoTokenizer.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="tokenizer",
        use_fast=False,
    )
    tokenizer_two = AutoTokenizer.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="tokenizer_2",
        use_fast=False,
    )
    text_encoder_cls_one = import_model_class_from_model_name_or_path(args.pretrained_model_name_or_path)
    text_encoder_cls_two = import_model_class_from_model_name_or_path(args.pretrained_model_name_or_path, subfolder="text_encoder_2")

    text_encoder_one = text_encoder_cls_one.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="text_encoder",
    )
    text_encoder_two = text_encoder_cls_two.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="text_encoder_2",
    )
    text_tokenizer = [tokenizer_one, tokenizer_two]
    text_encoder = [text_encoder_one, text_encoder_two]

    
    tag_token = tokenizer_one.encode(args.key_word)[1]

    return text_tokenizer, text_encoder, tag_token

def prepare_text_emb(text_tokenizer, text_encoders, prompt, tag_token, return_index=False):

    tokens = token_prompt(text_tokenizer, prompt)
    if return_index:
        index = torch.where(tokens[0] == tag_token)[1]
    else :
        index = None

    embeds, pooled_embeds = encode_prompt(text_encoders, tokens)

    return index, embeds, pooled_embeds


def prepare(args):
    text_tokenizer, text_encoder, tag_token = prepare_text_encoder(args)

    noise_scheduler = DDPMScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler")
    vae = AutoencoderKL.from_pretrained(args.pretrained_model_name_or_path, subfolder="vae", revision=None)
    unet = OriginalUNet2DConditionModel.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="unet",
    )
    id_encoder_clip = PhotoMakerIDEncoder()
    photomaker_path = hf_hub_download(repo_id="TencentARC/PhotoMaker", filename="photomaker-v1.bin", repo_type="model")
    id_encoder_clip, unet = load_photomaker(photomaker_path, clip_id_encoder=id_encoder_clip, unet=unet)
    controlnet = ControlNetModel.from_pretrained(args.controlnet_model_name_or_path)
    mix = Mix()
    mix.from_pretrained(args.mix_path)
    
    app = FaceAnalysis2(name='antelopev2', root='./', providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
    app.prepare(ctx_id=0, det_size=(512, 512))

    vae.requires_grad_(False)
    unet.requires_grad_(False)
    mix.requires_grad_(False)
    id_encoder_clip.requires_grad_(False)
    controlnet.requires_grad_(False)

    weight_dtype = torch.float16
    device = "cuda"
    vae.to(device, dtype=torch.float16)
    unet.to(device, dtype=weight_dtype)
    mix.to(device, dtype=weight_dtype)
    id_encoder_clip.to(device, dtype=weight_dtype)
    controlnet.to(device, dtype=weight_dtype)

    pipeline = StableDiffusionXLControlNetPipeline.from_pretrained(
        args.pretrained_model_name_or_path,
        vae=vae,
        controlnet=controlnet,
        unet=unet,
        torch_dtype=weight_dtype,
    )
    pipeline.scheduler = noise_scheduler
    pipeline = pipeline.to(device=device)
    return pipeline, id_encoder_clip, app, vae, noise_scheduler, mix, text_tokenizer, text_encoder, tag_token


def main(args):

    pipeline, id_encoder_clip, app, vae, noise_scheduler, mix, text_tokenizer, text_encoder, tag_token = prepare(args)
        
    @torch.no_grad()
    def process(
        control_img: Image.Image,
        pos_prompt: str, 
        neg_prompt: str,
        num_samples: int,
        ref_image: List[Image.Image],  
        strength: float,
        cfg_scale: float,
        steps: int,
        use_color_fix: bool,
        seed: int,
    ) -> List[np.ndarray]:
        
        device = "cuda"
        
        mask_index, pos_prompt_embeds, pos_pooled_prompt_embeds = prepare_text_emb(text_tokenizer, text_encoder, pos_prompt, tag_token, return_index=True)
        _, neg_prompt_embeds, neg_pooled_prompt_embeds = prepare_text_emb(text_tokenizer, text_encoder, neg_prompt, tag_token, return_index=False)

        generator = torch.Generator(device=device).manual_seed(seed)
        
        print(
            f"control image shape={control_img.size}\n"
            f"num_samples={num_samples}\n"
            f"strength={strength}\n"
            f"cdf scale={cfg_scale}, steps={steps}, use_color_fix={use_color_fix}\n"
            f"seed={seed}\n"
        )
        clip_processor = CLIPImageProcessor()
        
        control_img = control_img.resize((512,512)) 
        h, w = control_img.height, control_img.width

        
        weight_dtype = torch.float16
        
        ref_id_embs = []
        ref_clip_embs = []
        for i, ref in enumerate(ref_image):
            ref = np.array(ref[0])
            try:
                detect_face = analyze_faces(app, ref)[0]
            except:
                raise ValueError(f"Can't detect face: {i}")
            
            ref_emb = detect_face['embedding']
            ref_emb = torch.tensor(ref_emb).to(device)
            ref_emb = ref_emb / torch.norm(ref_emb, dim=0, keepdim=True)  # normalize embedding
            ref_id_embs.append(ref_emb.unsqueeze(dim=0).to(dtype=weight_dtype))

            crop_face = Image.fromarray(ref).crop(detect_face["bbox"])
            crop_face = clip_processor(crop_face)["pixel_values"][0]
            crop_face = torch.tensor(crop_face).to(device).unsqueeze(dim=0)

            ref_clip_emb = id_encoder_clip(crop_face)
            ref_clip_embs.append(ref_clip_emb.to(dtype=weight_dtype))
        ref_id_embs = torch.cat(ref_id_embs, dim=0)
        ref_clip_embs = torch.cat(ref_clip_embs, dim=0)
        embs = mix(clip_emb=ref_clip_embs, id_emb=ref_id_embs)
        pref = pos_prompt_embeds[:, :mask_index, :].clone().to(device)
        sufx = pos_prompt_embeds[:, mask_index+1:, :].clone().to(device)
        _pos_prompt_embeds = torch.cat([pref, embs.unsqueeze(dim=0), sufx], dim=1)[:, :77, :]
                

        preds = []
        for i in tqdm(range(num_samples)):
            
            control = np.array(control_img) / 255.0
            control = control * 2.0 - 1.0
            control = torch.tensor(control).permute(2, 0, 1).unsqueeze(dim=0).to(device=device, dtype=weight_dtype)
            latents = vae.encode(control.to(dtype=torch.float16)).latent_dist.sample()
            latents = latents.to(dtype=weight_dtype) * vae.config.scaling_factor
            noise = torch.randn_like(latents)
            bsz = latents.shape[0]
            timesteps = torch.randint(noise_scheduler.config.num_train_timesteps - 1, noise_scheduler.config.num_train_timesteps, (bsz,), device=latents.device)
            timesteps = timesteps.long()
            noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)
            image = pipeline(
                latents=noisy_latents,
                image=control,
                guidance_scale=float(cfg_scale),
                controlnet_conditioning_scale=float(strength),
                prompt_embeds=_pos_prompt_embeds,
                pooled_prompt_embeds=pos_pooled_prompt_embeds,
                negative_prompt_embeds=neg_prompt_embeds,
                negative_pooled_prompt_embeds=neg_pooled_prompt_embeds,
                num_inference_steps=int(steps),
                generator=generator,
            ).images[0]

            sr = torch.tensor((np.array(image) / 255.) * 2. - 1.).permute(2, 0, 1).unsqueeze(dim=0)
            if use_color_fix:
                sr = wavelet_reconstruction(sr.to(torch.float32).cpu(), control.to(torch.float32).cpu())

            sr = sr.squeeze().permute(1, 2, 0) * 127.5 + 127.5
            sr = sr.cpu().numpy().clip(0, 255).astype(np.uint8)

            from datetime import datetime
            current_time = datetime.now()
            time_str = current_time.strftime("%Y%m%d%H%M%S")
            
            Image.fromarray(sr).save(os.path.join("./app_result", time_str + ".png") )
            preds.append((np.array(sr), f"result_{i}.png"))
        return preds

    MARKDOWN = \
    """
    ## FaceMe: Robust Blind Face Restoration With Personal Identification
    """

    block = gr.Blocks().queue()
    with block:
        with gr.Row():
            gr.Markdown(MARKDOWN)
        with gr.Row():
            with gr.Column():
                input_image = gr.Image(sources="upload", type="pil")
                run_button = gr.Button(value="Run")
                with gr.Accordion("Options", open=True):
                    ref_image = gr.Gallery(label="reference", columns=4, type="pil", interactive=True)
                    pos_prompt = gr.Textbox(label="Positive Prompt", placeholder="Please enter your prompt (must contain only the keyword: 'face')", value="a photo of face. High quality facial image, realistic eyes with detailed texture, normal nose, soft expression, smooth skin, skin texture, soft lighting.", interactive=True)
                    neg_prompt = gr.Textbox(label="Negative Prompt", placeholder="Enter your negative prompt here...", value="painting, oil painting, illustration, drawing, art, sketch, oil painting, cartoon, CG Style, 3D render, unreal engine, blurring, dirty, messy, worst quality, low quality, frames, watermark,  signature, jpeg artifacts, deformed, lowres, over-smooth", interactive=True)
                    num_samples = gr.Slider(label="Number Of Samples", minimum=1, maximum=4, value=1, step=1)
                    cfg_scale = gr.Slider(label="Classifier Free Guidance Scale (Set a value larger than 1 to enable it!)", minimum=0.1, maximum=30.0, value=5.0, step=0.1)
                    strength = gr.Slider(label="Control Strength", minimum=0.0, maximum=2.0, value=1.0, step=0.01)
                    steps = gr.Slider(label="Steps", minimum=1, maximum=100, value=50, step=1)
                    use_color_fix = gr.Checkbox(label="Use Color Correction", value=True)
                    seed = gr.Slider(label="Seed", minimum=0, maximum=2147483647, step=1, value=233)
            with gr.Column():
                result_gallery = gr.Gallery(label="Output", show_label=False, elem_id="gallery", scale=2, height="auto")
        inputs = [
            input_image,
            pos_prompt, 
            neg_prompt,
            num_samples,
            ref_image,
            strength,
            cfg_scale,
            steps,
            use_color_fix,
            seed,
        ]

        run_button.click(fn=process, inputs=inputs, outputs=[result_gallery])
        
    block.launch(share=True, debug=True)




if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="FaceMe simple example.")
    parser.add_argument("--pretrained_model_name_or_path", type=str, default="SG161222/RealVisXL_V3.0")
    parser.add_argument("--controlnet_model_name_or_path", type=str, default=None)
    parser.add_argument("--mix_path", type=str, default=None)
    parser.add_argument("--pos_prompt", type=str, default='A photo of face.')
    parser.add_argument("--neg_prompt", type=str, default='A photo of face.')
    parser.add_argument("--key_word", type=str, default='face')
    args = parser.parse_args()

    assert args.key_word in args.pos_prompt

    main(args)

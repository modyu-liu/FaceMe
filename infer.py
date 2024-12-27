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
from peft import LoraConfig, get_peft_model
from arch.idencoder import PhotoMakerIDEncoder, Mix
from utils.wavelet_color_fix import wavelet_reconstruction
from utils.insightface_package import FaceAnalysis2, analyze_faces
from diffusers.utils import convert_unet_state_dict_to_peft
from typing import Any, Callable, Dict, List, Optional, Union, Tuple
from diffusers.utils import _get_model_file
from safetensors import safe_open


def parse_args():
    parser = argparse.ArgumentParser(description="FaceMe simple example.")
    parser.add_argument("--pretrained_model_name_or_path", type=str)
    parser.add_argument("--controlnet_model_name_or_path", type=str)
    parser.add_argument("--mix_path", type=str)
    parser.add_argument("--seed", type=int, default=233)
    parser.add_argument("--input_dir", type=str, default=None)
    parser.add_argument("--result_dir", type=str, default=None)
    parser.add_argument("--ref_dir", type=str, default=None)
    parser.add_argument("--pos_prompt", type=str, default='A photo of face.')
    parser.add_argument("--neg_prompt", type=str, default='A photo of face.')
    parser.add_argument("--key_word", type=str, default='face')
    parser.add_argument("--color_correction", action="store_true", help="Enable color correction")
    args = parser.parse_args()

    assert args.key_word in args.pos_prompt

    os.makedirs(args.result_dir, exist_ok=True)
    return args

def load_photomaker_adapter(
        pretrained_model_name_or_path_or_dict: Union[str, Dict[str, torch.Tensor]],
        weight_name: str,
        subfolder: str = '',
        **kwargs,
    ):
        # Load the main state dict first.
        cache_dir = kwargs.pop("cache_dir", None)
        force_download = kwargs.pop("force_download", False)
        resume_download = kwargs.pop("resume_download", False)
        proxies = kwargs.pop("proxies", None)
        local_files_only = kwargs.pop("local_files_only", None)
        token = kwargs.pop("token", None)
        revision = kwargs.pop("revision", None)

        user_agent = {
            "file_type": "attn_procs_weights",
            "framework": "pytorch",
        }

        if not isinstance(pretrained_model_name_or_path_or_dict, dict):
            model_file = _get_model_file(
                pretrained_model_name_or_path_or_dict,
                weights_name=weight_name,
                cache_dir=cache_dir,
                force_download=force_download,
                resume_download=resume_download,
                proxies=proxies,
                local_files_only=local_files_only,
                use_auth_token=token,
                revision=revision,
                subfolder=subfolder,
                user_agent=user_agent,
            )
            if weight_name.endswith(".safetensors"):
                state_dict = {"id_encoder": {}, "lora_weights": {}}
                with safe_open(model_file, framework="pt", device="cpu") as f:
                    for key in f.keys():
                        if key.startswith("id_encoder."):
                            state_dict["id_encoder"][key.replace("id_encoder.", "")] = f.get_tensor(key)
                        elif key.startswith("lora_weights."):
                            state_dict["lora_weights"][key.replace("lora_weights.", "")] = f.get_tensor(key)
            else:
                state_dict = torch.load(model_file, map_location="cpu")
        else:
            state_dict = pretrained_model_name_or_path_or_dict

        keys = list(state_dict.keys())
        if keys != ["id_encoder", "lora_weights"]:
            raise ValueError("Required keys are (`id_encoder` and `lora_weights`) missing from the state dict.")

        return state_dict["id_encoder"], state_dict["lora_weights"]


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


def prepare_prompt(args):
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
    text_encoder = [text_encoder_one, text_encoder_two]

    pos_tokens_one = tokenizer_one(args.pos_prompt, padding="max_length", max_length=tokenizer_one.model_max_length,
                                   truncation=True, return_tensors="pt").input_ids
    pos_tokens_two = tokenizer_two(args.pos_prompt, padding="max_length", max_length=tokenizer_two.model_max_length,
                                   truncation=True, return_tensors="pt").input_ids

    token_id_one = tokenizer_one.encode(args.key_word)[1]
    token_id_two = tokenizer_two.encode(args.key_word)[1]

    index = torch.where(pos_tokens_one == token_id_one)[1]

    neg_tokens_one = tokenizer_one(args.neg_prompt, padding="max_length", max_length=tokenizer_one.model_max_length,
                                   truncation=True, return_tensors="pt").input_ids
    neg_tokens_two = tokenizer_two(args.neg_prompt, padding="max_length", max_length=tokenizer_two.model_max_length,
                                   truncation=True, return_tensors="pt").input_ids

    pos_prompt_embeds, pos_pooled_prompt_embeds = encode_prompt(text_encoders=text_encoder, text_input_ids_list=[pos_tokens_one, pos_tokens_two])
    neg_prompt_embeds, neg_pooled_prompt_embeds = encode_prompt(text_encoders=text_encoder, text_input_ids_list=[neg_tokens_one, neg_tokens_two])

    return index, pos_prompt_embeds, pos_pooled_prompt_embeds, neg_prompt_embeds, neg_pooled_prompt_embeds


def main(args):
    mask_index, pos_prompt_embeds, pos_pooled_prompt_embeds, neg_prompt_embeds, neg_pooled_prompt_embeds = prepare_prompt(args)
    noise_scheduler = DDPMScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler")
    vae = AutoencoderKL.from_pretrained(args.pretrained_model_name_or_path, subfolder="vae", revision=None)
    unet = OriginalUNet2DConditionModel.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="unet",
    )

    photomaker_path = hf_hub_download(repo_id="TencentARC/PhotoMaker", filename="photomaker-v1.bin", repo_type="model")
    id_encoder_clip_state_dict, lora_state_dict = load_photomaker_adapter(
        os.path.dirname(photomaker_path),
        subfolder="",
        weight_name=os.path.basename(photomaker_path),
    )

    unet_lora_config = LoraConfig(
        r=64,
        lora_alpha=64,
        init_lora_weights="gaussian",
        target_modules=["to_k", "to_q", "to_v", "to_out.0"],
    )
    unet = get_peft_model(unet, unet_lora_config)
    lora_state_dict = {f'{k.replace("unet.", "base_model.model.").replace(".weight", ".default.weight")}': v for k, v in lora_state_dict.items() if k.startswith("unet.")}
    lora_state_dict = convert_unet_state_dict_to_peft(lora_state_dict)
    for name, param in unet.named_parameters():
        if name in lora_state_dict.keys():
            param = lora_state_dict[name]
        else:
            if name.find("lora") != -1:
                print(name)
    unet.merge_and_unload()
    
    id_encoder_clip = PhotoMakerIDEncoder()
    id_encoder_clip.load_state_dict(id_encoder_clip_state_dict, strict=False)
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
    generator = torch.Generator(device=device).manual_seed(args.seed)
    pipeline = pipeline.to(device=device)
    clip_processor = CLIPImageProcessor()

    input_files = [os.path.join(args.input_dir, file) for file in os.listdir(args.input_dir)]
    ref_files = [os.path.join(args.ref_dir, file) for file in os.listdir(args.ref_dir)]

    ref_id_embs = []
    ref_clip_embs = []
    for ref_file in ref_files:
        ref = cv2.cvtColor(cv2.imread(ref_file), cv2.COLOR_BGR2RGB)
        try:
            detect_face = analyze_faces(app, ref)[0]
        except:
            raise ValueError(f"Can't detect face: {ref_file}")
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


    for input_file in tqdm(input_files):
        control = cv2.cvtColor(cv2.imread(input_file), cv2.COLOR_BGR2RGB)
        control = control / 255.0
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
            prompt_embeds=_pos_prompt_embeds,
            pooled_prompt_embeds=pos_pooled_prompt_embeds,
            negative_prompt_embeds=neg_prompt_embeds,
            negative_pooled_prompt_embeds=neg_pooled_prompt_embeds,
            num_inference_steps=50,
            generator=generator,
        ).images[0]

        sr = torch.tensor((np.array(image) / 255.) * 2. - 1.).permute(2, 0, 1).unsqueeze(dim=0)
        if args.color_correction:
            sr = wavelet_reconstruction(sr.to(torch.float32).cpu(), control.to(torch.float32).cpu())

        sr = sr.squeeze().permute(1, 2, 0) * 127.5 + 127.5
        sr = sr.cpu().numpy().clip(0, 255).astype(np.uint8)

        result = cv2.cvtColor(sr, cv2.COLOR_BGR2RGB)
        cv2.imwrite(os.path.join(args.result_dir, os.path.basename(input_file)), result)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FaceMe simple example.")
    parser.add_argument("--pretrained_model_name_or_path", type=str)
    parser.add_argument("--controlnet_model_name_or_path", type=str)
    parser.add_argument("--mix_path", type=str)
    parser.add_argument("--seed", type=int, default=236)
    parser.add_argument("--input_dir", type=str, default=None)
    parser.add_argument("--result_dir", type=str, default=None)
    parser.add_argument("--ref_dir", type=str, default=None)
    parser.add_argument("--pos_prompt", type=str, default='A photo of face.')
    parser.add_argument("--neg_prompt", type=str, default='A photo of face.')
    parser.add_argument("--key_word", type=str, default='face')
    parser.add_argument("--color_correction", action="store_true", help="Enable color correction")
    args = parser.parse_args()

    assert args.key_word in args.pos_prompt
    os.makedirs(args.result_dir, exist_ok=True)

    main(args)

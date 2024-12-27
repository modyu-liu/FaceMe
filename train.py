import argparse
import os
import torch
import torch.nn.functional as F
import json 
import math
import random
import itertools
from tqdm.auto import tqdm
from transformers import AutoTokenizer, CLIPTextModel, CLIPTextModelWithProjection
from diffusers import UNet2DConditionModel as OriginalUNet2DConditionModel
from diffusers import (
    AutoencoderKL,
    ControlNetModel,
    DDPMScheduler,
)
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed
from diffusers.optimization import get_scheduler
from diffusers import AutoencoderKL, DDPMScheduler
from diffusers.utils import check_min_version, is_wandb_available, convert_unet_state_dict_to_peft
from diffusers.utils.torch_utils import is_compiled_module
from huggingface_hub import create_repo, upload_folder , hf_hub_download , snapshot_download
from utils.load_photomaker import load_photomaker
from arch.idencoder import Mix
if is_wandb_available():
    import wandb

logger = get_logger(__name__, log_level="INFO")

def load_config(config_path):
    """
    Load the configuration file and return the configuration dictionary.
    """
    with open(config_path, "r") as file:
        config = json.load(file)
    return config


def make_train_dataset(args, tokenizer, text_encoder):
    def encode_prompt(text_encoders, text_input_ids_list=None):
        prompt_embeds_list = []
        for i, text_encoder in enumerate(text_encoders):
            prompt_embeds = text_encoder(
                text_input_ids_list[i].to(text_encoder.device), output_hidden_states=True, return_dict=False
            )
            # We are only ALWAYS interested in the pooled output of the final text encoder
            pooled_prompt_embeds = prompt_embeds[0]
            prompt_embeds = prompt_embeds[-1][-2]
            bs_embed, seq_len, _ = prompt_embeds.shape
            prompt_embeds = prompt_embeds.view(bs_embed, seq_len, -1)
            prompt_embeds_list.append(prompt_embeds)
        prompt_embeds = torch.concat(prompt_embeds_list, dim=-1)
        pooled_prompt_embeds = pooled_prompt_embeds.view(bs_embed, -1)
        return prompt_embeds, pooled_prompt_embeds

    
    tokens_one = tokenizer[0]("A photo of face.",padding="max_length",
                              max_length=tokenizer[0].model_max_length,
                              truncation=True,
                              return_tensors="pt",).input_ids
    tokens_two = tokenizer[1]("A photo of face.",padding="max_length",
                              max_length=tokenizer[1].model_max_length,
                              truncation=True,
                              return_tensors="pt",).input_ids
    prompt_embeds , pooled_prompt_embeds = encode_prompt(text_encoders=text_encoder, text_input_ids_list=[tokens_one, tokens_two])
    crops_coords_top_left = (0, 0)
    original_size = (args.resolution, args.resolution)
    target_size = (args.resolution, args.resolution)
    add_time_ids = list(original_size + crops_coords_top_left + target_size)
    add_time_ids = torch.tensor([add_time_ids])
    
    from dataset import FaceMeDataset
    train_dataset = FaceMeDataset(file_json=args.train_data_dir, 
                                  prompt_embeds=prompt_embeds.squeeze(dim=0), 
                                  pooled_prompt_embeds=pooled_prompt_embeds.squeeze(dim=0), 
                                  tokens_one=tokens_one.squeeze(dim=0), 
                                  add_time_ids=add_time_ids.squeeze(dim=0), )    
   
    return train_dataset

def main(args):

    logging_dir = os.path.join(args.output_dir, "log")
    accelerator_project_config = ProjectConfiguration(project_dir=args.output_dir, logging_dir=logging_dir)
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_config=accelerator_project_config,
    )

    if args.seed is not None:
        set_seed(args.seed)

    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)

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
    
    token_id_one = tokenizer_one.encode("face")[1]
    token_id_two = tokenizer_two.encode("face")[1]
    
    text_encoder_one = CLIPTextModel.from_pretrained(args.pretrained_model_name_or_path, subfolder="text_encoder", )
    text_encoder_two = CLIPTextModelWithProjection.from_pretrained(args.pretrained_model_name_or_path, subfolder="text_encoder_2", )

    noise_scheduler = DDPMScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler")
    
    vae = AutoencoderKL.from_pretrained(args.pretrained_model_name_or_path, subfolder="vae",)
    unet = OriginalUNet2DConditionModel.from_pretrained(args.pretrained_model_name_or_path, subfolder="unet", )

    #merge photomaker weights
    photomaker_path = hf_hub_download(repo_id="TencentARC/PhotoMaker", filename="photomaker-v1.bin", repo_type="model")
    _, unet = load_photomaker(photomaker_path, clip_id_encoder=None, unet=unet)
     ###

    ## Initializing controlnet weights from unet 
    controlnet = ControlNetModel.from_unet(unet)

    # add id mix 
    mix = Mix()
    if args.mix_pretrained_path is not None:
        mix.from_pretrained(args.mix_pretrained_path)
    ###

    vae.requires_grad_(False)
    text_encoder_one.requires_grad_(False)
    text_encoder_two.requires_grad_(False)
    unet.requires_grad_(False)
    if args.mix_pretrained_path is not None:
        mix.requires_grad_(False)
    else:
        mix.requires_grad_(True)    
    controlnet.requires_grad_(True)

    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    vae.to(accelerator.device, dtype=weight_dtype)
    unet.to(accelerator.device, dtype=weight_dtype)
    text_encoder_one.to(accelerator.device, dtype=weight_dtype)
    text_encoder_two.to(accelerator.device, dtype=weight_dtype)
    if args.mix_pretrained_path is not None:
        mix.to(accelerator.device, dtype=weight_dtype)

    if args.mix_pretrained_path is not None:
        params_to_optimize = itertools.chain(controlnet.parameters())
    else :
        params_to_optimize = itertools.chain(controlnet.parameters(), mix.parameters())
    optimizer = torch.optim.AdamW(
        params_to_optimize,
        lr=args.learning_rate,
        betas=(0.9, 0.999),
        weight_decay=1e-2,
        eps=1e-08,
    )

    train_dataset = make_train_dataset(args, [tokenizer_one, tokenizer_two], [text_encoder_one, text_encoder_two])
    def custom_collate_fn(batch):
        gt = torch.stack([torch.tensor(item['target']) for item in batch])
        control = torch.stack([torch.tensor(item['control']) for item in batch])
        prompt_embeds = torch.stack([item['prompt_embeds'] for item in batch])
        pooled_prompt_embeds = torch.stack([item['pooled_prompt_embeds'] for item in batch])
        tokens_one = torch.stack([item['tokens_one'] for item in batch])
        add_time_ids = torch.stack([item['add_time_ids'] for item in batch])

        if args.mix_pretrained_path is not None and random.random() > float(args.null_prompt_p):
            return dict(target=gt,control=control, prompt_embeds=prompt_embeds, pooled_prompt_embeds=pooled_prompt_embeds ,add_time_ids=add_time_ids)
        else :  
            ref_id_emb = torch.stack([item['ref_id_emb'] for item in batch])
            ref_clip_emb = torch.stack([item['ref_clip_emb'] for item in batch])
            
            random_num = random.randint(1, 4)        
            ref_id_emb = ref_id_emb[:, :random_num, :]
            ref_clip_emb = ref_clip_emb[:, :random_num, :]
            index = torch.where(tokens_one == token_id_one)[1][0]
            pref = prompt_embeds[:, :index, :]
            sufx = prompt_embeds[:, index + 1:, :]
            
            return dict(target=gt,control=control, ref_id_emb=ref_id_emb, ref_clip_emb=ref_clip_emb, pref=pref, sufx=sufx, pooled_prompt_embeds=pooled_prompt_embeds ,add_time_ids=add_time_ids)
            
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        shuffle=True,
        collate_fn=custom_collate_fn,
        batch_size=args.train_batch_size,
        num_workers=args.num_workers,
    )

    # Scheduler and math around the number of training steps.
    # Check the PR https://github.com/huggingface/diffusers/pull/8312 for detailed explanation.
    num_warmup_steps_for_scheduler = args.lr_warmup_steps * accelerator.num_processes
    if args.max_train_steps is None:
        len_train_dataloader_after_sharding = math.ceil(len(train_dataloader) / accelerator.num_processes)
        num_update_steps_per_epoch = math.ceil(len_train_dataloader_after_sharding / args.gradient_accumulation_steps)
        num_training_steps_for_scheduler = args.num_train_epochs * num_update_steps_per_epoch * accelerator.num_processes
    else:
        num_training_steps_for_scheduler = args.max_train_steps * accelerator.num_processes

    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=num_warmup_steps_for_scheduler,
        num_training_steps=num_training_steps_for_scheduler,

    )
    if args.mix_pretrained_path is not None:
        controlnet, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
            controlnet, optimizer, train_dataloader, lr_scheduler
        )   
    else :   
        controlnet, mix, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
            controlnet, mix, optimizer, train_dataloader, lr_scheduler
        )

  
    def unwrap_model(model):
        model = accelerator.unwrap_model(model)
        model = model._orig_mod if is_compiled_module(model) else model
        return model

    ## register 
    def save_model_hook(models, weights, output_dir):
        if accelerator.is_main_process:
            for model in models:    
                if isinstance(unwrap_model(model), ControlNetModel):
                    model.save_pretrained(os.path.join(output_dir, 'controlnet'))
                    accelerator.print("success save controlnet!")
                elif isinstance(unwrap_model(model), Mix):
                    model.save_pretrained(os.path.join(output_dir, 'mix'))
                    accelerator.print("success save id mix!")
                else:
                    raise ValueError(f"unexpected save model: {model.__class__}")
                if weights:
                    weights.pop()
    def load_model_hook(models, input_dir):
        while len(models) > 0:
            model = models.pop()
            if isinstance(unwrap_model(model), ControlNetModel):
                model.from_pretrained(os.path.join(input_dir, 'controlnet'))
                accelerator.print("success load controlnet!")
            elif isinstance(unwrap_model(model), Mix):
                model.from_pretrained(os.path.join(input_dir, 'mix'))
                accelerator.print("success load id mix!")
            else:
                raise ValueError(f"unexpected load model: {model.__class__}")
    accelerator.register_save_state_pre_hook(save_model_hook)
    accelerator.register_load_state_pre_hook(load_model_hook)

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        if num_training_steps_for_scheduler != args.max_train_steps * accelerator.num_processes:
            logger.warning(
                f"The length of the 'train_dataloader' after 'accelerator.prepare' ({len(train_dataloader)}) does not match "
                f"the expected length ({len_train_dataloader_after_sharding}) when the learning rate scheduler was created. "
                f"This inconsistency may result in the learning rate scheduler not functioning properly."
            )
    # Afterwards we recalculate our number of training epochs
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)
    
    if accelerator.is_main_process:
        tracker_config = dict(vars(args))
        accelerator.init_trackers(args.exp_name, config=tracker_config)


    # Train!
    total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    global_step = 0
    first_epoch = 0
    initial_global_step = 0

    progress_bar = tqdm(
        range(0, args.max_train_steps),
        initial=initial_global_step,
        desc="Steps",
        # Only show the progress bar once on each machine.
        disable=not accelerator.is_local_main_process,
    )

    for epoch in range(first_epoch, args.num_train_epochs):
        epoch_loss = 0.0
        for step, batch in enumerate(train_dataloader):
            with accelerator.accumulate(controlnet):
                # Convert images to latent space
                latents = vae.encode(batch["target"].to(dtype=weight_dtype)).latent_dist.sample()
                latents = latents * vae.config.scaling_factor

                # Sample noise that we'll add to the latents
                noise = torch.randn_like(latents)
                bsz = latents.shape[0]
        
                # Sample a random timestep for each image
                timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bsz,), device=latents.device)
                timesteps = timesteps.long()

                # Add noise to the latents according to the noise magnitude at each timestep
                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

                controlnet_image = batch["control"].to(dtype=weight_dtype)
                
                if 'prompt_embeds' in batch.keys():
                    id_prompt_embeds = batch['prompt_embeds'].to(dtype=weight_dtype)
                else :
                    ref_clip_emb = batch['ref_clip_emb'].to(dtype=weight_dtype)
                    ref_id_emb = batch['ref_id_emb'].to(dtype=weight_dtype) 
                    pref = batch['pref'].to(dtype=weight_dtype)
                    sufx = batch['sufx'].to(dtype=weight_dtype)
    
                    mix_emb = mix(clip_emb=ref_clip_emb , id_emb=ref_id_emb)
                    id_prompt_embeds = torch.cat([pref, mix_emb, sufx] , dim=1)[:,:77,:]
                    id_prompt_embeds = id_prompt_embeds.to(dtype=weight_dtype)
                        
                unet_added_conditions = {'text_embeds':batch['pooled_prompt_embeds'].to(dtype=weight_dtype), 'time_ids': batch['add_time_ids']}
                
                down_block_res_samples, mid_block_res_sample = controlnet(
                    noisy_latents,
                    timesteps,
                    encoder_hidden_states= id_prompt_embeds ,
                    added_cond_kwargs= unet_added_conditions,
                    controlnet_cond=controlnet_image,
                    return_dict=False,
                )
                model_pred = unet(
                    noisy_latents,
                    timesteps,
                    encoder_hidden_states= id_prompt_embeds,
                    added_cond_kwargs= unet_added_conditions,
                    down_block_additional_residuals=[
                        sample.to(dtype=weight_dtype) for sample in down_block_res_samples
                    ],
                    mid_block_additional_residual=mid_block_res_sample.to(dtype=weight_dtype),
                    return_dict=False,
                )[0]

                # Get the target for loss depending on the prediction type
                if noise_scheduler.config.prediction_type == "epsilon":
                    target = noise
                elif noise_scheduler.config.prediction_type == "v_prediction":
                    target = noise_scheduler.get_velocity(latents, noise, timesteps)
                else:
                    raise ValueError(f"Unknown prediction type {noise_scheduler.config.prediction_type}")
                
                loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")
                accelerator.backward(loss)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1
                if accelerator.is_main_process:
                    
                    if global_step % args.checkpoint_steps == 0:
                        save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                        accelerator.save_state(save_path)
                        logger.info(f"Saved state to {save_path}")

             

            logs = {"loss": loss.detach().item(),  "lr": lr_scheduler.get_last_lr()[0]}
            epoch_loss += loss.detach().item()
            
            progress_bar.set_postfix(**logs)
            accelerator.log(logs, step=global_step)

            if global_step >= args.max_train_steps:
                break
            
        epoch_loss /= len(train_dataloader)
        accelerator.log({"epoch_loss": epoch_loss}, step=global_step)
    accelerator.wait_for_everyone()
    accelerator.end_training()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--pretrained_model_name_or_path", type=str, )
    parser.add_argument("--mix_pretrained_path", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default="./result")
    parser.add_argument("--resolution", type=int, default=512,)
    parser.add_argument("--train_batch_size", type=int, default=4)
    parser.add_argument("--num_train_epochs", type=int, default=200)
    parser.add_argument("--max_train_steps", type=int, default=None) 
    parser.add_argument("--checkpoint_steps", type=int, default=50000) 
    parser.add_argument("--learning_rate", type=float, default=5e-5)
    parser.add_argument("--lr_warmup_steps", type=int, default=500)
    parser.add_argument("--lr_scheduler", type=str, default="constant",)    
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--mixed_precision", type=str, default="fp16")
    parser.add_argument("--seed", type=str, default=233)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--report_to", type=str, default="wandb")
    parser.add_argument("--train_data_dir", type=str, )
    parser.add_argument("--null_prompt_p", type=float, default=0.5)
    parser.add_argument("--exp_name", type=str, default="faceme")
    
    args = parser.parse_args()
    if args.resolution % 8 != 0:
        raise ValueError(
            "`--resolution` must be divisible by 8 for consistently sized encoded images between the VAE and the controlnet encoder."
        )
    
    main(args)
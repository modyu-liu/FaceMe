import torch
import os
from diffusers.utils import _get_model_file
from safetensors import safe_open
from typing import Any, Callable, Dict, List, Optional, Union, Tuple


def load_photomaker_adapter(
        pretrained_model_name_or_path_or_dict: Union[str, Dict[str, torch.Tensor]],
        weight_name: str,
        subfolder: str = '',
        **kwargs,
    ):
        # Load the main state dict first.
        cache_dir = kwargs.pop("cache_dir", None)
        force_download = kwargs.pop("force_download", False)
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

        print(f"Loading PhotoMaker components [1] id_encoder from [{pretrained_model_name_or_path_or_dict}]...")
        return state_dict["id_encoder"], state_dict["lora_weights"]



def load_photomaker(photomaker_path: str, clip_id_encoder=None, unet=None):
    
    clip_id_state_dict, unet_lora_state_dict = load_photomaker_adapter(
        os.path.dirname(photomaker_path),
        subfolder="",
        weight_name=os.path.basename(photomaker_path),
    )

    if clip_id_encoder is not None:
        clip_id_encoder.load_state_dict(clip_id_state_dict, strict=False)

    if unet is not None:
        unet = apply_lora_to_unet(unet, unet_lora_state_dict)

    return clip_id_encoder, unet


def apply_lora_to_unet(unet, lora_state_dict):
    from peft import LoraConfig, get_peft_model

    unet_lora_config = LoraConfig(
        r=64,
        lora_alpha=64,
        init_lora_weights="gaussian",
        target_modules=["to_k", "to_q", "to_v", "to_out.0"],
    )

    unet = get_peft_model(unet, unet_lora_config)

    formatted_lora_state_dict = {
        f'{k.replace("unet.", "base_model.model.").replace(".weight", ".default.weight")}': v
        for k, v in lora_state_dict.items() if k.startswith("unet.")
    }

    for name, param in unet.named_parameters():
        if name in formatted_lora_state_dict:
            param.data.copy_(formatted_lora_state_dict[name])

    unet.merge_and_unload()

    return unet
import torch
import torch.nn as nn
from transformers.models.clip.modeling_clip import CLIPVisionModelWithProjection
from transformers.models.clip.configuration_clip import CLIPVisionConfig
import os
from safetensors.torch import save_file, load_file


VISION_CONFIG_DICT = {
    "hidden_size": 1024,
    "intermediate_size": 4096,
    "num_attention_heads": 16,
    "num_hidden_layers": 24,
    "patch_size": 14,
    "projection_dim": 768
}

class PhotoMakerIDEncoder(CLIPVisionModelWithProjection):
    def __init__(self):
        super().__init__(CLIPVisionConfig(**VISION_CONFIG_DICT))
        self.visual_projection_2 = nn.Linear(1024, 1280, bias=False)
        
    def forward(self, id_pixel_values):
        # Assuming only one reference image is input
        #b , c , h , w = id_pixel_values.shape
        
        shared_id_embeds = self.vision_model(id_pixel_values)[1]
        id_embeds = self.visual_projection(shared_id_embeds)
        id_embeds_2 = self.visual_projection_2(shared_id_embeds)
        
        id_embeds = torch.cat((id_embeds, id_embeds_2), dim=-1)
        
        return id_embeds
    
class MLP(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_dim, use_residual=True):
        super().__init__()
        if use_residual:
            assert in_dim == out_dim
        self.layernorm = nn.LayerNorm(in_dim)
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, out_dim)
        self.use_residual = use_residual
        self.act_fn = nn.GELU()

    def forward(self, x):
        residual = x
        x = self.layernorm(x)
        x = self.fc1(x)
        x = self.act_fn(x)
        x = self.fc2(x)
        if self.use_residual:
            x = x + residual
        return x


class MLPs(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.mlp1 = MLP(in_dim, in_dim * 2, in_dim * 2, False)
        self.mlp2 = MLP(in_dim * 2, in_dim * 2, in_dim * 2, True)
        self.mlp3 = MLP(in_dim * 2, in_dim * 4, in_dim * 4, False)
        self.mlp4 = MLP(in_dim * 4, in_dim * 4, in_dim * 4, True)
    def forward(self, x):
        x = self.mlp1(x)
        x = self.mlp2(x)
        x = self.mlp3(x)
        x = self.mlp4(x)
        return x 
    
class Mix(nn.Module):
    def __init__(self, embed_dim=2048):
        super().__init__()
        self.pro_id = MLPs(512)
        self.mlp1 = MLP(embed_dim * 2, embed_dim, embed_dim, use_residual=False)
        self.mlp2 = MLP(embed_dim, embed_dim, embed_dim, use_residual=True)
        self.norm1 = nn.LayerNorm(embed_dim)
        
    def forward(self, clip_emb, id_emb):
        id_emb = self.pro_id(id_emb)
        emb = self.mlp1(torch.cat([clip_emb, id_emb], dim=-1))
        emb = self.norm1(emb)
        emb = self.mlp2(emb)
        return emb



    def from_pretrained(self, save_dir):
        # Load encoder parameters
        mix_path = os.path.join(save_dir, "mix.safetensors")
        state_dict = load_file(mix_path)
        self.load_state_dict(state_dict)
        
    def save_pretrained(self, save_dir):
        os.makedirs(save_dir, exist_ok=True)
        # Save encoder parameters
        save_path = os.path.join(save_dir, "mix.safetensors")
        save_file(self.state_dict(), save_path)
        print(f"Mix saved to {save_path}")
    
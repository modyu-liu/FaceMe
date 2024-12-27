import math
import random
import numpy as np
import cv2
import torch.utils.data as data
from utils.degradation import random_mixed_kernels
from utils.degradation import (
    random_mixed_kernels,
    random_add_gaussian_noise,
    random_add_jpg_compression
)
import json
import torch


class FaceMeDataset(data.Dataset):
    
    def __init__(
        self,
        file_json: str, 
        prompt_embeds=None, 
        pooled_prompt_embeds=None, 
        tokens_one=None, 
        add_time_ids=None,
        blur_kernel_size = 41,
        kernel_list = ['iso', 'aniso'],
        kernel_prob = [0.5, 0.5],
        blur_sigma = [0.2, 10],
        downsample_range = [1, 16], # 1,16
        noise_range = [0,15], # 0 15
        jpeg_range =  [30,100],
    ):
        super().__init__()
        self.data = []
        with open(file_json , 'rt') as f:
            for line in f:
                self.data.append(json.loads(line))
        
        # degradation configurations
        self.blur_kernel_size = blur_kernel_size
        self.kernel_list = kernel_list
        self.kernel_prob = kernel_prob
        self.blur_sigma = blur_sigma
        self.downsample_range = downsample_range
        self.noise_range = noise_range
        self.jpeg_range = jpeg_range
        ####
        
        self.prompt_embeds=prompt_embeds 
        self.pooled_prompt_embeds=pooled_prompt_embeds 
        self.tokens_one=tokens_one 
        self.add_time_ids=add_time_ids


    def __getitem__(self, index: int):
        
        data = self.data[index]
        gt_path = data["target"]
        gt_emb_path = data['target_emb']
        ref_emb_list = data['ref_emb']
        ref_id_emb_list = ref_emb_list[0]
        ref_clip_emb_list = ref_emb_list[1]

        img_gt = cv2.imread(gt_path)
        img_gt = cv2.cvtColor(img_gt, cv2.COLOR_BGR2RGB)
        
        # ------------------------ Sample Reference Image ------------------------ #
        ref_id_emb_list.append(gt_emb_path[0])
        ref_clip_emb_list.append(gt_emb_path[1])

        ref_emb_list = list(zip(ref_id_emb_list, ref_clip_emb_list))

        current_len = len(ref_emb_list)
        target_len = 4
        
        if current_len < target_len:
            additional_emb = random.choices(ref_emb_list, k=target_len - current_len)
            ref_emb_list.extend(additional_emb)
           
        ref_emb_path = random.sample(ref_emb_list, target_len)
        random.shuffle(ref_emb_path)
        
        ref_id_embs = []
        ref_clip_embs = []
        for id_emb_path, clip_emb_path in ref_emb_path:
            id_emb = torch.load(id_emb_path,map_location=torch.device('cpu')).requires_grad_(False)
            clip_emb = torch.load(clip_emb_path,map_location=torch.device('cpu')).requires_grad_(False)
            if id_emb.ndimension() == 1:
                id_emb = id_emb.unsqueeze(dim=0)
            ref_id_embs.append(id_emb)
            ref_clip_embs.append(clip_emb)
    
        ref_id_embs = torch.cat(ref_id_embs, dim=0)
        ref_clip_embs = torch.cat(ref_clip_embs, dim=0)
        # ------------------------------------------------ #
        

        # ------------------------ generate lq image ------------------------ #
        img_gt = (img_gt / 255.0).astype(np.float32)
        h, w, _ = img_gt.shape
        # blur
        kernel = random_mixed_kernels(
            self.kernel_list,
            self.kernel_prob,
            self.blur_kernel_size,
            self.blur_sigma,
            self.blur_sigma,
            [-math.pi, math.pi],
            noise_range=None
        )
        img_lq = cv2.filter2D(img_gt, -1, kernel)
        # downsample
        scale = np.random.uniform(self.downsample_range[0], self.downsample_range[1])
        img_lq = cv2.resize(img_lq, (int(w // scale), int(h // scale)), interpolation=cv2.INTER_LINEAR)
        # noise
        if self.noise_range is not None:
            img_lq = random_add_gaussian_noise(img_lq, self.noise_range)
        # jpeg compression
        if self.jpeg_range is not None:
            img_lq = random_add_jpg_compression(img_lq, self.jpeg_range)
        # resize to original size
        img_lq = cv2.resize(img_lq, (w, h), interpolation=cv2.INTER_LINEAR)
        # ------------------------------------------------ #
        
        # GT [-1, 1] LQ [-1, 1]
        gt = (img_gt * 2 - 1).astype(np.float32)
        lq = (img_lq.clip(0, 1) * 2 - 1).astype(np.float32)
        # HWC to CHW
        gt = np.transpose(gt, (2, 0, 1))
        lq = np.transpose(lq, (2, 0, 1))
        #
        
        return {'target': gt, 'control': lq, 'ref_id_emb': ref_id_embs, 'ref_clip_emb': ref_clip_embs, 
            'prompt_embeds': self.prompt_embeds,'pooled_prompt_embeds':self.pooled_prompt_embeds,'tokens_one':self.tokens_one,'add_time_ids':self.add_time_ids}                                
    
    
    def __len__(self) -> int:
        return len(self.data)
    

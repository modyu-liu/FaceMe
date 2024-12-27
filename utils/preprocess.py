import os
import argparse
import cv2
from PIL import Image
import torch
import numpy as np
from transformers import CLIPImageProcessor
from tqdm.auto import tqdm
from huggingface_hub import hf_hub_download
from arch.idencoder import PhotoMakerIDEncoder
from utils.load_photomaker import load_photomaker
from utils.insightface_package import FaceAnalysis2, analyze_faces


def main(args):

    if args.dataset_name == 'FFHQ':
        files = [os.path.join(args.input_dir, file) for file in os.listdir(args.input_dir)]
    elif args.dataset_name == 'FFHQRef':
        files = []
        for pos1 in os.listdir(args.input_dir):
            for pos2 in os.listdir(os.path.join(args.input_dir, pos1)):
                for file in os.listdir(os.path.join(args.input_dir, pos1, pos2)):
                    files.append(os.path.join(args.input_dir, pos1, pos2, file))

    photomaker_path = hf_hub_download(repo_id="TencentARC/PhotoMaker", filename="photomaker-v1.bin", repo_type="model")
    id_encoder_clip = PhotoMakerIDEncoder()
    id_encoder_clip, _ = load_photomaker(photomaker_path, clip_id_encoder=id_encoder_clip, unet=None)
    id_encoder_clip = id_encoder_clip.to("cuda")
    clip_processor = CLIPImageProcessor()

    app = FaceAnalysis2(name='antelopev2', root='./', providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
    app.prepare(ctx_id=0, det_size=(512, 512))

    for path in tqdm(files):
        if args.dataset_name == 'FFHQ':
            basename = os.path.basename(path)
        elif args.dataset_name == 'FFHQRef':
            subdir , basename = path.split('/')[-2:]

        img = cv2.imread(path)
        img = cv2.cvtColor(img , cv2.COLOR_BGR2RGB)
        faces = analyze_faces(app, img)

        if len(faces) == 0 :
            print(f"{path} can't detect face !")
            continue
        faces = sorted(faces, key=lambda x : (x["bbox"][2] - x["bbox"][0]) * (x["bbox"][3] - x["bbox"][1]))[-1]
        emb = torch.tensor(faces['embedding']).to("cuda")
        emb = emb/torch.norm(emb, dim=0, keepdim=True)   # normalize embedding
        emb = emb.unsqueeze(dim=0)

        x1, y1, x2, y2 = faces["bbox"]
        crop_img = Image.fromarray(img).crop((x1,y1,x2,y2))
        crop_img = clip_processor(crop_img)['pixel_values'][0]
        crop_img = torch.tensor(crop_img).unsqueeze(dim=0).to("cuda")
        clip_emb = id_encoder_clip(crop_img)
        
        emb = emb.detach().cpu().numpy()
        clip_emb = clip_emb.detach().cpu().numpy()
        
        if args.dataset_name == 'FFHQ':
            np.save(os.path.join(args.clip_emb_save_dir, basename.split('.')[0] + '.npy'), clip_emb)
            np.save(os.path.join(args.id_emb_save_dir, basename.split('.')[0] + '.npy'), emb)
        elif args.dataset_name == 'FFHQRef' :
            os.makedirs(os.path.join(args.clip_emb_save_dir, basename.split('.')[0]), exist_ok=True)
            os.makedirs(os.path.join(args.id_emb_save_dir, basename.split('.')[0]), exist_ok=True)
            np.save(os.path.join(args.clip_emb_save_dir, basename.split('.')[0], subdir[-3:] + '.npy'), clip_emb)
            np.save(os.path.join(args.id_emb_save_dir, basename.split('.')[0], subdir[-3:] + '.npy'), emb)
            

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, default=None)
    parser.add_argument("--id_emb_save_dir", type=str, default=None)
    parser.add_argument("--clip_emb_save_dir", type=str, default=None)
    parser.add_argument("--dataset_name", type=str, choices=["FFHQ", "FFHQRef"], default='FFHQRef')
    args = parser.parse_args()
    os.makedirs(args.id_emb_save_dir, exist_ok=True)
    os.makedirs(args.clip_emb_save_dir, exist_ok=True)
    
    main(args)



import os
import argparse
import json
from tqdm import tqdm

def main(args):
        
    pair_data = []

    for file in tqdm(os.listdir(args.ffhq_dir)):
        num = file.split('.')[-2]

        target_path = os.path.join(args.ffhq_dir , file)

        target_id_emb_path = os.path.join(args.ffhq_emb_dir, "id_emb" , num + '.npy')
        target_clip_emb_path = os.path.join(args.ffhq_emb_dir, "clip_emb", num + '.npy')
        
        ref_id_emb_dir = os.path.join(args.ffhqref_emb_dir, "id_emb" , num)
        ref_clip_emb_dir = os.path.join(args.ffhqref_emb_dir, "clip_emb", num)
        
        ref_id_emb_paths = [os.path.join(ref_id_emb_dir, f) for f in os.listdir(ref_id_emb_dir)]
        ref_clip_emb_paths = [os.path.join(ref_clip_emb_dir, f) for f in os.listdir(ref_clip_emb_dir)]
        
        ref_id_emb_paths = sorted(ref_id_emb_paths)
        ref_clip_emb_paths = sorted(ref_clip_emb_paths)

        pair_data.append(dict(target=target_path, target_emb=(target_id_emb_path, target_clip_emb_path), ref_emb=(ref_id_emb_paths, ref_clip_emb_paths)))

        
    with open(os.path.join(args.save_dir, "train.json") , 'w') as f :
        for d in pair_data :
            json.dump(d , f)
            f.write('\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--ffhq_dir", type=str)
    parser.add_argument("--ffhq_emb_dir", type=str)
    parser.add_argument("--ffhqref_emb_dir", type=str)
    parser.add_argument("--save_dir", type=str)
    args = parser.parse_args()
    os.makedirs(args.save_dir , exist_ok=True)

    main(args)

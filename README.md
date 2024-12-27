# :fire: FaceMe: Robust Blind Face Restoration With Personal Identification (AAAI2025)

<a href=''><img src='https://img.shields.io/badge/Paper-arxiv-b31b1b.svg'></a> &nbsp;&nbsp;
<a href='https://modyu-liu.github.io/FaceMe_Homepage/'><img src='https://img.shields.io/badge/Project page-FaceMe-1bb41b.svg'></a> &nbsp;&nbsp;
<a href='https://huggingface.co/datasets/thomas2modyu/FaceMe'><img src='https://img.shields.io/badge/Dataset-huggingface-ffff00.svg'></a> &nbsp;&nbsp;
<a href=''><img src='https://img.shields.io/badge/Demo-huggingface-ffd700.svg'></a> &nbsp;&nbsp;



This is the official PyTorch codes for the paper:

>**FaceMe: Robust Blind Face Restoration With Personal Identification**<br>  [Siyu Liu<sup>1,*</sup>](https://github.com/modyu-liu), [Zhengpeng Duan<sup>1,*</sup>](https://adam-duan.github.io/), [Jia OuYang<sup>3</sup>](), [Jiayi Fu<sup>1</sup>](), [Hyunhee Park<sup>4</sup>](), [Zikun Liu<sup>3</sup>](), [Chunle Guo<sup>1,2,&dagger;</sup>](https://scholar.google.com/citations?user=RZLYwR0AAAAJ&hl=en), [Chongyi Li<sup>1,2</sup>](https://li-chongyi.github.io/) <br>
> <sup>1</sup> VCIP, CS, Nankai University, <sup>2</sup> NKIARI, Shenzhen Futian, <sup>3</sup> Samsung Research, China, Beijing (SRC-B), <sup>4</sup> The Department of Camera Innovation Group, Samsung Electronics
> <sup>*</sup>Denotes equal contribution, <sup>&dagger;</sup>Corresponding author.

![teaser_img](.assets/teaser.png)


:star: If FaceMe is helpful to your images or projects, please help star this repo. Thank you! :point_left:

---

## :boom: News

- **2024.12.27** Create this repo and release related code of our paper.

## :runner: TODO
- [ ] Launch an online demo
- [x] Release Checkpoints
- [x] Release the reference images of Wider-Test/LFW-Test/WebPhoto-Test we used 
- [x] Release FFHQRef dataset 
- [x] Release training and inference code


## :wrench: Dependencies and Installation

1. Clone repo

```bash
git clone https://github.com/modyu-liu/FaceMe
cd FaceMe 
```

2. Install packages
```bash
conda create -n faceme python=3.9
conda activate faceme
pip install -r requirements.txt
```


## :surfer: Quick Inference


**Step1: Download Checkpoints**

- Download the [[ControlNet and Mix](https://huggingface.co/thomas2modyu/FaceMe)] checkpoints and place them in the following directories: `weights/controlnet` and `weights/mix`.
- Download the [[antelopev2](https://github.com/deepinsight/insightface)] checkpoints and place it in the `models/` directory.
- All other required checkpoints will be downloaded automatically.

**Step2: Prepare testing data**

Place low-quality facial images in `[LQ DIR]:data/lq/` and high-quality reference facial images in `[REF DIR]:data/ref/`.

**Step 3: Running testing command**

```bash
python infer.py \
    --pretrained_model_name_or_path "SG161222/RealVisXL_V3.0" \  # Specify the pretrained model path
    --mix_path [MIX PATH] \                                    # Provide the path to the Mix‘s checkpoint
    --controlnet_model_name_or_path [CONTROLNET PATH] \        # Provide the path to the ControlNet's checkpoint
    --input_dir [LQ DIR] \                                     # Directory containing low-quality input images
    --ref_dir [REF DIR] \                                      # Directory containing high-quality reference images
    --result_dir [RESULT DIR] \                                # Directory to save the resulting outputs
    --color_correction \                                       # Apply color correction to the outputs
    --seed=233                                                 # Set a seed for reproducibility
```
Replace the placeholders `[MIX PATH]`, `[CONTROLNET PATH]`, `[LQ DIR]`, `[REF DIR]`, and `[RESULT DIR]` with their respective paths before running the command.

**Step 4: Check the results**

The processed results will be saved in the `[RESULT DIR]` directory.

**:seedling: Gradio Demo**
```bash
python demo.py \
    --pretrained_model_name_or_path "SG161222/RealVisXL_V3.0" \
    --mix_path [MIX PATH]\
    --controlnet_model_name_or_path [CONTROLNET PATH] \
```


## :muscle: Train

**Step1: Prepare training data**

The FaceMe model is trained using the  `FFHQ` dataset and its reference counterpart, `FFHQRef`. <a href='https://huggingface.co/datasets/thomas2modyu/FaceMe'><img src='https://img.shields.io/badge/Dataset-huggingface-ffff00.svg'></a> &nbsp;&nbsp;

**Dataset Structure**
```
-FFHQ DIR
    -00000.png
    -00001.png
    ...
-FFHQRef DIR
    -target_pose_0
        -target_pose_0_0
            -00000.png
            ...
        -target_pose_0_1
            -00000.png
            ...
        ...
    -target_pose_1
        -target_pose_1_0
            -00000.png
            ...
```

**Step2: Preprocess training data**

Using **ArcFace** and **CLIP image encoder** to extract identity embeddings.

**Command to Preprocess Data**
```bash
python utils/preprocess.py \
    --input_dir [FFHQ/FFHQRef DIR] \
    --id_emb_save_dir [SAVEDIR]/id_emb/ \
    --clip_emb_save_dir [SAVEDIR]/clip_emb/ \
    --dataset_name ['FFHQ'/'FFHQRef']
```
**Data Structure After Preprocessing**

```
-FFHQ SAVEDIR
    -id_emb
        -00000.npy
        -00001.npy
        ...
    -clip_emb
        -00000.npy
        -00001.npy
        ...
-FFHQRef SAVEDIR
    -id_emb
        -00000
            0_0.npy
            0_1.npy
            ...
        -00001
            ...
        ...
    -clip_emb
        -00000
            0_0.npy
            ...
        ...
```
**Step3: Create train json**

Generate a JSON file to record all training data paths for easy reference during training.

**Command to Create JSON**

```bash
python utils/create_train_json.py 
    --ffhq_dir [FFHQ DIR] # Directory containing the FFHQ dataset.
    --ffhq_emb_dir [FFHQ SAVEDIR] # Directory where FFHQ embeddings (id_emb and clip_emb) are saved.
    --ffhqref_emb_dir [FFHQRef SAVEDIR] # Directory where FFHQRef embeddings (id_emb and clip_emb) are saved.
    --save_dir [JSON SAVEDIR] # Path to save the generated JSON file.
```
**Step4: Start train**

Use the following command to start the training process:

```bash
accelerate launch train.py \
    --pretrained_model_name_or_path "SG161222/RealVisXL_V3.0" \
    --mix_pretrained_path [optional]{Stage1:None, Stage2:[YOUR SAVEDIR]} \ #  Path to the pretrained Mix model. For Stage 1, use None. for Stage 2, provide the directory path [YOUR SAVEDIR].
    --output_dir [YOUR SAVEDIR] \ # Directory to save the training outputs, such as model checkpoints.
    --train_data_dir [JSON SAVEDIR]/train.json \ # Path to the JSON file containing all training data paths (train.json created in Step 3). 
    --resolution 512 \
    --report_to "wandb" \
    --learning_rate 5e-5 \
    --train_batch_size 4 \
    --mixed_precision fp16 \
    --num_workers 4 \
    --gradient_accumulation_steps 1 \
    --num_train_epochs 100 \
    --checkpoint_steps 10000 \
```

## :book: Citation

If you find our repo useful for your research, please consider citing our paper:

```bibtex
@misc{ 
}
```

## :postbox: Contact

For technical questions, please contact `liusiyu29[AT]mail.nankai.edu.cn`


![visitors](https://visitor-badge.laobi.icu/badge?page_id=modyu-liu/FaceMe)

# [ICCV'25] TARO: Timestep-Adaptive Representation Alignment with Onset-Aware Conditioning for Synchronized Video-to-Audio Synthesis
<br>

**[Tri Ton](https://triton99.github.io/)<sup>1</sup>, [Ji Woo Hong](https://jiwoohong93.github.io/)<sup>1</sup>, [Chang D. Yoo](https://sanctusfactory.com/family.php)<sup>1†</sup>** 
<br>
<sup>1</sup>KAIST, South Korea
<br>
†Corresponding authors

<p align="center">
        <a href="https://triton99.github.io/taro-site/" target='_blank'>
        <img src="https://img.shields.io/badge/🐳-Project%20Page-blue">
        </a>
        <a href="https://arxiv.org/abs/2504.05684" target='_blank'>
        <img src="https://img.shields.io/badge/arXiv-2312.13528-b31b1b.svg">
        </a>
        <img alt="GitHub Repo stars" src="https://img.shields.io/github/stars/triton99/TARO">
</p>

## 📣 News
- **[09/2025]**: Training & Inference code released.
- **[06/2025]**: TARO accepted to ICCV 2025 🎉.
- **[04/2024]**: Paper uploaded to arXiv. Check out the manuscript [here](https://arxiv.org/abs/2504.05684).(https://arxiv.org/abs/2504.05684).

## To-Dos
- [x] Release model weights on Google Drive.
- [x] Release inference code
- [x] Release training code & dataset preparation

## ⚙️ Environmental Setups
1. Clone TARO.
```bash
git clone https://github.com/triton99/TARO
cd TARO
```

2. Create the environment.
```bash
conda create -n taro python==3.10
conda activate taro
pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu121

# Training
pip install --force pip==24.0
git clone https://github.com/pytorch/fairseq
cd fairseq
pip install --editable ./ --no-build-isolation
cd ..

git clone https://github.com/cwx-worst-one/EAT.git

# Inference
pip3 install -r requirements.txt
```

## 📁 Data Preparations
Please download the [VGGSound dataset](https://www.robots.ox.ac.uk/~vgg/data/vggsound/), extract the videos, and organize them into two folders: one with .mp4 files and one with corresponding .wav files (matching base filenames). 

Update the path variables at the top of the preprocessing scripts to point to your folders, then run:
```bash
./preprocess_video.sh

./preprocess_audio.sh
```

After processing, the data will have the following structure:
```bash
VGGSound/train
    ├── videos
    │   ├── abc.mp4
    │   └── ...
    ├── audios
    │   ├── abc.wav
    │   └── ...
    ├── cavp_feats
    │   ├── abc.npz
    │   └── ...
    ├── onset_feats
    │   ├── abc.npz
    │   └── ...
    ├── melspec
    │   ├── abc.npy
    │   └── ...
    └── fbank
    │   ├── abc.npy
    │   └── ...
```


## 🚀 Getting Started

### Download Checkpoints

The pretrained TARO checkpoint can be downloaded on [Google Drive](https://drive.google.com/drive/folders/1YqLsEtVYeSchhAh-wKS-BWuB6MK6_mJB?usp=sharing).

The CAVP checkpoint can be downloaded from [Diff-Foley](https://github.com/luosiallen/Diff-Foley).

The onset checkpoint can be downloaded from [SyncFusion](https://github.com/mcomunita/syncfusion).

### Training
```bash
./train.sh
```

### Inference
To run the inference code, you can use the following command:
```bash
python infer.py \
    --video_path ./test.mp4 \
    --save_folder_path ./output \
    --cavp_config_path ./cavp/model/cavp.yaml \
    --cavp_ckpt_path ./cavp_epoch66.ckpt \
    --onset_ckpt_path ./onset_model.ckpt \
    --model_ckpt_path ./taro_ckpt.pt
```

## 📖 Citing TARO

If you find our repository useful, please consider giving it a star ⭐ and citing our paper in your work:

```bibtex
@inproceedings{ton2025taro,
  title     = {TARO: Timestep-Adaptive Representation Alignment with Onset-Aware Conditioning for Synchronized Video-to-Audio Synthesis},
  author    = {Ton, Tri and Hong, Ji Woo and Yoo, Chang D},
  year      = {2025},
  booktitle = {International Conference on Computer Vision (ICCV)},
}
```

## 🤗 Acknowledgements

Our code is based on [REPA](https://github.com/sihyun-yu/REPA), [Diff-Foley](https://github.com/luosiallen/Diff-Foley), and [SyncFusion](https://github.com/mcomunita/syncfusion). We thank the authors for their excellent work!

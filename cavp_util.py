import torch
import subprocess
from pathlib import Path
import os
import cv2
import numpy as np
import torchvision.transforms as transforms
from PIL import Image
from tqdm import tqdm
from omegaconf import OmegaConf
import importlib


def which_ffmpeg() -> str:
    '''Determines the path to ffmpeg library

    Returns:
        str -- path to the library
    '''
    result = subprocess.run(['which', 'ffmpeg'], stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    ffmpeg_path = result.stdout.decode('utf-8').replace('\n', '')
    return ffmpeg_path

def reencode_video_with_diff_fps(video_path: str, tmp_path: str, extraction_fps: int, start_second, truncate_second) -> str:
    '''Reencodes the video given the path and saves it to the tmp_path folder.

    Args:
        video_path (str): original video
        tmp_path (str): the folder where tmp files are stored (will be appended with a proper filename).
        extraction_fps (int): target fps value

    Returns:
        str: The path where the tmp file is stored. To be used to load the video from
    '''
    assert which_ffmpeg() != '', 'Is ffmpeg installed? Check if the conda environment is activated.'
    os.makedirs(tmp_path, exist_ok=True)

    # form the path to tmp directory
    new_path = os.path.join(tmp_path, f'{Path(video_path).stem}_new_fps_{str(extraction_fps)}_truncate_{start_second}_{truncate_second}.mp4')
    cmd = f'{which_ffmpeg()} -hide_banner -loglevel panic '
    cmd += f'-y -ss {start_second} -t {truncate_second} -i {video_path} -an -filter:v fps=fps={extraction_fps} {new_path}'
    subprocess.call(cmd.split())
    return new_path

def instantiate_from_config(config, reload=False):
    if not "target" in config:
        if config == '__is_first_stage__':
            return None
        elif config == "__is_unconditional__":
            return None
        raise KeyError("Expected key `target` to instantiate.")
    return get_obj_from_str(config["target"], reload=reload)(**config.get("params", dict()))

def get_obj_from_str(string, reload=False):
    module, cls = string.rsplit(".", 1)
    if reload:
        module_imp = importlib.import_module(module)
        importlib.reload(module_imp)
    return getattr(importlib.import_module(module, package=None), cls)


class Extract_CAVP_Features(torch.nn.Module):

    def __init__(self, device=None, tmp_path="./", video_shape=(224,224), config_path=None, ckpt_path=None):
        super(Extract_CAVP_Features, self).__init__()
        self.fps = 4
        self.batch_size = 40
        self.device = device
        self.tmp_path = tmp_path

        # Initalize CAVP model:
        config = OmegaConf.load(config_path)
        self.stage1_model = instantiate_from_config(config.model).to(device)

        # Loading Model from:
        assert ckpt_path is not None
        self.init_first_from_ckpt(ckpt_path)
        self.stage1_model.eval()
        
        # Transform:
        self.img_transform = transforms.Compose([
            transforms.Resize(video_shape),
            transforms.ToTensor(),
        ])


    def init_first_from_ckpt(self, path):
        model = torch.load(path, map_location="cpu")
        if "state_dict" in list(model.keys()):
            model = model["state_dict"]
        # Remove: module prefix
        new_model = {}
        for key in model.keys():
            new_key = key.replace("module.","")
            new_model[new_key] = model[key]
        self.stage1_model.load_state_dict(new_model, strict=False)


    @torch.no_grad()
    def forward(self, video_path, tmp_path="./tmp_folder"):
        start_second = 0
        truncate_second = 10
        self.tmp_path = tmp_path

        # Load the video, change fps:
        video_path_low_fps = reencode_video_with_diff_fps(video_path, self.tmp_path, self.fps, start_second, truncate_second)

        # read the video:
        cap = cv2.VideoCapture(video_path_low_fps)

        feat_batch_list = []
        video_feats = []
        first_frame = True
        # pbar = tqdm(cap.get(7))
        i = 0
        while cap.isOpened():
            i += 1
            # pbar.set_description("Processing Frames: {} Total: {}".format(i, cap.get(7)))
            frames_exists, rgb = cap.read()
            
            if first_frame:
                if not frames_exists:
                    continue
            first_frame = False

            if frames_exists:
                rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
                rgb_tensor = self.img_transform(Image.fromarray(rgb)).unsqueeze(0).to(self.device)
                feat_batch_list.append(rgb_tensor)      # 32 x 3 x 224 x 224
                
                # Forward:
                if len(feat_batch_list) == self.batch_size:
                    # Stage1 Model:
                    input_feats = torch.cat(feat_batch_list,0).unsqueeze(0).to(self.device)
                    contrastive_video_feats = self.stage1_model.encode_video(input_feats, normalize=True, pool=False)
                    video_feats.extend(contrastive_video_feats.detach().cpu().numpy())
                    feat_batch_list = []
            else:
                if len(feat_batch_list) != 0:
                    input_feats = torch.cat(feat_batch_list,0).unsqueeze(0).to(self.device)
                    contrastive_video_feats = self.stage1_model.encode_video(input_feats, normalize=True, pool=False)
                    video_feats.extend(contrastive_video_feats.detach().cpu().numpy())
                cap.release()
                break

        # Remove the file
        os.remove(video_path_low_fps)
        video_contrastive_feats = np.concatenate(video_feats)
        return video_contrastive_feats


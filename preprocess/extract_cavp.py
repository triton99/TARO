import os
import torch
import numpy as np
from tqdm import tqdm
from argparse import ArgumentParser

import sys
sys.path.append(os.getcwd())
from cavp_util import Extract_CAVP_Features

def main():
    parser = ArgumentParser(description="Inference script parameters")
    parser.add_argument("--video_folder_path", type=str, default="./input_videos", required=True, help="Path to the input video folder")
    parser.add_argument("--save_folder_path", type=str, default="./output", help="Folder to save output files")
    parser.add_argument("--cavp_config_path", type=str, default="./cavp.yaml", help="Path to CAVP config file")
    parser.add_argument("--cavp_ckpt_path", type=str, default="./cavp_epoch66.ckpt", help="Path to CAVP checkpoint file")

    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    extract_cavp = Extract_CAVP_Features(device=device, config_path=args.cavp_config_path, ckpt_path=args.cavp_ckpt_path)

    os.makedirs(os.path.join(args.save_folder_path, "cavp_feats"), exist_ok=True)

    data_list = [file for file in os.listdir(args.video_folder_path) if file.endswith(".mp4")]
    data_list = sorted(data_list)

    for _, video_file in enumerate(tqdm(data_list, desc="Extracting CAVP features", total=len(data_list))):
        video_path = os.path.join(args.video_folder_path, video_file)
        try:
            cavp_feats = extract_cavp(video_path, tmp_path=args.save_folder_path)
            # Save cavp_feats as npz file
            base_name = os.path.splitext(os.path.basename(video_file))[0]
            np.savez(os.path.join(args.save_folder_path, "cavp_feats", f"{base_name}.npz"), cavp_feats)
        except Exception as e:
            print(f"Error processing {video_file}: {e}")

    print("========================================FINISH CAVP EXTRACTION===========================================")


if __name__ == "__main__":
    main()

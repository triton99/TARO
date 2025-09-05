import os
import torch
import numpy as np
from tqdm import tqdm
from argparse import ArgumentParser

import sys
sys.path.append(os.getcwd())
from onset_util import VideoOnsetNet, extract_onset

def main():
    parser = ArgumentParser(description="Inference script parameters")
    parser.add_argument("--video_folder_path", type=str, default="./input_videos", required=True, help="Path to the input video folder")
    parser.add_argument("--save_folder_path", type=str, default="./output", help="Folder to save output files")
    parser.add_argument("--onset_ckpt_path", type=str, default="./onset_ckpt.ckpt", help="Path to onset checkpoint")

    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    # Load the pre-trained onset detection model
    state_dict = torch.load(args.onset_ckpt_path)["state_dict"]
    new_state_dict = {}
    for key, value in state_dict.items():
        if "model.net.model" in key:
            new_key = key.replace("model.net.model", "net.model")  # Adjust the key as needed
        elif "model.fc." in key:
            new_key = key.replace("model.fc", "fc")  # Adjust the key as needed
        new_state_dict[new_key] = value
    onset_model = VideoOnsetNet(False).to(device)
    onset_model.load_state_dict(new_state_dict)
    onset_model.eval()

    os.makedirs(os.path.join(args.save_folder_path, "onset_feats"), exist_ok=True)

    data_list = [file for file in os.listdir(args.video_folder_path) if file.endswith(".mp4")]
    data_list = sorted(data_list)

    for _, video_file in enumerate(tqdm(data_list, desc="Extracting Onset features", total=len(data_list))):
        video_path = os.path.join(args.video_folder_path, video_file)
        try:
            onset_feats = extract_onset(video_path, onset_model, tmp_path=args.save_folder_path, device=device)
            # Save cavp_feats as npz file
            base_name = os.path.splitext(os.path.basename(video_file))[0]
            np.savez(os.path.join(args.save_folder_path, "onset_feats", f"{base_name}.npz"), onset_feats)
        except Exception as e:
            print(f"Error processing {video_file}: {e}")

    print("========================================FINISH CAVP EXTRACTION===========================================")


if __name__ == "__main__":
    main()

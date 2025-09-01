import torch
import os
import numpy as np
import random
import soundfile as sf
import ffmpeg

from argparse import ArgumentParser
from diffusers import AudioLDM2Pipeline
from models import MMDiT
from samplers import euler_sampler, euler_maruyama_sampler
from cavp_util import Extract_CAVP_Features 
from onset_util import extract_onset

def set_global_seed(seed):
    np.random.seed(seed % (2**32))
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

def main():
    parser = ArgumentParser(description="Inference script parameters")
    parser.add_argument("--video_path", type=str, default="./test.mp4", required=True, help="Path to the input video file")
    parser.add_argument("--save_folder_path", type=str, default="./output", help="Folder to save output files")
    parser.add_argument("--cavp_config_path", type=str, default="./cavp.yaml", help="Path to CAVP config file")
    parser.add_argument("--cavp_ckpt_path", type=str, default="./cavp_epoch66.ckpt", help="Path to CAVP checkpoint file")
    parser.add_argument("--onset_ckpt_path", type=str, default="./onset_model.ckpt", help="Path to onset model checkpoint file")
    parser.add_argument("--model_ckpt_path", type=str, default="./taro_ckpt.pt", help="Path to MMDiT model checkpoint file")

    args = parser.parse_args()
    os.makedirs(args.save_folder_path, exist_ok=True)

    seed = 0
    set_global_seed(seed)
    torch.set_grad_enabled(False)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    weight_dtype = torch.bfloat16

    # Load models
    extract_cavp = Extract_CAVP_Features(device=device, config_path=args.cavp_config_path, ckpt_path=args.cavp_ckpt_path)

    model = MMDiT(
        adm_in_channels=120,
        z_dims = [768],
        encoder_depth=4,
    ).to(device)

    state_dict = torch.load(args.model_ckpt_path, map_location=device)['ema']
    model.load_state_dict(state_dict)
    model.eval()
    model.to(weight_dtype)
    model_audioldm = AudioLDM2Pipeline.from_pretrained("cvssp/audioldm2")
    vae = model_audioldm.vae.to(device)
    vae.eval()

    vocoder = model_audioldm.vocoder.to(device)
    
    # Extract Features
    video_name = os.path.basename(args.video_path).split(".")[0]

    cavp_feats = extract_cavp(args.video_path, tmp_path=args.save_folder_path)
    onset_feats = extract_onset(args.video_path, args.onset_ckpt_path, tmp_path=args.save_folder_path, device=device)

    # Parameters for inference
    sr = 16000
    truncate = 131072
    fps = 4

    truncate_frame = int(fps * truncate / sr)
    truncate_onset = 120

    cfg_scale = 8
    mode = "sde"
    num_steps = 25
    heun = False
    guidance_low = 0.0
    guidance_high = 0.7
    path_type = "linear"

    latent_size = (204, 16)
    latents_scale = torch.tensor(
        [0.18215, 0.18215, 0.18215, 0.18215, 0.18215, 0.18215, 0.18215, 0.18215]
        ).view(1, 8, 1, 1).to(device)

    # Start inference
    video_feats = torch.from_numpy(cavp_feats[:truncate_frame]).unsqueeze(0).to(device).to(weight_dtype)
    onset_feats = torch.from_numpy(onset_feats[:truncate_onset]).unsqueeze(0).to(device).to(weight_dtype)

    z = torch.randn(len(video_feats), model.in_channels, latent_size[0], latent_size[1], device=device).to(weight_dtype)

    # Sample audios
    sampling_kwargs = dict(
        model=model, 
        latents=z,
        y=onset_feats,
        context=video_feats,
        num_steps=num_steps, 
        heun=heun,
        cfg_scale=cfg_scale,
        guidance_low=guidance_low,
        guidance_high=guidance_high,
        path_type=path_type,
    )

    with torch.no_grad():
        if mode == "sde":
            samples = euler_maruyama_sampler(**sampling_kwargs)
        elif mode == "ode":
            samples = euler_sampler(**sampling_kwargs)
        else:
            raise NotImplementedError()

        samples = vae.decode(samples / latents_scale).sample
        wav_samples = vocoder(samples.squeeze()).detach().cpu().numpy()

        # Save the audio
        sf.write(os.path.join(args.save_folder_path, video_name + ".wav"), wav_samples, sr)

        # Save the video with the generated audio
        trimmed_video_file_path = os.path.join(args.save_folder_path, video_name + "_trimmed.mp4")
        trimmed_audio_file_path = os.path.join(args.save_folder_path, video_name + ".wav")
        output_path = os.path.join(args.save_folder_path, video_name + "_wa.mp4")

        # Trim the video to match the audio duration
        ffmpeg.input(args.video_path, ss=0, t=truncate / sr).output(trimmed_video_file_path, vcodec='libx264', an=None).run(overwrite_output=True)

        # Combine trimmed video and generated audio
        input_video = ffmpeg.input(trimmed_video_file_path)
        input_audio = ffmpeg.input(trimmed_audio_file_path)
        ffmpeg.output(input_video, input_audio, output_path, vcodec='libx264', acodec='aac', strict='experimental').run(overwrite_output=True)
        os.remove(trimmed_video_file_path)

    print("========================================FINISH INFERENCE===========================================")

if __name__ == "__main__":
    main()


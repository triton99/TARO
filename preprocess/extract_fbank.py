import os
import torch
import torchaudio
import numpy as np
from tqdm import tqdm
import soundfile as sf
from argparse import ArgumentParser

def main():
    parser = ArgumentParser(description="Inference script parameters")
    parser.add_argument("--wav_folder_path", type=str, default="./input_wavs", required=True, help="Path to the input video folder")
    parser.add_argument("--save_folder_path", type=str, default="./output", help="Folder to save output files")

    args = parser.parse_args()

    os.makedirs(os.path.join(args.save_folder_path, "fbank"), exist_ok=True)

    target_length = 1024
    norm_mean = -4.268
    norm_std = 4.569

    # Loop over all .wav files in the audio folder
    for filename in tqdm(os.listdir(args.wav_folder_path)):
        if filename.endswith('.wav'):
            # Load the audio file
            source_file = os.path.join(args.wav_folder_path, filename)
            wav, sr = sf.read(source_file)
            if len(wav.shape) > 1:
                wav = wav[:, 0]

            source = torch.from_numpy(wav).float()
            if not sr == 16e3:
                source = torchaudio.functional.resample(source, orig_freq=sr, new_freq=16000).float()

            source = source - source.mean()
            source = source.unsqueeze(dim=0)
            source = torchaudio.compliance.kaldi.fbank(source, htk_compat=True, sample_frequency=16000, use_energy=False,
                window_type='hanning', num_mel_bins=128, dither=0.0, frame_shift=10).unsqueeze(dim=0)

            n_frames = source.shape[1]
            diff = target_length - n_frames
            if diff > 0:
                m = torch.nn.ZeroPad2d((0, 0, 0, diff)) 
                source = m(source)
            elif diff < 0:
                source = source[:,0:target_length, :]
            source = (source - norm_mean) / (norm_std * 2)

            # Save the spectrogram as .npy file
            output_filename = os.path.splitext(filename)[0] + '.npy'
            output_path = os.path.join(args.save_folder_path, "fbank", output_filename)

            np.save(output_path, source.squeeze(0).numpy())

    print("========================================FINISH FBANK EXTRACTION===========================================")

if __name__ == "__main__":
    main()

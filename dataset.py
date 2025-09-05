import os

import numpy as np
import torch
import random
import math

from torch.utils.data import Dataset

class audio_video_spec_fullset_Dataset(Dataset):
    # Only Load audio dataset: for training Stage1: Audio Npy Dataset
    def __init__(self, split, data_dir):
        super().__init__()
        debug_num=False

        if split == "train":
            self.split = "train"
        elif split == "valid" or split == 'test':
            self.split = "test"

        # Default params:
        self.min_duration = 2
        self.sr = 16000
        self.duration = 10
        self.truncate = 130560
        self.fps = 4
        self.fix_frames = False
        self.hop_len = 160
        self.onset_truncate = 120


        # spec_dir: spectrogram path
        # feat_dir: CAVP feature path
        # fbank_dir: fbank feature path
        # onset_dir: onset feature path
        dataset_spec_dir = os.path.join(data_dir, "melspec", self.split)
        dataset_feat_dir = os.path.join(data_dir, "cavp_feats", self.split)
        dataset_fbank_dir = os.path.join(data_dir, "fbank", self.split)
        dataset_onset_dir = os.path.join(data_dir, "onset_feats", "train")
        list_onset = os.listdir(dataset_onset_dir)
        list_onset = list(map(lambda x: x.split('.')[0], list_onset))


        with open(os.path.join(data_dir, '{}_list.txt'.format(self.split)), "r") as f:
            data_list = f.readlines()
            data_list = list(map(lambda x: x.strip(), data_list))
            data_list = list(set(data_list) & set(list_onset))

            spec_list = list(map(lambda x: os.path.join(dataset_spec_dir, x) + ".npy", data_list))      # spec
            feat_list = list(map(lambda x: os.path.join(dataset_feat_dir, x) + ".npz",     data_list))      # feat
            fbank_list = list(map(lambda x: os.path.join(dataset_fbank_dir, x) + ".npy",     data_list))      # fbank
            onset_list = list(map(lambda x: os.path.join(dataset_onset_dir, x) + ".npy",     data_list))      # onset


        # Merge Data:
        self.data_list = data_list
        self.spec_list = spec_list 
        self.feat_list = feat_list
        self.fbank_list = fbank_list
        self.onset_list = onset_list


        assert len(self.data_list) == len(self.spec_list) == len(self.feat_list)


        shuffle_idx = np.random.permutation(np.arange(len(self.data_list)))
        self.data_list = [self.data_list[i] for i in shuffle_idx]
        self.spec_list = [self.spec_list[i] for i in shuffle_idx]
        self.feat_list = [self.feat_list[i] for i in shuffle_idx]
        self.fbank_list = [self.fbank_list[i] for i in shuffle_idx]
        self.onset_list = [self.onset_list[i] for i in shuffle_idx]


        if debug_num:
            self.data_list = self.data_list[:debug_num]
            self.spec_list = self.spec_list[:debug_num]
            self.feat_list = self.feat_list[:debug_num]
            self.fbank_list = self.fbank_list[:debug_num]
            self.onset_list = self.onset_list[:debug_num]
        print('Split: {}  Sample Num: {}'.format(split, len(self.data_list)))


    def __len__(self):
        return len(self.data_list)
    

    def load_spec_and_feat(self, spec_path, video_feat_path, fbank_path, onset_path):
        """Load audio spec and video feat"""
        spec_raw = np.load(spec_path).astype(np.float32).T                    # channel: 1
        video_feat = np.load(video_feat_path)['arr_0'].astype(np.float32)
        fbank = np.load(fbank_path).astype(np.float32)
        onset = np.load(onset_path).astype(np.float32).reshape(-1)


        # Padding the samples:
        spec_len = self.sr * self.duration / self.hop_len
        fbank_len = int(spec_len / spec_raw.shape[1] * len(fbank))
        if spec_raw.shape[1] < spec_len:
            fbank = np.tile(fbank, (math.ceil(spec_len / spec_raw.shape[1]), 1))
            spec_raw = np.tile(spec_raw, math.ceil(spec_len / spec_raw.shape[1]))
        spec_raw = spec_raw[:, :int(spec_len)]
        fbank = fbank[:fbank_len]
        
        feat_len = self.fps * self.duration
        if video_feat.shape[0] < feat_len:
            video_feat = np.tile(video_feat, (math.ceil(feat_len / video_feat.shape[0]), 1))
        video_feat = video_feat[:int(feat_len)]

        onset_len = 15 * self.duration
        if onset.shape[0] < onset_len:
            onset = np.tile(onset, (math.ceil(onset_len / onset.shape[0])))
        onset = onset[:int(onset_len)]

        return spec_raw, video_feat, fbank, onset


    def mix_audio_and_feat(self, spec1=None, spec2=None, video_feat1=None, video_feat2=None, fbank1=None, fbank2=None, onset1=None, onset2=None, video_info_dict={}, mode='single'):
        """ Return Mix Spec and Mix video feat"""
        if mode == "single":
            # spec1:
            if not self.fix_frames:
                start_idx = random.randint(0, self.sr * self.duration - self.truncate - 1)  # audio start
            else:
                start_idx = 0

            start_frame = int(self.fps * start_idx / self.sr)
            truncate_frame = int(self.fps * self.truncate / self.sr)

            start_onset = int(15 * start_idx / self.sr)
            truncate_onset = self.onset_truncate

            # Spec Start & Truncate:
            spec_start = int(start_idx / self.hop_len)
            spec_truncate = int(self.truncate / self.hop_len)

            # Fbank Start & Truncate:
            fbank_start = int((spec_start / spec1.shape[1]) * len(fbank1))
            fbank_truncate = int((spec_truncate / spec1.shape[1]) * len(fbank1))

            spec1 = spec1[:, spec_start : spec_start + spec_truncate]
            video_feat1 = video_feat1[start_frame: start_frame + truncate_frame]
            fbank1 = fbank1[fbank_start: fbank_start + fbank_truncate]
            onset1 = onset1[start_onset: start_onset + truncate_onset]

            # info_dict:
            video_info_dict['video_time1'] = str(start_frame) + '_' + str(start_frame+truncate_frame)   # Start frame, end frame
            video_info_dict['video_time2'] = ""
            return spec1, video_feat1, fbank1, onset1, video_info_dict
        
        elif mode == "concat":
            total_spec_len = int(self.truncate / self.hop_len)
            # Random Trucate len:
            spec1_truncate_len = random.randint(self.min_duration * self.sr // self.hop_len, total_spec_len - self.min_duration * self.sr // self.hop_len - 1)
            spec2_truncate_len = total_spec_len - spec1_truncate_len

            # Sample spec clip:
            spec_start1 = random.randint(0, total_spec_len - spec1_truncate_len - 1)
            spec_start2 = random.randint(0, total_spec_len - spec2_truncate_len - 1)
            spec_end1, spec_end2 = spec_start1 + spec1_truncate_len, spec_start2 + spec2_truncate_len

            start1_fbank, truncate1_fbank = int((spec_start1 / spec1.shape[1]) * len(fbank1)), int((spec1_truncate_len / spec1.shape[1]) * len(fbank1))
            start2_fbank, truncate2_fbank = int((spec_start2 / spec2.shape[1]) * len(fbank2)), int((spec2_truncate_len / spec2.shape[1]) * len(fbank2))

            # concat spec:
            spec1, spec2 = spec1[:, spec_start1 : spec_end1], spec2[:, spec_start2 : spec_end2]
            concat_audio_spec = np.concatenate([spec1, spec2], axis=1)  

            # Concat Video Feat:
            start1_frame, truncate1_frame = int(self.fps * spec_start1 * self.hop_len / self.sr), int(self.fps * spec1_truncate_len * self.hop_len / self.sr)
            start2_frame, truncate2_frame = int(self.fps * spec_start2 * self.hop_len / self.sr), int(self.fps * self.truncate / self.sr) - truncate1_frame
            video_feat1, video_feat2 = video_feat1[start1_frame : start1_frame + truncate1_frame], video_feat2[start2_frame : start2_frame + truncate2_frame]
            concat_video_feat = np.concatenate([video_feat1, video_feat2])

            # Concat Fbank:
            fbank1, fbank2 = fbank1[start1_fbank : start1_fbank + truncate1_fbank], fbank2[start2_fbank : start2_fbank + truncate2_fbank]
            concat_fbank = np.concatenate([fbank1, fbank2])

            # Concat Onset:
            start1_onset, truncate1_onset = int(15 * spec_start1 * self.hop_len / self.sr), int(15 * spec1_truncate_len * self.hop_len / self.sr)
            start2_onset, truncate2_onset = int(15 * spec_start2 * self.hop_len / self.sr), self.onset_truncate - truncate1_onset
            onset_feat1, onset_feat2 = onset1[start1_onset : start1_onset + truncate1_onset], onset2[start2_onset : start2_onset + truncate2_onset]
            concat_onset = np.concatenate([onset_feat1, onset_feat2])

            video_info_dict['video_time1'] = str(start1_frame) + '_' + str(start1_frame+truncate1_frame)   # Start frame, end frame
            video_info_dict['video_time2'] = str(start2_frame) + '_' + str(start2_frame+truncate2_frame)
            return concat_audio_spec, concat_video_feat, concat_fbank, concat_onset, video_info_dict



    def __getitem__(self, idx):
        audio_name1 = self.data_list[idx]
        spec_npy_path1 = self.spec_list[idx]
        video_feat_path1 = self.feat_list[idx]
        fbank_path1 = self.fbank_list[idx]
        onset_path1 = self.onset_list[idx]


        # select other video:
        flag = False
        if random.uniform(0, 1) < 0.5:
            flag = True
            random_idx = idx
            while random_idx == idx:
                random_idx = random.randint(0, len(self.data_list)-1)
            audio_name2 = self.data_list[random_idx]
            spec_npy_path2 = self.spec_list[random_idx]
            video_feat_path2 = self.feat_list[random_idx]
            fbank_path2 = self.fbank_list[random_idx]
            onset_path2 = self.onset_list[random_idx]


        # Load the Spec and Feat:
        spec1, video_feat1, fbank1, onset1 = self.load_spec_and_feat(spec_npy_path1, video_feat_path1, fbank_path1, onset_path1)

        if flag:
            spec2, video_feat2, fbank2, onset2 = self.load_spec_and_feat(spec_npy_path2, video_feat_path2, fbank_path2, onset_path2)
            video_info_dict = {'audio_name1':audio_name1, 'audio_name2': audio_name2}
            mix_spec, mix_video_feat, mix_fbank, mix_onset, mix_info = self.mix_audio_and_feat(spec1, spec2, video_feat1, video_feat2, fbank1, fbank2, onset1, onset2, video_info_dict, mode='concat')
        else:
            video_info_dict = {'audio_name1':audio_name1, 'audio_name2': ""}
            mix_spec, mix_video_feat, mix_fbank, mix_onset, mix_info = self.mix_audio_and_feat(spec1=spec1, video_feat1=video_feat1, fbank1=fbank1, onset1=onset1, video_info_dict=video_info_dict, mode='single')


        norm_mean = -4.268
        norm_std = 4.569
        target_length = 1024
        n_frames = mix_fbank.shape[0]
        mix_fbank = torch.from_numpy(mix_fbank).contiguous()
        diff = target_length - n_frames
        if diff > 0:
            m = torch.nn.ZeroPad2d((0, 0, 0, diff))
            mix_fbank = m(mix_fbank)
            mix_fbank[n_frames:] = (mix_fbank[n_frames:] - norm_mean) / (norm_std * 2)
        elif diff < 0:
            mix_fbank = mix_fbank[0:target_length, :]

        mix_spec = mix_spec[None]
        mix_spec = torch.from_numpy(mix_spec).contiguous()
        mix_video_feat = torch.from_numpy(mix_video_feat).contiguous()
        mix_onset = torch.from_numpy(mix_onset).contiguous()

        data_dict = {}
        data_dict['mix_spec'] = mix_spec
        data_dict['mix_video_feat'] = mix_video_feat
        data_dict['mix_fbank'] = mix_fbank
        data_dict['mix_onset'] = mix_onset
        data_dict['mix_info_dict'] = mix_info     
        return data_dict



class audio_video_spec_fullset_Dataset_Train(audio_video_spec_fullset_Dataset):
    def __init__(self, data_dir):
        super().__init__(split='train', data_dir=data_dir)



def collate_fn_taro(data):
    mix_spec = torch.stack([example["mix_spec"] for example in data])
    mix_video_feat = torch.stack([example["mix_video_feat"] for example in data])
    mix_fbank = torch.stack([example["mix_fbank"] for example in data])
    mix_onset = torch.stack([example["mix_onset"] for example in data])
    mix_info_dict = [example["mix_info_dict"] for example in data]   

    return {
        "mix_spec": mix_spec,
        "mix_video_feat": mix_video_feat,
        "mix_fbank": mix_fbank,
        "mix_onset": mix_onset,
        "mix_info_dict": mix_info_dict,
    }



import os
from modules.basic_utils import load_json
from torch.utils.data import Dataset
from config.base_config import Config
from datasets.video_capture import VideoCapture


class ANetDataset(Dataset):
    """
        videos_dir: directory where all videos are stored
        config: AllConfig object
        split_type: 'train'/'test'
        img_transforms: Composition of transforms
        Notes: for test split, we return one video, caption pair for each caption belonging to that video
               so when we run test inference for t2v task we simply average on all these pairs.
    """

    def __init__(self, config: Config, split_type = 'train', img_transforms=None):
        self.config = config
        self.videos_dir = config.videos_dir
        self.img_transforms = img_transforms
        self.split_type = split_type

        train_db_file = 'data/ActivityNet/train.json'
        test_db_file = 'data/ActivityNet/val_1.json'

        train_file = 'data/ActivityNet/train_ids.json'
        test_file = 'data/ActivityNet/val_ids.json'

        # build vid to paragraph
        temp_dict = load_json(train_db_file)
        temp_dict.update(load_json(test_db_file))
       
        self.vid2caption = {}
        for key, value in temp_dict.items():
            paragraph = " ".join(value["sentences"])

            self.vid2caption[key] = paragraph

        if split_type == 'train':
            self.train_vids = load_json(train_file)
            self._construct_all_train_pairs()
            self.num_frames = self.config.num_frames
            self.video_sample_type = self.config.video_sample_type
        else:
            self.test_vids = load_json(test_file)
            self._construct_all_test_pairs()
            self.num_frames = self.config.num_test_frames
            self.video_sample_type = self.config.video_sample_type_test

    def __getitem__(self, index):
        if self.split_type == 'train':
            video_path, caption, video_id = self._get_vidpath_and_caption_by_index_train(index)
        else:
            video_path, caption, video_id = self._get_vidpath_and_caption_by_index_test(index)

        imgs, idxs = VideoCapture.load_frames_from_video(video_path,
                                                         self.num_frames, 
                                                         self.config.num_prompts,
                                                         self.config.video_sample_type)

        # process images of video
        if self.img_transforms is not None:
            imgs = self.img_transforms(imgs)

        ret = {
            'video_id': video_id,
            'video': imgs,
            'text': caption
        }

        return ret

    def _get_vidpath_and_caption_by_index_train(self, index):
        vid, caption = self.all_train_pairs[index]

        video_path = os.path.join(self.videos_dir, vid + '.mp4')
        if not os.path.exists(video_path):
            video_path = os.path.join(self.videos_dir, vid + '.mkv')
        return video_path, caption, vid

    def _get_vidpath_and_caption_by_index_test(self, index):
        vid, caption = self.all_test_pairs[index]
        video_path = os.path.join(self.videos_dir, vid + '.mp4')
        if not os.path.exists(video_path):
            video_path = os.path.join(self.videos_dir, vid + '.mkv')
        return video_path, caption, vid

    def __len__(self):
        if self.split_type == 'train':
            return len(self.all_train_pairs)
        return len(self.all_test_pairs)

    def _construct_all_train_pairs(self):
        self.all_train_pairs = []
        for vid in self.train_vids:
            if vid in self.vid2caption.keys():
                self.all_train_pairs.append([vid, self.vid2caption[vid]])

    def _construct_all_test_pairs(self):
        self.all_test_pairs = []
        for vid in self.test_vids:
            if vid in self.vid2caption.keys():
                self.all_test_pairs.append([vid, self.vid2caption[vid]])

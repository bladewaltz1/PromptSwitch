import torch.nn as nn
from transformers import CLIPModel

from config.base_config import Config
from modules.baseline_pooling import BaselinePooling


class CLIPBaseline(nn.Module):
    def __init__(self, config: Config):
        super(CLIPBaseline, self).__init__()
        self.config = config
        self.clip = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        assert config.pooling_type == config.pooling_type_test
        self.pool_frames = BaselinePooling(config.pooling_type, config)
        self.pool_frames_test = self.pool_frames

        params_optimizer = list(self.named_parameters())
        self.clip_params = [p for n, p in params_optimizer if "clip." in n]
        self.noclip_params = [p for n, p in params_optimizer if "clip." not in n]

    def forward(self, data, return_all_frames=False):
        batch_size = data['video'].shape[0]
        text_data = data['text']
        video_data = data['video']
        video_data = video_data.reshape(-1, 3, self.config.input_res, self.config.input_res)

        text_features = self.clip.get_text_features(**text_data)
        video_features = self.clip.get_image_features(video_data)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        video_features = video_features / video_features.norm(dim=-1, keepdim=True)
        video_features = video_features.reshape(batch_size, self.config.num_frames, -1)

        video_features_pooled = self.pool_frames(text_features, video_features)

        if return_all_frames:
            return text_features, video_features, video_features_pooled

        return text_features, video_features_pooled

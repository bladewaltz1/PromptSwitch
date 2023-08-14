from config.base_config import Config
from model.clip_baseline import CLIPBaseline
from model.clip_transformer import CLIPTransformer
from model.prompt_clip import PromptCLIP


class ModelFactory:
    @staticmethod
    def get_model(config: Config):
        if config.arch == 'clip_baseline':
            return CLIPBaseline(config)
        elif config.arch == 'clip_transformer':
            return CLIPTransformer(config)
        elif config.arch == 'prompt_clip':
            return PromptCLIP(config)
        else:
            raise NotImplemented

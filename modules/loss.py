import torch.nn as nn
import torch
import torch.nn.functional as F

from config.base_config import Config

from .tokenizer import clip_tokenizer


class CLIPLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, sims, logit_scale):
        """
        Inputs: cosine similarities
            sims: n x n (text is dim-0)
            logit_scale: 1 x 1
        """
        logit_scale = logit_scale.exp()
        logits = sims * logit_scale
        
        t2v_log_sm = F.log_softmax(logits, dim=1)
        t2v_neg_ce = torch.diag(t2v_log_sm)
        t2v_loss = -t2v_neg_ce.mean()

        v2t_log_sm = F.log_softmax(logits, dim=0)
        v2t_neg_ce = torch.diag(v2t_log_sm)
        v2t_loss = -v2t_neg_ce.mean()

        return (t2v_loss + v2t_loss) / 2.0


class CaptionLoss(nn.Module):
    def __init__(self, config):
        super().__init__()
        weight = torch.ones(clip_tokenizer.vocab_size)
        frequent_words = "in on of to about this that a an the and there here is are . , \
                          <|endoftext|> <|startoftext|>"
        frequent_ids = clip_tokenizer.convert_tokens_to_ids(
            clip_tokenizer.tokenize(frequent_words)
        )
        weight[frequent_ids] = config.frequent_word_weight
        self.register_buffer('weight', weight)
        self.mult = config.caption_loss_mult

    def forward(self, pred_logits, input_ids):
        mask = input_ids[:, :-1] != clip_tokenizer.eos_token_id
        pred_logits = pred_logits[mask]
        target_ids = input_ids[:, 1:][mask]
        return F.cross_entropy(pred_logits, 
                               target_ids, 
                               weight=self.weight) * self.mult


class LossFactory:
    @staticmethod
    def get_loss(config: Config):
        if config.loss == 'clip':
            return {'clip': CLIPLoss()}
        elif config.loss == 'clip+caption':
            return {'clip': CLIPLoss(),
                    'caption': CaptionLoss(config)}
        else:
            raise NotImplementedError

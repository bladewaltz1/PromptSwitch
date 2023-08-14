from collections import defaultdict, deque

import numpy as np
import torch
from tqdm import tqdm

from config.base_config import Config
from modules.metrics import (sim_matrix_training, sim_matrix_inference, 
                             generate_embeds_per_video_id, 
                             t2v_metrics, v2t_metrics)
from modules.tokenizer import clip_tokenizer
from trainer.base_trainer import BaseTrainer


class Trainer(BaseTrainer):
    """
    Trainer class
    Note:
        Inherited from BaseTrainer.
    """

    def __init__(self, model, loss, optimizer, config: Config, train_data_loader, 
                 valid_data_loader, lr_scheduler=None, writer=None, use_ema=False):

        super().__init__(model, loss, optimizer, config, writer, use_ema)
        self.train_data_loader = train_data_loader
        self.valid_data_loader = valid_data_loader
        self.lr_scheduler = lr_scheduler

        self.pooling_type = config.pooling_type
        self.pooling_type_test = config.pooling_type_test
        self.window_metric = defaultdict(lambda: deque(maxlen=config.eval_window_size))
        self.best_window = -1.0
        self.best = -1.0

    def _train_epoch(self, epoch):
        """
        Training logic for an epoch
        :param epoch: Current training epoch.
        :return: A log that contains all information you want to save.
        """
        self.model.train()
        total_loss = 0.0
        num_steps = len(self.train_data_loader)
        eval_steps = np.linspace(0, num_steps-1, self.evals_per_epoch+1, dtype=int)[1:]

        for batch_idx, data in enumerate(self.train_data_loader):
            # then assume we must tokenize the input, e.g. its a string
            data['text'] = clip_tokenizer(data['text'], return_tensors='pt', padding=True,
                                          truncation=True)
            if isinstance(data['text'], torch.Tensor):
                data['text'] = data['text'].to(self.device)
            else:
                data['text'] = {key: val.to(self.device) for key, val in data['text'].items()}

            data['video'] = data['video'].to(self.device)

            model_output = self.model(data)
            text_embeds = model_output['text_features']
            video_embeds_pooled = model_output['video_features_pooled']
            sims = sim_matrix_training(text_embeds, video_embeds_pooled, self.pooling_type)
            loss = self.loss['clip'](sims, self.model.clip.logit_scale)

            if 'caption' in self.loss.keys():
                pred_logits = model_output['pred_logits']
                input_ids = model_output['input_ids']
                loss_caption = self.loss['caption'](pred_logits, input_ids)
            loss_all = loss + loss_caption
            loss_all.backward()

            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            if self.lr_scheduler is not None:
                self.lr_scheduler.step()
            self.optimizer.zero_grad()

            if self.use_ema:
                self.model_ema.update(self.model)

            torch.clamp_(self.model.clip.logit_scale.data, max=np.log(100))

            self.global_step += 1
            if self.writer is not None:
                self.writer.add_scalar('train/loss_train', loss.detach().item(), self.global_step)
                self.writer.add_scalar('train/loss_cap', loss_caption.detach().item(), self.global_step)

            total_loss += loss.detach().item()

            if batch_idx % self.log_step == 0:
                print('Train Epoch: {} dl: {}/{} Loss Ctr: {:.6f} Loss Cap: {:.6f}'.format(
                    epoch,
                    batch_idx,
                    num_steps-1,
                    loss.detach().item(),
                    loss_caption.detach().item()))

            if batch_idx in eval_steps:
                if self.use_ema:
                    model = self.model_ema.module
                else:
                    model = self.model
                val_res = self._valid_epoch_step(model, epoch, batch_idx, num_steps-1)
                self.model.train()

                if val_res['R1-t2v-window'] > self.best_window:
                    self.best_window = val_res['R1-t2v-window']
                    self._save_checkpoint(epoch, save_best=True)

                if val_res['R1-t2v'] > self.best:
                    self.best = val_res['R1-t2v']

                print(" Current Best Window Average R@1 is {}".format(self.best_window))
                print(" Current Best R@1 is {}\n\n".format(self.best))

        res = {
            'loss_train':  total_loss / num_steps
        }

        return res

    def _valid_epoch_step(self, model, epoch, step, num_steps):
        """
        Validate at a step when training an epoch at a certain step
        :return: A log that contains information about validation
        """
        model.eval()
        total_val_loss = 0.0
        text_embed_arr = []
        vid_embed_arr = []
        all_vid_ids = []

        with torch.no_grad():
            for _, data in tqdm(enumerate(self.valid_data_loader)):
                data['text'] = clip_tokenizer(data['text'], return_tensors='pt', padding=True, 
                                              truncation=True)
                if isinstance(data['text'], torch.Tensor):
                    data['text'] = data['text'].to(self.device)
                else:
                    data['text'] = {key: val.to(self.device) for key, val in data['text'].items()}

                data['video'] = data['video'].to(self.device)

                model_output = model(data, return_all_frames=True)
                text_embed = model_output['text_features']
                vid_embed = model_output['video_features']
                vid_embed_pooled = model_output['video_features_pooled']

                text_embed_arr.append(text_embed.cpu())
                vid_embed_arr.append(vid_embed.cpu())
                sims_batch = sim_matrix_training(text_embed, vid_embed_pooled, self.pooling_type_test)

                curr_loss = self.loss['clip'](sims_batch, model.clip.logit_scale)
                total_val_loss += curr_loss.item()

                for v_id in data['video_id']:
                    all_vid_ids.append(v_id)

            text_embeds = torch.cat(text_embed_arr)
            vid_embeds = torch.cat(vid_embed_arr)

            # Since we have all pairs, remove duplicate videos when there's multiple captions per video
            vid_embeds_per_video_id = {}
            for idx, v_id in enumerate(all_vid_ids):
                if v_id not in vid_embeds_per_video_id:
                    vid_embeds_per_video_id[v_id] = vid_embeds[idx]

            vid_embeds = torch.stack([vid_embeds_per_video_id[v_id] for v_id in vid_embeds_per_video_id])

            # Pool frames for inference once we have all texts and videos
            model.pool_frames.cpu()
            vid_embeds_pooled = model.pool_frames_test(text_embeds, vid_embeds)
            model.pool_frames.cuda()

            text_embeds_per_video_id, vid_embeds_pooled_per_video_id = generate_embeds_per_video_id(text_embeds, 
                    vid_embeds_pooled, all_vid_ids, self.pooling_type_test)

            sims = sim_matrix_inference(text_embeds_per_video_id, vid_embeds_pooled_per_video_id, self.pooling_type_test)

            total_val_loss = total_val_loss / len(self.valid_data_loader)

            res = t2v_metrics(sims)
            res.update(v2t_metrics(sims))

            # Compute window metrics
            for m in res:
                self.window_metric[m].append(res[m])

            # Compute average of window metrics
            for m in self.window_metric:
                res[m + "-window"] = np.mean(self.window_metric[m])

            print(f"-----Val Epoch: {epoch}, dl: {step}/{num_steps}-----\n",
                  "-------------------- t2v ------------------------ \n",
                  f"R@1: {res['R1-t2v']} (window: {res['R1-t2v-window']})\n", 
                  f"R@5: {res['R5-t2v']} (window: {res['R5-t2v-window']})\n", 
                  f"R@10: {res['R10-t2v']} (window: {res['R10-t2v-window']})\n",
                  f"MedR: {res['MedR-t2v']} (window: {res['MedR-t2v-window']})\n",
                  f"MeanR: {res['MeanR-t2v']} (window: {res['MeanR-t2v-window']})\n",
                  "-------------------- v2t ------------------------ \n",
                  f"R@1: {res['R1-v2t']} (window: {res['R1-v2t-window']})\n", 
                  f"R@5: {res['R5-v2t']} (window: {res['R5-v2t-window']})\n", 
                  f"R@10: {res['R10-v2t']} (window: {res['R10-v2t-window']})\n",
                  f"MedR: {res['MedR-v2t']} (window: {res['MedR-v2t-window']})\n",
                  f"MeanR: {res['MeanR-v2t']} (window: {res['MeanR-v2t-window']})\n",
                  "------------------------------------------------- \n",
                  f"Loss: {total_val_loss}")

            res['loss_val'] =  total_val_loss

            if self.writer is not None:
                for m in res:
                    self.writer.add_scalar(f'val/{m}', res[m], self.global_step)

            return res

from torch.utils.data import DataLoader
from torch.utils.data._utils.collate import default_collate

from config.base_config import Config
from datasets.model_transforms import init_transform_dict
from datasets.msrvtt_dataset import MSRVTTDataset
from datasets.msvd_dataset import MSVDDataset
from datasets.lsmdc_dataset import LSMDCDataset
from datasets.anet_dataset import ANetDataset


class DataFactory:

    @staticmethod
    def get_data_loader(config: Config, split_type='train'):
        img_transforms = init_transform_dict(config.input_res)
        train_img_tfms = img_transforms['clip_train']
        test_img_tfms = img_transforms['clip_test']

        if config.dataset_name == "MSRVTT":
            if split_type == 'train':
                dataset = MSRVTTDataset(config, split_type, train_img_tfms)
                return DataLoader(dataset, batch_size=config.batch_size,
                           shuffle=True, num_workers=config.num_workers,
                           collate_fn=collate_fn)
            else:
                dataset = MSRVTTDataset(config, split_type, test_img_tfms)
                return DataLoader(dataset, batch_size=config.test_batch_size,
                           shuffle=False, num_workers=config.num_workers)

        elif config.dataset_name == "MSVD":
            if split_type == 'train':
                dataset = MSVDDataset(config, split_type, train_img_tfms)
                return DataLoader(dataset, batch_size=config.batch_size,
                            shuffle=True, num_workers=config.num_workers,
                            collate_fn=collate_fn)
            else:
                dataset = MSVDDataset(config, split_type, test_img_tfms)
                return DataLoader(dataset, batch_size=config.test_batch_size,
                            shuffle=False, num_workers=config.num_workers)

        elif config.dataset_name == 'LSMDC':
            if split_type == 'train':
                dataset = LSMDCDataset(config, split_type, train_img_tfms)
                return DataLoader(dataset, batch_size=config.batch_size,
                            shuffle=True, num_workers=config.num_workers,
                            collate_fn=collate_fn)
            else:
                dataset = LSMDCDataset(config, split_type, test_img_tfms)
                return DataLoader(dataset, batch_size=config.test_batch_size,
                            shuffle=False, num_workers=config.num_workers)

        elif config.dataset_name == 'ActivityNet':
            if split_type == 'train':
                dataset = ANetDataset(config, split_type, train_img_tfms)
                return DataLoader(dataset, batch_size=config.batch_size,
                            shuffle=True, num_workers=config.num_workers,
                            collate_fn=collate_fn)
            else:
                dataset = ANetDataset(config, split_type, test_img_tfms)
                return DataLoader(dataset, batch_size=config.test_batch_size,
                            shuffle=False, num_workers=config.num_workers)

        else:
            raise NotImplementedError


def collate_fn(batch):
    video_ids = []
    new_batch = []
    for item in batch:
        video_id = item['video_id']
        if video_id not in video_ids:
            video_ids.append(video_id)
            new_batch.append({
                'video_id': video_id, 
                'video': item['video'], 
                'text': item['text']
            })
    return default_collate(new_batch)

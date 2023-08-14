import numpy as np
import torch


def sim_matrix_training(text_embeds, vid_embeds_pooled, pooling_type):
    """
    Computes the similarity matrix using pooled video frames
    
    Output
        sims: num_texts x num_vids
    """
    text_embeds = text_embeds / text_embeds.norm(dim=-1, keepdim=True)
    vid_embeds_pooled = vid_embeds_pooled / vid_embeds_pooled.norm(dim=-1, keepdim=True)

    if pooling_type in ['avg', 'subsampled']:
        sims = torch.mm(text_embeds, vid_embeds_pooled.t())

    else:
        # num_texts x embed_dim x num_vids
        vid_embeds_pooled = vid_embeds_pooled.permute(1,2,0)
        # num_texts x 1 x embed_dim
        text_embeds = text_embeds.unsqueeze(1)

        sims = torch.bmm(text_embeds, vid_embeds_pooled).squeeze(1)

    return sims


def sim_matrix_inference(text_embeds_per_video_id, vid_embeds_pooled_per_video_id, pooling_type):
    """
    Computes the similarity matrix using pooled video frames using all texts per video

    Output
        sims: num_vids x max_text_per_vid x num_vids
    """
    text_embeds_per_video_id = text_embeds_per_video_id / text_embeds_per_video_id.norm(dim=-1, keepdim=True)
    vid_embeds_pooled_per_video_id = vid_embeds_pooled_per_video_id / vid_embeds_pooled_per_video_id.norm(dim=-1, keepdim=True)

    if pooling_type in ['avg', 'subsampled']:
        # text_embeds_per_video_id -> num_vids x max_text_per_vid x embed_dim
        # vid_embeds_pooled_per_video_id -> num_vids x embed_dim

        sims = text_embeds_per_video_id @ vid_embeds_pooled_per_video_id.t()

    else:
        # text_embeds_per_video_id -> num_vids x max_text_per_vid x embed_dim
        # vid_embeds_pooled_per_video_id -> num_vids x num_vids x max_text_per_vid x embed_dim
        num_vids, max_text_per_vid, embed_dim = text_embeds_per_video_id.shape

        # num_vids x max_text_per_vid x embed_dim x num_vids
        vid_embeds_pooled_per_video_id = vid_embeds_pooled_per_video_id.permute(1,2,3,0)
        vid_embeds_pooled_per_video_id = vid_embeds_pooled_per_video_id.view(num_vids*max_text_per_vid, embed_dim, num_vids)
        # num_vids x max_text_per_vid x 1 x embed_dim
        text_embeds_per_video_id = text_embeds_per_video_id.unsqueeze(2)
        text_embeds_per_video_id = text_embeds_per_video_id.view(num_vids*max_text_per_vid, 1, embed_dim)

        sims = torch.bmm(text_embeds_per_video_id, vid_embeds_pooled_per_video_id)
        sims = sims.view(num_vids, max_text_per_vid, 1, num_vids).squeeze(2)

    return sims


def generate_embeds_per_video_id(text_embeds, vid_embeds_pooled, all_vid_ids, pooling_type):
    # Construct dictionary of text embeds per unique video id
    text_embeds_per_video_id = {}

    for idx, v_id in enumerate(all_vid_ids):
        if v_id in text_embeds_per_video_id:
            text_embeds_per_video_id[v_id].append(text_embeds[idx])
        else:
            text_embeds_per_video_id[v_id] = [text_embeds[idx]]

    for v_id in text_embeds_per_video_id:
        text_embeds_per_video_id[v_id] = torch.stack(text_embeds_per_video_id[v_id])

    # num_vids x max_text_per_vid x embed_dim
    text_embeds_per_video_id = pad_and_stack_dict_to_tensor(text_embeds_per_video_id,
        text_embeds_per_video_id.keys(), text_embeds.shape[-1])

    if pooling_type in ['avg', 'subsampled']:
        # num_vids x embed_dim
        vid_embeds_pooled_per_video_id = vid_embeds_pooled

    else:
        # Construct dictionary of video embeds for each text per video_id
        vid_embeds_pooled_per_video_id = []

        for i in range(vid_embeds_pooled.shape[0]):
            vid_embeds_pooled_per_video_id.append({})
            for idx, v_id in enumerate(all_vid_ids):
                if v_id in vid_embeds_pooled_per_video_id[i]:
                    vid_embeds_pooled_per_video_id[i][v_id].append(vid_embeds_pooled[i, idx, :])
                else:
                    vid_embeds_pooled_per_video_id[i][v_id] = [vid_embeds_pooled[i, idx, :]]

        for i in range(len(vid_embeds_pooled_per_video_id)):
            for v_id in vid_embeds_pooled_per_video_id[i]:
                vid_embeds_pooled_per_video_id[i][v_id] = torch.stack(vid_embeds_pooled_per_video_id[i][v_id])

            # num_vids x max_text_per_vid x embed_dim
            vid_embeds_pooled_per_video_id[i] = pad_and_stack_dict_to_tensor(vid_embeds_pooled_per_video_id[i],
                    vid_embeds_pooled_per_video_id[i].keys(), vid_embeds_pooled.shape[-1])

        # num_vids x num_vids x max_text_per_vid x embed_dim
        vid_embeds_pooled_per_video_id = torch.stack(vid_embeds_pooled_per_video_id)

    return text_embeds_per_video_id, vid_embeds_pooled_per_video_id


def t2v_metrics(sims):
    # Permute sims so it represents a sequence of text-video similarity matrices.
    # Then obtain the double argsort to position the rank on the diagonal
    stacked_sims = sims.permute(1,0,2)

    sims_sort = torch.argsort(stacked_sims, dim=-1, descending=True)
    sims_sort_2 = torch.argsort(sims_sort, dim=-1, descending=False)

    ranks = torch.flatten(torch.diagonal(sims_sort_2, dim1=1, dim2=2))

    # Now we need to extract valid ranks, as some belong to inf padding values
    valid_check = torch.flatten(torch.diagonal(sims, dim1 = 0, dim2 = 2))
    mask = ~ torch.logical_or(torch.isinf(valid_check), torch.isnan(valid_check))
    valid_ranks = ranks[mask]

    return compute_metrics(valid_ranks.numpy(), 't2v')


def v2t_metrics(sims):
    # Code to avoid nans
    sims[sims!=sims] = float('-inf')
    # Forms a similarity matrix
    sims, _ = torch.max(sims, dim = 1)
    sims = sims.t()

    sims_sort = torch.argsort(sims, dim=-1, descending=True)
    sims_sort_2 = torch.argsort(sims_sort, dim=-1, descending=False)

    ranks = torch.diag(sims_sort_2).numpy() # diagonal

    return compute_metrics(ranks, 'v2t')


def compute_metrics(lst, prefix='t2v'):
    metrics = {}
    metrics[f"R1-{prefix}"] = 100 * float(np.sum(lst == 0)) / len(lst)
    metrics[f"R5-{prefix}"] = 100 * float(np.sum(lst < 5)) / len(lst)
    metrics[f"R10-{prefix}"] = 100 * float(np.sum(lst < 10)) / len(lst)
    metrics[f"R50-{prefix}"] = 100 * float(np.sum(lst < 50)) / len(lst)
    metrics[f"R100-{prefix}"] = 100 * float(np.sum(lst < 100)) / len(lst)
    metrics[f"MedR-{prefix}"] = np.median(lst) + 1
    metrics[f"MeanR-{prefix}"] = np.mean(lst) + 1
    return metrics


def pad_and_stack_dict_to_tensor(input, order, d=512):
    max_length = max([input[k].shape[0] for k in input])

    padded_input = {k: torch.cat([input[k], torch.full((max_length - input[k].shape[0], d), 
                                                        float("-inf"), device = input[k].device)]) for k in input}

    padded_stacked_input = torch.stack([padded_input[k] for k in order], dim = 0)
    return padded_stacked_input

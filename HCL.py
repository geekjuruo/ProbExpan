import os
import pickle
import torch
import numpy as np
from torch.utils.data import Dataset, ConcatDataset

class Pairs(Dataset):
    def __init__(self, eid2sents, max_len, eid1, eid2=None):
        self.eid1 = eid1
        self.sents1 = eid2sents[eid1]
        if eid2 is None:
            self.sents2 = eid2sents[eid1]
        else:
            self.sents2 = eid2sents[eid2]

        self.num_pair = min(max(len(self.sents1), len(self.sents2)), max_len)
        if len(self.sents1) < self.num_pair:
            self.indexs1 = np.random.choice(len(self.sents1), self.num_pair, replace=True)
        else:
            self.indexs1 = np.random.choice(len(self.sents1), self.num_pair, replace=False)

        if len(self.sents2) < self.num_pair:
            self.indexs2 = np.random.choice(len(self.sents2), self.num_pair, replace=True)
        else:
            self.indexs2 = np.random.choice(len(self.sents2), self.num_pair, replace=False)

    def __len__(self):
        return self.num_pair

    def __getitem__(self, index):
        flag_1 = False
        flag_2 = False
        cnt_1 = 0
        cnt_2 = 0
        while(True):
            token_ids1 = self.sents1[self.indexs1[(index+cnt_1) % self.num_pair]]
            token_ids2 = self.sents2[self.indexs2[(index+cnt_2) % self.num_pair]]

            if len(token_ids1) < 200:
                flag_1 = True
            else:
                cnt_1 += 1
            if len(token_ids2) < 200:
                flag_2 = True
            else:
                cnt_2 += 1
            if flag_1 and flag_2:
                break

        return token_ids1, token_ids2


def get_cl_dataset(cls2eids, eid2sents, max_sents):
    dataset_list = []
    eid_pairs = set()
    for cls_name in cls2eids:
        eids = list(cls2eids[cls_name])
        for eid in eids:
            if (eid, eid) not in eid_pairs:
                this_dataset = Pairs(eid2sents, max_sents, eid)
                dataset_list.append(this_dataset)
                eid_pairs.add((eid, eid))
            if len(cls_name.split(' ')) == 1:
                for cnt in range(2):
                    while True:
                        eid2 = np.random.choice(eids)
                        eid_pair = (eid, eid2) if eid <= eid2 else (eid2, eid)
                        if eid_pair not in eid_pairs:
                            eid_pairs.add(eid_pair)
                            break
                    this_dataset = Pairs(eid2sents, max_sents, eid, eid2)
                    dataset_list.append(this_dataset)
            else:
                for cnt in range(1):
                    this_dataset = Pairs(eid2sents, max_sents, eid)
                    dataset_list.append(this_dataset)
    dataset = ConcatDataset(dataset_list)
    return dataset


def cl_collate_fn(batch):
    batch_ids1, batch_ids2 = zip(*batch)

    batch_max_length1 = max(len(ids) for ids in batch_ids1)
    batch_ids1 = torch.tensor([ids + [0 for _ in range(batch_max_length1 - len(ids))] for ids in batch_ids1]).long()
    batch_max_length2 = max(len(ids) for ids in batch_ids2)
    batch_ids2 = torch.tensor([ids + [0 for _ in range(batch_max_length2 - len(ids))] for ids in batch_ids2]).long()

    return batch_ids1, batch_ids2


def get_negative_mask(batch_size):
    negative_mask = torch.ones((batch_size, 2 * batch_size), dtype=bool)
    for i in range(batch_size):
        negative_mask[i, i] = 0
        negative_mask[i, i + batch_size] = 0

    negative_mask = torch.cat((negative_mask, negative_mask), 0)
    return negative_mask


def cl_criterion(out_1, out_2, tau_plus, batch_size, beta, estimator='hard', temperature=0.5):
    # neg score
    out = torch.cat([out_1, out_2], dim=0)
    neg = torch.exp(torch.mm(out, out.t().contiguous()) / temperature)
    old_neg = neg.clone()
    mask = get_negative_mask(batch_size).cuda()
    neg = neg.masked_select(mask).view(2 * batch_size, -1)

    # pos score
    pos = torch.exp(torch.sum(out_1 * out_2, dim=-1) / temperature)
    pos = torch.cat([pos, pos], dim=0)

    # negative samples similarity scoring
    if estimator == 'hard':
        N = batch_size * 2 - 2
        imp = (beta * neg.log()).exp()
        reweight_neg = (imp * neg).sum(dim=-1) / imp.mean(dim=-1)
        Ng = (-tau_plus * N * pos + reweight_neg) / (1 - tau_plus)
        # constrain (optional)
        Ng = torch.clamp(Ng, min=N * np.e ** (-1 / temperature))
    elif estimator == 'easy':
        Ng = neg.sum(dim=-1)
    else:
        raise Exception('Invalid estimator selected. Please use any of [hard, easy]')

    # contrastive loss
    loss = (- torch.log(pos / (pos + Ng))).mean()

    return loss


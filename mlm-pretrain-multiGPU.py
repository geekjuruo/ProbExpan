import os
import argparse
import sys
import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from transformers import BertTokenizer, BertModel, BertConfig, AdamW
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import Dataset, ConcatDataset, DataLoader
from utils import *

class Model(nn.Module):

    def __init__(self, len_vocab, mask_token_id, model_name='bert-base-uncased', dim=768):
        super().__init__()

        bert_config = BertConfig.from_pretrained(model_name)
        bert_config.output_hidden_states = True
        self.bert = BertModel.from_pretrained(model_name, add_pooling_layer=False)

        self.head = nn.Sequential(nn.Linear(dim, dim),
                                  nn.GELU(),
                                  nn.Linear(dim, len_vocab),
                                  nn.LogSoftmax(dim=-1))

        self.mask_token_id = mask_token_id

    def forward(self, x, CL=False):
        mask_pos = (x == self.mask_token_id).nonzero(as_tuple=True)
        pad_mask = (x != 0).long()
        bert_output = self.bert(x, pad_mask)

        hidden_stats = bert_output[0]
        # print(hidden_stats.shape)
        set_embeddings = hidden_stats[mask_pos]
        # print(set_embeddings.shape)

        set_distributions = self.head(set_embeddings)

        return set_distributions

class Eid2Data(Dataset):
    def __init__(self, eid, eid2sents, label_indexs, siz=None):
        self.eid = eid
        self.sents = eid2sents[eid]
        self.label_indexs = label_indexs

        if siz is not None:
            if siz <= len(self.sents):
                indexs = np.random.choice(len(self.sents), siz, replace=False)
                self.sents = [self.sents[i] for i in indexs]
            else:
                indexs = np.random.choice(len(self.sents), siz, replace=True)
                self.sents = [self.sents[i] for i in indexs]

        self.num_sents = len(self.sents)

    def __len__(self):
        return self.num_sents

    def __getitem__(self, index):
        token_ids = self.sents[index]
        labels = self.label_indexs
        return token_ids, labels


def collate_fn(batch):
    batch_ids, batch_labels = zip(*batch)

    batch_max_length = max(len(ids) for ids in batch_ids)
    batch_ids = torch.tensor([ids + [0 for _ in range(batch_max_length - len(ids))] for ids in batch_ids]).long()

    return batch_ids, batch_labels

def run_epoch(model, data_iter, loss_compute, optimizer, log_step=200, master=False):
    total_loss_predict = 0

    # masked enity prediction task
    for i, batch in enumerate(data_iter):
        out = model.forward(batch[0].cuda())

        optimizer.zero_grad()
        loss_predict = loss_compute(out, batch[1])
        loss_predict.backward()
        optimizer.step()

        total_loss_predict += loss_predict.item()
        if (i + 1) % log_step == 0 and master:
            print("Step: %5d        Loss: %.4f" % (i + 1, total_loss_predict / log_step))
            total_loss_predict = 0

class Loss_Compute(nn.Module):
    def __init__(self, criterion, len_vocab, smoothing=0):
        super(Loss_Compute, self).__init__()
        self.criterion = criterion
        self.len_vocab = len_vocab
        self.smoothing = smoothing

    def forward(self, output, batch_labels):
        dists = []
        for labels in batch_labels:
            len_set = len(labels)
            dist = torch.zeros(self.len_vocab)
            dist.fill_(self.smoothing / (self.len_vocab - len_set))
            dist.scatter_(0, torch.tensor(labels), (1 - self.smoothing) / len_set)
            dists.append(dist)
        tensor_dists = torch.stack(dists).cuda()

        return self.criterion(output, tensor_dists)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # Each process runs on 1 GPU device specified by the local_rank argument.
    parser.add_argument("--local_rank", type=int)
    parser.add_argument('-dataset', default='../data/wiki', help='path to dataset folder')
    parser.add_argument('-vocab', default='entity2id.txt', help='vocab file')
    parser.add_argument('-pkl_e2s', default='entity2sents.pkl', help='sents file for pretraining')
    parser.add_argument('-save_path', default='../model_wiki', help='path to place model parameters')
    parser.add_argument('-epoch', default=5)
    parser.add_argument('-lr', default=1e-5)
    parser.add_argument('-batchsize', default=128)
    parser.add_argument('-smoothing', default=0.075)
    parser.add_argument('-num_sent', default=256)
    args = parser.parse_args()
    print(args)

    epoch = args.epoch
    lr = args.lr
    batchsize = args.batchsize
    smoothing = args.smoothing
    num_sent = args.num_sent
    
    save_path = args.save_path
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    
    torch.cuda.set_device(args.local_rank)
    # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
    torch.distributed.init_process_group(backend='nccl')

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=False)
    mask_token_id = tokenizer.mask_token_id

    # dict of entity names, list of entity ids, dict of line index
    eid2name, _, _ = load_vocab(os.path.join(args.dataset, args.vocab))

    # dict: eid to sentences
    eid2sents = pickle.load(open(os.path.join(args.dataset, args.pkl_e2s), 'rb'))
    list_eids = list(eid2sents.keys())
    len_vocab = len(list_eids)
    eid2index = {eid: i for i, eid in enumerate(list_eids)}

    model = Model(len_vocab, mask_token_id)
    # freeze part of BERT
    unfreeze_layers = [('encoder.layer.%d' % i) for i in range(11, 12)] + ['head']
    for name, param in model.named_parameters():
        param.requires_grad = False
        for ele in unfreeze_layers:
            if ele in name:
                param.requires_grad = True
                break
    model.cuda()
    model = nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank], output_device=args.local_rank)

    # optimizer
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
         'weight_decay': 0.01},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
         'weight_decay': 0.0}
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=lr)

    loss_compute = Loss_Compute(nn.KLDivLoss(reduction='batchmean'), len_vocab, smoothing=smoothing)

    for i in range(1, epoch+1):
        if args.local_rank == 0:
            print('\n[Epoch %2d]' % i)
        
        # rebuild dataset before each epoch
        list_dataset = []
        for eid in list_eids:
            this_dataset = Eid2Data(eid, eid2sents, [eid2index[eid]], num_sent)
            list_dataset.append(this_dataset)
        dataset = ConcatDataset(list_dataset)

        sampler = DistributedSampler(dataset)
        dataloader = DataLoader(dataset, batch_size=batchsize, collate_fn=collate_fn, sampler=sampler)

        run_epoch(model, dataloader, loss_compute, optimizer, log_step=200, master=(args.local_rank == 0))
        if args.local_rank == 0:
            torch.save(model.module.state_dict(), os.path.join(args.save_path, 'epoch%d' % i))

import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertTokenizer, BertModel, BertConfig, AdamW
from torch.utils.data import Dataset, ConcatDataset, DataLoader
from utils import *
from HCL import cl_criterion, get_cl_dataset, cl_collate_fn
import time
import pickle


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

        self.projection_head = nn.Sequential(nn.Linear(dim, 256, bias=False), nn.BatchNorm1d(256),
                               nn.ReLU(inplace=True), nn.Linear(256, 128, bias=True))

        self.mask_token_id = mask_token_id

    def forward(self, x, CL=False):
        mask_pos = (x == self.mask_token_id).nonzero(as_tuple=True)
        pad_mask = (x != 0).long()
        bert_output = self.bert(x, pad_mask)

        hidden_stats = bert_output[0]
        # print(hidden_stats.shape)
        set_embeddings = hidden_stats[mask_pos]
        # print(set_embeddings.shape)

        if CL:
            set_distributions = None
            projection = self.projection_head(set_embeddings)
            projection = F.normalize(projection, dim=-1)
        else:
            set_distributions = self.head(set_embeddings)
            projection = None

        return set_distributions, projection


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


def run_epoch(model, data_iter, loss_compute, optimizer, log_step=100):
    total_loss_predict = 0

    # masked enity prediction task
    for i, batch in enumerate(data_iter):
        out, _ = model.forward(batch[0].cuda())

        optimizer.zero_grad()
        loss_predict = loss_compute(out, batch[1])
        loss_predict.backward()
        optimizer.step()

        total_loss_predict += loss_predict.item()
        if (i + 1) % log_step == 0:
            print("Step: %4d        Loss: %.4f" % (i + 1, total_loss_predict / log_step))
            total_loss_predict = 0

def run_epoch_cl(model, cl_data_iter, cl_loss_compute, optimizer, log_step=100):
    total_loss_cl = 0

    # contrastive learning
    for i, batch in enumerate(cl_data_iter):
        _, out_1 = model.forward(batch[0].cuda(), CL=True)
        _, out_2 = model.forward(batch[1].cuda(), CL=True)

        optimizer.zero_grad()
        # For Wiki
        loss_cl = cl_loss_compute(out_1, out_2, tau_plus=0.05, batch_size=256, beta=1) * 7
        # # For APR
        # loss_cl = cl_loss_compute(out_1, out_2, tau_plus=0.1, batch_size=256, beta=1) * 7
        loss_cl.backward()
        optimizer.step()

        total_loss_cl += loss_cl.item() / 7
        if (i + 1) % log_step == 0:
            print("CL Step: %4d     CL Loss: %.4f" % (i + 1, total_loss_cl / log_step))
            total_loss_cl = 0


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

class Expan(object):

    def __init__(self, args, cls_names, model_name='bert-base-uncased'):

        self.tokenizer = BertTokenizer.from_pretrained(model_name, do_lower_case=False)
        self.mask_token_id = self.tokenizer.mask_token_id

        # dict of entity names, list of entity ids, dict of line index
        self.eid2name, _, _ = load_vocab(os.path.join(args.dataset, args.vocab))

        # dict: eid to sentences
        self.eid2sents = pickle.load(open(os.path.join(args.dataset, args.pkl_e2s), 'rb'))
        self.list_eids = list(self.eid2sents.keys())
        self.len_vocab = len(self.list_eids)
        self.eid2index = {eid: i for i, eid in enumerate(self.list_eids)}

        self.model = Model(self.len_vocab, self.mask_token_id, model_name=model_name)

        self.cls_names = cls_names
        self.num_cls = len(cls_names)

        self.pkl_path_e2d = os.path.join(args.dataset, args.pkl_e2d)
        self.eindex2dists = None

        self.pkl_path_e2d = os.path.join(args.dataset, args.pkl_e2d)
        self.pkl_path_e2logd = os.path.join(args.dataset, args.pkl_e2d + '_log')
        self.eindex2dist = None
        self.eindex2logdist = None
        if os.path.exists(self.pkl_path_e2d):
            self.eindex2dist = pickle.load(open(self.pkl_path_e2d, 'rb'))
        if os.path.exists(self.pkl_path_e2logd):
            self.eindex2logdist = pickle.load(open(self.pkl_path_e2logd, 'rb'))

        self.eid2MeanLogDist = dict()
        self.eid2dist = dict()

        self.cls2eids = None
        if os.path.exists(os.path.join(args.dataset, args.pkl_cls2eids)):
            self.cls2eids = pickle.load(open(os.path.join(args.dataset, args.pkl_cls2eids), 'rb'))

    # Pretraining model with Contrastive Loss and Masked Entity Prediction Loss
    def pretrain(self, save_path, lr=1e-5, epoch=5, batchsize=128, num_sen_per_entity=256, smoothing=0.1):
        if self.model is None:
            self.model = Model(self.len_vocab, self.mask_token_id, model_name="./bert")

        if not os.path.exists(save_path):
            # os.mkdir(save_path)
            os.makedirs(save_path)

        # freeze part of BERT
        unfreeze_layers = ['encoder.layer.11', 'head', 'projection_head']
        for name, param in self.model.named_parameters():
            param.requires_grad = False
            # print(name, param.size())
            for ele in unfreeze_layers:
                if ele in name:
                    param.requires_grad = True
                    break

        # optimizer
        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay)],
             'weight_decay': 0.01},
            {'params': [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay)],
             'weight_decay': 0.0}
        ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=lr)

        loss_compute = Loss_Compute(nn.KLDivLoss(reduction='batchmean'), self.len_vocab,
                                    smoothing=smoothing)

        self.model.cuda()

        for i in range(0, epoch):
            print('\n[Epoch %d]' % (i + 1))

            # rebuild dataset before each epoch
            list_dataset = []
            for eid in self.list_eids:
                this_dataset = Eid2Data(eid, self.eid2sents, [self.eid2index[eid]], num_sen_per_entity)
                list_dataset.append(this_dataset)

            dataset = ConcatDataset(list_dataset)
            data_loader = DataLoader(dataset, batch_size=batchsize, shuffle=True, collate_fn=collate_fn)

            if self.cls2eids is not None:
                # Contrastive Learning
                for cnt in range(2):
                    dataset_cl = get_cl_dataset(self.cls2eids, self.eid2sents, num_sen_per_entity)
                    data_loader_cl = DataLoader(dataset_cl, batch_size=256, shuffle=True, collate_fn=cl_collate_fn, drop_last=True)
                    run_epoch_cl(self.model, data_loader_cl, cl_criterion, optimizer, log_step=200)
                    print('')

            # Masked Entity Prediction Learning
            run_epoch(self.model, data_loader, loss_compute, optimizer, log_step=200)

            # Save Model Parameters
            model_pkl_name = "epoch_%d.pkl" % (i + 1)
            if self.cls2eids is not None:
                model_pkl_name = "cl_epoch_%d.pkl" % (i + 1)
            torch.save(self.model.state_dict(), os.path.join(save_path, model_pkl_name))


    def expand(self, query_sets, target_size=103, ranking=True, mu=9,
               init_win_size=1, win_grow_rate=2.5, win_grow_step=20, total_iter=1):
        pre_expanded_sets = [None for _ in range(self.num_cls)]
        expanded_sets = query_sets
        cnt_iter = 0
        flag_stop = False
        pre_cursor = 13

        assert os.path.exists(self.pkl_path_e2d)
        self.eindex2dists = pickle.load(open(self.pkl_path_e2d, 'rb'))

        print('Start expanding:')
        for i in range(self.num_cls):
            print(str([self.eid2name[eid] for eid in query_sets[i]]))
        print('')

        while cnt_iter < total_iter and flag_stop is False:
            print('[Iteration %d]' % (cnt_iter + 1))
            flag_stop = True
            seed_sets = []
            cursor = target_size

            # check whether the expanded_set of each class is changed in last iteration
            # if so, renew seed set
            for i, expanded_set in enumerate(expanded_sets):
                changed = False
                if cnt_iter == 0:
                    seed_set = expanded_set
                    changed = True
                elif cnt_iter == 1:
                    seed_set = expanded_set[:13]
                    changed = True
                else:
                    # seed set is updated as the longest common set between pre_expanded_set and expanded_set
                    for j in range(pre_cursor, target_size):
                        for k in range(3, j):
                            if pre_expanded_sets[i][k] not in expanded_set[:j]:
                                changed = True
                                break
                        if changed and j < cursor:
                            cursor = j
                            pre_cursor = cursor
                            break
                    seed_set = expanded_set
                seed_sets.append(seed_set)

                if changed:
                    flag_stop = False
                else:
                    print(self.cls_names[i] + '  UNCHANGED')

            # truncate seed sets to same length
            if cnt_iter > 1:
                print('Cursor: ', cursor)
                print('')
                seed_sets = [seed_set[:cursor] for seed_set in seed_sets]

            pre_expanded_sets = expanded_sets
            expanded_sets = self.expand_(seed_sets, target_size, ranking, mu + cnt_iter * 2,
                                         init_win_size, win_grow_rate, win_grow_step)

            for i, expanded_set in enumerate(expanded_sets):
                print(self.cls_names[i])
                print(str([self.eid2name[eid] for eid in expanded_set]))
            print('\n')
            cnt_iter += 1

        return [eid_set[3:] for eid_set in expanded_sets]

    def expand_(self, seed_sets, target_size, ranking, mu, init_win_size, win_grow_rate, win_grow_step):
        expanded_sets = seed_sets

        eid_out_of_sets = set()
        for eid in self.list_eids:
            eid_out_of_sets.add(eid)
        for eid_set in seed_sets:
            for eid in eid_set:
                eid_out_of_sets.remove(eid)

        rounds = len(expanded_sets[0]) - 3
        while len(expanded_sets[0]) < target_size:
            if rounds < win_grow_step:
                size_window = init_win_size
            elif rounds < 50:
                size_window = init_win_size + (rounds / win_grow_step) * win_grow_rate
            else:
                size_window = init_win_size + (rounds / win_grow_step) * win_grow_rate * 1.25
            if rounds >= 100:
                size_window = 1
            rounds += 1

            """ Expand """
            for i, cls in enumerate(self.cls_names):
                scores = np.zeros(self.len_vocab)
                eid_set = expanded_sets[i]
                for eid in eid_set:
                    mean_dist = self.get_mean_log_dist(eid)
                    scores += mean_dist

                indexs = np.argsort(-scores)

                """ Window Search """
                cnt = 0
                tgt_eid = None
                min_KL_div = float('inf')
                set_dist = np.zeros(self.len_vocab)
                len_set = len(expanded_sets[0])

                for index in indexs:
                    if cnt >= int(size_window):
                        break
                    eid = self.list_eids[index]
                    if eid in eid_out_of_sets:
                        cnt += 1
                        feature_dist = self.get_feature_dist(eid)
                        mean_prob = np.mean(feature_dist)
                        set_dist[:] = mean_prob
                        set_dist[index] = feature_dist[index]

                        if len_set <= 41:
                            set_dist[[self.eid2index[eid] for eid in eid_set[:len_set]]] = mean_prob * 1000
                        else:
                            set_dist[[self.eid2index[eid] for eid in eid_set[:41]]] = mean_prob * 1000
                            set_dist[[self.eid2index[eid] for eid in eid_set[41:min(len_set, 75)]]] = mean_prob * 500
                            if len_set > 75:
                                set_dist[[self.eid2index[eid] for eid in eid_set[75:len_set]]] = mean_prob * 300

                        KL_div = KL_divergence(set_dist, feature_dist)
                        if KL_div < min_KL_div:
                            min_KL_div = KL_div
                            tgt_eid = eid
                            # print(cls, self.eid2name[eid], np.max(feature_dist), mean_prob)

                expanded_sets[i].append(tgt_eid)
                eid_out_of_sets.remove(tgt_eid)

        if ranking:
            """ Re-Ranking """
            ranked_expanded_sets = []
            dt = np.dtype([('eid', 'int'), ('rev_score', 'float32')])

            for i, eid_set in enumerate(expanded_sets):
                # rank-score on original entity set
                scores = [mu / r for r in range(1, target_size - 2)]

                # sort with KL-div bewtween set distribution and entity's feature distribution
                set_dist = np.zeros(self.len_vocab)

                eids = eid_set[3:]
                KLdivs = []
                for eid in eids:
                    feature_dist = self.get_feature_dist(eid)
                    mean_prob = np.mean(feature_dist)
                    set_dist[:] = mean_prob
                    set_dist[self.eid2index[eid]] = feature_dist[self.eid2index[eid]]

                    set_dist[[self.eid2index[eid] for eid in eid_set[:20]]] = mean_prob * 1600
                    set_dist[[self.eid2index[eid] for eid in eid_set[20:35]]] = mean_prob * 900
                    set_dist[[self.eid2index[eid] for eid in eid_set[35:70]]] = mean_prob * 600
                    set_dist[[self.eid2index[eid] for eid in eid_set[70:]]] = mean_prob * 150
                    diverg = KL_divergence(set_dist, feature_dist)
                    # print(self.eid2name[eid], np.max(feature_dist), np.mean(feature_dist), diverg)
                    KLdivs.append(diverg)
                # print('')
                arg_sorted_z = np.argsort(KLdivs)
                for j, r in enumerate(arg_sorted_z):
                    scores[r] += 1 / (j + 1)

                z = list(zip(eids, -np.array(scores)))
                z = np.array(z, dtype=dt)
                sorted_z = np.sort(z, order='rev_score')
                ranked_set = eid_set[:3] + [x[0] for x in sorted_z]
                ranked_expanded_sets.append(ranked_set)
        else:
            ranked_expanded_sets = expanded_sets

        return ranked_expanded_sets

    def load_model(self, path):
        self.model.load_state_dict(torch.load(path))

    def make_eindex2dist(self, batchsize=256, model_id=None):
        self.model.cuda()
        eindex2logdist = []
        eindex2dist = []
        pkl_path_e2d = self.pkl_path_e2d
        pkl_path_e2logd = self.pkl_path_e2logd
        if model_id is not None:
            pkl_path_e2d += str(model_id)
            pkl_path_e2logd += str(model_id)
        print('Total entities: %d' % len(self.eid2sents))
        print('Making %s and %s ...' % (pkl_path_e2d, pkl_path_e2logd))
        for i, eid in enumerate(self.eid2sents):
            list_dists = []
            dataset = Eid2Data(eid, self.eid2sents, [])
            data_loader = DataLoader(dataset, batch_size=batchsize, collate_fn=collate_fn)

            with torch.no_grad():
                for j, batch in enumerate(data_loader):
                    output = self.model.forward(batch[0].cuda())
                    output = output[1] if output[0] is None else output[0]
                    list_dists.append(output)

            log_dists = torch.cat(list_dists).cpu().numpy()
            eindex2logdist.append(np.mean(log_dists, axis=0))
            dists = np.exp(log_dists)
            eindex2dist.append(np.mean(dists, axis=0))

            torch.cuda.empty_cache()
            if i % 2000 == 0:
                print(i)

        print('Writing to disk ...')
        pickle.dump(eindex2logdist, open(pkl_path_e2logd, 'wb'))
        pickle.dump(eindex2dist, open(pkl_path_e2d, 'wb'))
        self.eindex2logdist = eindex2logdist
        self.eindex2dist = eindex2dist

    def ensemble_eindex2dists(self, model_ids):
        pkl_name = self.pkl_path_e2d
        # for model_id in model_ids[:-1]:
        #     pkl_name += str(model_id) + '-'
        # pkl_name += str(model_ids[-1])
        log_pkl_name = pkl_name + '_log'
        print('Making %s and %s ...' % (pkl_name, log_pkl_name))
        eindex2dist = pickle.load(open(self.pkl_path_e2d + str(model_ids[0]), 'rb'))
        eindex2logdist = pickle.load(open(self.pkl_path_e2logd + str(model_ids[0]), 'rb'))
        assert len(eindex2dist) == len(eindex2logdist)
        print(model_ids[0])
        for model_id in model_ids[1:]:
            this_eindex2dist = pickle.load(open(self.pkl_path_e2d + str(model_id), 'rb'))
            this_eindex2logdist = pickle.load(open(self.pkl_path_e2logd + str(model_id), 'rb'))
            for i in range(len(eindex2dist)):
                eindex2dist[i] = eindex2dist[i] + this_eindex2dist[i]
                eindex2logdist[i] = eindex2logdist[i] + this_eindex2logdist[i]
            print(model_id)
        for i in range(len(eindex2dist)):
            eindex2dist[i] *= (1 / len(model_ids))
            eindex2logdist[i] *= (1 / len(model_ids))

        pickle.dump(eindex2dist, open(pkl_name, 'wb'))
        pickle.dump(eindex2logdist, open(log_pkl_name, 'wb'))
        self.eindex2logdist = eindex2logdist
        self.eindex2dist = eindex2dist

    # Calculate Model's Score
    def eval_model(self, save_path, fn_model, model_id, seeds_sets):
        pkl_path_e2d = self.pkl_path_e2d + str(model_id)
        if not os.path.exists(pkl_path_e2d):
            self.load_model(os.path.join(save_path + str(model_id), fn_model))
            self.model.cuda()
        else:
            self.eindex2dist = pickle.load(open(pkl_path_e2d, 'rb'))
        scores = []
        for cls_name in self.cls_names:
            print(cls_name)
            divergs = []
            seeds = set()
            for seed_set in seeds_sets[cls_name]:
                for eid in seed_set:
                    seeds.add(eid)

            for eid in seeds:
                for eid2 in seeds:
                    if eid == eid2:
                        continue

                    if os.path.exists(pkl_path_e2d):
                        dist = self.get_feature_dist(eid)
                        dist2 = self.get_feature_dist(eid2)
                    else:
                        dist = self.get_feature_dist2(eid)
                        dist2 = self.get_feature_dist2(eid2)

                    dist[self.eid2index[eid2]] = dist2[self.eid2index[eid2]]
                    dist2[self.eid2index[eid]] = dist[self.eid2index[eid]]

                    diverg = KL_divergence(dist, dist2)
                    divergs.append(diverg)

            mean_diverg = np.mean(divergs)
            print(mean_diverg, '')
            scores.append(mean_diverg)

        score = geometric_mean(scores)
        return score

    def get_mean_log_dist(self, eid):
        mean_log_dist = self.eindex2logdist[self.eid2index[eid]]
        mean_log_dist = standardization(mean_log_dist)
        return mean_log_dist

    def get_feature_dist(self, eid):
        feature_dist = self.eindex2dist[self.eid2index[eid]]
        return feature_dist

    def get_feature_dist2(self, eid, batchsize=128):
        dataset = Eid2Data(eid, self.eid2sents, [])
        data_loader = DataLoader(dataset, batch_size=batchsize, collate_fn=collate_fn)
        list_dists = []
        for _, batch in enumerate(data_loader):
            with torch.no_grad():
                output = self.model.forward(batch[0].cuda())
                output = output[1] if output[0] is None else output[0]
                list_dists.append(output)
        dist = torch.cat(list_dists).cpu().numpy()
        dist = np.mean(np.exp(dist), axis=0)
        return dist


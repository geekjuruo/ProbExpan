import os
import torch
import argparse
from utils import *
from Expan import Expan
import sys

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-dataset', default='../data/wiki', help='path to dataset folder')
    parser.add_argument('-vocab', default='entity2id.txt', help='vocab file')
    parser.add_argument('-pkl_e2s', default='entity2sents.pkl', help='name of entity2sents pkl file')
    parser.add_argument('-pretrained_model', default=None, help='name of pretrained model parameters')
    parser.add_argument('-save_path', default='../model_wiki', help='path to place model parameters')
    parser.add_argument('-pkl_e2d', default='entity2dist', help='name of entity2dist pkl file')
    parser.add_argument('-output', default='ensemble+winodw+rank', help='file name for output')
    parser.add_argument('-ensemble', default=True)
    parser.add_argument('-num_model', default=5)
    parser.add_argument('-num_top_model', default=2)
    parser.add_argument('-CL', action="store_true")
    parser.add_argument('-pkl_cls2eids', default='cls2eids.pkl', help='name of cls2eids pkl file')
    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    print(args)

    if not os.path.exists(os.path.join(args.dataset, args.output)):
        os.mkdir(os.path.join(args.dataset, args.output))

    class_names = []
    query_sets = dict()
    gt = dict()
    num_query_per_class = 0
    for file in os.listdir(os.path.join(args.dataset, 'query')):
        class_name = file.split('.')[0]
        class_names.append(class_name)
        query_sets[class_name] = []
        gt[class_name] = set()
        num_query_per_class = 0

        with open(os.path.join(args.dataset, 'query', file), encoding='utf-8') as f:
            for line in f:
                if line == 'EXIT\n':
                    break
                num_query_per_class += 1
                temp = line.strip().split(' ')
                query_sets[class_name].append([int(eid) for eid in temp])

        with open(os.path.join(args.dataset, 'gt', file), encoding='utf-8') as f:
            for line in f:
                temp = line.strip().split('\t')
                eid = int(temp[0])
                if int(temp[2]) >= 1:
                    gt[class_name].add(eid)

    expan = Expan(args, class_names)

    if args.ensemble:
        top_model_ids = []
        model_ids = np.array(range(1, args.num_model+1))

    if args.pretrained_model is not None:
        if args.ensemble:
            scores = []
            print('Scores:')
            for i in model_ids:
                score = expan.eval_model(args.save_path, args.pretrained_model, i, query_sets)
                scores.append(score)
                print('[Model %d] %f\n' % (i, score))
            top_model_ids = model_ids[np.argsort(scores)[:args.num_top_model]]
            print(top_model_ids)

    else:
        if args.ensemble:
            for i in model_ids:
                print('[Model %d]' % i)
                if args.CL:
                    expan.pretrain(args.save_path+str(i), lr=1.5e-5, epoch=6, batchsize=256, num_sen_per_entity=256, smoothing=0.075)
                else:
                    expan.pretrain(args.save_path+str(i), lr=1e-5, epoch=5, batchsize=128, num_sen_per_entity=256, smoothing=0.075)
                print('')
            exit(0)
        else:
            if args.CL:
                expan.pretrain(args.save_path, lr=1.5e-5, epoch=6, batchsize=256, num_sen_per_entity=256, smoothing=0.075)
            else:
                expan.pretrain(args.save_path, lr=1e-5, epoch=5, batchsize=128, num_sen_per_entity=256, smoothing=0.075)
            exit(0)

    if not os.path.exists(os.path.join(args.dataset, args.pkl_e2d)):
        if args.ensemble:
            for i in top_model_ids:
                if not os.path.exists(os.path.join(args.dataset, args.pkl_e2d + str(i))):
                    expan.load_model(os.path.join(args.save_path + str(i), args.pretrained_model))
                    expan.make_eindex2dists(batchsize=128, model_id=i)
            expan.ensemble_eindex2dists(top_model_ids)
        else:
            expan.load_model(os.path.join(args.save_path, args.pretrained_model))
            expan.make_eindex2dists(batchsize=128)

    '''
    Expanding and Evalutation
    '''

    MAPs = [0, 0, 0, 0]
    num_class = len(class_names)
    target_size = 103
    if not args.CL:
        target_size = 203

    with open(os.path.join(args.dataset, args.output, 'summary.txt'), 'w') as file_summary:
        for i in range(0, num_query_per_class):
            print('\n[Test %d]' % (i+1))
            query_set = [query_sets[cls_name][i] for cls_name in class_names]
            # Hyperparamter setting for wiki
            expanded = expan.expand(query_set, target_size=target_size, ranking=True, mu=9, win_grow_rate=2.5, win_grow_step=20)
            AP10s, AP20s, AP50s, AP100s = [[], [], [], []]
            for j, cls in enumerate(class_names):
                with open(os.path.join(args.dataset, args.output, f'{i}_{cls}'), 'w') as f:
                    AP10, AP20, AP50, AP100 = [apk(gt[cls], expanded[j], n) for n in [10, 20, 50, 100]]
                    AP10s.append(AP10)
                    AP20s.append(AP20)
                    AP50s.append(AP50)
                    AP100s.append(AP100)

                    print(AP10, AP20, AP50, AP100, file=f)
                    print('', file=f)
                    for eid in expanded[j]:
                        print(f'{eid}\t{expan.eid2name[eid]}', file=f)

            MAPs[0] += sum(AP10s) / num_class
            MAPs[1] += sum(AP20s) / num_class
            MAPs[2] += sum(AP50s) / num_class
            MAPs[3] += sum(AP100s) / num_class
            for j, cls in enumerate(class_names):
                print('[%s]\t %.6f %.6f %.6f %.6f' % (cls, AP10s[j], AP20s[j], AP50s[j], AP100s[j]))
            print('[TEST %d]' % (i + 1), file=file_summary)
            print('MAP %.6f %.6f %.6f %.6f\n' %
                  (sum(AP10s) / num_class, sum(AP20s) / num_class, sum(AP50s) / num_class, sum(AP100s) / num_class),
                  file=file_summary)

        print('\nTotal MAP %.6f %.6f %.6f %.6f\n' %
              (MAPs[0] / num_query_per_class, MAPs[1] / num_query_per_class,
               MAPs[2] / num_query_per_class, MAPs[3] / num_query_per_class), file=file_summary)


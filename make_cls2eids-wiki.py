import os
import argparse
import pickle
from utils import *

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-dataset', default='../data/wiki', help='path to dataset folder')
    parser.add_argument('-pkl_cls2eids', default='cls2eids.pkl', help='name of cls2eids pkl file')
    parser.add_argument('-path_expand_result', default='ensemble+winodw+rank')
    parser.add_argument('-path_num_cls2eids', default='num_cls2eids-wiki.txt')
    args = parser.parse_args()

    cls2eids = dict()
    gt_eids = dict()
    classes = []
    for file in os.listdir(os.path.join(args.dataset, 'gt')):
        name_cls = file.split('.')[0]
        name_neg_cls = 'neg ' + name_cls
        cls2eids[name_cls] = set()
        cls2eids[name_neg_cls] = set()
        gt_eids[name_cls] = []
        classes.append(name_cls)
        with open(os.path.join(args.dataset, 'gt', file), encoding='utf-8') as f:
            for line in f:
                temp = line.strip().split('\t')
                eid = int(temp[0])
                if int(temp[2]) > 0:
                    gt_eids[name_cls].append(eid)

    # get positive entities from pre-expanded results
    for cls_name in classes:
        for file in os.listdir(os.path.join(args.dataset, args.path_expand_result)):
            if cls_name not in file:
                continue
            with open(os.path.join(args.dataset, args.path_expand_result, file), encoding='utf-8') as f:
                for i, line in enumerate(f):
                    if 2 <= i < 7:
                        temp = line.strip().split('\t')
                        cls2eids[cls_name].add(int(temp[0]))
                    if 7 <= i < 12 and cls_name != 'sportsleagues':
                        temp = line.strip().split('\t')
                        cls2eids[cls_name].add(int(temp[0]))

    # get positive entities from seeds
    for file in os.listdir(os.path.join(args.dataset, 'query')):
        cls_name = file.split('.')[0]
        with open(os.path.join(args.dataset, 'query', file), encoding='utf-8') as f:
            for line in f:
                if line == 'EXIT\n':
                    break
                temp = line.strip().split(' ')
                for eid in temp:
                    cls2eids[cls_name].add(int(eid))

    # get negative entities from seeds
    neg_nums = dict()
    for cls_name in classes:
        neg_nums[cls_name] = dict()
        name_neg_cls = 'neg ' + cls_name
        for i in range(5):
            neg_nums[cls_name][i] = 0
        for file in os.listdir(os.path.join(args.dataset, args.path_expand_result)):
            if cls_name not in file:
                continue
            with open(os.path.join(args.dataset, args.path_expand_result, file), encoding='utf-8') as f:
                for i, line in enumerate(f):
                    if i >= 2:
                        temp = line.strip().split('\t')
                        eid = int(temp[0])
                        if 112 <= i < 122:
                            if cls_name in ['countries', 'sportsleagues', 'china_provinces', 'us_states']:
                                cls2eids[name_neg_cls].add(eid)
                        if 172 <= i < 202 and i % 2:
                            cls2eids[name_neg_cls].add(eid)

    # print(str(classes))
    # for i in range(5):
    #     num = 0
    #     temp = '[' + str(100 + i*20 + 1) + '-' + str(100 + (i+1)*20) + ']'
    #     for cls_name in classes:
    #         temp += '\t' + ('%2d' % neg_nums[cls_name][i])
    #         num += neg_nums[cls_name][i]
    #     temp += '\t' + str('%2d' % num)
    #     print(temp)

    with open(args.path_num_cls2eids, 'w') as f:
        for i, cls_name in enumerate(cls2eids):
            print('[%s]\t%d' % (cls_name, len(cls2eids[cls_name])), file=f)
            if i % 2:
                print('', file=f)

    pickle.dump(cls2eids, open(os.path.join(args.dataset, args.pkl_cls2eids), 'wb'))

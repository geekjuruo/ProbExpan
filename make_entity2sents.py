import os
from transformers import BertTokenizer
import argparse
import pickle
from utils import *

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-dataset', default='../data/wiki', help='path to dataset folder')
    parser.add_argument('-vocab', default='entity2id.txt', help='vocab file')
    parser.add_argument('-sent', default='sentences.json', help='sent file')
    parser.add_argument('-pkl_e2s', default='entity2sents.pkl', help='name of entity2sents pkl file')
    parser.add_argument('-path_num_sents', default='num_sents-wiki.txt')
    parser.add_argument('-max_len', default=150, help='max sentence len')
    args = parser.parse_args()


    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=False)
    mask_token = tokenizer.mask_token

    eid2name, vocab, eid2idx = load_vocab(os.path.join(args.dataset, args.vocab))

    entity2sents = dict()
    for eid in vocab:
        entity2sents[eid] = []

    filename = os.path.join(args.dataset, args.sent)
    total_line = get_num_lines(filename)
    with open(filename, 'r', encoding='utf-8') as f:
        for line in tqdm(f, total=total_line):
            obj = json.loads(line)
            if len(obj['entityMentions']) == 0 or len(obj['tokens']) > args.max_len:
                continue
            raw_sent = [token.lower() for token in obj['tokens']]
            for entity in obj['entityMentions']:
                eid = entity['entityId']
                sent = copy.deepcopy(raw_sent)
                sent[entity['start']:entity['end'] + 1] = [mask_token]
                entity2sents[eid].append(tokenizer.encode(sent))

    drop_eids = []
    with open(args.path_num_sents, 'w') as f:
        print(len(entity2sents), file=f)
        print('', file=f)

        total_size = 0
        cnt = 0
        for eid in entity2sents:
            siz = len(entity2sents[eid])
            if siz < 2:
                drop_eids.append(eid)
            else:
                total_size += siz
                print('%d\t%s\t%d' % (cnt, eid2name[eid], siz), file=f)
                cnt += 1

        print('\nTotal entities %d' % cnt, file=f)
        print('Total sents %d' % total_size, file=f)

    for eid in drop_eids:
        entity2sents.pop(eid)

    print(len(entity2sents))
    pickle.dump(entity2sents, open(os.path.join(args.dataset, args.pkl_e2s), 'wb'))

# ProbExpan

Pytorch Implementation for SIGIR2022 full paper "Contrastive Learning with Hard Negative Entities for Entity Set Expansion".

## Prerequisites
#### python >= 3.7
#### pytorch >= 1.6.0

## Data

The download links of datasets used in our experiments are all public available in their original paper mentioned in Appendix B. After downloading the dataset, put them under the folder "./data/"

#### Following instructions use Wiki as defalut dataset

## Data Preprocessing

Run
```
python make_entity2sents.py
```
to get "./data/wiki/entity2sents.pkl"

## Learning Phase 1 and 2 

To train multiple models with Masked Entity Prediction task and ensemble top models, run
```
python main.py -num_model 5 -num_top_model 2
```

After pretraining, run
```
python main.py -num_model 5 -num_top_model 2 -pretrained_model epoch_5.pkl
```

Expansion result will be saved under "./data/wiki/ensemble+winodw+rank"

## Learning Phase 3 and 4

To train multiple models with Contrastive Loss and ensemble top models, first run
```
python make_cls2eids-wiki.py
```

then run 
```
python main.py -CL -num_model 5 -num_top_model 2 -output cl+ensemble+winodw+rank
```

After pretraining, run
```
python main.py -CL -num_model 5 -num_top_model 2 -output cl+ensemble+winodw+rank -pretrained_model epoch_5.pkl
```

Expansion result will be saved under "./data/wiki/cl+ensemble+winodw+rank"

## Multi GPU

To train single model with Masked Entity Prediction task on Multi GPU, run
```
python -m torch.distributed.launch --nproc_per_node=[NUM_GPU] mlm-pretrain-multiGPU.py
```



## Citation

If you consider our paper or code useful, please cite our paper:

```
@inproceedings{10.1145/3477495.3531954,
author = {Li, Yinghui and Li, Yangning and He, Yuxin and Yu, Tianyu and Shen, Ying and Zheng, Hai-Tao},
title = {Contrastive Learning with Hard Negative Entities for Entity Set Expansion},
year = {2022},
isbn = {9781450387323},
publisher = {Association for Computing Machinery},
address = {New York, NY, USA},
url = {https://doi.org/10.1145/3477495.3531954},
doi = {10.1145/3477495.3531954},
abstract = {Entity Set Expansion (ESE) is a promising task which aims to expand entities of the target semantic class described by a small seed entity set. Various NLP and IR applications will benefit from ESE due to its ability to discover knowledge. Although previous ESE methods have achieved great progress, most of them still lack the ability to handle hard negative entities (i.e., entities that are difficult to distinguish from the target entities), since two entities may or may not belong to the same semantic class based on different granularity levels we analyze on. To address this challenge, we devise an entity-level masked language model with contrastive learning to refine the representation of entities. In addition, we propose the ProbExpan, a novel probabilistic ESE framework utilizing the entity representation obtained by the aforementioned language model to expand entities. Extensive experiments and detailed analyses on three datasets show that our method outperforms previous state-of-the-art methods.},
booktitle = {Proceedings of the 45th International ACM SIGIR Conference on Research and Development in Information Retrieval},
pages = {1077–1086},
numpages = {10},
keywords = {knowledge discovery, entity set expansion, contrastive learning},
location = {Madrid, Spain},
series = {SIGIR '22}
}
```
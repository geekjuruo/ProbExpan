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

Expansion result will be saved under "./data/wiki/cl+ensemble+winodw+rank"

## Multi GPU

To train single model with Masked Entity Prediction task on Multi GPU, run
```
python -m torch.distributed.launch --nproc_per_node=[NUM_GPU] mlm-pretrain-multiGPU.py
```

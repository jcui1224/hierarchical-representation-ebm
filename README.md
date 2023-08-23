# Learning Hierarchical Features with Joint Latent Space Energy-Based Prior ([Project Page](https://jcui1224.github.io/hierarchical-representation-ebm-proj/))



## Train

```python
CUDA_VISIBLE_DEVICES=gpu0 python train_hs.py / train_fid.py
```

### Test

```python
CUDA_VISIBLE_DEVICES=gpu0 python sandbox-FIDScore.py / sandbox-HierarchicalSampling.py
```

Checkpoint is relaseed. Please update the checkpoint directory accordingly in sandbox-FIDScore.py or sandbox-HierarchicalSampling.py.




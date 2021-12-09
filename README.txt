# LR-GCCF-AAAI20
Revisiting Graph based Collaborative Filtering: A Linear Residual Graph Convolutional Network Approach

![Overall_framework](images/Overall_framework.jpg)


Graph Convolutional Networks (GCNs) are state-of-the-art graph based representation learning models by iteratively stacking multiple layers of convolution aggregation operations and non-linear activation operations. Recently, in Collaborative Filtering (CF) domain, by treating the user-item interaction behavior as a bipartite graph, some researchers model the higher-layer collaborative signals with GCNs to alleviate the data sparsity issue in CF, and show superior performance compared to traditional works. However, these GCN based recommendation models suffer from the complexity and training difficulty with non-linear activations for large user-item graphs, and usually could not model deep layers of graph convolutions due to the over smoothing problem in the iterative process. In this paper, we revisit these graph based collaborative filtering models from two aspects. First, we empirically show that removing non-linearities would enhance recommendation performance, which is consistent with the theories in simple graph convolutional networks. Second, we propose a residual network structure that is specifically designed for CF with user-item interaction modeling, which alleviates the over smoothing problem in graph convolution aggregation operation with sparse data. The proposed model is a linear model and it is easy to train, scales to large datasets, and yields better efficiency and effectiveness on two real datasets.

We provide PyTorch implementations for LR-GCCF model.


## Train/test

- Train a model:

```python
#!./LR-GCCF/code
cd LR-GCCF
cd code
#for amazon dataset
python train_amazon.py
#for gowalla dataset
python train_gowalla.py
```

- Test a model:

```python
#!./LR-GCCF/code
cd LR-GCCF
cd code
#for amazon dataset
python test_amazon.py
#for gowalla dataset
python test_gowalla.py
```
# 10708-Project

## Intro
The Graph Convolutional Networks (GCN) originally proposed by Kipf and Welling are continued studied on different papers towards different directions and extent. In a lot of those area, it was shown that GCNs are effective graph model for semi-supervised learning. However, the original model was transductive meaning the premise for the constitution of the data is relatively harsh. Along with other defects like high-scale expansion along the network, FastGCN was introduced in 2018. By interpret the each layer as an integral operator of the input embedding over some probability measures, the writer used Monte Carlo approaches to further estimate the integral operator. We show the original model and some of the improvements and trails we made on the original FastGCN.

The codes are adapted from [FastGCN in PyTorch](https://github.com/Gkunnan97/FastGCN_pytorch).


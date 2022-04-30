# 10708-Project

## Intro
In this paper we attempt to modify Fast-GCN in order to improve both the speed and accuracy of research paper classification. We propose two separate methods for generating a sampling distribution at each layer in the GCN that we hypothesize could outperform the sampling method in Fast-GCN. Our results, however, show that in the case of the Pubmed and Cora datasets, our methods do not in fact outperform Fast-GCN. We explore and discuss potential reasons for this below in the paper. Additionally, we show that in the case of the Pubmed dataset, simpler methods like a Random Forest significantly outperform all graphical convolutional networks, while in the Cora dataset, graphical convolutional networks perform better. We explore reasons behind the discrepancy between the performances of both classes of models on these two similar datasets.

The codes are adapted from [FastGCN in PyTorch](https://github.com/Gkunnan97/FastGCN_pytorch).



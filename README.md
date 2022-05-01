# 10708-Project

## Intro
In this paper we attempt to modify Fast-GCN in order to improve both the speed and accuracy of research paper classification. We propose two separate methods for generating a sampling distribution at each layer in the GCN that we hypothesize could outperform the sampling method in Fast-GCN. Our results, however, show that in the case of the Pubmed and Cora datasets, our methods do not in fact outperform Fast-GCN. We explore and discuss potential reasons for this below in the paper. Additionally, we show that in the case of the Pubmed dataset, simpler methods like a Random Forest significantly outperform all graphical convolutional networks, while in the Cora dataset, graphical convolutional networks perform better. We explore reasons behind the discrepancy between the performances of both classes of models on these two similar datasets.

The codes are adapted from [FastGCN in PyTorch](https://github.com/Gkunnan97/FastGCN_pytorch).

## Background/Dataset
|              | Nodes | Edges | Classes | Features |
|:-------------|:------:|:------:|:------:|:------:|
|Cora     |  2,708 |  5,429 |  7 |  1433 |
|PubMed       |  19,717 |  44,338 |  3 |  500 |

For our baselines, we ran 3 different state of art Graph Convolutional Network models (Vanilla GCN, Fast-GCN, and AS-GCN). We found that Fast-GCN was significantly faster than the other methods on both Cora and Pubmed. However, the test accuracy of Fast-GCN was slightly worse than the other two methods.

![Test Accuracy of 3 models over Cora Dataset](https://github.com/Mr-Msz/10708-Project/blob/main/Images/Pre_Test_Acc_Cora.png)
![Train Time of 3 models over Cora Dataset](https://github.com/Mr-Msz/10708-Project/blob/main/Images/Pre_Train_Time_Cora.png)
<p align="center">Figure (1) Train/Test Outcome on Cora Dataset</p>

 
![Test Accuracy of 3 models over Pubmed Dataset](https://github.com/Mr-Msz/10708-Project/blob/main/Images/Pre_Test_Acc_Pub.png)
![Test Accuracy of 3 models over Pubmed Dataset](https://github.com/Mr-Msz/10708-Project/blob/main/Images/Pre_Train_Time_Pub.png)
<p align="center">Figure (2) Train/Test Outcome on PubMed Dataset</p>

## Experiment & Result
Experiment can be found in this [report](https://github.com/Mr-Msz/10708-Project/blob/main/10708_Final_Project.pdf) (Section 4)
###Result
(1) c value:
For our first alternative sampling method, we tried different values of the hyper-parameter c in order to find the optimal weighting between uniform sampling and our calculated probability distribution. We found that c = 0.2 worked the best on both Pubmed and Cora dataset, and plotted the results:

<p align="center">
  <img src="https://github.com/Mr-Msz/10708-Project/blob/main/Images/c-graph.png" />
</p>
<p align="center">Figure 3: Test Accuracy with different value of c</p>

(2) Performance:
After determining the optimal value of c (0.2), we began comparing the performance of different methods on the two dataset. The following table is the final performance for each of the methods (test accuracy):
|Proposed & Baseline| FastGCN | FastGCN with Alternative Sampling | Hybrid Methods |
|:-------------|:------:|:------:|:------:|
|Cora     |  0.854 |  0.860 |   0.680 |
|PubMed       |  0.844 |  0.847 |   0.886 |

|Conventional ML| Logistic Regression | Multi-layer Perceptron | Random Forest |
|:-------------|:------:|:------:|:------:|
|Cora     |  0.363 |   0.737 |    0.722 |
|PubMed       |  0.854 |   0.880 |    0.901 |

Below we present a comparison between GCN-Related Techniques in terms of train and test accuracy over time. Each figure contains 4 lines; one for each of the following: Vanilla FastGCN Methods, FastGCN with Alternative Sampling Methods, and the Hybrid Method.

<p align="center">
  <img src="https://github.com/Mr-Msz/10708-Project/blob/main/Images/Train_Acc_Cora.png" />
  <img src="https://github.com/Mr-Msz/10708-Project/blob/main/Images/Test_Acc_Cora.png" />
   <img src="https://github.com/Mr-Msz/10708-Project/blob/main/Images/Train_Time_Cora.png" />
</p>
<p align="center">Figure (4) Train/Test Outcome on Cora Dataset</p>

<p align="center">
  <img src="https://github.com/Mr-Msz/10708-Project/blob/main/Images/Train_Acc_Pubmed.png" />
  <img src="https://github.com/Mr-Msz/10708-Project/blob/main/Images/Test_Acc_Pubmed.png" />
   <img src="https://github.com/Mr-Msz/10708-Project/blob/main/Images/Train_Time_Pubmed.png" />
</p>
<p align="center">Figure (4) Train/Test Outcome on Pubmed Dataset</p>

9 References and citations
References
[1] Jie Chen, Tengfei Ma, and Cao Xiao. “Fastgcn: fast learning with graph convolutional networks via importance sampling”. In: arXiv preprint arXiv:1801.10247 (2018).
[2] Lei Chen et al. “Revisiting graph based collaborative filtering: A linear residual graph convolutional network approach”. In: Proceedings of the AAAI conference on artificial intelligence. Vol. 34. 01. 2020, pp. 27–34.
[3] Wei-Lin Chiang et al. “Cluster-gcn: An efficient algorithm for training deep and large graph convolutional networks”. In: Proceedings of the 25th ACM SIGKDD International Conference on Knowledge Discovery & Data Mining. 2019, pp. 257–266.
[4] Thomas N Kipf and Max Welling. “Semi-supervised classification with graph convolutional networks”. In: arXiv preprint arXiv:1609.02907 (2016).
[5] Maosen Li et al. “Actional-structural graph convolutional networks for skeleton-based action recognition”. In: Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. 2019, pp. 3595–3603.
[6] Xin Liu et al. “Sampling methods for efficient training of graph convolutional networks: A survey”. In: IEEE/CAA Journal of Automatica Sinica 9.2 (2021), pp. 205–234.
[7] Cheng Wan et al. “BNS-GCN: Efficient Full-Graph Training of Graph Convolutional Networks with Partition-Parallelism and Random Boundary Node Sampling Sampling”. In: Proceedings of Machine Learning and Systems 4 (2022).
[8] Felix Wu et al. “Simplifying graph convolutional networks”. In: International conference on machine learning. PMLR. 2019, pp. 6861–6871.
[9] Liang Yao, Chengsheng Mao, and Yuan Luo. “Graph convolutional networks for text classification”. In: Proceedings of the AAAI conference on artificial intelligence. Vol. 33. 01. 2019, pp. 7370–7377.
[10] Si Zhang et al. “Graph convolutional networks: a comprehensive review”. In: Computational Social Networks 6.1 (2019), pp. 1–23.

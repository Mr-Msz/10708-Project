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
<p align="center">Figure (a)</p>

 
![Test Accuracy of 3 models over Pubmed Dataset](https://github.com/Mr-Msz/10708-Project/blob/main/Images/Pre_Test_Acc_Pub.png)
![Test Accuracy of 3 models over Pubmed Dataset](https://github.com/Mr-Msz/10708-Project/blob/main/Images/Pre_Train_Time_Pub.png)

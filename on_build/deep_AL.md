# Deep Active Learning

Deep neural networks is popular in machine learning community.


## Difficulty to utilize AL on neural networks


# Current works

## Survey/Review


- A Survey of Active Learning for Text Classification using Deep Neural Networks [[2020]](https://arxiv.org/pdf/2008.07267.pdf)
- A Survey of Deep Active Learning [[2020]](https://arxiv.org/pdf/2009.00236.pdf)
- [From Model-driven to Data-driven: A Survey on Active Deep Learning [2021]](https://arxiv.org/pdf/2101.09933.pdf)

## The taxonomy for the current works


## Criticism

Several works compare the current DeepAL methods, and state that their experiments are flawed.
We think these papers are very interesting.
So here we give the details of each paper

### 1. Parting with Illusions about Deep Active Learning [2019]

This work state that current state-of-art DeepAL works doesn't consider the parallel setting such as "Semi-supervised learning", "data augmentation" etc.
So they hold a comparative study on several AL strategies with SL and SSL training paradigms.
They hold the experiments on two tasks: image classification and semantic segmentation.

Results from classification task:
- AL works well with data augmentation, but data augmentation blurs the differences between AL strategies: they all perform largely the same.
- Combining SSL and AL can be yields an improvement over raw SSL.
- Relative ranking of the AL methods changes completely on different datasets
- AL selection strategy is counter-productive in the low-budget regime, even worse than Random Sampling.
- SSL-AL method clearly outperforms fine-tuning of a pre-trained ImageNet network in both high-and low-budget settings.

Results from semantic segmentation task:
- Random selection with SSL performs best

Overall conclusion:
- Current evaluation protocol used in active learning is sub-optimal which in turn leads to wrong conclusions about the methods’ performance.
- Modern semi-supervised learning algorithms applied in the conventional active learning setting show a higher relative performance increase.
- State-of-the-art active learning approaches often fail to outperform simple random sampling, especially when the labeling budget is small.

### 2. Towards Robust and Reproducible Active Learning Using Neural Networks [2020]
  Comparative works.
  A comparative study over state-of-art Deep AL methods.
  In short, it states that compared to the well-regularized RSB, state-of-the-art AL methods evaluated in this paper do not achieve any noticeable gain.


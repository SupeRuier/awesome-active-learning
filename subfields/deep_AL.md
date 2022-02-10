# Deep Active Learning

Deep neural network is popular in machine learning community.
Many researchers focus on how to utilize AL with neural networks or deep models in many tasks.

## Survey/Review

There already are several surveys for the DeepAL.
- A Survey of Active Learning for Text Classification using Deep Neural Networks [[2020]](https://arxiv.org/pdf/2008.07267.pdf)
- A Survey of Deep Active Learning [[2020]](https://arxiv.org/pdf/2009.00236.pdf)
- [From Model-driven to Data-driven: A Survey on Active Deep Learning [2021]](https://arxiv.org/pdf/2101.09933.pdf)


## Difficulty to utilize AL on neural networks

1. To perform active learning, a model has to be able to learn from small amounts of data and represent its uncertainty over unseen data.
2. Many AL acquisition functions rely on model uncertainty. But in deep learning we rarely represent such model uncertainty.

# Current works

There are many works about Deep AL in the literature.
Here we only list the ones focused on the strategies or the framework design.

The taxonomy here is similar to the taxonomy [here](pb_classification.md).
However, due to the outstanding performance of semi-supervised learning in the deep learning literature, there are works include SSL into the AL framework.

| SL or SSL | Strategy Types                               | Description                                  | Famous Works                  |
| --------- | -------------------------------------------- | -------------------------------------------- | ----------------------------- |
| SL-based  | Informativeness                              | Measure the informativeness of each instance | EGL/MC-Dropout/ENS/BAIT       |
|           | Representativeness-impart                    | Represent the underlying distribution        | Core-set/BatchBALD/BADGE/VAAL |
|           | Learn to score                               | Learn a evaluation function directly.        | LL                            |
| SSL-based | (We didn't specify the strategy types here.) | The SSL is conclude into the framework       |                               |

## Supervised learning based

Supervised learning based methods means the model only use the labeled instances to train the model.
Here the taxonomy is similar to the one without using neural networks.

### Informativeness

The works under this category are focusing on how to evaluate the uncertainty of an instance for a neural network.

Uncertainty-based:
- A new active labeling method for deep learning [IJCNN, 2014]
- [Probabilistic Backpropagation for Scalable Learning of Bayesian Neural Networks [ICML, 2015]](http://proceedings.mlr.press/v37/hernandez-lobatoc15.pdf):
  Use an active learning scenario which is necessary to produce accurate estimates of uncertainty for obtaining good performance to estimates of the posterior variance on the weights produced by PBP(the proposed methods for BNN).
- [Active Discriminative Text Representation Learning [AAAI, 2017]](https://www.aaai.org/ocs/index.php/AAAI/AAAI17/paper/viewPaper/14174):
  Propose a new active learning (AL) method for text classification with convolutional neural networks (CNNs).
  Select instances that contain words likely to most affect the embeddings.
  Achieve this by calculating the expected gradient length (**EGL**) with respect to the embeddings for each word comprising the remaining unlabeled sentences.
- [Deep Bayesian Active Learning with Image Data [ICML, 2017]](https://dl.acm.org/doi/10.5555/3305381.3305504): (272)
  **MC-Dropout**.
  Address the difficulty of combining AL and Deep learning.
  Deep model depends on large amount of data.
  Deep learning methods rarely represent model uncertainty.
  This paper combine Bayesian deep learning into the active learning framework.
  Out perform than other kernel methods in image classification.
  Take the top b points with the highest BALD acquisition score.
- [Adversarial active learning for deep networks: a margin based approach [ICML, 2018]](https://arxiv.org/pdf/1802.09841.pdf)61:
  Based on theoretical works on margin theory for active learning, we know that such examples may help to considerably decrease the number of annotations. 
  While measuring the exact distance to the decision boundaries is intractable, we propose to rely on adversarial examples.
- [Towards Better Uncertainty Sampling: Active Learning with Multiple Views for Deep Convolutional Neural Network [ICME, 2019]](https://ieeexplore.ieee.org/abstract/document/8784806/)
- [DEAL: Deep Evidential Active Learning for Image Classification [2020, Arxiv]](https://arxiv.org/pdf/2007.11344.pdf):
  Recent AL methods for CNNs do not perform consistently well and are often computationally expensive.
  Replace the softmax standard output of a CNN with the parameters of a Dirichlet density.
  This paper have a summary of the previous works on deep AL.

Disagreement-based:
- [The power of ensembles for active learning in image classification [2018, CVPR]](https://openaccess.thecvf.com/content_cvpr_2018/papers/Beluch_The_Power_of_CVPR_2018_paper.pdf):
  **ENS**.

Fisher information:
- [Gone Fishing: Neural Active Learning with Fisher Embeddings [2021, Arxiv]](https://arxiv.org/pdf/2106.09675.pdf): 
  **BAIT**.
  This work also utilize greedy selection to build an AL batch.

Performance gain:
- Deep Active Learning by Leveraging Training Dynamics [2021]

### Representativeness-impart

Take into account the representativeness of the data.
This could be evaluated on the distribution.
Besides, the diversity of a batch could also make the selection more representative (batch mode selection).

There are following sub-categories:
- Density-based sampling 
- Diversity-based sampling (batch mode)
- Discriminator guided Sampling
- Expected loss on unlabeled data
- Graph-based

Density-based sampling:
- [Active learning for convolutional neural networks: A core-set approach [ICLR, 2018]](https://arxiv.org/abs/1708.00489):
  Define the problem of active learning as **core-set** selection, i.e. choosing set of points such that a model learned over the selected subset is competitive for the remaining data points.
  The empirical analysis suggests that they (Deep Bayesian Active Learning with Image Data [ICML, 2017]) do not scale to large-scale datasets because of batch sampling.
  Provide a rigorous bound between an average loss over any given subset of the dataset and the remaining data points via the geometry of the data points.
  And choose a subset such that this bound is minimized (loss minimization).
  Try to find a set of points to query labels such that when we learn a model, the performance of the model on the labelled subset and that on the whole dataset will be as close as possible.
  Batch active learning.
- [Ask-n-Learn: Active Learning via Reliable Gradient Representations for Image Classification [2020, Arxiv]](https://arxiv.org/pdf/2009.14448.pdf): 
  Use kmeans++ on the learned gradient embeddings to select instances.
- [Deep Active Learning by Model Interpretability [2020, Arxiv]](https://arxiv.org/pdf/2007.12100.pdf):
  In this paper, inspired by piece-wise linear interpretability in DNN, they introduce the linear separable regions of samples to the problem of active learning, and propose a novel Deep Active learning approach by Model Interpretability (DAMI).
  They use the local piece-wise interpretation in MLP as the representation of each sample, and directly run K-Center clustering to select and label samples
- [Diffusion-based Deep Active Learning[2020, Arxiv]](https://arxiv.org/pdf/2003.10339.pdf): 
  Build graph by the first hidden layer in DNN. 
  The selection is performed on the graph. 
  Consider a random walk as a mean to assign a label.

Diversity-based sampling (batch mode):
- [BatchBALD: Efficient and Diverse Batch Acquisition for Deep Bayesian Active Learning [NeurIPS, 2019]](http://papers.nips.cc/paper/8925-batchbald-efficient-and-diverse-batch-acquisition-for-deep-bayesian-active-learning):
  Batch acquisition with diverse but informative instances for deep bayesian network.
  Use a tractable approximation to the mutual information between a batch of points and model parameters as an acquisition function
- [Deep Batch Active Learning by Diverse, Uncertain Gradient Lower Bounds [2020, ICLR]](https://arxiv.org/abs/1906.03671)：
  **BADGE**. 
  Good paper, compared many previous Deep AL.
  Very representative for Deep AL.
  Capture uncertainty and diversity.
  Measure uncertainty through the magnitude of the resulting gradient with respect to parameters of the final (output) layer.
  To capture diversity, we collect a batch of examples where these gradients span a diverse set of directions by use k-means++ (made to produce a good initialization for k-means clustering).
- Density Weighted Diversity Based Query Strategy for Active Learning [2021, CSCWD]

Discriminator guided Sampling:
- [Variational Adversarial Active Learning [ICCV, 2019]](http://openaccess.thecvf.com/content_ICCV_2019/html/Sinha_Variational_Adversarial_Active_Learning_ICCV_2019_paper.html): 
  Use the idea from adversarial learning.
  **VAAL**.
  The intuition is to train a discriminator to decide which instance is least similar the labeled instance.
  This discriminator works as the selector.
  (The VAE is trained to fool the adversarial network to believe that all the examples are from the labeled data while the adversarial classifier is trained to differentiate labeled from unlabeled samples.) 
- [Deep Adversarial Active Learning With Model Uncertainty For Image Classification [2020, ICIP]](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9190726&tag=1): 
  Still distinguish between labeled and unlabeled data with a adversarial loss, but they try to use select instances dissimilar to the labeled data with higher prediction uncertainty. 
  This work is inspired by *Variational adversarial active learning*.
- [State-Relabeling Adversarial Active Learning [2020, CVPR]](https://arxiv.org/pdf/2004.04943.pdf):
  Train a discriminator as AL strategy.
  The discriminator is trained by a combined representation (supervised & unsupervised embedding) and the uncertainty calculated from the supervised model.
  The final selection is operated on the combined representation with the discriminator.
- [Task-Aware Variational Adversarial Active Learning [2021, CVPR]](https://openaccess.thecvf.com/content/CVPR2021/papers/Kim_Task-Aware_Variational_Adversarial_Active_Learning_CVPR_2021_paper.pdf)：
  Add a loss predictor upon VAAL.
  The intuition of loss predictor is similar to *Learning loss for active learning*.

Expected loss on unlabeled data:
- [Deep Active Learning Explored Across Diverse Label Spaces [Dissertation, 2018]](https://repository.asu.edu/attachments/201065/content/Ranganathan_asu_0010E_17759.pdf):
  Deep belief networks + AL with the most informative batch of unlabeled data points.
  The DBN is trained by combining both the labeled and unlabeled data with the aim to obtain least entropy on the unlabeled data and also least cross-entropy on the labeled data.
  CNN + AL in the paper.

Graph-based:
- [Sequential Graph Convolutional Network for Active Learning [2021, CVPR]](https://openaccess.thecvf.com/content/CVPR2021/papers/Caramalau_Sequential_Graph_Convolutional_Network_for_Active_Learning_CVPR_2021_paper.pdf)

Submodular-based:
- SIMILAR: Submodular Information Measures Based Active Learning In Realistic Scenarios [2021, Arxiv]

### Learn to score

- [Multi-criteria active deep learning for image classification [Knowledge-Based Systems, 2019]](https://www.sciencedirect.com/science/article/pii/S0950705119300747)3：
  Integrate different query strategies as well as make a performance balance among classes.
  The strategies are adapted.
- [Learning loss for active learning [2019, CVPR]](https://openaccess.thecvf.com/content_CVPR_2019/html/Yoo_Learning_Loss_for_Active_Learning_CVPR_2019_paper.html):
  **LL**.
  Predict the loss after selecting each instance.
- [Deep Active Learning with Adaptive Acquisition [Arxiv, 2019]](https://arxiv.org/abs/1906.11471)

## Semi/self-supervised learning imparted

Most of works only use the labeled instances to train the model.
There are several works utilize the unlabeled instances and use a semi-supervised training paradigm to build the framework.
Besides, self-supervised learning would also bring a huge performance improvement.
Here the works are categorized by **how the unlabeled instances are used**.

General work:
- Towards Good Practices for Efficiently Annotating Large-Scale Image Classification Datasets [2021, CVPR]
- [Consistency-Based Semi-supervised Active Learning: Towards Minimizing Labeling Cost [2021, ECCV]](https://link.springer.com/content/pdf/10.1007%2F978-3-030-58607-2_30.pdf)

Pseudo-labels：
- [Cost-effective active learning for deep image classification [IEEE TCSVT, 2016]](https://ieeexplore.ieee.org/abstract/document/7508942): (180)
  Besides AL, it also provide pseudo labels to the high confidence examples.
- [Rethinking deep active learning: Using unlabeled data at model training [2020, ICPR]](https://arxiv.org/abs/1911.08177)
- [AdaReNet: Adaptive Reweighted Semi-supervised Active Learning to Accelerate Label Acquisition [2021, PETRA]](https://dl.acm.org/doi/pdf/10.1145/3453892.3461321)
- [Semi-supervised Active Learning for Semi-supervised Models: Exploit Adversarial Examples with Graph-based Virtual Labels [2021, ICCV]](https://openaccess.thecvf.com/content/ICCV2021/papers/Guo_Semi-Supervised_Active_Learning_for_Semi-Supervised_Models_Exploit_Adversarial_Examples_With_ICCV_2021_paper.pdf): Take into account the stability of the pseudo-labels inferred by a GCN label propagator.
- ATM: An Uncertainty-aware Active Self-training Framework for Label-efficient Text Classification [2021]

Data Augmentation:
- [Parting with Illusions about Deep Active Learning [2019, Arxiv]](https://arxiv.org/pdf/1912.05361.pdf):
  Take into account the data augmentation and SSL in the literature.
  It considers the an image classification task and a semantic segmentation task.
- [Bayesian Generative Active Deep Learning [ICML, 2019]](https://arxiv.org/abs/1904.11643):
  Data augmentation and active learning at the same time.
  The labeled set not only have queried data but also the generated data in each loop.
  The classifier and the GAN model are trained jointly. 

Labeled-unlabeled data indistinguishable:
- [Deep Active Learning: Unified and Principled Method for Query and Training [2020, ICAIS]](https://arxiv.org/abs/1911.09162)
- [Visual Transformer for Task-aware Active Learning [2021]](https://arxiv.org/pdf/2106.03801.pdf)

Consistency (stay same after a distortion):
- [Consistency-Based Semi-supervised Active Learning: Towards Minimizing Labeling Cost [2020, ECCV]](https://link.springer.com/chapter/10.1007/978-3-030-58607-2_30): the selection is based on the consistency-task instead of the classification task.
- [Semi-Supervised Active Learning with Temporal Output Discrepancy [2021, Arxiv]](https://arxiv.org/pdf/2107.14153.pdf)

Contrastive Loss:
- [Improving Robustness and Efficiency in Active Learning with Contrastive Loss [2021, Arxiv]](https://arxiv.org/pdf/2109.06873.pdf): Select instances least similar to the labeled ones with the same class.
- Highly Efficient Representation and Active Learning Framework and Its Application to Imbalanced Medical Image Classification [2021, NeurIPS]: Self-supervised representation learning with a multi-class GP classifier.

Both self and semi-supervised imparted:
- Self-supervised Semi-supervised Learning for Data Labeling and Quality Evaluation [2021, Arxiv]: Use random selection with BYOL and Label Propagation.


# Criticism/Discovery on Deep AL

Several works compare the current DeepAL methods, and state that their experiments are flawed.
We think these papers are very interesting.
So here we give the details of each paper.

## 1. Parting with Illusions about Deep Active Learning [2019]

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

## 2. Towards Robust and Reproducible Active Learning Using Neural Networks [2020]

This work state the performance of random sampling baseline and AL strategies vary significantly over different papers.
With the goal of improving the reproducibility and robustness of AL methods, in this study they evaluate the performance of these methods for image classification compared to a random sampling in a fair experimental environment.
They also state that most AL works overlooked the regularization which would reduce the generalization error.
They hold the comparative study on different regularization setting.
(parameter norm penalty, random augmentation (RA), stochastic weighted averaging (SWA), and shake-shake (SS))

Results from image classification task:
- The performance of RS is significantly better than what they state in the other works. And there is no strategy performs clearly better than RS.
- With different AL batch size, the performance of strategies is inconsistent.
- AL methods do not outperform RS, and it isn't robust on class imbalanced setting.
- Models trained with RA and SWA consistently achieve significant performance gains across all AL iterations and exhibit appreciably smaller variance across multiple runs of the experiments.
- Consider the selected instances from VGG16 to ResNet18 and WRN-28-2, the performance varies. RS still performs well.

## 3. Effective Evaluation of Deep Active Learning on Image Classification Tasks [2021, Open Review]

Point out four issues for current AL works:
1. Contradictory observations on the performance of different AL algorithms
2. Unintended exclusion of important generalization approaches such as data augmentation and SGD for optimization
3. A lack of study of evaluation facets like the labeling efficiency of AL
4. Little or no clarity on the scenarios in which AL outperforms random sampling (RS).

They presented a unified re-implementation of state-of-the-art AL algorithms in the context of image classification.
Besides, they did a careful analysis of how existing approaches perform with varying amounts of redundancy and number of examples per class.

Point out some important details are not clear:
> - Should model training be restarted after every round, or can one fine-tune the current model? 
> - What is the effect of using carefully crafted seed sets in AL? 
> - Is there an ideal AL batch size, or does the batch size actually matter? 
> - When is AL more effective than random sampling, and what are the factors that affect the labeling efficiency of AL algorithms? Lastly, 
> - How scalable are AL algorithms; specifically, what is the amount of time taken by model training versus that taken by AL selection, and can this be made more compute and energy-efficient?

Key finding and takeaways:
> - Data augmentation and other generalization approaches have a considerable impact on the test performance as well as on the labeling efficiency.
> - SGD performs and generalizes better than Adam consistently in AL.
> - In the presence of data augmentation and the SGD optimizer, there is no significant advantage of diversity over very simple uncertainty sampling (at least in most standard academic data sets).
> - When we make the data set artificially redundant (either by repeating data points or by using near repetitions through augmentations), we see that BADGE starts outperforming uncertainty sampling.
> - The number of instances per class has an important impact on the performance of AL algorithms: The fewer the examples there are per class, the smaller the room there is for AL to improve over random sampling
> - The initialization of the labeled seed set (e.g., a random versus a more diverse or representative seed set) has little to no impact on the performance of AL after a few rounds; 
> - Reasonably sized choices of the AL batch size also have little to no impact.
> - Updating the models from previous rounds (fine-tuning) versus retraining the models from scratch negatively impacts the performance of AL only in the early selection rounds
> - The most time-consuming and energy-inefficient part of the AL loop is the model (re)training.

## 4. Reducing Label Effort: Self-Supervised meets Active Learning [2021, ICCV]

It shows that the improvements brought from AL are far less than that from self-training.
Only when the number of labeled instances approaches 50%, the gap could disappear.

## 5.  On the marginal benefit of active learning: Does self-supervision eat its cake? [2021, ICASSP]: 
Fail to observe any additional benefit of state-of-the-art active learning algorithms when combined with state-of-the-art S4L techniques.

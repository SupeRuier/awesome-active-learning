# Active Learning Summary
In this repository, previous works of active learning were categorized. 

- [Active Learning Summary](#active-learning-summary)
- [At the Beginning](#at-the-beginning)
- [Brief Introduction](#brief-introduction)
  - [Scenarios](#scenarios)
    - [Pool-based AL](#pool-based-al)
    - [Stream-based AL](#stream-based-al)
    - [Query Synthesis](#query-synthesis)
- [Previous Works](#previous-works)
  - [Query Strategy](#query-strategy)
    - [Heterogeneity/Disagreement-Based Models](#heterogeneitydisagreement-based-models)
      - [Uncertainty](#uncertainty)
      - [Margin](#margin)
      - [Entropy](#entropy)
      - [Ensemble Models](#ensemble-models)
    - [Performance-Based Models](#performance-based-models)
      - [Information or Performance Gain](#information-or-performance-gain)
      - [Expected Error Reduction](#expected-error-reduction)
      - [Expected Variance Reduction](#expected-variance-reduction)
    - [Representativeness-imparted Models](#representativeness-imparted-models)
      - [Information Density Frameworks](#information-density-frameworks)
    - [Meta Active Learning](#meta-active-learning)
    - [Others](#others)
  - [Models](#models)
    - [SVM/LR](#svmlr)
    - [Bayesian/Probabilistic](#bayesianprobabilistic)
    - [Gaussian Progress](#gaussian-progress)
    - [Neural Network](#neural-network)
  - [- [Deep Adversarial Active Learning With Model Uncertainty For Image Classification [2020, ICIP]](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9190726&tag=1): Still distinguish between labeled and unlabeled data with a adversarial loss, but they try to use select instances dissimilar to the labeled data with higher prediction uncertainty. This work is inspired by *Variational adversarial active learning*.](#--deep-adversarial-active-learning-with-model-uncertainty-for-image-classification-2020-icip-still-distinguish-between-labeled-and-unlabeled-data-with-a-adversarial-loss-but-they-try-to-use-select-instances-dissimilar-to-the-labeled-data-with-higher-prediction-uncertainty-this-work-is-inspired-by-variational-adversarial-active-learning)
  - [Theoretical Support for Active Learning](#theoretical-support-for-active-learning)
  - [Problem Settings Change (Combine AL with other settings)](#problem-settings-change-combine-al-with-other-settings)
    - [Multi-Class Active Learning](#multi-class-active-learning)
    - [Multi-Task Active Learning](#multi-task-active-learning)
      - [Multi-Label Active Learning](#multi-label-active-learning)
      - [With pre-defined constrains](#with-pre-defined-constrains)
    - [Multi-Domain Active Learning](#multi-domain-active-learning)
    - [Active Domain Adaptation](#active-domain-adaptation)
    - [Active Learning for Recommendation](#active-learning-for-recommendation)
    - [Active Learning for Remote Sensing Image Classification](#active-learning-for-remote-sensing-image-classification)
    - [Active Meta-Learning](#active-meta-learning)
    - [Semi-Supervised Active Learning](#semi-supervised-active-learning)
    - [Active Reinforcement Learning](#active-reinforcement-learning)
    - [Generative Adversarial Network with Active Learning](#generative-adversarial-network-with-active-learning)
    - [Others](#others-1)
  - [Practical Considerations](#practical-considerations)
    - [Batch mode selection](#batch-mode-selection)
    - [Varying Costs](#varying-costs)
    - [Noise Labelers](#noise-labelers)
    - [Multiple Labelers](#multiple-labelers)


# At the Beginning

Active learning is used to reduce the annotation cost in machine learning process.
There have been several surveys for this topic.
The main ideas and the scenarios are introduced in these surveys.

- Active learning: theory and applications [[2001]](https://ai.stanford.edu/~koller/Papers/Tong:2001.pdf.gz)
- **Active Learning Literature Survey (Recommend to read)**[[2009]](https://minds.wisconsin.edu/handle/1793/60660)
- A survey on instance selection for active learning [[2012]](https://link.springer.com/article/10.1007/s10115-012-0507-8)
- Active Learning: A Survey [[2014]](https://www.taylorfrancis.com/books/e/9780429102639/chapters/10.1201/b17320-27)

In the rest of this note, we collect the works in the following categories.
A short summary for each work might be provided.

# Brief Introduction
## Scenarios
### Pool-based AL
### Stream-based AL
### Query Synthesis

# Previous Works

## Query Strategy
### Heterogeneity/Disagreement-Based Models
#### Uncertainty
Uncertainty sampling simply queries an instance of which the predicted class value possesses a minimum probability among all candidate instances.
- Heterogeneous uncertainty sampling for supervised learning [1994, ICML]
- A sequential algorithm for training text classifiers [1994, SIGIR]
- [Support Vector Machine Active Learning with Applications to Text Classification [2001, JMLR]](http://www.jmlr.org/papers/v2/tong01a.html): 
  Version space reduction with SVM.
#### Margin
Margin in classification problems is calculated as diﬀerence between the ﬁrst and second highest class probability.
- Selective sampling with redundant views [2000, AAAI]
#### Entropy
- [Committee-based sampling for training probabilistic classifiers [1995, ICML]](https://dl.acm.org/doi/10.1145/1015330.1015385)
- Employing em and pool-based active learning for text classification [1998, ICML]
- Diverse ensembles for active learning [ICML, 2004]: 
  Use margins to measure ensemble disagreement but generalizes the idea to multi-class problems
- Active learning for probability estimation using jensen-shannon divergence [2005, ECML]
#### Ensemble Models
- Query learning strategies using boosting and bagging [1998, ICML]: 
  Query-by-Bagging, Query-by-Boosting.
- Selective sampling with redundant views [2000, AAAI]: 
  Co-testing.
### Performance-Based Models
#### Information or Performance Gain
#### Expected Error Reduction
#### Expected Variance Reduction
### Representativeness-imparted Models
#### Information Density Frameworks
### Meta Active Learning
### Others

-------------------
## Models
### SVM/LR
Most common models, we won't waste time here.
Most of classic strategies are based on these models.
### Bayesian/Probabilistic
- Employing EM and Pool-Based Active Learning for Text Classiﬁcation [[1998. ICML]](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.50.10&rep=rep1&type=pdf): 
  EM + Query-by-Committee (QBC-with-EM)
### Gaussian Progress
- Active instance sampling via matrix partition [2010, NIPS]: Gaussian Process. Maximizing a natural mutual information criterion between the labeled and unlabeled instances. No comparison with others.(69 citations)
- [Bayesian active learning for classification and preference learning [Arxiv, 2011]](https://arxiv.org/abs/1112.5745):
  Propose an approach that expresses information gain in terms of predictive entropies, and apply this method to the Gaussian Process Classifier (GPC).
  This method is referred as *BALD*.
  Capture how strongly the model predictions for a given data point and the model parameters are coupled, implying that ﬁnding out about the true label of data points with high mutual information would also inform us about the true model parameters.
- Adaptive active learning for image classiﬁcation [CVPR, 2013]
- [Active learning with Gaussian Processes for object categorization [2007, ICCV]](https://ieeexplore.ieee.org/abstract/document/4408844): Consider both the distance from the boundary as well as the variance in selecting the points; this is only possible due to the availability of the predictive distribution in GP regression. A significant boost in classification performance is possible, especially when the amount of training data for a category is ultimately very small.(303 citations)
- Safe active learning for time-series modeling with gaussian processes [2018, NIPS]
- Actively learning gaussian process dynamics



### Neural Network
- A new active labeling method for deep learning [IJCNN, 2014]
- Captcha recognition with active deep learning [Neural Computation, 2015]
- [Probabilistic Backpropagation for Scalable Learning of Bayesian Neural Networks [ICML, 2015]](http://proceedings.mlr.press/v37/hernandez-lobatoc15.pdf):
  Use an active learning scenario which is necessary to produce accurate estimates of uncertainty for obtaining good performance to estimates of the posterior variance on the weights produced by PBP(the proposed methods for BNN).
- [Cost-effective active learning for deep image classification [IEEE Transactions on Circuits and Systems for Video Technology, 2016]](https://ieeexplore.ieee.org/abstract/document/7508942): (180)
  Besides AL, it also provide pseudo labels to the high confidence examples.
- [Deep learning approach for active classification of electrocardiogram signals [2016, Information Science]](https://reader.elsevier.com/reader/sd/pii/S0020025516300184?token=EBB87D490BCDC26916121FCCCBAC34EFC879C7908C40ACF69667DCE1136B957C4608146ABABFCD7F438D7E7C8E4BA49C): (280 citations)
- [Deep Bayesian Active Learning with Image Data [ICML, 2017]](https://dl.acm.org/doi/10.5555/3305381.3305504): (272)
  Address the difficulty of combining AL and Deep learning.
  Deep model depends on large amount of data.
  Deep learning methods rarely represent model uncertainty.
  This paper combine **Bayesian deep learning** into the active learning framework.
  Out perform than other kernel methods in image classification.
  Take the top b points with the highest BALD acquisition score.
- [Active Discriminative Text Representation Learning [AAAI, 2017]](https://www.aaai.org/ocs/index.php/AAAI/AAAI17/paper/viewPaper/14174):
  Propose a new active learning (AL) method for text classiﬁcation with convolutional neural networks (CNNs).
  Select instances that contain words likely to most affect the embeddings.
  Achieve this by calculating the expected gradient length (EGL) with respect to the embeddings for each word comprising the remaining unlabeled sentences.
- [Fine-tuning Convolutional Neural Networks for Biomedical Image Analysis: Actively and Incrementally [CVPR, 2017]](http://openaccess.thecvf.com/content_cvpr_2017/html/Zhou_Fine-Tuning_Convolutional_Neural_CVPR_2017_paper.html):
  The CNN is fine-tuned in each active learning iteration **incrementally**.
  Assume that each candidate (predefined) takes one of possible labels Y.
  This assumption might make it more difficult to generalize.
- [Deep Active Learning Explored Across Diverse Label Spaces [Dissertation, 2018]](https://repository.asu.edu/attachments/201065/content/Ranganathan_asu_0010E_17759.pdf):
  Deep belief networks + AL with the most informative batch of unlabeled data points.
  The DBN is trained by combining both the labeled and unlabeled data with the aim to obtain least entropy on the unlabeled data and also least cross-entropy on the labeled data.
  CNN + AL in the paper [Deep active learning for image classification [ICIP, 2017]](https://ieeexplore.ieee.org/abstract/document/8297020).
- [Active learning for convolutional neural networks: A core-set approach [ICLR, 2018]](https://arxiv.org/abs/1708.00489):
  Define the problem of active learning as core-set selection, i.e. choosing set of points such that a model learned over the selected subset is competitive for the remaining data points.
  The empirical analysis suggests that they (Deep Bayesian Active Learning with Image Data [ICML, 2017]) do not scale to large-scale datasets because of batch sampling.
  Provide a rigorous bound between an average loss over any given subset of the dataset and the remaining data points via the geometry of the data points.
  And choose a subset such that this bound is minimized (loss minimization).
  Try to ﬁnd a set of points to query labels (s 1 ) such that when we learn a model, the performance of the model on the labelled subset and that on the whole dataset will be as close as possible.
  Batch active learning.
- [Deep active learning for named entity recognition [ICLR, 2018]](https://arxiv.org/abs/1707.05928)80:
  Incremental manner.
  Uncertainty-based heuristic, select those sentences for which the length-normalized log probability of the current prediction is the lowest.
- [Adversarial active learning for deep networks: a margin based approach [ICML, 2018]](https://arxiv.org/pdf/1802.09841.pdf)61:
  Based on theoretical works on margin theory for active learning, we know that such examples may help to considerably decrease the number of annotations. 
  While measuring the exact distance to the decision boundaries is intractable, we propose to rely on adversarial examples.
- [Deep Batch Active Learning by Diverse, Uncertain Gradient Lower Bounds [Arxiv, 2019][2020, ICLR]](https://arxiv.org/abs/1906.03671)：
  *BADGE*. 
  Good paper, compared many previous Deep AL.
  Very representative for Deep AL.
  Capture uncertainty and diversity.
  Measure uncertainty through the magnitude of the resulting gradient with respect to parameters of the ﬁnal (output) layer.
  To capture diversity, we collect a batch of examples where these gradients span a diverse set of directions by use k-means++ (made to produce a good initialization for k-means clustering).
- [Overcoming Practical Issues of Deep Active Learning and its Applications on Named Entity Recognition [Arxiv, 2019]](https://arxiv.org/abs/1911.07335)0
- [Deep Active Learning for Anchor User Prediction [IJCAI, 2019]](https://arxiv.org/abs/1906.07318)0
- [An Active Deep Learning Approach for Minimally Supervised PolSAR Image Classification [IEEE Transactions on Geoscience and Remote Sensing, 2019]](https://ieeexplore.ieee.org/abstract/document/8784406)4
- [Deep Active Learning for Axon-Myelin Segmentation on Histology Data [Arxiv, 2019]](https://arxiv.org/abs/1907.05143)1
- [Deep Active Learning with Adaptive Acquisition [Arxiv, 2019]](https://arxiv.org/abs/1906.11471)1
- [Towards Better Uncertainty Sampling: Active Learning with Multiple Views for Deep Convolutional Neural Network [ICME, 2019]](https://ieeexplore.ieee.org/abstract/document/8784806/)0
- [Active Deep Learning for Activity Recognition with Context Aware Annotator Selection [SIGKDD, 2019]](https://dl.acm.org/doi/abs/10.1145/3292500.3330688)1
- [BatchBALD: Efficient and Diverse Batch Acquisition for Deep Bayesian Active Learning [NIPS, 2019]](http://papers.nips.cc/paper/8925-batchbald-efficient-and-diverse-batch-acquisition-for-deep-bayesian-active-learning)3:
  Batch acquisition with diverse but informative instances for deep bayesian network.
  Use a tractable approximation to the mutual information between a batch of points and model parameters as an acquisition function
- [A deep active learning system for species identification and counting in camera trap images [Arxiv, 2019]](https://arxiv.org/abs/1910.09716)0
- [Disentanglement based Active Learning [Arxiv, 2019]](https://arxiv.org/abs/1912.07018)0
- [Deep Active Learning: Unified and Principled Method for Query and Training [Arxiv, 2019]](https://arxiv.org/abs/1911.09162)0
- [Rethinking deep active learning: Using unlabeled data at model training [Arxiv, 2019]](https://arxiv.org/abs/1911.08177)1
- [Bayesian Generative Active Deep Learning [Arxiv, 2019]](https://arxiv.org/abs/1904.11643)4:
  Data augmentation and active learning at the same time.
  The labeled set not only have queried data but also the generated data in each loop.
  The classifier and the GAN model are trained jointly. 
- [Deeper Connections between Neural Networks and Gaussian Processes Speed-up Active Learning [IJCAI, 2019]](https://arxiv.org/abs/1902.10350)0
- [Variational Adversarial Active Learning [ICCV, 2019]](http://openaccess.thecvf.com/content_ICCV_2019/html/Sinha_Variational_Adversarial_Active_Learning_ICCV_2019_paper.html): 
  Use the idea from adversarial learning.
  The intuition is to train a discriminator to decide which instance is least similar the labeled instance.
  This discriminator works as the selector.
  (The VAE is trained to fool the adversarial network to believe that all the examples are from the labeled data while the adversarial classifier is trained to differentiate labeled from unlabeled samples.) 52
- [Multi-criteria active deep learning for image classification [Knowledge-Based Systems, 2019]](https://www.sciencedirect.com/science/article/pii/S0950705119300747)3：
  Integrate different query strategies as well as make a performance balance among classes.
  The strategies are adapted.
- [Active and Incremental Learning with Weak Supervision [KI-Künstliche Intelligenz, 2020]](https://link.springer.com/article/10.1007/s13218-020-00631-4)0
- [Accelerating the Training of Convolutional Neural Networks for Image Segmentation with Deep Active Learning [Dissertation, 2020]](https://uwspace.uwaterloo.ca/handle/10012/15537)
- [QActor: On-line Active Learning for Noisy Labeled Stream Data [Arxiv, 2020]](https://arxiv.org/abs/2001.10399)0
- [Diffusion-based Deep Active Learning[2020, Arxiv]](https://arxiv.org/pdf/2003.10339.pdf): Build graph by the first hidden layer in DNN. The selection is performed on the graph. 
  Consider a random walk as a mean to assign a label
- [State-Relabeling Adversarial Active Learning [2020, Arxiv]](https://arxiv.org/pdf/2004.04943.pdf):
  Train a discriminator as AL strategy.
  The discriminator is trained by a combined representation (supervised & unsupervised embedding) and the uncertainty calculated from the supervised model.
  The final selection is operated on the combined representation with the discriminator.
- [DEAL: Deep Evidential Active Learning for Image Classification [2020, Arxiv]](https://arxiv.org/pdf/2007.11344.pdf):
  Recent AL methods for CNNs do not perform consistently well and are often computationally expensive.
  Replace the softmax standard output of a CNN with the parameters of a Dirichlet density.
  This paper have a summary of the previous works on deep AL.
- [Deep Active Learning by Model Interpretability [2020, Arxiv]](https://arxiv.org/pdf/2007.12100.pdf):
  In this paper, inspired by piece-wise linear interpretability in DNN, they introduce the linear separable regions of samples to the problem of active learning, and propose a novel Deep Active learning approach by Model Interpretability (DAMI).
- [Ask-n-Learn: Active Learning via Reliable Gradient Representations for Image Classification [2020, Arxiv]](https://arxiv.org/pdf/2009.14448.pdf): Use kmeans++ on the learned gradient embeddings to select instances.
- [Deep Adversarial Active Learning With Model Uncertainty For Image Classification [2020, ICIP]](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9190726&tag=1): Still distinguish between labeled and unlabeled data with a adversarial loss, but they try to use select instances dissimilar to the labeled data with higher prediction uncertainty. This work is inspired by *Variational adversarial active learning*.
-------------------
## Theoretical Support for Active Learning
Not really familiar with this.
Might fill this slot at the end.


-------------------
## Problem Settings Change (Combine AL with other settings)
### Multi-Class Active Learning 
- [Active Hidden Markov Models for Information Extraction [2001, International Symposium on Intelligent Data Analysis]](https://link.springer.com/chapter/10.1007/3-540-44816-0_31):
  Margin methods compares the most likely and second likely labels. 
- [Diverse ensembles for active learning [ICML, 2004]](https://dl.acm.org/doi/10.1145/1015330.1015385): 
  Use margins to measure ensemble disagreement but generalizes the idea to multi-class problems. 
  C4.5 as base learner.
- [Multi-class Ensemble-Based Active Learning [2006, ECML]](https://link.springer.com/chapter/10.1007/11871842_68): 
  Extract the most valuable samples by margin-based disagreement, uncertainty, sampling-based disagreement, or specific disagreement. 
  C4.5 as base learner.
- [Multi-Class Active Learning for Image Classification [CVPR, 2009]](https://ieeexplore.ieee.org/abstract/document/5206627):
  A comparison of BvSB and Entropy for active multi-classification. 
  They use one-vs-one SVM.
  **Well accepted solution for multi-class classification.**
  (Very simple paper, just a comparison, nothing else)
- [Active learning for large multi-class problems [CVPR, 2009]](https://ieeexplore.ieee.org/abstract/document/5206651):
  Introduce a probabilistic variant of the K-nearest neighbor (KNN) method for classiﬁcation that can be further used for active learning in multi-class scenarios.
- [A batch-mode active learning technique based on multiple uncertainty for SVM classifier [2011, Geoscience and Remote Sensing Letters]](https://ieeexplore.ieee.org/abstract/document/6092438/):
  Batch-mode SVM-based method. 
  Not only consider the smallest distances to the decision hyperplanes but also take into account the distances to other hyperplanes.
- [Scalable Active Learning for Multiclass Image Classification [TPAMI, 2012]](https://ieeexplore.ieee.org/abstract/document/6127880/): 
  Convert an active multi-class classification problem into a series active binary classification problem. 
  One-vs-one SVM. 
  Oracle answer if the query selection match the class of the other selected image
- [An active learning-based SVM multi-class classification model [2015, Pattern Recognition]](https://www.sciencedirect.com/science/article/pii/S003132031400497X):
  Use ove-vs-rest SVM, and select from the three types of unclear region (CBA, CCA, CNA).


### Multi-Task Active Learning  
Works under this category normally have multiple tasks on single domain.
When each task in multi-task learning is a classification task, MTAL degenerate to multi-label active learning.

#### Multi-Label Active Learning
- Multi-label svm active learning for image classification. [ICIP, 2004]:
  First MLAL work.
  Query all the labels of the selected instance.
- Two-dimensional active learning for image classiﬁcation [CVPR, 2008]:
  Authors show that querying instance-label pairs is more effective.
  Use most generation error reduction criteria. iteratively select the ones to minimize Multi-Labeled Bayesian Error Bound.
- Effective multi-label active learning for text classification [SIGKDD, 2009]:
  Multi-label text classification and query all the labeled of the instance.
  Approximated by the size of version space, and the reduction rate of the size of version space is optimized with Support Vector Machines (SVM). 
  Expected loss for multi-label data is approximated by summing up losses on all labels according to the most conﬁdent result of label prediction.
- Optimal batch selection for active learning in multi-label classification [ACMMM, 2011]:
  Select a batch of points that each individual point furnishes high information and the selected batch of points have minimal redundancy.
  Design an uncertainty vector and uncertainty matrix to measure the redundancy between unlabeled points.
  Use SVM as underlying model.
- Active learning with multi-label svm classification [IJCAI, 2013]:
  Query all the labels of the selected instance.
  Measure the informativeness of an instance by combining the label cardinality inconsistency and the separation margin with a tradeoff parameter.
- Active query driven by uncertainty and diversity for incremental multi-label learning [ICDM, 2013]:
  Exploit both uncertainty and diversity in the instance space and label space with an incremental multi-label classification model. 
  Pick instance then pick label.
- Multilabel Image Classification via High-Order Label Correlation Driven Active Learning [TIP, 2014]:
  Consider high order correlations (more than pairwise correlation).
  Query item-label pair.
- Active learning by querying informative and representative examples [NIPS 2010/TPAMI, 2014 (with multi-label)]:
  Count unlabeled instances into query object(loss) function (so that representative), and select the instance would have the lowest loss.
- Multi-Label Active Learning: Query Type Matters [IJCAI, 2015]：
  Iteratively select one instance along with a pair of labels, and then query their relevance ordering
- Multi-Label Deep Active Learning with Label Correlation [ICIP, 2018]:
  CNN is used to produce high level representation of the image and the LSTM models the label dependencies.
  Then select k samples furnishing the maximum entropies to form batch B.
- Effective active learning strategy for multi-label learning [Neural Computing, 2018]:
  An review paper.

#### With pre-defined constrains
- Multi-Task Active Learning with Output Constraints [AAAI, 2010]: 
  Query item-label pairs.
  They design a reward function to calculate VOI (value of information).
  Constrains are known and provided at the beginning.
  Use Naive Bayes as underlying model.
- Cost-Effective Active Learning for Hierarchical Multi-Label Classification [IJCAI, 2018]:
  Hierarchical structure are predefined.
  Use batch mode selection.
  Query item-label pairs, and the informativeness of instance-label pair counted in the contribution of ancestor and dependent.
  Use one-vs-all linear SVM.


### Multi-Domain Active Learning
Works under this category normally are on different domains.
There are not many work in this field.
- [Multi-domain active learning for text classification [KDD, 2012]](https://dl.acm.org/doi/10.1145/2339530.2339701):
  Different domains share a subspace.
  The classification model contains domain unique and domain shared parts.
  Active learning select the instance would have maximum expected loss reduction.
  The loss reduction is estimated by the version space of SVM.
  They said it could handle the multi-class situation, but didn't describe how.
- [Multi-domain active learning for recommendation [AAAI, 2016]](https://www.aaai.org/ocs/index.php/AAAI/AAAI16/paper/viewPaper/12369): 
  Use Rating-Matrix Generative Model (RMGM) for multi-domain recommendation (a probability model).
  The global generalization error introduced by an user-item pair could be divided to domain-independent and domain specific part.

### Active Domain Adaptation
### Active Learning for Recommendation
- A survey of active learning in collaborative filtering recommender systems
### Active Learning for Remote Sensing Image Classification
- A survey of active learning algorithms for supervised remote sensing image classification
### Active Meta-Learning
- Learning How to Actively Learn: A Deep Imitation Learning Approach
### Semi-Supervised Active Learning
### Active Reinforcement Learning
### Generative Adversarial Network with Active Learning
### Others
- [Projection based Active Gaussian Process Regression for Pareto Front Modeling [Arxiv, 2020]](https://arxiv.org/pdf/2001.07072.pdf):
  Active learning and multi-objective optimization.

---------------------
## Practical Considerations
### Batch mode selection
- Batch Mode Active Learning and Its Application to Medical Image Classiﬁcation [[2006, ICML]](https://dl.acm.org/doi/10.1145/1143844.1143897): Largest reduction in the Fisher information + submodular functions. Multi-class classification. Kernel logistic regressions (KLR) and the support vector machines (SVM).
- Semi-Supervised SVM Batch Mode Active Learning for Image Retrieval [CVPR, 2008]
### Varying Costs
### Noise Labelers
### Multiple Labelers
- [Active cross-query learning: A reliable labeling mechanism via crowdsourcing for smart surveillance [Computer Communications, 2020]](https://www.sciencedirect.com/science/article/pii/S014036641931730X):
  Each labeling task is repeated several times to complete the cross-query learning.

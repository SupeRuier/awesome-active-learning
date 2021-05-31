# AL strategies on Different models

In this chapter, we care about how to apply AL on specific models.

# Models

## SVM/LR
Most common models, we won't waste time here.
Most of classic strategies are based on these models.

## Bayesian/Probabilistic
- Employing EM and Pool-Based Active Learning for Text Classiﬁcation [[1998. ICML]](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.50.10&rep=rep1&type=pdf): 
  EM + Query-by-Committee (QBC-with-EM)

## Gaussian Progress
- Active instance sampling via matrix partition [2010, NIPS]: Gaussian Process. Maximizing a natural mutual information criterion between the labeled and unlabeled instances. No comparison with others.(69 citations)
- [Bayesian active learning for classification and preference learning [Arxiv, 2011]](https://arxiv.org/abs/1112.5745):
  Propose an approach that expresses information gain in terms of predictive entropies, and apply this method to the Gaussian Process Classifier (GPC).
  This method is referred as *BALD*.
  Capture how strongly the model predictions for a given data point and the model parameters are coupled, implying that ﬁnding out about the true label of data points with high mutual information would also inform us about the true model parameters.
- Adaptive active learning for image classiﬁcation [CVPR, 2013]
- [Active learning with Gaussian Processes for object categorization [2007, ICCV]](https://ieeexplore.ieee.org/abstract/document/4408844): Consider both the distance from the boundary as well as the variance in selecting the points; this is only possible due to the availability of the predictive distribution in GP regression. A significant boost in classification performance is possible, especially when the amount of training data for a category is ultimately very small.(303 citations)
- Safe active learning for time-series modeling with gaussian processes [2018, NIPS]
- Actively learning gaussian process dynamics

## Decision Trees
- [Active Learning with Direct Query Construction [KDD, 2008]](https://dl.acm.org/doi/pdf/10.1145/1401890.1401950)


## Neural Network
- A new active labeling method for deep learning [IJCNN, 2014]
- Captcha recognition with active deep learning [Neural Computation, 2015]
- [Probabilistic Backpropagation for Scalable Learning of Bayesian Neural Networks [ICML, 2015]](http://proceedings.mlr.press/v37/hernandez-lobatoc15.pdf):
  Use an active learning scenario which is necessary to produce accurate estimates of uncertainty for obtaining good performance to estimates of the posterior variance on the weights produced by PBP(the proposed methods for BNN).
- [Cost-effective active learning for deep image classification [IEEE TCSVT, 2016]](https://ieeexplore.ieee.org/abstract/document/7508942): (180)
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
- [Parting with Illusions about Deep Active Learning [2019, Arxiv]](https://arxiv.org/pdf/1912.05361.pdf):
  Take into account the data augmentation and SSL in the literature.
  It considers the an image classification task and a semantic segmentation task.
- [Active and Incremental Learning with Weak Supervision [KI-Künstliche Intelligenz, 2020]](https://link.springer.com/article/10.1007/s13218-020-00631-4)0
- [Accelerating the Training of Convolutional Neural Networks for Image Segmentation with Deep Active Learning [Dissertation, 2020]](https://uwspace.uwaterloo.ca/handle/10012/15537)
- [QActor: On-line Active Learning for Noisy Labeled Stream Data [Arxiv, 2020]](https://arxiv.org/abs/2001.10399)0
- [Diffusion-based Deep Active Learning[2020, Arxiv]](https://arxiv.org/pdf/2003.10339.pdf): Build graph by the first hidden layer in DNN. The selection is performed on the graph. 
  Consider a random walk as a mean to assign a label
- [State-Relabeling Adversarial Active Learning [2020, CVPR]](https://arxiv.org/pdf/2004.04943.pdf):
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
- [Towards Robust and Reproducible Active Learning Using Neural Networks [2020]](https://arxiv.org/pdf/2002.09564.pdf):
  A comparative study over state-of-art Deep AL methods.
  In short, it states that compared to the well-regularized RSB, state-of-the-art AL methods evaluated in this paper do not achieve any noticeable gain.
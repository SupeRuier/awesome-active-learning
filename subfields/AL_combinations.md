# AL with other Research Problems

Active learning is also been used with other Learning/Research paradigms.
These works could be considered as the applications of AL with other **research problems**.
For the information of the real applications of AL in **industry**, please check [AL Applications](subfields/AL_applications.md).

*(Note that the works list in this pages are works I browsed in background reading. 
There might be more important works in the corresponding fields not listed here.
Besides, I can't ensure the works are representative in the fields.
So if you have any comments and recommendations, pls let me know.)*

Reducing the labeling cost is a common need in many research fields.
The works here are almost

<!-- TODO: revise the order -->
- [AL with other Research Problems](#al-with-other-research-problems)
  - [Computer Vision (CV)](#computer-vision-cv)
  - [Natural Language Processing (NLP)](#natural-language-processing-nlp)
  - [Domain adaptation/Transfer learning](#domain-adaptationtransfer-learning)
  - [Metric learning/Pairwise comparison/Similarity learning](#metric-learningpairwise-comparisonsimilarity-learning)
  - [One/Few-shot learning](#onefew-shot-learning)
  - [Graph Processing](#graph-processing)
  - [Clustering](#clustering)
  - [Recommendation](#recommendation)
  - [Reinforcement Learning](#reinforcement-learning)
  - [Robotics](#robotics)
  - [Model Interpretability](#model-interpretability)
  - [Causal Analysis](#causal-analysis)
  - [Anomaly Detection](#anomaly-detection)
  - [Ordinal Regression/Classification](#ordinal-regressionclassification)
  - [Label De-noising](#label-de-noising)
  - [Model Selection](#model-selection)
  - [Software Engineering](#software-engineering)
  - [Positive and unlabeled (PU) learning](#positive-and-unlabeled-pu-learning)
  - [Human Learning](#human-learning)
  - [Sequence Labeling](#sequence-labeling)
  - [Sample Selection for Optimization Problems](#sample-selection-for-optimization-problems)
  - [Multi-Fidelity Machine Learning](#multi-fidelity-machine-learning)
  - [Generative Adversarial Network Training](#generative-adversarial-network-training)

## Computer Vision (CV)

CV is quite a wide conception.
Here we only post several subtypes in the fields.

Image segmentation/Semantic Segmentation：
- Geometry in active learning for binary and multi-class [2019, Computer vision and image understanding]
- Contextual Diversity for Active Learning [2020, Arxiv]
- Semi-supervised Active Learning for Instance Segmentation via Scoring Predictions
- [Attention, Suggestion and Annotation: A Deep Active Learning Framework for Biomedical Image Segmentation [2020, MICCAI]](https://link.springer.com/chapter/10.1007/978-3-030-59710-8_1)
- [Deep Active Learning for Joint Classification & Segmentation with Weak Annotator [2020]](https://arxiv.org/pdf/2010.04889.pdf): Leverage the unlabeled images to improve model accuracy with less oracle-annotation. AL for segmentation of images selected for pixel-level annotation.
- [Difficulty-aware Active Learning for Semantic Segmentation [2020]](https://arxiv.org/pdf/2010.08705.pdf)
- Embodied Visual Active Learning for Semantic Segmentation [2020]

Object Detection: 
- Deep Active Learning for Remote Sensing Object Detection [2020, Arxiv]
- [Active Object Detection in Sonar Images [2020, IEEE Access]](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=9106398)
- [Importance of Self-Consistency in Active Learning for Semantic Segmentation [2020, Arxiv]](https://arxiv.org/pdf/2008.01860.pdf)

Image Captioning:
- Structural Semantic Adversarial Active Learning for Image Captioning [2020, ACMMM]

Action Recognition:
- [Sparse Semi-Supervised Action Recognition with Active Learning [2020]](https://arxiv.org/pdf/2012.01740.pdf)

Video Object Detection:
- [Temporal Coherence for Active Learning in Videos [2019, ICCVW]](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9022609&tag=1)
- [Active learning method for temporal action localization in untrimmed videos [2020, US Patent]](https://patents.google.com/patent/US20190325275A1/en)

## Natural Language Processing (NLP)

NLP is also quite a wide conception.
Here we only post several subtypes in the fields.

Sentiment/text classification:
- Active learning for imbalanced sentiment classification [2012, EMNLP-CoNLL]
- [Deep Active Learning with Simulated Rationales for Text Classiﬁcation [2020, PRAI]](https://link.springer.com/chapter/10.1007/978-3-030-59830-3_32): Utilize auxiliary task to reduce the data scarce affect at the beginning of the AL process.

Named entity recognition: 
- Learning How to Actively Learn: A Deep Imitation Learning Approach [2018, ACL]
- LTP: A New Active Learning Strategy for CRF-Based Named Entity Recognition [2020, Arxiv]

Semantic Parsing:
- [Uncertainty and Traffic-Aware Active Learning for Semantic Parsing [2020]](https://assets.amazon.science/af/ca/4c43ed0c4932a3a8365693e68420/uncertainty-and-traffic-aware-active-learning-for-semantic-parsing.pdf)

Classifier Pruning:
- [FIND: Human-in-the-Loop Debugging Deep Text Classifiers [2020, EMNLP]](https://www.aclweb.org/anthology/2020.emnlp-main.24.pdf): Visualize each extracted feature as a word cloud. Human decide wether to block the corresponding feature.

Neural Machine Translation:
- [Active Learning Approaches to Enhancing Neural Machine Translation [2020, EMNLP]](https://www.aclweb.org/anthology/2020.findings-emnlp.162.pdf): The first to do a large-scale study on actively training Transformer for NMT.

Sequence Tagging:
- [Active Learning for Sequence Tagging with Deep Pre-trained Models and Bayesian Uncertainty Estimates [2021]](https://arxiv.org/pdf/2101.08133.pdf)

## Domain adaptation/Transfer learning

Normally when we use AL in domain adaptation, we can obtain several true labels of the unlabeled instances on source/target domain.
And for transfer learning, the concept is even wider.
Here we refer active transfer learning to any forms of transfer, eg. data or model.

Domain adaptation:
- Active learning for cross-domain sentiment classification [2013, IJCAI]
- Transfer Learning with Active Queries from Source Domain [2016, IJCAI]
- Active Sentiment Domain Adaptation [2017, ACL]
- Active Adversarial Domain Adaptation [2020, WACV]

Transfer learning:
- Accelerating active learning with transfer learning [2013, ICDM]
- Hierarchical Active Transfer Learning [2015, SIAM]
- Semi-Supervised Active Learning with Cross-Class Sample Transfer [2016, IJCAI]
- Active learning with cross-class knowledge transfer [2016, AAAI]
- Active learning with cross-class similarity transfer [2017, AAAI]
- Rapid Performance Gain through Active Model Reuse [IJCAI, 2019]

## Metric learning/Pairwise comparison/Similarity learning 

- [Active Ordinal Querying for Tuplewise Similarity Learning [2020 AAAI]](https://www.aaai.org/Papers/AAAI/2020GB/AAAI-CanalG.9650.pdf): Introduce AL to their new similarity methods (InfoTuple). Active Sampling for Pairwise Comparisons via Approximate Message Passing and Information Gain Maximization.
- Batch Decorrelation for Active Metric Learning [2020, IJCAI]

## One/Few-shot learning

One/few-shot learning is to learn a model on a new dataset with one for few available instances.
AL could be used to select the instances in the new dataset.

- Active one-shot learning [2017, Arxiv]
- Augmented Memory Networks for Streaming-Based Active One-Shot Learning [2019, Arxiv]
- Active one-shot learning with Prototypical Networks [2019, ESANN]
- Active one-shot learning by a deep Q-network strategy [2020, Neurocomputing]
- On the Utility of Active Instance Selection for Few-Shot Learning [2020]: Show via these “upper bounds” that we do not have a significant room for improving few-shot models through actively selecting instances.

## Graph Processing

Graph Embedding/Network representation learning:
- Active discriminative network representation learning [2018, IJCAI]

Graph node classification:
- Active learning for streaming networked data [2014, ACM International Conference on Conference on Information and Knowledge Management]
- MetAL: Active Semi-Supervised Learning on Graphs via Meta Learning [2020, Arxiv]
- Active Learning for Node Classiﬁcation: An Evaluation [2020, Entropy MDPI]
- Active Learning for Node Classification: The Additional Learning Ability from Unlabelled Nodes [2020]

Link Prediction:
- Complex Query Answering with Neural Link Predictors [2020]

Node response prediction:
- [Meta-Active Learning for Node Response Prediction in Graphs [2020]](https://scholar.google.com/scholar?hl=zh-CN&as_sdt=0%2C5&q=META-ACTIVE+LEARNING+FOR+NODE+RESPONSE+PREDICTION+IN+GRAPHS&btnG=)

Graph transfer:
- [Graph Policy Network for Transferable Active Learning on Graphs [2020, Arxiv]](https://arxiv.org/pdf/2006.13463.pdf)
- [Active Domain Transfer on Network Embedding [2020, Proceedings of The Web Conference ]](https://arxiv.org/pdf/2007.11230.pdf)
- [Active Learning on Graphs with Geodesically Convex Classes [2020, MLG]](http://www.mlgworkshop.org/2020/papers/MLG2020_paper_40.pdf)

## Clustering

AL could support clustering by provide supervised information.

- Semi-Supervised Selective Affinity Propagation Ensemble Clustering With Active Constraints [2020, IEEE Access]
- Active Learning for Constrained Document Clustering with Uncertainty Region [Complexity, 2020]: Must link & cannot link.
- Cautious Active Clustering [2020]
- Improving evolutionary constrained clustering using Active Learning [2020, Knowledge-Based Systems]

Review:
- Interactive clustering: a scoping review [2020, Artificial Intelligence Review]

## Recommendation

Review:
- A survey of active learning in collaborative filtering recommender systems

## Reinforcement Learning

Active exploration
- Model-Based Active Exploration
- [SAMBA: Safe Model-Based & Active Reinforcement Learning [2020, Arxiv]](https://arxiv.org/pdf/2006.09436.pdf)


## Robotics

Human-Robot Interaction:
- [Teacher-Learner Interaction for Robot Active Learning [2020, Thesis]](https://aaltodoc.aalto.fi/bitstream/handle/123456789/46843/isbn9789526400556.pdf?sequence=1&isAllowed=y)

Robot motion planning:
- [Active Learning of Signal Temporal Logic Specifications](https://people.kth.se/~linard/publications/active_learn_stl.pdf)
- [Online Body Schema Adaptation through Cost-Sensitive Active Learning](https://arxiv.org/pdf/2101.10892.pdf)


## Model Interpretability

- [ALEX: Active Learning based Enhancement of a Model’s EXplainability [2020, CIKM]](https://dl.acm.org/doi/pdf/10.1145/3340531.3417456)
- [Active Sampling for Learning Interpretable Surrogate Machine Learning Models [2020, DSAA]](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9260055&tag=1)
- [Explainable Active Learning (XAL): Toward AI Explanations as Interfaces for Machine Teachers [Arxiv, 2020]](https://arxiv.org/pdf/2001.09219.pdf)

## Causal Analysis

- [Root Cause Analysis for Self-organizing Cellular Network: an Active Learning Approach](https://link.springer.com/article/10.1007/s11036-020-01589-1)
- [Active Invariant Causal Prediction: Experiment Selection through Stability](https://arxiv.org/pdf/2006.05690.pdf)

## Anomaly Detection

- Meta-AAD: Active Anomaly Detection with Deep Reinforcement Learning [2020, Arxiv]

## Ordinal Regression/Classification

Could be consider as a regression where the relative orders of the instances matter.

- [Active Learning for Imbalanced Ordinal Regression [2020, IEEE Access]](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9208667)

## Label De-noising

- Active Deep Learning to Tune Down the Noise in Labels [2018, KDD]

## Model Selection

As active model selection, both pool based and stream based. 
Normally there only is model selection without training.

- Online Active Model Selection for Pre-trained Classifiers [2020]

## Software Engineering

Software Defects Prediction:
- [Empirical evaluation of the active learning strategies on software defects prediction [2016, ISSSR]](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9265897)
- [Improving high-impact bug report prediction with combination of interactive machine learning and active learning [2021]](https://reader.elsevier.com/reader/sd/pii/S0950584921000185?token=E1D095736314C62935E011266E971E6DA8289DDF6AC3CB3F57115363383EEED292B3A9C1B8CDD30E81FAAE08F8F0B9B4)

## Positive and unlabeled (PU) learning

A special case of binary classiﬁcation where a learner only has access to labeled positive examples and unlabeled examples.

- Class Prior Estimation in Active Positive and Unlabeled Learning [2020, IJCAI]

## Human Learning

- [Human Active Learning [2008, NIPS]](http://papers.nips.cc/paper/3456-human-active-learning)
- [Human Active Learning [2018]](https://www.intechopen.com/books/active-learning-beyond-the-future/human-active-learning)

## Sequence Labeling

- [SeqMix: Augmenting Active Sequence Labeling via Sequence Mixup [2020]](https://arxiv.org/pdf/2010.02322.pdf): Not only provide the selected instance, but also provide a generated sequence according to the selected one.

## Sample Selection for Optimization Problems

The background is that simulation (for evaluation) is quite expensive in many optimization problem.
Utilize active sampling to reduce the optimization cost.

- [Building energy optimization using surrogate model and active sampling [2020]](https://www.tandfonline.com/doi/pdf/10.1080/19401493.2020.1821094)
- [ALGA: Active Learning-Based Genetic Algorithm for Accelerating Structural Optimization [2020, AIAA]](https://arc.aiaa.org/doi/pdf/10.2514/1.J059240)

## Multi-Fidelity Machine Learning

- [Deep Multi-Fidelity Active Learning of High-Dimensional Outputs [2020]](https://arxiv.org/pdf/2012.00901.pdf)

## Generative Adversarial Network Training

AL could reduce the number of needed instances to train a GAN.
- Learning Class-Conditional GANs with Active Sampling [2019, SIGKDD]
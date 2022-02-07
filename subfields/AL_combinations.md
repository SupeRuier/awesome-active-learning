# AL with other AI Research Problems

Active learning is also been used with other Learning/Research paradigms.
These works could be considered as the applications of AL with other **AI research problems**.
For the information of the real applications of AL in **science** and **industry**, please check [AL Applications](subfields/AL_applications.md).

*(Note that the works list in this pages are works I browsed in background reading. 
There might be more important works in the corresponding fields not listed here.
Besides, I can't ensure the works are representative in the fields.
So if you have any comments and recommendations, pls let me know.)*

Reducing the labeling cost is a common need in many research fields.

<!-- TODO: revise the order -->
- [AL with other AI Research Problems](#al-with-other-ai-research-problems)
  - [Computer Vision (CV)](#computer-vision-cv)
  - [Natural Language Processing (NLP)](#natural-language-processing-nlp)
  - [Domain adaptation/Transfer learning](#domain-adaptationtransfer-learning)
  - [Metric learning/Pairwise comparison/Similarity learning](#metric-learningpairwise-comparisonsimilarity-learning)
  - [One/Few/Zero-shot learning or Meta-Learning](#onefewzero-shot-learning-or-meta-learning)
  - [Graph Processing](#graph-processing)
  - [Semi-supervised learning](#semi-supervised-learning)
  - [Online Learning System](#online-learning-system)
  - [Clustering](#clustering)
  - [Recommendation](#recommendation)
  - [Reinforcement Learning](#reinforcement-learning)
  - [Robotics](#robotics)
  - [Curriculum Learning](#curriculum-learning)
  - [Model Interpretability](#model-interpretability)
  - [Causal Analysis](#causal-analysis)
  - [Anomaly Detection](#anomaly-detection)
  - [Speech Recognition](#speech-recognition)
  - [Ordinal Regression/Classification](#ordinal-regressionclassification)
  - [Label Enhancement](#label-enhancement)
    - [Label De-noising](#label-de-noising)
    - [Clean Label Uncertainties](#clean-label-uncertainties)
  - [Model Selection](#model-selection)
  - [Information Retrieval](#information-retrieval)
  - [Software Engineering](#software-engineering)
  - [Positive and unlabeled (PU) learning](#positive-and-unlabeled-pu-learning)
  - [Human Learning](#human-learning)
  - [Sequence Labeling](#sequence-labeling)
  - [Optimization Problems](#optimization-problems)
    - [Multi-Objective Optimizations](#multi-objective-optimizations)
    - [Influence Maximization in Network](#influence-maximization-in-network)
  - [Multi-Fidelity Machine Learning](#multi-fidelity-machine-learning)
  - [Generative Adversarial Network Training](#generative-adversarial-network-training)
  - [Adversarial Attack](#adversarial-attack)
    - [Detection](#detection)
    - [Training](#training)
  - [Algorithm Fairness](#algorithm-fairness)
  - [Reliability Analysis](#reliability-analysis)
  - [Learning from Label Proportions (LLP)](#learning-from-label-proportions-llp)
  - [Instance Search (INS)](#instance-search-ins)
  - [Treatment Effect](#treatment-effect)

## Computer Vision (CV)

CV is quite a wide conception.
Here we only post several subtypes in the fields.

Image classification:
- [Deep active learning for image classification [ICIP, 2017]](https://ieeexplore.ieee.org/abstract/document/8297020).
- [The power of ensembles for active learning in image classification [2018, CVPR]](https://openaccess.thecvf.com/content_cvpr_2018/papers/Beluch_The_Power_of_CVPR_2018_paper.pdf)

Image Semantic Segmentation：
- Geometry in active learning for binary and multi-class [2019, Computer vision and image understanding]
- Contextual Diversity for Active Learning [2020, Arxiv]
- [Accelerating the Training of Convolutional Neural Networks for Image Segmentation with Deep Active Learning [2020, Dissertation]](https://uwspace.uwaterloo.ca/handle/10012/15537)
- Semi-supervised Active Learning for Instance Segmentation via Scoring Predictions
- [Attention, Suggestion and Annotation: A Deep Active Learning Framework for Biomedical Image Segmentation [2020, MICCAI]](https://link.springer.com/chapter/10.1007/978-3-030-59710-8_1)
- [Deep Active Learning for Joint Classification & Segmentation with Weak Annotator [2020]](https://arxiv.org/pdf/2010.04889.pdf): Leverage the unlabeled images to improve model accuracy with less oracle-annotation. AL for segmentation of images selected for pixel-level annotation.
- [Difficulty-aware Active Learning for Semantic Segmentation [2020]](https://arxiv.org/pdf/2010.08705.pdf)
- Embodied Visual Active Learning for Semantic Segmentation [2020]
- Active Learning with Bayesian UNet for Efﬁcient Semantic Image Segmentation [Journal of Imaging]
- Active Image Segmentation Propagation [2016, CVPR]
- [Reinforced active learning for image segmentation [2020, ICLR]](https://arxiv.org/pdf/2002.06583.pdf):
  An agent learns a policy to select a subset of small informative image regions (opposed to entire images) to be labeled, from a pool of unlabeled data.
- [ViewAL: Active Learning With Viewpoint Entropy for Semantic Segmentation [2020, CVPR]](https://openaccess.thecvf.com/content_CVPR_2020/papers/Siddiqui_ViewAL_Active_Learning_With_Viewpoint_Entropy_for_Semantic_Segmentation_CVPR_2020_paper.pdf)
- [Revisiting Superpixels for Active Learning in Semantic Segmentation with Realistic Annotation Costs [2021, CVPR]](https://openaccess.thecvf.com/content/CVPR2021/papers/Cai_Revisiting_Superpixels_for_Active_Learning_in_Semantic_Segmentation_With_Realistic_CVPR_2021_paper.pdf)
- MEAL: Manifold Embedding-based Active Learning [2021, Arxiv]
- Joint Semi-supervised and Active Learning for Segmentation of Gigapixel Pathology Images with Cost-Effective Labeling [2021, ICCV]

Semantic Segmentation with domain adaptation：
- Multi-Anchor Active Domain Adaptation for Semantic Segmentation [2021, ICCV]：
  One time selection, not in the conventional loop manner.

Object Detection: 
- [A deep active learning system for species identification and counting in camera trap images [Arxiv, 2019]](https://arxiv.org/abs/1910.09716)
- Deep Active Learning for Remote Sensing Object Detection [2020, Arxiv]
- [Active Object Detection in Sonar Images [2020, IEEE Access]](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=9106398)
- [Importance of Self-Consistency in Active Learning for Semantic Segmentation [2020, Arxiv]](https://arxiv.org/pdf/2008.01860.pdf)
- [Active and Incremental Learning with Weak Supervision [KI-Künstliche Intelligenz, 2020]](https://link.springer.com/article/10.1007/s13218-020-00631-4)0
- [Active Learning for Deep Object Detection via Probabilistic Modeling [2021, ICCV]](https://openaccess.thecvf.com/content/ICCV2021/papers/Choi_Active_Learning_for_Deep_Object_Detection_via_Probabilistic_Modeling_ICCV_2021_paper.pdf)
- [Multiple Instance Active Learning for Object Detection [2021]](https://arxiv.org/pdf/2104.02324.pdf)
- Active learning for annotation and recognition of faces in video [2021, Master's Thesis]
- Multiple Instance Active Learning for Object Detection [2021, CVPR]
- Region-level Active Learning for Cluttered Scenes [2021]
- Deep active learning for object detection [2021, Information Sciences]
- [QBox: Partial Transfer Learning With Active Querying for Object Detection [2021, TNNLS]](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9548667&tag=1)
- Towards Dynamic and Scalable Active Learning with Neural Architecture Adaption for Object Detection [2021, BMVC]:
  Add NAS into the AL loops.
- TALISMAN: Targeted Active Learning for Object Detection with Rare Classes and Slices using Submodular Mutual Information [2021]

Point Cloud Semantic Segmentation：
- Label-Efficient Point Cloud Semantic Segmentation: An Active Learning Approach [2021, CVPR]
- ReDAL: Region-based and Diversity-aware Active Learning for Point Cloud Semantic Segmentation [2021, ICCV]

Image Captioning:
- Structural Semantic Adversarial Active Learning for Image Captioning [2020, ACMMM]

Action Recognition:
- [Sparse Semi-Supervised Action Recognition with Active Learning [2020]](https://arxiv.org/pdf/2012.01740.pdf)

Video Object Detection:
- [Temporal Coherence for Active Learning in Videos [2019, ICCVW]](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9022609&tag=1)
- [Active learning method for temporal action localization in untrimmed videos [2020, US Patent]](https://patents.google.com/patent/US20190325275A1/en)

Visual Tracking:
- Active Learning for Deep Visual Tracking [2021]

Visual Question Answering:
- Mind Your Outliers! Investigating the Negative Impact of Outliers on Active Learning for Visual Question Answering [2021]
- Single-Modal Entropy based Active Learning for Visual Question Answering [2021]

Pose Estimation:
- Deep Active Learning For Human Pose Estimation Via Consistency Weighted Core-Set Approach [2021, ICIP]

## Natural Language Processing (NLP)

NLP is also quite a wide conception.
Here we only post several subtypes in the fields.

**Survey** for the whole field:
- [Putting Humans in the Natural Language Processing Loop: A Survey [2021]](https://arxiv.org/pdf/2103.04044.pdf)

Sentiment/text classification:
- Active learning for imbalanced sentiment classification [2012, EMNLP-CoNLL]
- [Deep Active Learning with Simulated Rationales for Text Classiﬁcation [2020, PRAI]](https://link.springer.com/chapter/10.1007/978-3-030-59830-3_32): Utilize auxiliary task to reduce the data scarce affect at the beginning of the AL process.
- Active Learning via Membership Query Synthesis for Semi-supervised Sentence Classification [2019, CoNLL]
- Deep Active Learning for Text Classification with Diverse Interpretations [2021, CIKM]

Named entity recognition: 
- Learning How to Actively Learn: A Deep Imitation Learning Approach [2018, ACL]
- [Deep active learning for named entity recognition [ICLR, 2018]](https://arxiv.org/abs/1707.05928):
  Incremental manner.
  Uncertainty-based heuristic, select those sentences for which the length-normalized log probability of the current prediction is the lowest.
- [Overcoming Practical Issues of Deep Active Learning and its Applications on Named Entity Recognition [Arxiv, 2019]](https://arxiv.org/abs/1911.07335)
- LTP: A New Active Learning Strategy for CRF-Based Named Entity Recognition [2020, Arxiv]
- Subsequence Based Deep Active Learning for Named Entity Recognition [2021]
- Deep Active Learning for Swedish Named Entity Recognition [2021, Master Thesis]

Parsing:
- [Uncertainty and Traffic-Aware Active Learning for Semantic Parsing [2020]](https://assets.amazon.science/af/ca/4c43ed0c4932a3a8365693e68420/uncertainty-and-traffic-aware-active-learning-for-semantic-parsing.pdf)
- [Diversity-Aware Batch Active Learning for Dependency Parsing [2021]](https://arxiv.org/pdf/2104.13936.pdf)

Classifier Pruning:
- [FIND: Human-in-the-Loop Debugging Deep Text Classifiers [2020, EMNLP]](https://www.aclweb.org/anthology/2020.emnlp-main.24.pdf): Visualize each extracted feature as a word cloud. Human decide wether to block the corresponding feature.

Neural Machine Translation:
- [Active Learning Approaches to Enhancing Neural Machine Translation [2020, EMNLP]](https://www.aclweb.org/anthology/2020.findings-emnlp.162.pdf): The first to do a large-scale study on actively training Transformer for NMT.
- Active Learning for Massively Parallel Translation of Constrained Text into Low Resource Languages [2021, Arxiv]

Sequence Tagging:
- [Active Learning for Sequence Tagging with Deep Pre-trained Models and Bayesian Uncertainty Estimates [2021]](https://arxiv.org/pdf/2101.08133.pdf)

Sequence Generation:
- [Adversarial Active Learning for Sequence Labeling and Generation [2018, IJCAI]](https://www.ijcai.org/proceedings/2018/0558.pdf)

Coreference Resolution：
- [Adaptive Active Learning for Coreference Resolution [2021]](https://arxiv.org/pdf/2104.07611.pdf)

Fine-Tuning for Downstream NLP Tasks:
- Active Learning for Effectively Fine-Tuning Transfer Learning to Downstream Task
- Bayesian Active Learning with Pretrained Language Models
- Multi-class Text Classification using BERT-based Active Learning

Question Answering:
- Improving Question Answering Performance Using Knowledge Distillation and Active Learning [2021]

Event extraction:
- Active Learning for Event Extraction with Memory-based Loss Prediction Model [2021]

## Domain adaptation/Transfer learning

Normally when we use AL in domain adaptation, we can obtain several true labels of the unlabeled instances on source/target domain.
And for transfer learning, the concept is even wider.
Here we refer active transfer learning to any forms of transfer, eg. data or model.

Domain adaptation:
- Active learning for cross-domain sentiment classification [2013, IJCAI]
- Transfer Learning with Active Queries from Source Domain [2016, IJCAI]
- Active Sentiment Domain Adaptation [2017, ACL]
- On Gleaning Knowledge from Multiple Domains for Active Learning [2017, IJCAI]
- Active Adversarial Domain Adaptation [2020, WACV]
- Transferable Query Selection for Active Domain Adaptation [2021]:
  Point out that the strategy of previous works are not transferable (they use the criteria from the source domain to guide the target domain selection).
- Zero-Round Active Learning [2021, Arxiv]: Cold-start.
- S3VAADA: Submodular Subset Selection for Virtual Adversarial Active Domain Adaptation [2021, ICCV]
- Active Domain Adaptation via Clustering Uncertainty-weighted Embeddings [2021, ICCV]
- Active Universal Domain Adaptation [2021, ICCV]: There are unknown class in the target domain.
- Active Learning for Domain Adaptation: An Energy-based Approach [2021]

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
- Maximizing Conditional Entropy for Batch-Mode Active Learning of Perceptual Metrics [2021]
- A Unified Batch Selection Policy for Active Metric Learning [2021, ECML PKDD]


## One/Few/Zero-shot learning or Meta-Learning

One/few-shot learning is to learn a model on a new dataset with one for few available instances.
AL could be used to select the instances to build the support set.

- For one/few-shot learning:
- Active one-shot learning [2017, Arxiv]
- A Meta-Learning Approach to One-Step Active-Learning [2017, Arxiv]
- Learning Algorithms for Active Learning [2017, ICML]
- Meta-learning for batch mode active learning [2018, ICLR Workshop]
- Augmented Memory Networks for Streaming-Based Active One-Shot Learning [2019, Arxiv]
- Active one-shot learning with Prototypical Networks [2019, ESANN]
- Active one-shot learning by a deep Q-network strategy [2020, Neurocomputing]
- On the Utility of Active Instance Selection for Few-Shot Learning [2020]: Show via these “upper bounds” that we do not have a significant room for improving few-shot models through actively selecting instances.

There are also works about zero-shot learning:
- Graph active learning for GCN-based zero-shot classification [2021, Neurocomputing]

## Graph Processing

Graph Embedding/Network representation learning:
- Active discriminative network representation learning [2018, IJCAI]

Graph node classification:
- Active learning for streaming networked data [2014, ACM International Conference on Conference on Information and Knowledge Management]
- Active Semi-Supervised Learning Using Sampling Theory for Graph Signals [2014, KDD]
- MetAL: Active Semi-Supervised Learning on Graphs via Meta Learning [2020, Arxiv]
- Active Learning for Node Classiﬁcation: An Evaluation [2020, Entropy MDPI]
- Active Learning for Node Classification: The Additional Learning Ability from Unlabelled Nodes [2020]
- Active Learning for Attributed Graphs [2020, Master Dissertation]
- [ALG: Fast and Accurate Active Learning Framework for Graph Convolutional Networks [2021, SIGMOD]](https://dl.acm.org/doi/pdf/10.1145/3448016.3457325)

Link Prediction:
- Complex Query Answering with Neural Link Predictors [2020]

Node response prediction:
- [Meta-Active Learning for Node Response Prediction in Graphs [2020]](https://scholar.google.com/scholar?hl=zh-CN&as_sdt=0%2C5&q=META-ACTIVE+LEARNING+FOR+NODE+RESPONSE+PREDICTION+IN+GRAPHS&btnG=)

Graph transfer/Network Alignment:
- [Graph Policy Network for Transferable Active Learning on Graphs [2020, Arxiv]](https://arxiv.org/pdf/2006.13463.pdf)
- [Active Domain Transfer on Network Embedding [2020, Proceedings of The Web Conference ]](https://arxiv.org/pdf/2007.11230.pdf)
- [Active Learning on Graphs with Geodesically Convex Classes [2020, MLG]](http://www.mlgworkshop.org/2020/papers/MLG2020_paper_40.pdf)
- [Attent: Active Attributed Network Alignment [2021, WWW]](https://idvxlab.com/papers/2021WWW_Attent_zhou.pdf)

Anchor user prediction:
- [Deep Active Learning for Anchor User Prediction [IJCAI, 2019]](https://arxiv.org/abs/1906.07318)

Entity resolution:
- [Graph-boosted Active Learning for Multi-Source Entity Resolution [2021]](https://www.uni-mannheim.de/media/Einrichtungen/dws/Files_Research/Web-based_Systems/pub/Primpeli-Bizer-ALMSER-ISWC2021-Preprint.pdf)

Entity Alignment:
- ActiveEA: Active Learning for Neural Entity Alignment [2021]

## Semi-supervised learning

In semi-supervised learning, there also are limited labeled data.
The goal of SSL is also utilizing the limited labeled data to achieve a good performance.
This goal is similar to active learning.
So the model parts of active learning could be switched to a SSL model.

- [Consistency-Based Semi-supervised Active Learning: Towards Minimizing Labeling Cost [2021, Springer]](https://link.springer.com/content/pdf/10.1007%2F978-3-030-58607-2_30.pdf)

## Online Learning System

- [Active learning for online training in imbalanced data streams under cold start [2021, In Workshop on Machine Learning in Finance (KDD ’21)]](https://arxiv.org/pdf/2107.07724.pdf)

## Clustering

AL could support clustering by provide supervised information.

- Semi-Supervised Selective Affinity Propagation Ensemble Clustering With Active Constraints [2020, IEEE Access]
- Active Learning for Constrained Document Clustering with Uncertainty Region [Complexity, 2020]: Must link & cannot link.
- Cautious Active Clustering [2020]
- Improving evolutionary constrained clustering using Active Learning [2020, Knowledge-Based Systems]
- An Active Learning Method Based on Variational Autoencoder and DBSCAN Clustering [2021]

Review:
- Interactive clustering: a scoping review [2020, Artificial Intelligence Review]

## Recommendation

Review:
- A survey of active learning in collaborative filtering recommender systems

Transfer Learning for Recommendation System:
- Active Transfer Learning for Recommendation System [2020, PhD Dissertation]

Positive Unlabeled learning:
- PU Active Learning for Recommender Systems [2021, Neural Processing Letters]

## Reinforcement Learning

Active exploration
- Model-Based Active Exploration
- [SAMBA: Safe Model-Based & Active Reinforcement Learning [2020, Arxiv]](https://arxiv.org/pdf/2006.09436.pdf)

Save training cost in the measure of time:
- [Active Reinforcement Learning over MDPs [2021]](https://arxiv.org/pdf/2108.02323.pdf)

Atari games:
- Width-Based Planning and Active Learning for Atari [2021]

## Robotics

Review:
- [Active learning in robotics: A review of control principles [2021, Mechatronics]](https://www.sciencedirect.com/science/article/pii/S0957415821000659)

Human-Robot Interaction:
- [Teacher-Learner Interaction for Robot Active Learning [2020, Thesis]](https://aaltodoc.aalto.fi/bitstream/handle/123456789/46843/isbn9789526400556.pdf?sequence=1&isAllowed=y)

Object Detection Learning:
- [Weakly-Supervised Object Detection Learning through Human-Robot Interaction [2021, Arxiv]](https://arxiv.org/pdf/2107.07901.pdf)

Robot motion planning:
- [Active Learning of Signal Temporal Logic Specifications](https://people.kth.se/~linard/publications/active_learn_stl.pdf)
- [Online Body Schema Adaptation through Cost-Sensitive Active Learning](https://arxiv.org/pdf/2101.10892.pdf)

Demonstrate Robots:
- Active Learning of Bayesian Probabilistic Movement Primitives [2021, IEEE ROBOTICS AND AUTOMATION LETTERS]

Active Exploration:
- SEAL: Self-supervised Embodied Active Learning using Exploration and 3D Consistency [2021, NeuraIPS]

## Curriculum Learning
- [Active Curriculum Learning [2021, InterNLP]](https://aclanthology.org/2021.internlp-1.pdf#page=52)

## Model Interpretability

- [ALEX: Active Learning based Enhancement of a Model’s EXplainability [2020, CIKM]](https://dl.acm.org/doi/pdf/10.1145/3340531.3417456)
- [Active Sampling for Learning Interpretable Surrogate Machine Learning Models [2020, DSAA]](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9260055&tag=1)
- [Explainable Active Learning (XAL): Toward AI Explanations as Interfaces for Machine Teachers [Arxiv, 2020]](https://arxiv.org/pdf/2001.09219.pdf)
- [Human-in-the-loop Extraction of Interpretable Concepts in Deep Learning Models [2021]](https://arxiv.org/pdf/2108.03738.pdf)

## Causal Analysis

- [Root Cause Analysis for Self-organizing Cellular Network: an Active Learning Approach](https://link.springer.com/article/10.1007/s11036-020-01589-1)
- [Active Invariant Causal Prediction: Experiment Selection through Stability](https://arxiv.org/pdf/2006.05690.pdf)

## Anomaly Detection

- Meta-AAD: Active Anomaly Detection with Deep Reinforcement Learning [2020, Arxiv]

## Speech Recognition

- [Loss Prediction: End-to-End Active Learning Approach For Speech Recognition [2021, Arxiv]](https://arxiv.org/pdf/2107.04289.pdf)

## Ordinal Regression/Classification

Could be consider as a regression where the relative orders of the instances matter.

- [Active Learning for Imbalanced Ordinal Regression [2020, IEEE Access]](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9208667)
- [Supervised Anomaly Detection via Conditional Generative Adversarial Network and Ensemble Active Learning [2021]](https://arxiv.org/pdf/2104.11952.pdf)

## Label Enhancement

### Label De-noising

- Active Deep Learning to Tune Down the Noise in Labels [2018, KDD]

### Clean Label Uncertainties

Enhance the quality of generated from weakly supervised model.

- [CHEF: A Cheap and Fast Pipeline for Iteratively Cleaning Label Uncertainties [2021, Arxiv]](https://arxiv.org/pdf/2107.08588.pdf)

## Model Selection

As active model selection, both pool based and stream based. 
Normally there only is model selection without training.

- Online Active Model Selection for Pre-trained Classifiers [2020]

## Information Retrieval

- Efficient Test Collection Construction via Active Learning [2021, ICTIR]: The active selection here is for IR evaluation.

## Software Engineering

Software Defects Prediction:
- [Empirical evaluation of the active learning strategies on software defects prediction [2016, ISSSR]](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9265897)
- [Improving high-impact bug report prediction with combination of interactive machine learning and active learning [2021]](https://reader.elsevier.com/reader/sd/pii/S0950584921000185?token=E1D095736314C62935E011266E971E6DA8289DDF6AC3CB3F57115363383EEED292B3A9C1B8CDD30E81FAAE08F8F0B9B4)

## Positive and unlabeled (PU) learning

A special case of binary classiﬁcation where a learner only has access to labeled positive examples and unlabeled examples.

- Class Prior Estimation in Active Positive and Unlabeled Learning [2020, IJCAI]

## Human Learning

- [Human Active Learning [2008, NeurIPS]](http://papers.nips.cc/paper/3456-human-active-learning)
- [Human Active Learning [2018]](https://www.intechopen.com/books/active-learning-beyond-the-future/human-active-learning)

## Sequence Labeling

- [SeqMix: Augmenting Active Sequence Labeling via Sequence Mixup [2020]](https://arxiv.org/pdf/2010.02322.pdf): Not only provide the selected instance, but also provide a generated sequence according to the selected one.

## Optimization Problems

The background is that simulation (for evaluation) is quite expensive in many optimization problem.
Utilize active sampling to reduce the optimization cost.

Works:
- [Building energy optimization using surrogate model and active sampling [2020]](https://www.tandfonline.com/doi/pdf/10.1080/19401493.2020.1821094)
- [ALGA: Active Learning-Based Genetic Algorithm for Accelerating Structural Optimization [2020, AIAA]](https://arc.aiaa.org/doi/pdf/10.2514/1.J059240)

### Multi-Objective Optimizations

Works:
- [Active Learning for Multi-Objective Optimization [2013, ICML]](http://proceedings.mlr.press/v28/zuluaga13.pdf)

### Influence Maximization in Network

Works:
- Near-optimal Batch Mode Active Learning and Adaptive Submodular Optimization [2013, ICML]

## Multi-Fidelity Machine Learning

- [Deep Multi-Fidelity Active Learning of High-Dimensional Outputs [2020]](https://arxiv.org/pdf/2012.00901.pdf)

## Generative Adversarial Network Training

AL could reduce the number of needed instances to train a GAN.
- Learning Class-Conditional GANs with Active Sampling [2019, SIGKDD]

## Adversarial Attack 

### Detection
- [Active Machine Learning Adversarial Attack Detection in the User Feedback Process [2021, IEEE Access]](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=9366529)
- Robust Active Learning: Sample-Efficient Training of Robust Deep Learning Models [2021]

### Training

- Models in the Loop: Aiding Crowdworkers with Generative Annotation Assistants [2021]

## Algorithm Fairness

Addressing fairness at the data collection and dataset preparation stages therefore becomes an essential part of training fairer algorithms.
For example, the ImageNet was crawled from image databases without considering sensitive attributes such as race or gender. In consequence, models trained (or pre-trained) on this dataset are prone to mimic societal biases.

- [Can Active Learning Preemptively Mitigate Fairness Issues [2021, ICLR-RAI]](https://arxiv.org/abs/2104.06879)

## Reliability Analysis

- Sequential active learning of low-dimensional model representations for reliability analysis [2021, Arxiv]

## Learning from Label Proportions (LLP)

- Active learning from label proportions via pSVM [2021, Neurocomputing]

## Instance Search (INS)

- Confidence-Aware Active Feedback for Efficient Instance Search [2021]

## Treatment Effect

- Causal-BALD: Deep Bayesian Active Learning of Outcomes to Infer Treatment-Effects from Observational Data [2021, NeuraIPS]
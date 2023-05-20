# AL with other AI Research Problems

Active learning is also been used with other Learning/Research paradigms.
These works could be considered as the applications of AL with other **AI research problems**.
For the information of the real applications of AL in **science** and **industry**, please check [AL Applications](../contents/AL_applications.md).

*(Note that the works list in this pages are works I browsed in background reading. 
There might be more important works in the corresponding fields not listed here.
Besides, I can't ensure the works are representative in the fields.
So if you have any comments and recommendations, pls let me know.)*

Reducing the labeling cost is a common need in many research fields.

- [AL with other AI Research Problems](#al-with-other-ai-research-problems)
- [Popular Fields](#popular-fields)
  - [Computer Vision (CV)](#computer-vision-cv)
  - [Natural Language Processing (NLP)](#natural-language-processing-nlp)
  - [Domain adaptation/Transfer learning](#domain-adaptationtransfer-learning)
  - [One/Few/Zero-shot learning or Meta-Learning](#onefewzero-shot-learning-or-meta-learning)
  - [Graph Processing](#graph-processing)
  - [Metric learning/Pairwise comparison/Similarity learning](#metric-learningpairwise-comparisonsimilarity-learning)
  - [Recommendation](#recommendation)
  - [Reinforcement Learning](#reinforcement-learning)
  - [Robotics](#robotics)
  - [Model Interpretability](#model-interpretability)
  - [Clustering](#clustering)
- [Other Fields (alphabetical order)](#other-fields-alphabetical-order)
  - [Adversarial Attack](#adversarial-attack)
  - [Algorithm Fairness](#algorithm-fairness)
  - [Anomaly Detection](#anomaly-detection)
  - [Audio Processing](#audio-processing)
  - [Causal Analysis](#causal-analysis)
  - [Choice Model](#choice-model)
  - [Continual Learning](#continual-learning)
  - [Curriculum Learning](#curriculum-learning)
  - [Entity Matching](#entity-matching)
  - [Federated Learning](#federated-learning)
  - [Hedge](#hedge)
  - [Human Learning](#human-learning)
  - [Information Retrieval](#information-retrieval)
  - [Instance Search (INS)](#instance-search-ins)
  - [Generative Adversarial Network Training](#generative-adversarial-network-training)
  - [Knowledge Distillation](#knowledge-distillation)
  - [Label Enhancement](#label-enhancement)
  - [Learning from Label Proportions (LLP)](#learning-from-label-proportions-llp)
  - [Model Selection](#model-selection)
  - [Multi-Fidelity Machine Learning](#multi-fidelity-machine-learning)
  - [Online Learning System](#online-learning-system)
  - [Optimization Problems](#optimization-problems)
    - [Multi-Objective Optimizations](#multi-objective-optimizations)
    - [Influence Maximization in Network](#influence-maximization-in-network)
  - [Ordinal Regression/Classification](#ordinal-regressionclassification)
  - [Positive and Unlabeled (PU) Learning](#positive-and-unlabeled-pu-learning)
  - [Prompt Engineering](#prompt-engineering)
  - [Reliability Analysis](#reliability-analysis)
  - [Relation Extraction](#relation-extraction)
  - [Sequence Labeling](#sequence-labeling)
  - [Software Engineering](#software-engineering)
  - [Spiking Neural Network](#spiking-neural-network)
  - [Speech Recognition](#speech-recognition)
  - [Symbolic Regression](#symbolic-regression)
  - [Treatment Effect](#treatment-effect)

# Popular Fields

## Computer Vision (CV)

CV is quite a wide conception.
Here we only post several subtypes in the fields.

**Survey** for the whole field:
- Deep Active Learning for Computer Vision: Past and Future [2022]

Image classification:
- [Deep active learning for image classification [ICIP, 2017]](https://ieeexplore.ieee.org/abstract/document/8297020).
- [The power of ensembles for active learning in image classification [2018, CVPR]](https://openaccess.thecvf.com/content_cvpr_2018/papers/Beluch_The_Power_of_CVPR_2018_paper.pdf)
- MoBYv2AL: Self-supervised Active Learning for Image Classification [2022, BMVC]

Image Semantic Segmentation：
- Geometry in active learning for binary and multi-class [2019, Computer vision and image understanding]
- Contextual Diversity for Active Learning [2020, Arxiv]
- [Accelerating the Training of Convolutional Neural Networks for Image Segmentation with Deep Active Learning [2020, Dissertation]](https://uwspace.uwaterloo.ca/handle/10012/15537)
- Semi-supervised Active Learning for Instance Segmentation via Scoring Predictions
- [Attention, Suggestion and Annotation: A Deep Active Learning Framework for Biomedical Image Segmentation [2020, MICCAI]](https://link.springer.com/chapter/10.1007/978-3-030-59710-8_1)
- [Deep Active Learning for Joint Classification & Segmentation with Weak Annotator [2020]](https://arxiv.org/pdf/2010.04889.pdf): Leverage the unlabeled images to improve model accuracy with less oracle-annotation. AL for segmentation of images selected for pixel-level annotation.
- [Difficulty-aware Active Learning for Semantic Segmentation [2020]](https://arxiv.org/pdf/2010.08705.pdf)
- Embodied Visual Active Learning for Semantic Segmentation [2020]
- Active Learning with Bayesian UNet for Efficient Semantic Image Segmentation [Journal of Imaging]
- Active Image Segmentation Propagation [2016, CVPR]
- [Reinforced active learning for image segmentation [2020, ICLR]](https://arxiv.org/pdf/2002.06583.pdf):
  An agent learns a policy to select a subset of small informative image regions (opposed to entire images) to be labeled, from a pool of unlabeled data.
- [ViewAL: Active Learning With Viewpoint Entropy for Semantic Segmentation [2020, CVPR]](https://openaccess.thecvf.com/content_CVPR_2020/papers/Siddiqui_ViewAL_Active_Learning_With_Viewpoint_Entropy_for_Semantic_Segmentation_CVPR_2020_paper.pdf)
- [Revisiting Superpixels for Active Learning in Semantic Segmentation with Realistic Annotation Costs [2021, CVPR]](https://openaccess.thecvf.com/content/CVPR2021/papers/Cai_Revisiting_Superpixels_for_Active_Learning_in_Semantic_Segmentation_With_Realistic_CVPR_2021_paper.pdf)
- MEAL: Manifold Embedding-based Active Learning [2021, Arxiv]
- Joint Semi-supervised and Active Learning for Segmentation of Gigapixel Pathology Images with Cost-Effective Labeling [2021, ICCV]
- An Active and Contrastive Learning Framework for Fine-Grained Off-Road Semantic Segmentation [2022]
- Active Pointly-Supervised Instance Segmentation [2022]
- RADIAL: Random Sampling from Intelligent Pool for Active Learning [2022, ICML workshop]
- Revisiting Deep Active Learning for Semantic Segmentation [2023]
- Adaptive Superpixel for Active Learning in Semantic Segmentation [2023]
- Best Practices in Active Learning for Semantic Segmentation [2023]: A comparative study.

Semantic Segmentation with domain adaptation：
- Multi-Anchor Active Domain Adaptation for Semantic Segmentation [2021, ICCV]：
  One time selection, not in the conventional loop manner.
- ADeADA: Adaptive Density-aware Active Domain Adaptation for Semantic Segmentation [2022]:
  Acquire labels of the samples with high probability density in the target domain yet with low probability density in the source domain
- Active Domain Adaptation with Multi-level Contrastive Units for Semantic Segmentation [2022]
- Labeling Where Adapting Fails: Cross-Domain Semantic Segmentation with Point Supervision via Active Selection [2022]:
  Pixel-level selections.
- Pixel Exclusion: Uncertainty-aware Boundary Discovery for Active Cross-Domain Semantic Segmentation [2022, MM]
- MADAv2: Advanced Multi-Anchor Based Active Domain Adaptation Segmentation [2023]

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
- Entropy-based Active Learning for Object Detection with Progressive Diversity Constraint [2022, CVPR]
- Weakly Supervised Object Detection Based on Active Learning [2022, NPL]
- Active Learning Strategies for Weakly-Supervised Object Detection [2022]
- MUS-CDB: Mixed Uncertainty Sampling with Class Distribution Balancing for Active Annotation in Aerial Object Detection [2022]
- Box-Level Active Detection [2023]
- MuRAL: Multi-Scale Region-based Active Learning for Object Detection [2023]
- Hybrid Active Learning via Deep Clustering for Video Action Detection [2023, CVPR]

Point Cloud Semantic Segmentation：
- Label-Efficient Point Cloud Semantic Segmentation: An Active Learning Approach [2021, CVPR]
- ReDAL: Region-based and Diversity-aware Active Learning for Point Cloud Semantic Segmentation [2021, ICCV]
- Active Learning for Point Cloud Semantic Segmentation via Spatial-Structural Diversity Reasoning [2022]
- You Never Get a Second Chance To Make a Good First Impression: Seeding Active Learning for 3D Semantic Segmentation [2023]
- A multi-granularity semisupervised active learning for point cloud semantic segmentation [2023, NCA]

Image Captioning:
- Structural Semantic Adversarial Active Learning for Image Captioning [2020, ACMMM]

Action Recognition:
- [Sparse Semi-Supervised Action Recognition with Active Learning [2020]](https://arxiv.org/pdf/2012.01740.pdf)

Video Object Detection:
- [Temporal Coherence for Active Learning in Videos [2019, ICCVW]](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9022609&tag=1)
- [Active learning method for temporal action localization in untrimmed videos [2020, US Patent]](https://patents.google.com/patent/US20190325275A1/en)
- Active Learning with Effective Scoring Functions for Semi-Supervised Temporal Action Localization [2022]

Video Captioning:
- MAViC: Multimodal Active Learning for Video Captioning [2022]

Video Action Spotting:
- Towards Active Learning for Action Spotting in Association Football Videos [2023]

3D Object Detection:
- Exploring Diversity-based Active Learning for 3D Object Detection in Autonomous Driving [2022]
- Exploring Active 3D Object Detection from a Generalization Perspective [2023, ICLR]
- Bi3D: Bi-domain Active Learning for Cross-domain 3D Object Detection [2023, CVPR]

3D Semantic Segmentation:
- LiDAL: Inter-frame Uncertainty Based Active Learning for 3D LiDAR Semantic Segmentation [2022, ECCV]

Visual Tracking:
- Active Learning for Deep Visual Tracking [2021]
- Pseudo Loss Active Learning for Deep Visual Tracking [2022, Pattern Recognition]

Visual Question Answering:
- Mind Your Outliers! Investigating the Negative Impact of Outliers on Active Learning for Visual Question Answering [2021]
- Single-Modal Entropy based Active Learning for Visual Question Answering [2021]

Pose Estimation:
- Deep Active Learning For Human Pose Estimation Via Consistency Weighted Core-Set Approach [2021, ICIP]
- Active Learning with Pseudo-Labels for Multi-View 3D Pose Estimation [2021]
- Meta Agent Teaming Active Learning for Pose Estimation [2022, CVPR]
- VL4Pose: Active Learning Through Out-Of-Distribution Detection For Pose Estimation [2022]
- Rethinking the Data Annotation Process for Multi-view 3D Pose Estimation with Active Learning and Self-Training [2023, WACV]

Optical Flow Prediction:
- Optical Flow Training under Limited Label Budget via Active Learning [2022]

Emotion Classification:

In this field, the acquired labels could be highly subjective.
- An Exploration of Active Learning for Affective Digital Phenotyping [2022]

Micro-Expression Recognition:
- Tackling Micro-Expression Data Shortage via Dataset Alignment and Active Learning [2022, IEEE Trans Multimedia]

Collision Prediction & Handling:

- N-Penetrate: Active Learning of Neural Collision Handler for Complex 3D Mesh Deformations [2022, ICML]

## Natural Language Processing (NLP)

NLP is also quite a wide conception.
Here we only post several subtypes in the fields.

**Survey** for the whole field:
- [Putting Humans in the Natural Language Processing Loop: A Survey [2021]](https://arxiv.org/pdf/2103.04044.pdf)
- A Survey of Active Learning for Natural Language Processing [2022]

Sentiment/text classification:
- Active learning for imbalanced sentiment classification [2012, EMNLP-CoNLL]
- [Deep Active Learning with Simulated Rationales for Text Classification [2020, PRAI]](https://link.springer.com/chapter/10.1007/978-3-030-59830-3_32): Utilize auxiliary task to reduce the data scarce affect at the beginning of the AL process.
- Active Learning via Membership Query Synthesis for Semi-supervised Sentence Classification [2019, CoNLL]
- Active Learning for BERT: An Empirical Study [2020, EMNLP]
- Deep Active Learning for Text Classification with Diverse Interpretations [2021, CIKM]
- Revisiting Uncertainty-based Query Strategies for Active Learning with Transformers [2022]: Fine-tune the pretrained transformer based models.
- Active Learning on Pre-trained Language Model with Task-Independent Triplet Loss [2022]

Named entity recognition (Information Extraction): 
- Learning How to Actively Learn: A Deep Imitation Learning Approach [2018, ACL]
- [Deep active learning for named entity recognition [ICLR, 2018]](https://arxiv.org/abs/1707.05928):
  Incremental manner.
  Uncertainty-based heuristic, select those sentences for which the length-normalized log probability of the current prediction is the lowest.
- [Overcoming Practical Issues of Deep Active Learning and its Applications on Named Entity Recognition [Arxiv, 2019]](https://arxiv.org/abs/1911.07335)
- LTP: A New Active Learning Strategy for CRF-Based Named Entity Recognition [2020, Arxiv]
- Subsequence Based Deep Active Learning for Named Entity Recognition [2021]
- Deep Active Learning for Swedish Named Entity Recognition [2021, Master Thesis]
- FAMIE: A Fast Active Learning Framework for Multilingual Information Extraction [2022]
- Subsequence Based Deep Active Learning for Named Entity Recognition [2022, ACL/IJCNLP]
- Active Learning for Name Entity Recognition with External Knowledge [2023, ACM Trans. Asian Low-Resour. Lang. Inf. Process.]

Parsing:
- [Uncertainty and Traffic-Aware Active Learning for Semantic Parsing [2020]](https://assets.amazon.science/af/ca/4c43ed0c4932a3a8365693e68420/uncertainty-and-traffic-aware-active-learning-for-semantic-parsing.pdf)
- [Diversity-Aware Batch Active Learning for Dependency Parsing [2021]](https://arxiv.org/pdf/2104.13936.pdf)
- Active Programming by Example with a Natural Language Prior [2022]
- Active Learning for Multilingual Semantic Parser [2023]

Classifier Pruning:
- [FIND: Human-in-the-Loop Debugging Deep Text Classifiers [2020, EMNLP]](https://www.aclweb.org/anthology/2020.emnlp-main.24.pdf): Visualize each extracted feature as a word cloud. Human decide wether to block the corresponding feature.

Neural Machine Translation:
- [Active Learning Approaches to Enhancing Neural Machine Translation [2020, EMNLP]](https://www.aclweb.org/anthology/2020.findings-emnlp.162.pdf): The first to do a large-scale study on actively training Transformer for NMT.
- Active Learning for Massively Parallel Translation of Constrained Text into Low Resource Languages [2021, Arxiv]
- Active Learning with Expert Advice for Real World MT [2022]

Sequence Tagging:
- [Active Learning for Sequence Tagging with Deep Pre-trained Models and Bayesian Uncertainty Estimates [2021]](https://arxiv.org/pdf/2101.08133.pdf)

Sequence Generation:
- [Adversarial Active Learning for Sequence Labeling and Generation [2018, IJCAI]](https://www.ijcai.org/proceedings/2018/0558.pdf)

Coreference Resolution：
- [Adapting Coreference Resolution Models through Active Learning [2022, ACL]](https://aclanthology.org/2022.acl-long.519.pdf)

Learning with Pre-Trained Model:
- Active Learning for Effectively Fine-Tuning Transfer Learning to Downstream Task
- Bayesian Active Learning with Pretrained Language Models
- Multi-class Text Classification using BERT-based Active Learning
- On the Importance of Effectively Adapting Pretrained Language Models for Active Learning [2022, ACL]
- ACTUNE: Uncertainty-Based Active Self-Training for Active Fine-Tuning of Pretrained Language Models [2022, NAACL-HLT]:
  AL + SelfSL.
- Smooth Sailing: Improving Active Learning for Pre-trained Language Models with Representation Smoothness Analysis [2022]
- Low-resource Interactive Active Labeling for Fine-tuning Language Models [2022, EMNLP]

Question Answering:
- Improving Question Answering Performance Using Knowledge Distillation and Active Learning [2021]

Event Extraction (Information Extraction):
- Active Learning for Event Extraction with Memory-based Loss Prediction Model [2021]

Rational Learning:
- A Rationale-Centric Framework for Human-in-the-loop Machine Learning [2022]
- Active Learning on Pre-trained Language Model with Task-Independent Triplet Loss [2022]

Argument Structure Extraction:
- Efficient Argument Structure Extraction with Transfer Learning and Active Learning [2022]

Claim Verification：
- Active PETs: Active Data Annotation Prioritisation for Few-Shot Claim Verification with Pattern Exploiting Training [2022]

Abuse Detection:
- Is More Data Better? Re-thinking the Importance of Efficiency in Abusive Language Detection with Transformers-Based Active Learning [2022, COLING workshop TRAC]

Abstractive Text Summarization:
- Active Learning for Abstractive Text Summarization [2023]

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
- Active Learning Over Multiple Domains in Natural Language Tasks [2021]
- Improving Semi-Supervised Domain Adaptation Using Effective Target Selection and Semantics [2021, CVPR]
- Discriminative active learning for domain adaptation [2021, KBS]
- Learning Distinctive Margin toward Active Domain Adaptation [2022]: 
  The feature gradient from both loss and query function share similar update direction and yield high query score
- Loss-based Sequential Learning for Active Domain Adaptation [2022]:
  Semi-supervised domain adaptation on target domain.
  Use predict loss to select.
- DistillAdapt: Source-Free Active Visual Domain Adaptation [2022]:
  Access source model instead of source data.
- Active Model Adaptation Under Unknown Shift [2022]: The distribution is shifted from the source domain.
- Combating Label Distribution Shift for Active Domain Adaptation [2022]: Consider label distribution mismatch.
- Source-Free Active Domain Adaptation via Energy-Based Locality Preserving Transfer [2022, MM]
- Active Multi-Task Representation Learning [2022, ICML]: Select from multiple source domains.
- TL-ADA: Transferable Loss-based Active Domain Adaptation [2023, Neural Networks]
- Dirichlet-based Uncertainty Calibration for Active Domain Adaptation [2023, ICLR]
- MHPL: Minimum Happy Points Learning for Active Source Free Domain Adaptation [2023, CVPR]
- Divide and Adapt: Active Domain Adaptation via Customized Learning [2023, CVPR]

Transfer learning:
- Accelerating active learning with transfer learning [2013, ICDM]
- Hierarchical Active Transfer Learning [2015, SIAM]
- Semi-Supervised Active Learning with Cross-Class Sample Transfer [2016, IJCAI]
- Active learning with cross-class knowledge transfer [2016, AAAI]
- Active learning with cross-class similarity transfer [2017, AAAI]
- Rapid Performance Gain through Active Model Reuse [IJCAI, 2019]


## One/Few/Zero-shot learning or Meta-Learning

One/few-shot learning is to learn a model on a new dataset with one for few available instances.
AL could be used to select the instances to build the support set.

For one/few-shot learning:
- Active one-shot learning [2017, Arxiv]
- A Meta-Learning Approach to One-Step Active-Learning [2017, Arxiv]
- Learning Algorithms for Active Learning [2017, ICML]
- Meta-learning for batch mode active learning [2018, ICLR Workshop]
- Augmented Memory Networks for Streaming-Based Active One-Shot Learning [2019, Arxiv]
- Active one-shot learning with Prototypical Networks [2019, ESANN]
- Active one-shot learning by a deep Q-network strategy [2020, Neurocomputing]
- On the Utility of Active Instance Selection for Few-Shot Learning [2020]: 
  Show via these “upper bounds” that we do not have a significant room for improving few-shot models through actively selecting instances.
- Beyond Simple Meta-Learning: Multi-Purpose Models for Multi-Domain, Active and Continual Few-Shot Learning [2022]
- Active Few-Shot Learning with FASL [2022]:
  Find that AL methods do not yield strong improvements over a random baseline when applied to datasets with balanced label distributions. 
  However, experiments on modified datasets with a skewed label distributions as well as naturally unbalanced datasets show the value of AL methods.
- Active Few-Shot Learning for Sound Event Detection [2022]
- Active Transfer Prototypical Network: An Efficient Labeling Algorithm for Time-Series Data [2022]
- Active Few-Shot Classification: a New Paradigm for Data-Scarce Learning Settings [2022]
- MEAL: Stable and Active Learning for Few-Shot Prompting [2022]
- Few-shot initializing of Active Learner via Meta-Learning [2022, EMNLP]
- Active Learning for Efficient Few-Shot Classification [2023, ICASSP]

There are also works about zero-shot learning:
- Graph active learning for GCN-based zero-shot classification [2021, Neurocomputing]

## Graph Processing

Graph Embedding/Network representation learning:
- Active discriminative network representation learning [2018, IJCAI]

Graph node classification:
- Active learning for streaming networked data [2014, ACM International Conference on Conference on Information and Knowledge Management]
- Active Semi-Supervised Learning Using Sampling Theory for Graph Signals [2014, KDD]
- MetAL: Active Semi-Supervised Learning on Graphs via Meta Learning [2020, Arxiv]
- Active Learning for Node Classification: An Evaluation [2020, Entropy MDPI]
- Active Learning for Node Classification: The Additional Learning Ability from Unlabelled Nodes [2020]
- Active Learning for Attributed Graphs [2020, Master Dissertation]
- [ALG: Fast and Accurate Active Learning Framework for Graph Convolutional Networks [2021, SIGMOD]](https://dl.acm.org/doi/pdf/10.1145/3448016.3457325)
- Partition-Based Active Learning for Graph Neural Networks [2022]
- LSCALE: Latent Space Clustering-Based Active Learning for Node Classification [2022, ECMLPKDD]
- Active Learning for Node Classification using a Convex Optimization approach [2022, BigDataService]
- SmartQuery: An Active Learning Framework for Graph Neural Networks through Hybrid Uncertainty Reduction [2022, CIKM]

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
- Impact of the Characteristics of Multi-Source Entity Matching Tasks on the Performance of Active Learning Methods [2022]
- Active Temporal Knowledge Graph Alignment [2023, IJSWIS]
- Deep Active Alignment of Knowledge Graph Entities and Schemata [2023, Proc. ACM Manag. Data]

## Metric learning/Pairwise comparison/Similarity learning 

- [Active Ordinal Querying for Tuplewise Similarity Learning [2020 AAAI]](https://www.aaai.org/Papers/AAAI/2020GB/AAAI-CanalG.9650.pdf): Introduce AL to their new similarity methods (InfoTuple). Active Sampling for Pairwise Comparisons via Approximate Message Passing and Information Gain Maximization.
- Batch Decorrelation for Active Metric Learning [2020, IJCAI]
- Maximizing Conditional Entropy for Batch-Mode Active Learning of Perceptual Metrics [2021]
- A Unified Batch Selection Policy for Active Metric Learning [2021, ECML PKDD]
- Active metric learning and classification using similarity queries [2022]

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
- How to Train Your Agent: Active Learning from Human Preferences and Justifications in Safety-Critical Environments [2022, AAMAS]
- Active Exploration for Inverse Reinforcement Learning [2022]

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
- SEAL: Self-supervised Embodied Active Learning using Exploration and 3D Consistency [2021, NeurIPS]

Semantic Mapping:
- An Informative Path Planning Framework for Active Learning in UAV-based Semantic Mapping [2023]

## Model Interpretability

- [ALEX: Active Learning based Enhancement of a Model’s EXplainability [2020, CIKM]](https://dl.acm.org/doi/pdf/10.1145/3340531.3417456)
- [Active Sampling for Learning Interpretable Surrogate Machine Learning Models [2020, DSAA]](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9260055&tag=1)
- [Explainable Active Learning (XAL): Toward AI Explanations as Interfaces for Machine Teachers [Arxiv, 2020]](https://arxiv.org/pdf/2001.09219.pdf)
- [Human-in-the-loop Extraction of Interpretable Concepts in Deep Learning Models [2021]](https://arxiv.org/pdf/2108.03738.pdf)

## Clustering

AL could support clustering by provide supervised information.

- Semi-Supervised Selective Affinity Propagation Ensemble Clustering With Active Constraints [2020, IEEE Access]
- Active Learning for Constrained Document Clustering with Uncertainty Region [Complexity, 2020]: Must link & cannot link.
- Cautious Active Clustering [2020]
- Improving evolutionary constrained clustering using Active Learning [2020, Knowledge-Based Systems]
- An Active Learning Method Based on Variational Autoencoder and DBSCAN Clustering [2021]
- Active constrained deep embedded clustering with dual source [2022, Applied Intelligence]
- Active deep image clustering [2022, KBS]
- Active Learning with Positive and Negative Pairwise Feedback [2023]
- Active Clustering Ensemble With Self-Paced Learning [2023, TNNLS]

Review:
- Interactive clustering: a scoping review [2020, Artificial Intelligence Review]


# Other Fields (alphabetical order)

## Adversarial Attack 

Detection:
- [Active Machine Learning Adversarial Attack Detection in the User Feedback Process [2021, IEEE Access]](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=9366529)
- Robust Active Learning: Sample-Efficient Training of Robust Deep Learning Models [2021]

Training:
- Models in the Loop: Aiding Crowdworkers with Generative Annotation Assistants [2021]

## Algorithm Fairness

Addressing fairness at the data collection and dataset preparation stages therefore becomes an essential part of training fairer algorithms.
For example, the ImageNet was crawled from image databases without considering sensitive attributes such as race or gender. In consequence, models trained (or pre-trained) on this dataset are prone to mimic societal biases.

- [Can Active Learning Preemptively Mitigate Fairness Issues [2021, ICLR-RAI]](https://arxiv.org/abs/2104.06879)

## Anomaly Detection

- Meta-AAD: Active Anomaly Detection with Deep Reinforcement Learning [2020, Arxiv]
- A Semi-Supervised VAE Based Active Anomaly Detection Framework in Multivariate Time Series for Online Systems [2022, WWW]
- Situation-Aware Multivariate Time Series Anomaly Detection Through Active Learning and Contrast VAE-Based Models in Large Distributed Systems [2022]

## Audio Processing

- Active Correction for Incremental Speaker Diarization of a Collection with Human in the Loop [2022, Applied Science]
- Investigating Active-learning-based Training Data Selection for Speech Spoofing Countermeasure [2022]
- An efficient framework for constructing speech emotion corpus based on integrated active learning strategies [2022, IEEE Trans. Affect. Comput.]


## Causal Analysis

- [Root Cause Analysis for Self-organizing Cellular Network: an Active Learning Approach](https://link.springer.com/article/10.1007/s11036-020-01589-1)
- [Active Invariant Causal Prediction: Experiment Selection through Stability](https://arxiv.org/pdf/2006.05690.pdf)

## Choice Model

- Active Learning for Non-Parametric Choice Models [2022]

## Continual Learning
- Active Continual Learning: Labelling Queries in a Sequence of Tasks [2023]

## Curriculum Learning
- [Active Curriculum Learning [2021, InterNLP]](https://aclanthology.org/2021.internlp-1.pdf#page

## Entity Matching
- Deep entity matching with adversarial active learning [2022, VLDB]

## Federated Learning

- Federated Active Learning (F-AL): an Efficient Annotation Strategy for Federated Learning [2022]
- LG-FAL : Federated Active Learning Strategy using Local and Global Models [2022, ICML workshop]
- Knowledge-Aware Federated Active Learning with Non-IID Data [2022]
- Federated deep active learning for attention-based transaction classiﬁcation [2022, Applied Intelligence]
- Re-thinking Federated Active Learning based on Inter-class Diversity [2023]

## Hedge

Hedge is quite simple to explain in words: the algorithm combines the predictions of all the experts on a given round by taking their weighted average, where the weight of an expert exponentially decays according to the number of previous mistakes.
The goal is to find a weight for the experts' predictions to make the final decision.

- ActiveHedge: Hedge meets Active Learning [2022, ICML]

## Human Learning

- [Human Active Learning [2008, NeurIPS]](http://papers.nips.cc/paper/3456-human-active-learning)
- [Human Active Learning [2018]](https://www.intechopen.com/books/active-learning-beyond-the-future/human-active-learning)

## Information Retrieval

- Efficient Test Collection Construction via Active Learning [2021, ICTIR]: The active selection here is for IR evaluation.
- Are Binary Annotations Sufficient? Video Moment Retrieval via Hierarchical Uncertainty-based Active Learning [2023]

## Instance Search (INS)

- Confidence-Aware Active Feedback for Efficient Instance Search [2021]

## Generative Adversarial Network Training

AL could reduce the number of needed instances to train a GAN.
- Learning Class-Conditional GANs with Active Sampling [2019, SIGKDD]

## Knowledge Distillation

- Robust Active Distillation [2022]
- PVD-AL: Progressive Volume Distillation with Active Learning for Efficient Conversion Between Different NeRF Architectures [2023]

## Label Enhancement

Label De-noising:
- Active Deep Learning to Tune Down the Noise in Labels [2018, KDD]

Clean Label Uncertainties: Enhance the quality of generated from weakly supervised model.

- [CHEF: A Cheap and Fast Pipeline for Iteratively Cleaning Label Uncertainties [2021, Arxiv]](https://arxiv.org/pdf/2107.08588.pdf)

## Learning from Label Proportions (LLP)

- Active learning from label proportions via pSVM [2021, Neurocomputing]

## Model Selection

As active model selection, both pool based and stream based. 
Normally there is only model selection without training.

- Online Active Model Selection for Pre-trained Classifiers [2020]

## Multi-Fidelity Machine Learning

- [Deep Multi-Fidelity Active Learning of High-Dimensional Outputs [2020]](https://arxiv.org/pdf/2012.00901.pdf)
- [Batch Multi-Fidelity Active Learning with Budget Constraints [2022, NeruaIPS]](https://arxiv.org/pdf/2210.12704.pdf)
- Disentangled Multi-Fidelity Deep Bayesian Active Learning [2023]

## Online Learning System

- [Active learning for online training in imbalanced data streams under cold start [2021, In Workshop on Machine Learning in Finance (KDD ’21)]](https://arxiv.org/pdf/2107.07724.pdf)

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

## Ordinal Regression/Classification

Could be consider as a regression where the relative orders of the instances matter.

- [Active Learning for Imbalanced Ordinal Regression [2020, IEEE Access]](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9208667)
- [Supervised Anomaly Detection via Conditional Generative Adversarial Network and Ensemble Active Learning [2021]](https://arxiv.org/pdf/2104.11952.pdf)
- Active learning for ordinal classification based on expected cost minimization [2022, scientific reports]

## Positive and Unlabeled (PU) Learning

A special case of binary classification where a learner only has access to labeled positive examples and unlabeled examples.

- Class Prior Estimation in Active Positive and Unlabeled Learning [2020, IJCAI]

## Prompt Engineering

- Active Prompting with Chain-of-Thought for Large Language Models [2023]

## Reliability Analysis

- Sequential active learning of low-dimensional model representations for reliability analysis [2021, Arxiv]

## Relation Extraction

- Active Relation Discovery: Towards General and Label-aware Open Relation Extraction [2021, TKDE]

## Sequence Labeling

- [SeqMix: Augmenting Active Sequence Labeling via Sequence Mixup [2020]](https://arxiv.org/pdf/2010.02322.pdf): Not only provide the selected instance, but also provide a generated sequence according to the selected one.

## Software Engineering

Software Defects Prediction:
- [Empirical evaluation of the active learning strategies on software defects prediction [2016, ISSSR]](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9265897)
- [Improving high-impact bug report prediction with combination of interactive machine learning and active learning [2021]](https://reader.elsevier.com/reader/sd/pii/S0950584921000185?token=E1D095736314C62935E011266E971E6DA8289DDF6AC3CB3F57115363383EEED292B3A9C1B8CDD30E81FAAE08F8F0B9B4)

## Spiking Neural Network

- Bio-inspired Active Learning method in spiking neural network [2023, KBS]
- Effective Active Learning Method for Spiking Neural Networks [2023, TNNLS]

## Speech Recognition

- [Loss Prediction: End-to-End Active Learning Approach For Speech Recognition [2021, Arxiv]](https://arxiv.org/pdf/2107.04289.pdf)

## Symbolic Regression

- Online Symbolic Regression with Informative Query [2023]

## Treatment Effect

- Causal-BALD: Deep Bayesian Active Learning of Outcomes to Infer Treatment-Effects from Observational Data [2021, NeurIPS]

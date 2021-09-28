# AL Applications

In this chapter, we will divide the works into two sections: **scientific applications** and **industrial applications**.

*(Note that the works list in this pages are works I browsed in background reading. 
There might be more important works in the corresponding fields not listed here.
Besides, I can't ensure the works are representative in the fields.
So if you have any comments and recommendations, pls let me know.)*

- [AL Applications](#al-applications)
- [Scientific Applications](#scientific-applications)
  - [Biology](#biology)
  - [Materials](#materials)
  - [Astronomy](#astronomy)
  - [Chemistry](#chemistry)
  - [Math and Statistics](#math-and-statistics)
  - [Geology](#geology)
  - [Experiment Design/Experimental Condition Selection](#experiment-designexperimental-condition-selection)
- [Industrial Applications](#industrial-applications)
  - [Labeling System](#labeling-system)
  - [Remote Sensing](#remote-sensing)
  - [Medical Research](#medical-research)
    - [Medical Image Classification / Image Annotation](#medical-image-classification--image-annotation)
    - [Medical Image Segmentation](#medical-image-segmentation)
    - [Medical Symptom Recognition from Text](#medical-symptom-recognition-from-text)
    - [ECG classification](#ecg-classification)
    - [Retinal image analysis](#retinal-image-analysis)
  - [Drug Discovery](#drug-discovery)
  - [Spam Detection](#spam-detection)
  - [Person Re-identification](#person-re-identification)
  - [Mobile Health Monitoring / Disease detection](#mobile-health-monitoring--disease-detection)
  - [Privacy Policy Classification](#privacy-policy-classification)
  - [Dataset Engineering](#dataset-engineering)
    - [Dataset Building/ Data Annotation](#dataset-building-data-annotation)
    - [Data Enrichment](#data-enrichment)
  - [Software Engineering](#software-engineering)
  - [Answer Selection](#answer-selection)
  - [Dialog Policy Learning (intelligent system)](#dialog-policy-learning-intelligent-system)
  - [Communication](#communication)
  - [Brain Mapping in a High Performance Computing Environment](#brain-mapping-in-a-high-performance-computing-environment)
  - [Crowd Counting](#crowd-counting)
  - [Malware Detection](#malware-detection)
  - [Solvability Prediction in Power Systems](#solvability-prediction-in-power-systems)
  - [Disguised Faces Recognition](#disguised-faces-recognition)
  - [Gas Reservoir Prediction](#gas-reservoir-prediction)

# Scientific Applications 

The scientific applications of AL are about biology, chemistry, physics or things about experiment design and analysis.

## Biology

Structural Biology:
- Active machine learning for transmembrane helix prediction [2010, BMC Bioinform]
- Investigating Active Learning and Meta-Learning for Iterative Peptide Design [2020, JCM]
- Large scale active-learning-guided exploration for in vitro protein production optimization [2020, Nature Communications]
- Active learning to classify macromolecular structures in situ for less supervision in cryo-electron tomography [Bioinformatics]

Cell Classification:
- Active feature selection discovers minimal gene-sets for classifying cell-types and disease states in single-cell mRNA-seq data [2021]

## Materials

Molecular Dynamics
- [Active learning accelerates ab initio molecular dynamics on reactive energy surfaces [2020, Chem]
- [ASGN: An Active Semi-supervised Graph Neural Network for Molecular Property Prediction [2020. KDD]](https://dl.acm.org/doi/pdf/10.1145/3394486.3403117)
- Active Learning in Bayesian Neural Networks for Bandgap Predictions of Novel Van der Waals Heterostructures [2021, Advanced Intelligent Systems]

Materials Design and Discovery

- Bias free multi-objective active learning for materials design and discovery [2020]
- [Active learning for the power factor prediction in diamond-like thermoelectric materials [2020, npj Computational Materials]](https://www.nature.com/articles/s41524-020-00439-8)
- [Active Discovery of Catalysts for Sustainable Energy Storage [2021]](https://www.proquest.com/openview/660c2b7aa14e0b9228240134af6e089e/1?cbl=18750&diss=y&pq-origsite=gscholar)
- [Active discovery of organic semiconductors [2021, Nature communications]](https://www.nature.com/articles/s41467-021-22611-4)

Transition State Calculation:
- Active Learning for Transition State Calculation

## Astronomy

Spectroscopic Surveys
- [Active deep learning method for the discovery of objects of interest in large spectroscopic surveys](https://arxiv.org/pdf/2009.03219.pdf): We apply active learning classification methods supported by deep convolutional neural networks to automatically identify complex emission-line shapes in multi-million spectra archives.
- [Active learning with RESSPECT: Resource allocation for extragalactic astronomical transients [2020]](https://arxiv.org/pdf/2010.05941.pdf): The Recommendation System for Spectroscopic follow- up (RESSPECT) project aims to enable the construction of optimized training samples for the Rubin Observatory Legacy Survey of Space and Time (LSST), taking into account a realistic description of the astronomical data environment.

## Chemistry

Chemical natural language processing
- Active Learning Yields Better Training Data for Scientific Named Entity Recognition [2020]


## Math and Statistics

Model checking:
- [Active and sparse methods in smoothed model checking](https://arxiv.org/pdf/2104.09940.pdf)

## Geology

Lithology Identification:
- [Active Domain Adaptation With Application to Intelligent Logging Lithology Identification [IEEE TCYB]](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9357423&tag=1)
- Evaluation of active learning algorithms for formation lithology identification [2021, Journal of Petroleum Science and Engineering]

## Experiment Design/Experimental Condition Selection

- Active machine learning-driven experimentation to determine compound effects on protein patterns [2016, Elife]
- [Categorical Matrix Completion with Active Learning for High-throughput Screening [2020, IEEE Transactions on Computational Biology and Bioinformatics]](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9043585): Categorical matrix completion is designed to accurately impute the missing experiments while margin sampling is also implemented for uncertainty estimation.
- Active Learning Approach to Optimization of Experimental Control [2020, CHIN. PHYS. LETT.]: Also an optimization problem, where they try to get the optimal control parameters.
- AI-Assisted Scientific Data Collection with Iterative Human Feedback [2021]
- Active Learning for the Optimal Design of Multinomial Classification in Physics [2021]

# Industrial Applications

The industrial applications of AL are about the practical problems or specific requirements in the industry.

## Labeling System

This is the most direct industrial application.

Firms:
- [Scale](https://scale.com)
- [Figure8](https://en.wikipedia.org/wiki/Figure_Eight_Inc.) (It seems to be closed.)

Papers:
- [AI-Assisted Human Labeling: Batching for Efficiency without Overreliance [2021, ACM-HCI]](https://dl.acm.org/doi/pdf/10.1145/3449163)

## Remote Sensing 

Remote sensing is an field with really high labeling cost.
AL is well used in this field.

Several surveys of AL for remote sensing:
- A survey of active learning algorithms for supervised remote sensing image classification
- A Survey of Active Learning for Quantifying Vegetation Traits from Terrestrial Earth Observation Data [2021]

Papers:
- [An Active Deep Learning Approach for Minimally Supervised PolSAR Image Classification [IEEE Transactions on Geoscience and Remote Sensing, 2019]](https://ieeexplore.ieee.org/abstract/document/8784406)4
- Deep Active Learning for Remote Sensing Object Detection [2020, Arxiv]
- Deep Active Learning in Remote Sensing for data efficient Change Detection [2020, Arxiv]
- Classiﬁcation of Summer Crops Using Active Learning Techniques on Landsat Images in the Northwest of the Province of Buenos Aires [2020]
- [Online Semisupervised Active Classification for Multiview PolSAR Data [2020, IEEE TRANSACTIONS ON CYBERNETICS]](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9238410)
- [Hyperspectral Image Classiﬁcation with Feature-Oriented Adversarial Active Learning [2020, Remote Sensing]](https://www.mdpi.com/2072-4292/12/23/3879)
- Adversarial Discriminative Active Deep Learning for Domain Adaptation in Hyperspectral Images Classification [2021, INTERNATIONAL JOURNAL OF REMOTE SENSING]
- Active Deep Learning for Hyperspectral Image Classification With Uncertainty Learning [2021,Active Deep Learning for Hyperspectral Image Classification With Uncertainty Learning]

## Medical Research 

### Medical Image Classification / Image Annotation

- [Fine-tuning Convolutional Neural Networks for Biomedical Image Analysis: Actively and Incrementally [CVPR, 2017]](http://openaccess.thecvf.com/content_cvpr_2017/html/Zhou_Fine-Tuning_Convolutional_Neural_CVPR_2017_paper.html):
  The CNN is fine-tuned in each active learning iteration **incrementally**.
  Assume that each candidate (predefined) takes one of possible labels Y.
  This assumption might make it more difficult to generalize.
- [Semi-Supervised Active Learning for COVID-19 Lung Ultrasound Multi-symptom Classification [2020]](https://arxiv.org/pdf/2009.05436.pdf): Multi-label classification network (MSML) is proposed, and a human-machine interaction is exploited to confirm the final annotations that are used to fine-tune MSML with progressively labeled data. Still pool based.
- [An Adaptive Low-Rank Modeling-Based Active Learning Method for Medical Image Annotation [2020, IRBM]](https://www.sciencedirect.com/science/article/pii/S1959031820301056): Medical images: the intrinsic presence of noise in medical images, the large number of images, and the variety of imaging modalities. Low-rank modeling-based multi-label active learning (LRMMAL) method.
- A Transfer Learning Based Active Learning Framework for Brain Tumor Classification [2020]
- An Active Learning Method for Diabetic Retinopathy Classification with Uncertainty Quantification [2020]
- Deep Reinforcement Active Learning for Medical Image Classification [2020]
- MedSelect: Selective Labeling for Medical Image Classification Combining Meta-Learning with Deep Reinforcement Learning [2021]
- Representative Region Based Active Learning For Histological Classification Of Colorectal Cancer [2021, ISBI]
- Su-Sampling Based Active Learning For Large-Scale Histopathology Image [2021, ICIP]

### Medical Image Segmentation

- Suggestive annotation: A deep active learning framework for biomedical image segmentation [2017, MICCAI]
- [Diminishing Uncertainty within the Training Pool: Active Learning for Medical Image Segmentation [2020, IEEE TRANSACTIONS ON MEDICAL IMAGING]](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9310359)
- Efficient Active Learning for Image Classification and Segmentation Using a Sample Selection and Conditional Generative Adversarial Network [2018, MICCAI]
- [Deep Active Learning for Axon-Myelin Segmentation on Histology Data [Arxiv, 2019]](https://arxiv.org/abs/1907.05143)
- Labeling Cost Sensitive Batch Active Learning For Brain Tumor Segmentation [2021, ISBI]
- U-Net-Based Active Learning Framework for Enhancing Cancer Immunotherapy [2021, Master's Thesis]
- Reducing Annotating Load: Active Learning with Synthetic Images in Surgical Instrument Segmentation

### Medical Symptom Recognition from Text

- Medical symptom recognition from patient text: An active learning approach for long-tailed multilabel distributions [2020, ML4H]

### ECG classification

- [Deep learning approach for active classification of electrocardiogram signals [2016, Information Science]](https://reader.elsevier.com/reader/sd/pii/S0020025516300184?token=EBB87D490BCDC26916121FCCCBAC34EFC879C7908C40ACF69667DCE1136B957C4608146ABABFCD7F438D7E7C8E4BA49C): (280 citations)

### Retinal image analysis

- Human-in-the-loop for efficient training of retinal image analysis methods [2021, Master's Thesis]

## Drug Discovery

- Active-learning strategies in computer-assisted drug discovery. [2015, Drug Discov.]
- Active learning for computational chemogenomics [2017, Future Medicinal Chemistry]
- Evaluation of Categorical Matrix Completion Algorithms: Towards Improved Active Learning for Drug Discovery [2021, System Biology]

## Spam Detection

- [Camouflaged Chinese Spam Content Detection with Semi-supervised Generative Active Learning [2020, ACL]](https://www.aclweb.org/anthology/2020.acl-main.279.pdf)

## Person Re-identification

- [Rethinking data collection for person re-identification: active redundancy reduction [2021, Pattern Recognition]](https://reader.elsevier.com/reader/sd/pii/S0031320321000145?)

## Mobile Health Monitoring / Disease detection

- Collaborative Multi-Expert Active Learning for Mobile Health Monitoring: Architecture, Algorithms, and Evaluation [2020, Sensors]
- [Exploiting Active Learning in Novel Refractive Error Detection with Smartphones [2020, ACMMM]](https://dl.acm.org/doi/pdf/10.1145/3394171.3413748): Use their own created dataset (only 172 images).

## Privacy Policy Classification
- Deep Active Learning with Crowdsourcing Data for Privacy Policy Classification [2020, Arxiv]

## Dataset Engineering

### Dataset Building/ Data Annotation

- Lsun: Construction of a large-scale image dataset using deep learning with humans in the loop [2015, Arxiv]
- A Simple Yet Brisk And Efficient Active Learning Platform For Text Classification
- [Scale AI](https://scale.com)
- [Appen (Figure-Eight)](https://appen.com)

### Data Enrichment

- [ActiveDeeper: A Model-based Active Data Enrichment System](http://www.vldb.org/pvldb/vol13/p2885-zhao.pdf): Use keywords provide by the oracles to search in a database to build a dataset for the current task.

## Software Engineering

API Misuse Detection
- [Active Learning of Discriminative Subgraph Patterns for API Misuse Detection [2021, TSE]](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9392340)

## Answer Selection

- [Combination of Active Learning and Self-Paced Learning for Deep Answer Selection with Bayesian Neural Network [2020, ECAI]](http://ecai2020.eu/papers/449_paper.pdf)

## Dialog Policy Learning (intelligent system)

- [Dialog Policy Learning for Joint Clarification and Active Learning Queries [2020, Arxiv]](https://arxiv.org/pdf/2006.05456.pdf)

## Communication

Eg. 5g, Traffic Classification

- [Active Popularity Learning with Cache Hit Ratio Guarantees using a Matrix Completion Committee [2020]](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9217322&tag=1): The object is modeling the optimal cache strategy in 5g wireless communication. AL is to learn the content popularities (criterion for determining whether cached or not) since it allows the system to leverage the trade-off between exploration (caching new files) and exploitation (use known files to cache). Specifically, AL is used to do matrix completion on the missing entries of the demand matrix.
- [Active content popularity learning and caching optimization with hit ratio guarantees [2020, IEEE Access]](https://ieeexplore.ieee.org/abstract/document/9159587): Similar to the last one but with more details. A query is defined as the response received from the user terminal to the system.
- [Active Learning for Network Traffic Classification: A Technical Survey [2021, Arxiv]](https://arxiv.org/pdf/2106.06933.pdf)

## Brain Mapping in a High Performance Computing Environment

About neuron segmentation and tracing at scale.
- [Active Learning Pipeline for Brain Mapping in a High Performance Computing Environment [2020, Arxiv]](https://arxiv.org/pdf/2006.14684.pdf)

## Crowd Counting

- [Active Crowd Counting with Limited Supervision [2020, Arxiv]](https://arxiv.org/abs/2007.06334): An multi-domain crowd counting framework. Use discriminator to learn a feature extractor for the crowd density regression for all the domains.
- [Uncertainty Estimation and Sample Selection for Crowd Counting [2020, Arxiv]](https://arxiv.org/pdf/2009.14411.pdf)

## Malware Detection

- Q-learning and LSTM based deep active learning strategy for malware defense in industrial IoT applications [2021, Multimedia Tools and Applications]

## Solvability Prediction in Power Systems

- [Deep Active Learning for Solvability Prediction in Power Systems [2020]](https://arxiv.org/pdf/2007.13250.pdf)

## Disguised Faces Recognition

- [A-LINK: Recognizing Disguised Faces via Active Learning based Inter-Domain Knowledge [2019]](http://iab-rubric.org/papers/2019_BTAS_ALINK.pdf)

## Gas Reservoir Prediction

- [Reservoir prediction through cost⁃sensitive active learning [2020, JOURNAL OF NANJING UNIVERSITY]](https://jns.nju.edu.cn/CN/10.13232/j.cnki.jnju.2020.04.014)
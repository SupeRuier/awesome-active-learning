# Active learning summary

In this repository, previous works of active learning were categorized. 
We try to summarize the current AL works in a **problem-orientated approach** and a **technique-orientated approach**.
We also summarized the applications of AL.
The software resources and the relevant scholars are listed.

*(I can't ensure this summary covers all the representative works and ideas.
So if you have any comments and recommendations, pls let me know.)*

- [Active learning summary](#active-learning-summary)
- [At the Beginning](#at-the-beginning)
- [Problem-oriented approach](#problem-oriented-approach)
  - [Basic Problem Settings](#basic-problem-settings)
  - [Advanced Problem Setting](#advanced-problem-setting)
  - [Problem Settings with other Research Fields](#problem-settings-with-other-research-fields)
- [Technique-oriented approach](#technique-oriented-approach)
  - [Taxonomy on AL strategies](#taxonomy-on-al-strategies)
  - [AL strategies on Different models](#al-strategies-on-different-models)
- [Theoretical Support for Active Learning](#theoretical-support-for-active-learning)
- [Practical Considerations when Apply AL](#practical-considerations-when-apply-al)
- [Real-World Applications of AL](#real-world-applications-of-al)
- [Resources:](#resources)
  - [Software Packages/Libraries](#software-packageslibraries)
  - [Tutorials](#tutorials)
- [Groups/Scholars:](#groupsscholars)
- [Our Subjective Opintions on AL](#our-subjective-opintions-on-al)

# At the Beginning

Active learning is used to reduce the annotation cost in machine learning process.
It is under the assumption that some samples are more important for a given task than other samples.
There have been several surveys for this topic.
The main ideas and the scenarios are introduced in these surveys.

- Active learning: theory and applications [[2001]](https://ai.stanford.edu/~koller/Papers/Tong:2001.pdf.gz)
- **Active Learning Literature Survey (Recommend to read)**[[2009]](https://minds.wisconsin.edu/handle/1793/60660)
- A survey on instance selection for active learning [[2012]](https://link.springer.com/article/10.1007/s10115-012-0507-8)
- Active Learning: A Survey [[2014]](https://www.taylorfrancis.com/books/e/9780429102639/chapters/10.1201/b17320-27)
- Active Learning Query Strategies for Classification, Regression, and Clustering: A Survey [[2020, Journal of Computer Science and Technology]](https://link.springer.com/article/10.1007/s11390-020-9487-4)
- A Survey of Active Learning for Text Classification using Deep Neural Networks [[2020]](https://arxiv.org/pdf/2008.07267.pdf)
- A Survey of Deep Active Learning [[2020]](https://arxiv.org/pdf/2009.00236.pdf)
- ALdataset: a benchmark for pool-based active learning [[2020]](https://arxiv.org/pdf/2010.08161.pdf)
- Active Learning: Problem Settings and Recent Developments [[2020]](https://arxiv.org/pdf/2012.04225.pdf)

# Problem-oriented approach

If you are pursuing use AL to reduce the cost of annotation in a pre-defined problem setting, we summarized the previous works in a problem-oriented order.
In this section we are not focus on what the algorithm looks like.
However, we identify the exact problem settings and list the applicable methods in the previous works.

## Basic Problem Settings

According to scenarios and tasks, almost all the AL works could be divided into the following sub-problems.
Please check [here](AL_core.md) for more details.

|                | Pool-based                     | Stream-based         | Query synthesis |
| -------------- | ------------------------------ | -------------------- | --------------- |
| Classification | PB-classification (most works) | SB-classification    |                 |
| Regression     | PB-regression                  | SB-regression (rare) |                 |

## Advanced Problem Setting

Under these problem settings, there are also other dimensions.

- [Multi-class active learning](subfields/MCAL.md): In a classification task, each instance has one label from several possible classes.
- [Multi-label active learning](subfields/MLAL.md): In a classification task, each instance has multiple labels.
- [Multi-task active learning](subfields/MTAL.md): The model or set of models handles multiple different tasks simultaneously. For instance, handle two classification tasks at the same time, or one classification and one regression. Multi-domain: Similar to multi-task, but the data are from different datasets(domains). The model or set of models handles multiple datasets simultaneously.
- [Multi-domain active learning](subfields/MDAL.md): Similar to multi-task, but the data are from different datasets(domains). The model or set of models handles multiple datasets simultaneously.
- [Multi-view/modal active learning](subfields/MVAL.md): The instances might have different views (different sets of features). The model or set of models handles different views simultaneously.

## Problem Settings with other Research Fields

In many research fields, the problems we are meeting can't be easily divided into classification or regression.
So AL algorithms should be revised for different problem settings.
Here we summarized several related research fields which use AL to reduce the cost of annotation.

- Computer vision (CV)                                    
- Natural Language Processing (NLP)                       
- Domain adaptation/Transfer learning                     
- Graph data                                              
- Metric learning/Pairwise comparison/Similarity learning 
- One-shot learning                                       
- Clustering                                              
- Generative Adversarial Network                          
- De-noise                                                
- Causal Analysis                                         
- Positive and unlabeled (PU) learning                    
- Reinforcement Learning                                  

The list of works could see [here](subfields/AL_combinations.md)

# Technique-oriented approach

After we specifying the problems,the way to derive active learning algorithms should be considered.
In this section, we analyze the current works from two aspects：
- Taxonomy on AL strategies
- AL strategies on Different models

There might be overlaps between these two aspects. 
Here we just provide two different approaches for you to find the information you need.

## Taxonomy on AL strategies

In this section, we provide our taxonomy on the current AL strategies.
We hope this can bring you a intuition to design your own strategy or just choose the most appropriate strategy.

- Supervised
  - model output guided selection
  - model middle information guided selection
- Unsupervised
  - only base on the original features
- [Batch mode](subfields/AL_combinations.md)

Use other research field to improve AL.
The list of works could see [here](subfields/AL_combinations.md)
- Reinforcement Learning
- Meta-learning
- Quantum computing
- GAN

Please check [here](subfields/pb_classification.md) for more information.

## AL strategies on Different models

There are model-free AL strategies and model-dependent AL strategies.
However, even the same strategy might be implement differently on different models.
So we summarized several of the previous works for several well used models.
Hopefully this could help you improve the performance of your current AL strategy or try to find a appropriate framework for your current model.
Specifically, we care how to apply AL strategy on each type of models in this approach.

Model list:
- SVM/LR
- Bayesian/Probabilistic
- Gaussian Progress
- Neural Network

Please check [here](AL_technique.md) form more details (Not finished yet).

# Theoretical Support for Active Learning

Might fill this slot later.

# Practical Considerations when Apply AL

When we use AL in real life scenarios, the practical situation is not perfectly match our problem settings introduced above.
The data, the oracle, the scale and many other situations could be really different.
Here we list the considerations potentially occurred in AL.

| Type       | Practical Considerations                                         |
| ---------- | ---------------------------------------------------------------- |
| Data       | Imbalanced data                                                  |
|            | Cost-sensitive                                                   |
|            | Logged data                                                      |
|            | Feature missing data                                             |
|            | Multiple Correct Outputs                                         |
| Oracle     | The assumption change on single oracle (Noise/Special behaviors) |
|            | Multiple/Diverse labeler (ability/price)                         |
| Scale      | Large-scale                                                      |
| Other cost | Model's training cost                                            |

The list of works could see [here](subfields/practical_considerations.md).

# Real-World Applications of AL

AL has already been used in many real life applications.
For many reasons, the implementations in many companies are confidential.
But we can still find many applications from several published papers and websites.

If you are wondering how could AL be applied in many other fields, we summarized a list of works [here](subfields/AL_applications.md).

# Resources:

## Software Packages/Libraries
There already are several python AL project:
- [Google's active learning playground](https://github.com/google/active-learning)
- [A modular active learning framework for Python](https://github.com/modAL-python/modAL)
- [libact: Pool-based Active Learning in Python](https://github.com/ntucllab/libact)
- [ALiPy](https://github.com/NUAA-AL/ALiPy): 
  An AL tool-box from NUAA. 
  The project is leaded by Shengjun Huang.
- [pytorch_active_learning](https://github.com/rmunro/pytorch_active_learning)
- [Deep-active-learning](https://github.com/ej0cl6/deep-active-learning)

## Tutorials
- [active-learning-workshop](https://github.com/Azure/active-learning-workshop): 
  KDD 2018 Hands-on Tutorial: Active learning and transfer learning at scale with R and Python

# Groups/Scholars:
1. [Hsuan-Tien Lin](https://www.csie.ntu.edu.tw/~htlin/)
2. [Shengjun Huang](http://parnec.nuaa.edu.cn/huangsj/) (NUAA)
3. [Dongrui Wu](https://sites.google.com/site/drwuHUST/publications/completepubs) (Active Learning for Regression)
4. Raymond Mooney
5. [Yuchen Guo](http://ise.thss.tsinghua.edu.cn/MIG/gyc.html)

# Our Subjective Opintions on AL

Active learning has been researched over 3 decades, the fundamental theories and ideas are quite complete.
Current works are more focusing on combining AL and other research fields or looking for new applications.

In my point of view, there are several directions which are promising but not fully discovered:
- Deep active learning: from the popularity of the deep models
  - Cold start
  - Uncertainty measurement
  - Efficiently update model in AL process
- Stream based active learning: most works are about pool-based AL, but the data steam is also heavily-used in daily life. This types of works should be similar to online learning
  - Selection strategy.
- Large scale active learning: AL is invented for high labeling cost situation. However, most works are just try their algorithms on a relatively small dataset
- Practical considerations: Before utilize AL into realistic situation, there still are many technique problems
- Application scenarios
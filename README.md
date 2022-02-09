# Awesome Active Learning

<span style="color:red">**Hope you can find everything you need about active learning in this repository.**</span>
This is not only a curated list, but also a well-structured library for active learning.
This repository is constructed in a **problem-orientated** approach, the techniques are discussed under the corresponding problem settings.

Specifically, this repository includes:
- Reviews/Surveys of AL
- Definition of AL
- Problem Settings (Basic/Advanced/Other AI Fields)
- Theoretical Support
- Practical Considerations
- Real-World Applications
- Resources of Using/Learning AL
- Scholars/Groups

### Contributing
If you find any valuable researches, please feel free to pull request or contact [ruihe.cs@gmail.com](ruihe.cs@gmail.com) to update this repository.
Comments and suggestions are also welcome!

# At the Beginning (Reviews/Surveys/Benchmarks)

Active learning is used to reduce the annotation cost in machine learning process.
It is under the assumption that some samples are more important for a given task than other samples.
There have been several surveys for this topic.
They provided a good overview for the field.

- Active learning: theory and applications [[2001]](https://ai.stanford.edu/~koller/Papers/Tong:2001.pdf.gz)
- **Active Learning Literature Survey (Recommend to read)**[[2009]](https://minds.wisconsin.edu/handle/1793/60660)
- A survey on instance selection for active learning [[2012]](https://link.springer.com/article/10.1007/s10115-012-0507-8)
- Active Learning: A Survey [[2014]](https://www.taylorfrancis.com/books/e/9780429102639/chapters/10.1201/b17320-27)
- Active Learning Query Strategies for Classification, Regression, and Clustering: A Survey [[2020, Journal of Computer Science and Technology]](https://link.springer.com/article/10.1007/s11390-020-9487-4)
- A Survey of Active Learning for Text Classification using Deep Neural Networks [[2020]](https://arxiv.org/pdf/2008.07267.pdf)
- A Survey of Deep Active Learning [[2020]](https://arxiv.org/pdf/2009.00236.pdf)
- ALdataset: a benchmark for pool-based active learning [[2020]](https://arxiv.org/pdf/2010.08161.pdf)
- Active Learning: Problem Settings and Recent Developments [[2020]](https://arxiv.org/pdf/2012.04225.pdf)
- From Model-driven to Data-driven: A Survey on Active Deep Learning [[2021]](https://arxiv.org/pdf/2101.09933.pdf)
- [Understanding the Relationship between Interactions and Outcomes in Human-in-the-Loop Machine Learning [[2021]](http://harp.ri.cmu.edu/assets/pubs/hil_ml_survey_ijcai_2021.pdf): HIL, a wider framework.
- A Comparative Survey: Benchmarking for Pool-based Active Learning [[2021, IJCAI]](https://www.ijcai.org/proceedings/2021/0634.pdf)
- A Survey on Cost Types, Interaction Schemes, and Annotator Performance Models in Selection Algorithms for Active Learning in Classification [[2021]](https://arxiv.org/pdf/2109.11301.pdf)

# What is Active Learning?

# Problem settings

Firstly, we summarized the previous works in a problem-oriented order.
We note that, in this section we try to identify the exact problem settings and list the methods (or works) for the corresponding settings.
In other words, we need to understand what specific problems active learning is trying to solve.

We divided the problem settings into three types:
1. Basic Problem Settings
   - Under the basic scenarios: Pool-based/Stream-based/Query synthesis
   - Under the basic tasks: Classification/Regression
2. Advanced Problem Settings
   - Under many variants of machine learning problem settings.
3. Problem Settings from other Research Fields
   - With more complex tasks or problem settings from other research fields

## Basic Problem Settings (Three basic scenarios)

According to three types of scenarios and two basic tasks, almost all the AL works could be divided into the following sub-problems.
Please check [**here**](AL_problem.md) for more details.

|                | Pool-based                     | Stream-based         | Query synthesis |
| -------------- | ------------------------------ | -------------------- | :-------------: |
| Classification | PB-classification (most works) | SB-classification    |        -        |
| Regression     | PB-regression                  | SB-regression (rare) |        -        |

## Advanced Problem Settings

There are many variants of machine learning problem settings.
Under these problem settings, AL could be further applied.

- [Multi-class active learning](subfields/MCAL.md): In a classification task, each instance has one label from multiple classes (more than 2).
- [Multi-label active learning](subfields/MLAL.md): In a classification task, each instance has multiple labels.
- [Multi-task active learning](subfields/MTAL.md): The model or set of models handles multiple different tasks simultaneously. For instance, handle two classification tasks at the same time, or one classification and one regression. 
- [Multi-domain active learning](subfields/MDAL.md): Similar to multi-task, but the data are from different datasets(domains). The model or set of models handles multiple datasets simultaneously.
- [Multi-view/modal active learning](subfields/MVAL.md): The instances might have different views (different sets of features). The model or set of models handles different views simultaneously.

## Problem Settings from other AI Research Fields

In many AI research fields, the problem settings can't be simply divided into supervised classification or regression problem.
They either acquire different types of outputs or assume a unusual learning process.
So AL algorithms should be revised/developed for these problem settings.
Here we summarized several research fields which use AL to reduce the cost of annotation.

- Computer Vision (CV)
- Natural Language Processing (NLP)
- Domain adaptation/Transfer learning
- Metric learning/Pairwise comparison/Similarity learning
- One/Few/Zero-shot learning
- Graph Processing
- Clustering
(We didn't list all of them in this section.)

The full list of works could see [**here**](subfields/AL_combinations.md)

# Theoretical Support for Active Learning

<!-- TODO: Might fill this slot later. -->
(Not finished yet)

# Practical Considerations to Apply AL

When we use AL in real life scenarios, the practical situations usually are not perfectly matching our problem settings which are introduced above.
The data, the oracle, the scale and many other situations could be really different to the experimental settings.
In other words, this section is about what else need to be considered to meet the needs in practical problems for applying AL strategies.
Here we list the practical considerations for AL.

| Type               | Practical Considerations                                         |
| ------------------ | ---------------------------------------------------------------- |
| Data               | Imbalanced data                                                  |
|                    | Cost-sensitive case                                              |
|                    | Logged data                                                      |
|                    | Feature missing data                                             |
|                    | Multiple Correct Outputs                                         |
|                    | Unknown input classes                                            |
|                    | Different data types                                             |
|                    | Data with Perturbation                                           |
| Oracle             | The assumption change on single oracle (Noise/Special behaviors) |
|                    | Multiple/Diverse labeler (ability/price)                         |
| Workflow           | Cold start                                                       |
|                    | Stop criteria                                                    |
| Scale              | Large-scale                                                      |
| Training cost      | Take into account the training cost                              |
|                    | Incrementally Train                                              |
| Query types        | Provide other feedbacks other than just labels                   |
| Performance metric | Other than the learning curves                                   |


The list of works could see [**here**](subfields/practical_considerations.md).

# Real-World Applications of AL

We have introduced that AL could be used in many [other research fields](subfields/AL_combinations.md).
In fact, AL has already been used in many real-world applications.
For many reasons, the implementations in many companies are confidential.
But we can still find many applications from several published papers and websites.

Basically, there are two types of applications: **scientific applications** & **industrial applications**.
We summarized a list of works [**here**](subfields/AL_applications.md).

# Resources

## Software Packages/Libraries
There already are several python AL projects:
- [Google's active learning playground](https://github.com/google/active-learning)
- [A modular active learning framework for Python](https://github.com/modAL-python/modAL)
- [libact: Pool-based Active Learning in Python](https://github.com/ntucllab/libact)
- [ALiPy: Active Learning in Python](https://github.com/NUAA-AL/ALiPy): 
  An AL tool-box from NUAA. 
  The project is leaded by Shengjun Huang.
- [pytorch_active_learning](https://github.com/rmunro/pytorch_active_learning)
- [DeepAL](https://github.com/ej0cl6/deep-active-learning): [Here](https://arxiv.org/abs/2111.15258) for the introductions.
- [BaaL: Bayesian Active Learning](https://github.com/ElementAI/baal/)
- [Paladin](https://www.aclweb.org/anthology/2021.eacl-demos.28.pdf): An anotation tool for creating high-quality multi-label document-level datasets.
- [lrtc](https://github.com/IBM/low-resource-text-classification-framework): experimenting with text classification tasks
- [Small-text: Active Learning for Text Classification in Python](https://github.com/webis-de/small-text): State-of-the-art active learning for text classification

## Tutorials
- [active-learning-workshop](https://github.com/Azure/active-learning-workshop): 
  KDD 2018 Hands-on Tutorial: Active learning and transfer learning at scale with R and Python
- [Active Learning from Theory to Practice](https://www.youtube.com/watch?v=_Ql5vfOPxZU):
  ICML 2019 Tutorial.
- [Overview of Active Learning for Deep Learning](https://jacobgil.github.io/deeplearning/activelearning):
  Jacob Gildenblat.

# Groups/Scholars

We also list several scholars who are currently heavily contributing to this research direction.

1. [Hsuan-Tien Lin](https://www.csie.ntu.edu.tw/~htlin/)
2. [Shengjun Huang](http://parnec.nuaa.edu.cn/huangsj/) (NUAA)
3. [Dongrui Wu](https://sites.google.com/site/drwuHUST/publications/completepubs) (Active Learning for Regression)
4. Raymond Mooney
5. [Yuchen Guo](http://ise.thss.tsinghua.edu.cn/MIG/gyc.html)

And several young researchers:
- Jamshid Sourati [University of Chicago]
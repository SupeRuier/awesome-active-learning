# Practical considerations

Here are several types of practical considerations we might meet when we use AL.
According to the variation of the underlying assumptions, these works could be divided into the following types: data, oracle, scale, other cost.

*(Note that the works list in this pages are works I browsed in background reading. 
There might be more important works in the corresponding fields not listed here.
Besides, I can't ensure the works are representative in the fields.
So if you have any comments and recommendations, pls let me know.)*

- [Practical considerations](#practical-considerations)
- [The considerations of the data](#the-considerations-of-the-data)
  - [Imbalanced data](#imbalanced-data)
  - [Cost-sensitive case](#cost-sensitive-case)
  - [Logged data](#logged-data)
  - [Feature missing data](#feature-missing-data)
  - [Multiple Correct Outputs](#multiple-correct-outputs)
- [The considerations of the oracles](#the-considerations-of-the-oracles)
  - [The assumption change on single oracles (Noise/Special behaviors)](#the-assumption-change-on-single-oracles-noisespecial-behaviors)
  - [Multiple/Diverse labeler (ability/price)](#multiplediverse-labeler-abilityprice)
- [The considerations of the scale](#the-considerations-of-the-scale)
  - [Large-scale](#large-scale)
- [The considerations of the model training cost](#the-considerations-of-the-model-training-cost)


# The considerations of the data

In the most basic setting, we assume that the data pools are perfectly collected.
The collected data don't have any missing value, and are balanced in classes.
The types of variations of the assumptions on data could be the features, labels and importance.

## Imbalanced data

The datasets will not always be balanced in classes.
When the datasets are not balanced, the AL strategy should avoid keeping selecting the instances of the majority class.

Works:
- Active learning for imbalanced sentiment classification [2012, EMNLP-CoNLL]: Make sure the selected instances are balanced. Take into account certainty and uncertainty. (70)
- Certainty-based active learning for sampling imbalanced datasets [2013, Neurocomputing] (17)
- Online Adaptive Asymmetric Active Learning for Budgeted Imbalanced Data [2018, SIGKDD]: Different query probability for different labels (according to the current model). (17)
- Active Learning for Improving Decision-Making from Imbalanced Data [2019]
- Active Learning for Skewed Data Sets [2020, Arxiv]

## Cost-sensitive case

The cost of misclassification on different classes sometimes are different.
Besides, querying different instances may cost differently.
So the AL strategy should take these into account and try to reduce the overall cost at the end.

Works:
- Learning cost-sensitive active classifiers [2002, Artificial Intelligence] (206)
- Active learning for cost-sensitive classification [2017, ICML] (25)

## Logged data 

The learner has access to a logged labeled dataset that has been collected according to a known pre-determined policy, and the goal is to learn a classiﬁer that predicts the labels accurately over the entire population, not just conditioned on the logging policy.

Works:
- Active Learning with Logged Data [2018, Arxiv]

## Feature missing data  

In many practical cases, the instances have incomplete feature descriptions.
Filling these feature values are important to accurately classify the instances.
However, the feature acquisition could also be expensive.
For example, the examinations in the hospital.
So AL could be used to handle the missing feature values.
There are basically two types: active feature acquisition and classification.
Active feature acquisition obtains feature values at the training time but the active feature acquisition obtains feature values at the test time.

Works:
- Active feature-value acquisition for classifier induction [2004, ICDM] (89)
- Active learning with feedback on both features and instances [2006, JMLR] (7)
- Cost-sensitive feature acquisition and classification [2007, Pattern Recognition] (124)
- Active learning by labeling features [2009, EMNLP] (159)
- Active feature-value acquisition [2009, Management Science] (100)
- Learning by Actively Querying Strong Modal Features [2016, IJCAI] (3)
- Active Feature Acquisition with Supervised Matrix Completion [2018, KDD] (16)
- Joint Active Learning with Feature Selection via CUR Matrix Decomposition [2019, TPAMI] (25)
- Active Feature Acquisition for Opinion Stream Classiﬁcation under Drift [2020, CEUR Workshop]
- Active feature acquisition on data streams under feature drift [2020, Annals of Telecommunications]

## Multiple Correct Outputs

Sometimes, an instance will have multiple correct outputs.
This causes the previous uncertainly based measurements to over-estimate the uncertainty and sometimes perform worse than a random sampling baseline. 

Works:
- Deep Bayesian Active Learning for Multiple Correct Outputs [2019, Arxiv] (1)

# The considerations of the oracles

The oracle is one of the important part in AL loop.
Normally, we assume that we only have one perfect oracle with accurate feedback.
However, this assumption might not always hold true in the number of the oracles or in the quality of the oracles.

## The assumption change on single oracles (Noise/Special behaviors)

We used to assume each labeler is perfect that they can provide accurate answer without doubt.
However, they are more likely to provide noise labels, or they don't even sure about the answer.

Works：
- Get another label? improving data quality and data mining using multiple, noisy labelers [2008, KDD]: This paper addresses the repeated acquisition of labels for data items when the labeling is imperfect. 
- Active Learning from Crowds with Unsure Option [2015, IJCAI]: Allow the annotators to express that they are unsure about the assigned data instances by adding the “unsure” option.
- Active learning with oracle epiphany [2016, NIPS]: The oracle could suddenly decide how to label by the accumulative effect of seeing multiple similar queries.

## Multiple/Diverse labeler (ability/price) 

Crowd-sourcing is one of the heavy application of AL.
It requires AL to be applicable with multiple labelers with different expertise.
They might bring different levels of noise and ask different levels of pay.

Works:
- Knowledge transfer for multi-labeler active learning [2013, ECMLPKDD]: Model each labeler's expertise and only to query an instance’s label from the labeler with the best expertise.
- Active learning from weak and strong labelers [2015, NIPS]: Learn a classifier with low error on data labeled by the oracle, while using the weak labeler to reduce the number of label queries made to this labeler.
- An Interactive Multi-Label Consensus Labeling Model for Multiple Labeler Judgments [2018, AAAI]: The premise is that labels inferred with high consensus among labelers, might be closer to the ground truth. Proposed  a novel formulation that aims to collectively optimize the cost of labeling, labeler reliability, label-label correlation and inter-labeler consensus.
- Cost-effective active learning from diverse labelers [2017, AAAI]: The cost of a labeler is proportional to its overall labeling quality. But different labelers usually have diverse expertise, and thus it is likely that labelers with a low overall quality can provide accurate labels on some speciﬁc instances. Select labeler can provide an accurate label for the instance with a relative low cost.
- Interactive Learning with Proactive Cognition Enhancement for Crowd Workers [2020, AAAI]: Try to help workers improve their reliability. Add a machine teaching part. Generate exemplars for human learners with the help of the ground truth inference algorithms.


# The considerations of the scale

When the scale of the problem is large, AL will face several practical problems.

## Large-scale

In AL, we normally need to evaluate each unlabeled instance.
However, when the scale is very large, the evaluation process would be costly.

Works:
- Hashing hyperplane queries to near points with applications to large-scale active learning [2010, NIPS]: We consider the problem of retrieving the database points nearest to a given hyperplane query without exhaustively scanning the database. (85)
- Scaling Up Crowd-Sourcing to Very Large Datasets: A Case for Active Learning [2014, VLDB]: (161)
- Scalable active learning for multiclass image classification [2012, TPAMI]: Use locality sensitive hashing to provide a very fast approximation to active learning, which gives sublinear time scaling, allowing application to very large datasets. (107)
- Scalable Active Learning by Approximated Error Reduction [2018, KDD]: Enable an eﬃcient estimation of the error reduction without re-inferring labels of massive data points. Also utilize a hierarchical anchor graph to construct a small candidate set, which allows us to further accelerate the AER estimation.(8)
  
# The considerations of the model training cost

In AL process, the model would be retrained in every AL iteration, this would cause a heavy computational cost.
Several works also take into account the model's training cost.

Works:
- [Minimum Cost Active Labeling](https://arxiv.org/pdf/2006.13999.pdf)
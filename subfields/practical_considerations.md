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
  - [Biased data](#biased-data)
  - [Cost-sensitive case](#cost-sensitive-case)
  - [Logged data](#logged-data)
  - [Feature missing data](#feature-missing-data)
  - [Multiple Correct Outputs](#multiple-correct-outputs)
  - [Unknown input classes](#unknown-input-classes)
  - [Different data types](#different-data-types)
  - [Data with Perturbation](#data-with-perturbation)
- [The considerations of the oracles](#the-considerations-of-the-oracles)
  - [The assumption change on single oracles (Noise/Special behaviors)](#the-assumption-change-on-single-oracles-noisespecial-behaviors)
  - [Multiple/Diverse labeler (ability/price)](#multiplediverse-labeler-abilityprice)
- [The considerations of the scale](#the-considerations-of-the-scale)
  - [Large-scale](#large-scale)
- [The consideration of the workflow](#the-consideration-of-the-workflow)
  - [Cold start problem](#cold-start-problem)
  - [Stop criteria](#stop-criteria)
  - [Asynchronous Training](#asynchronous-training)
- [The considerations of the model training cost](#the-considerations-of-the-model-training-cost)
  - [Take into the training cost into the total cost](#take-into-the-training-cost-into-the-total-cost)
  - [Incrementally Train](#incrementally-train)
- [The consideration of query/feedback types](#the-consideration-of-queryfeedback-types)
- [The consideration of the performance metric](#the-consideration-of-the-performance-metric)
- [The Considerations of More Assumptions](#the-considerations-of-more-assumptions)
  - [The Assumption Change on the Single Model](#the-assumption-change-on-the-single-model)


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
- Balancing Exploration and Exploitation: A novel active learner for imbalanced data [2020, KBS]
- [Identifying Wrongly Predicted Samples: A Method for Active Learning [2020]](https://arxiv.org/pdf/2010.06890.pdf): A type of expected loss reduction strategy. Identify wrongly predicted samples by accepting the model prediction and then judging its effect on the generalization error. Results are better than other method in imbalanced setting (Not specifically designed for imbalanced setting).
- [VaB-AL: Incorporating Class Imbalance and Difficulty with Variational Bayes for Active Learning [2021, CVPR]](https://openaccess.thecvf.com/content/CVPR2021/papers/Choi_VaB-AL_Incorporating_Class_Imbalance_and_Difficulty_With_Variational_Bayes_for_CVPR_2021_paper.pdf): 
  Assume that the probability of a model making a mistake is highly related to the label.
- SIMILAR: Submodular Information Measures Based Active Learning In Realistic Scenarios [2021, Arxiv]
- Active learning with extreme learning machine for online imbalanced multiclass classification [2021, KBS]: Base on extreme learning machine.

## Biased data

Distribution of unlabeled train data is not aligned with the test data.

- [Deep Active Learning for Biased Datasets via Fisher Kernel Self-Supervision [2020, CVPR]](https://openaccess.thecvf.com/content_CVPR_2020/papers/Gudovskiy_Deep_Active_Learning_for_Biased_Datasets_via_Fisher_Kernel_Self-Supervision_CVPR_2020_paper.pdf)

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

## Unknown input classes

In a stream based setting, the coming instances might have labels not seen before.
AL strategies should detect and make responses to these instances.

- [Into the unknown: Active monitoring of neural networks [2020]](https://arxiv.org/pdf/2009.06429.pdf): in dynamic environments where unknown input classes appear frequently.

## Different data types

We usually deal with normal data vectors in conventional learning.
Sometimes, AL need to be used to handle several unusual data types.

- [Cost-Effective Active Semi-Supervised Learning on Multivariate Time Series Data With Crowds [2020, TRANSACTIONS ON SYSTEMS, MAN, AND CYBERNETICS: SYSTEMS]](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9199304)

## Data with Perturbation

- Improving Model Robustness by Adaptively Correcting Perturbation Levels with Active Queries [2021, AAAI]:
  To utilize the model on noised/perturbed data, the models are trained on the artificial noised/perturbed data.
  To create the dataset, the different noise level should be assigned to each instance, this is costly.
  Here use active selection to assign the noise to the most valuable instance (could be least or most perturbed).

# The considerations of the oracles

The oracle is one of the important part in AL loop.
Normally, we assume that we only have one perfect oracle with accurate feedback.
However, this assumption might not always hold true in the number of the oracles or in the quality of the oracles.

There is a PhD thesis for this topic:
- [Active Learning with Uncertain Annotators [2020]](https://books.google.com.hk/books?hl=zh-CN&lr=&id=4pIiEAAAQBAJ&oi=fnd&pg=PA49&ots=i2qwdJVK3_&sig=6R5FxbWyjfcAjHZQttXL6jQ6ws8&redir_esc=y&hl=zh-CN&sourceid=cndr#v=onepage&q&f=false)

## The assumption change on single oracles (Noise/Special behaviors)

We used to assume each labeler is perfect that they can provide accurate answer without doubt.
However, they are more likely to provide noise labels, or they don't even sure about the answer.

Works：
- Get another label? improving data quality and data mining using multiple, noisy labelers [2008, KDD]: This paper addresses the repeated acquisition of labels for data items when the labeling is imperfect. 
- [Active Learning from Crowds [2011, ICML]](https://icml.cc/2011/papers/596_icmlpaper.pdf): Multiple labelers, with varying expertise, are available for query- ing.
- Active Learning from Crowds with Unsure Option [2015, IJCAI]: Allow the annotators to express that they are unsure about the assigned data instances by adding the “unsure” option.
- Active learning with oracle epiphany [2016, NeurIPS]: The oracle could suddenly decide how to label by the accumulative effect of seeing multiple similar queries.
- [Exploiting Context for Robustness to Label Noise in Active Learning [2020, TIP]](https://arxiv.org/pdf/2010.09066.pdf)
- Evidential Nearest Neighbours in Active Learning [2021, IAL-ECML-PKDD]

## Multiple/Diverse labeler (ability/price) 

Crowd-sourcing is one of the heavy application of AL.
It requires AL to be applicable with multiple labelers with different expertise.
They might bring different levels of noise and ask different levels of pay.

Works:
- Knowledge transfer for multi-labeler active learning [2013, ECMLPKDD]: Model each labeler's expertise and only to query an instance’s label from the labeler with the best expertise.
- Active learning from weak and strong labelers [2015, NeurIPS]: Learn a classifier with low error on data labeled by the oracle, while using the weak labeler to reduce the number of label queries made to this labeler.
- Active Learning from Imperfect Labelers [2016, NeurIPS]:
  Theoretical work.
- Cost-effective active learning from diverse labelers [2017, AAAI]: The cost of a labeler is proportional to its overall labeling quality. But different labelers usually have diverse expertise, and thus it is likely that labelers with a low overall quality can provide accurate labels on some speciﬁc instances. Select labeler can provide an accurate label for the instance with a relative low cost.
- An Interactive Multi-Label Consensus Labeling Model for Multiple Labeler Judgments [2018, AAAI]: The premise is that labels inferred with high consensus among labelers, might be closer to the ground truth. Proposed  a novel formulation that aims to collectively optimize the cost of labeling, labeler reliability, label-label correlation and inter-labeler consensus.
- [Active Deep Learning for Activity Recognition with Context Aware Annotator Selection [SIGKDD, 2019]](https://dl.acm.org/doi/abs/10.1145/3292500.3330688)
- Interactive Learning with Proactive Cognition Enhancement for Crowd Workers [2020, AAAI]: Try to help workers improve their reliability. Add a machine teaching part. Generate exemplars for human learners with the help of the ground truth inference algorithms.
- [Cost-Effective Active Semi-Supervised Learning on Multivariate Time Series Data With Crowds [2020, TRANSACTIONS ON SYSTEMS, MAN, AND CYBERNETICS: SYSTEMS]](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9199304): Use evolutionary idea to select the oracle set hypothesis.
- [Active Learning for Noisy Data Streams Using Weak and Strong Labelers [2020]](https://arxiv.org/pdf/2010.14149.pdf)
- Cost-Accuracy Aware Adaptive Labeling for Active Learning [2020, AAAI]: Need to select instances and labelers.
- [Active cross-query learning: A reliable labeling mechanism via crowdsourcing for smart surveillance [Computer Communications, 2020]](https://www.sciencedirect.com/science/article/pii/S014036641931730X):
  Each labeling task is repeated several times to complete the cross-query learning.
  


# The considerations of the scale

When the scale of the problem is large, AL will face several practical problems.

## Large-scale

In AL, we normally need to evaluate each unlabeled instance.
However, when the scale is very large, the evaluation process would be costly.

Works:
- Hashing hyperplane queries to near points with applications to large-scale active learning [2010, NeurIPS]: We consider the problem of retrieving the database points nearest to a given hyperplane query without exhaustively scanning the database. (85)
- Scaling Up Crowd-Sourcing to Very Large Datasets: A Case for Active Learning [2014, VLDB]: (161)
- Scalable active learning for multiclass image classification [2012, TPAMI]: Use locality sensitive hashing to provide a very fast approximation to active learning, which gives sublinear time scaling, allowing application to very large datasets. (107)
- Scalable Active Learning by Approximated Error Reduction [2018, KDD]: Enable an eﬃcient estimation of the error reduction without re-inferring labels of massive data points. Also utilize a hierarchical anchor graph to construct a small candidate set, which allows us to further accelerate the AER estimation.(8)
- Quantum speedup for pool-based active learning [2019, QIP]
  
# The consideration of the workflow

## Cold start problem

Normally, in a cold start setting (no labeled instance at all at the beginning), AL is hard to work.
Prior work, like BADGE, often depend on model uncertainty or inference, but these measures can be unreliable if the model has not trained on enough data.
Usually, a portion of data are randomly selected at the beginning to train a super weak model to get into the AL loop (might still not enough for deep models).

Transfer an existing model:
- [Cold-start Active Learning through Self-supervised Language Modeling [2020]](https://arxiv.org/pdf/2010.09535.pdf)

## Stop criteria

When to stop the active learning process is important.
If the performance barely improves, the running learning process would waste the budget.

- Rebuilding Trust in Active Learning with Actionable Metrics [2020]: Use a contradiction metric as a proxy on model improvement.

## Asynchronous Training

Make the selection phase keep running during the model training.

- [Asynchronous Active Learning with Distributed Label Querying [2021, IJCAI]](https://www.ijcai.org/proceedings/2021/0354.pdf)

# The considerations of the model training cost

In AL process, the model would be retrained in every AL iteration, this would cause a heavy computational cost.

## Take into the training cost into the total cost
Several works also take into account the model's training cost.

Works:
- [Minimum Cost Active Labeling [2020]](https://arxiv.org/pdf/2006.13999.pdf)

## Incrementally Train
Not retain the model from the scratch but incrementally train the model.
Fine-tuning is one of the practical method.

Works:
- Active and incremental learning for semantic ALS point cloud segmentation [2020]: In this paper, they propose an active and incremental learning strategy to iteratively query informative point cloud data for manual annotation and the model is continuously trained to adapt to the newly labelled samples in each iteration.

# The consideration of query/feedback types

Conventionally, the oracles provide and only provide the accurate labels of the select instance.
However, in practice, it may not be convenient for oracles to provide labels.
In other situations, oracles might provide more information than labels.
So other interactions are allowed in active learning.

Works:
- [Ask Me Better Questions: Active Learning Queries Based on Rule Induction [2011, KDD]](https://dl.acm.org/doi/pdf/10.1145/2020408.2020559)
- Active Decision Boundary Annotation with Deep Generative Models [2017, ICCV]
- [Active Learning with n-ary Queries for Image Recognition [2019, WACV]](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=8658398): This work is under multi-classification setting. Providing the exact class label may be time consuming and error prone. The annotator merely needs to identify which of the selected n categories a given unlabeled sample belongs to (where n is much smaller than the actual number of classes).
- [Active Learning++: Incorporating Annotator’s Rationale using Local Model Explanation [2020, Arxiv]](https://arxiv.org/pdf/2009.04568.pdf): Beside the label, oracles also need to provide the rationale. In this paper, the rationale is the importance rank of the features. The oracles not only provide the label of the selected instance but also an importance rank of the features on the selected instance.
- [ALICE: Active Learning with Contrastive Natural Language Explanations [2020, arxiv]](https://arxiv.org/pdf/2009.10259.pdf): This is a work from Stanford. It use an class-based AL which the AL module selects most confusing class pairs instead of instances (select the b class pairs with the lowest JensenShannon Divergence distance). The expert would provide contrastive natural language explanations. The knowledge is extracted by semantic parsing. The architecture of the model contains an one-vs-rest global classifier and local classifier (conditional execution on the global classifier). The local classifiers are not only trained on the original figures but also the resized image patches obtained in the semantic examination grounding. An attention mechanism is used to train the local classifiers.
- [Active Learning of Classification Models from Enriched Label-related Feedback [2020, PhD Thesis]](http://d-scholarship.pitt.edu/39554/7/Xue%20Final%20ETD.pdf): The human annotator provides additional information (enriched label-related feedback) reflecting the relations among possible labels. The feedback includes probabilistic scores, ordinal Likert-scale categories, Ordered Class Set, Permutation Subsets.
- [Hierarchical Active Learning with Overlapping Regions [2020]](https://dl.acm.org/doi/pdf/10.1145/3340531.3412022)

# The consideration of the performance metric

In general, we still use visual comparison on learning curves to compare different strategies.
However, sometimes the curves are not clearly evaluated.
So several works try to analysis and improve the performance metric in AL.

Works:
- [Statistical comparisons of active learning strategies over multiple datasets [2018, KBS]](https://reader.elsevier.com/reader/sd/pii/S0950705118300492?token=205F260BCACAF95AB50110F9C7D04C204F3655534DACB447434150331DBDC420A8B45831D78B83BBE772FA273E855566):
  Propose two approaches to analysis AL strategies.
  The first approach is based on the analysis of the area under learning curve and the rate of performance change. 
  The second approach considers the intermediate results derived from the active learning iterations.
- [Rebuilding Trust in Active Learning with Actionable Metrics [2021, Arxiv]](https://arxiv.org/pdf/2012.11365.pdf):
  This work states the limitations of AL in practice.
  It evaluates AL strategies under different performance metrics.

# The Considerations of More Assumptions

## The Assumption Change on the Single Model

Not only select instance but also select models.

- Active learning with model selection [2014, AAAI]
- Deep active learning with a neural architecture search [2019, Neural IPS]
- [Dual Active Learning for Both Model and Data Selection [2021, IJCAI]](https://www.ijcai.org/proceedings/2021/0420.pdf)
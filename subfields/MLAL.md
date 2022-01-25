# Multi-Label Active Learning

Multi-label active learning (MLAL) is to use AL on multi-label learning (MLL) problems.
We note that MLAL could be seen as a degeneration from multi-task active learning (MTAL) when all the tasks are classification.
Sometimes these two fields are focus on the same MLAL problems but sometimes are not.
In this chapter, we only discuss the MLAL problems in spite of what terminology the papers use.
Different from the general MTAL, MLAL usually utilize the relationship between labels.

We can divide the MLAL works into three types by the query type:
- Query all the labels of the selected unlabeled instances.
- Query the specific label of the selected instance. More efficient and minimize the waste of budget.
- Others

| MLAL query types          | Works                       |
| ------------------------- | --------------------------- |
| Instance query            | MML/MMC/BMAL/Adaptive/CVIRS |
| Instance-label pair query | 2DAL/AUDI/QUIRE             |
| Others                    | AURO                        |

There also is another sub-problem where some constrains might be known in advance (pre-defined constrains).
There are also few works on this topic.

## Works

### Instance query

- [Multi-label svm active learning for image classiﬁcation [2004, ICIP]](https://ieeexplore.ieee.org/abstract/document/1421535/):
  **ML**, **MML**.
  First MLAL work.
  The selected image set consists of the images the sum of whose expected loss values are largest.
  (133)
- [Effective multi-label active learning for text classification [2009, SIGKDD]](https://dl.acm.org/doi/abs/10.1145/1557019.1557119):
  **MMC**
  The framework use SVM to predict, and a LR was used to predict the number of the labels of the instance.
  Select the unlabeled data which can lead to the largest reduction of the expected model loss (version space of SVM). 
  Expected loss for multi-label data is approximated by summing up losses on all labels according to the most conﬁdent result of label prediction.
  (150)
- [Optimal batch selection for active learning in multi-label classification [2011, ACMMM]](https://dl.acm.org/doi/abs/10.1145/2072298.2072028)
  **BMAL**
  Batch-mode MLAL.
  Selected batch of points have minimal redundancy among them.
  They design an uncertainty vector and uncertainty matrix to evaluate the redundancy between unlabeled points.
  (18)
- [Active learning with multi-label svm classification [2013, IJCAI]](https://www.aaai.org/ocs/index.php/IJCAI/IJCAI13/paper/viewPaper/6509):
  Measure the informativeness of an instance by combining the label cardinality inconsistency and the separation margin with a tradeoff parameter
  The two strategies are integrated into an adaptive framework of multi-label active learning
  **Adaptive**
  (88)
- [Multi-Label Deep Active Learning with Label Correlation [2018, ICIP]](https://ieeexplore.ieee.org/abstract/document/8451460/):
  Use LSTM to model the relations among labels.
  CNN is used to produce high level representation of the image and the LSTM models the label dependencies.
  Select k samples furnishing the maximum entropies to form batch B.
  (2)
- [Effective active learning strategy for multi-label learning [2018, Neurocomputing]](https://www.sciencedirect.com/science/article/pii/S0925231217313371):
  **The review part of this paper is quite good.**
  Uncertainty Sampling based on Category Vector Inconsistency and Ranking of Scores (**CVIRS**).
  This strategy selects those unlabelled examples having the most unified uncertainty computed by means of the rank aggregation problem formulated, and at the same time, the most inconsistent predicted category vectors.
  (19)
- [Granular Multilabel Batch Active Learning With Pairwise Label Correlation [2021, TSMC]](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9377714): 
  Granular batch mode-based ranking active model for the multilabel (GBRAML).
- A Gaussian Process-Bayesian Bernoulli Mixture Model for Multi-Label Active Learning [2021, NeuraIPS]:
  **GP-B2M**

### Instance-label pair query

- T[wo-dimensional active learning for image classification [2008, CVPR]](https://ieeexplore.ieee.org/abstract/document/4587383/):
  **2DAL**.
  Select sample-label pairs, rather than only samples, to minimize a multi-label Bayesian classification error bound.
  Propose a Kernelized Maximum Entropy Model (KMEM) to model label correlations.
  They showed that querying instance-label pairs is more effective.
  (143)
- [Active query driven by uncertainty and diversity for incremental multi-label learning [2013, ICDM]](https://ieeexplore.ieee.org/abstract/document/6729601/):
  **AUDI**.
  Propose an incremental MLL model by combining label ranking with threshold learning, and avoid retraining from scratch after every query.
  Propose to exploit both uncertainty and diversity in the instance space and label space with the incremental MLL model.
  In the instance space, they simultaneously evaluate the uncertainty with label cardinality inconsistency and the diversity with the number of labels not queried.
  In the labels space, the distance from a label to the thresholding dummy label is employed to evaluate the uncertainty.
  (42)
- [Multi-label Image Classification via High-Order Label Correlation Driven Active Learning [2014, TIP]](https://ieeexplore.ieee.org/abstract/document/6725629/):
  **HoAL**.
  Introduce auxiliary compositional label to measure the score containing multi-correlations.
  The cross-label uncertainty on unlabeled data is deﬁned based on KL divergence. 
  Both single-label uncertainty and cross-label uncertainty are uniﬁed by the cross entropy measure.
  The informative example-label pair selection is formulated as a continuous optimization problem over selection variables with the consideration of label correlations.
  (39)
- [Active learning by querying informative and representative examples [2014, TPAMI]](https://ieeexplore.ieee.org/abstract/document/6747346/):
  **QUIRE**.
  Introduce a label correlation matrix.
  Combining label correlation with the measures of representativeness and informativeness for query selection.
  (393)
- [Multi-label active learning based on submodular functions [2018, Neurocomputing]](https://www.sciencedirect.com/science/article/pii/S0925231218307070):
  Propose a query strategy by constructing a submodular function for the selected instance-label pairs, which can measure and combine the informativeness and representativeness
  (0)
- [Multiview Multi-Instance Multilabel Active Learning [2021, TNNLS]](https://repository.kaust.edu.sa/bitstream/handle/10754/667375/TNNLS-2020-P-14015%20%281%29.pdf?sequence=1&isAllowed=y)
- [Cost-effective Batch-mode Multi-label Active Learning [2021, Neurocomputing]](https://www.sciencedirect.com/science/article/pii/S0925231221012534)

### MLAL with pre-defined constrains
- [Multi-Task Active Learning with Output Constraints [2010, AAAI]](https://www.aaai.org/ocs/index.php/AAAI/AAAI10/paper/viewPaper/1947):
  Not like a conventional AL, it design a reward function to calculate VOI (value of information).
  Each possible labeling assignment for a task is first propagated to all relevant tasks reachable through constraints and the reward (VOI) is measured over all relevant reachable tasks.
  Generalize the entropy measure used in the classical single-task uncertainty sampling and also highlight the role of task inconsistency in multi-task active learning.
  (51)
- [Cost-Effective Active Learning for Hierarchical Multi-Label Classification [2018, IJCAI]](https://pdfs.semanticscholar.org/e7cf/fc7941957e8b1f790b9c106edf3fd892ad20.pdf):
  A batch mode selection.
  A label hierarchy is pre-defined.
  The informativeness of instance-label pair is counted in the contribution of ancestor and dependent. 
  (7)

### Others
- [Multi-Label Active Learning: Query Type Matters [2015, IJCAI]](https://www.aaai.org/ocs/index.php/IJCAI/IJCAI15/paper/viewPaper/10995):
  **AURO**.
  AL strategies that select example-label pairs avoid information redundancy, but they may ignore the interaction between labels and can obtain a limited supervision from each query.
  They iteratively select one instance along with a pair of labels, and then query their relevance ordering, i.e., ask the oracle which of the two labels is more relevant to the instance.
- Active Refinement for Multi-Label Learning: A Pseudo-Label Approach [2021]

# Multi-Instance-Multi-Label Active Learning

In this case, the task is to predict the labels of bags of instances.
The number of labels would be more than two.

- [Active Learning in Incomplete Label Multiple Instance Multiple Label Learning [2021, Arxiv]](https://arxiv.org/pdf/2107.10804.pdf)
- Cost-effective multi-instance multilabel active learning [2021]
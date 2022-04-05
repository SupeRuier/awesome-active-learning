# Multi-Task Active Learning

Multi-task learning (MTL) is to learn different tasks at the same time.
Multi-task active learning (MTAL) tries to use AL in multi-task learning.
Different from multi-domain learning (MDL), MTL stresses the relationships of tasks and the MDL stresses the relationships of domains.

MTAL could be divided into following parts:
- Homogeneous MTAL: all the sub-tasks are the same type.
- Heterogeneous MTAL: not all the sub-tasks are the same type.

As far as we know, most MTAL works are about Homogeneous MTAL on classification, which is known as the multi-label active learning ([MLAL](../contents/MLAL.md)) problem.
In this chapter, we would only list the works which are not included in MLAL.

## Works

### Homogeneous MTAL

- [Active learning for multi-task adaptive filtering [2010, ICML]](https://icml.cc/Conferences/2010/papers/620.pdf):
  They simplify the adaptive filtering as a text classification problem, each class would be a task (multi-class classification).
  Define a new scoring function called Utility Gain to estimate the perceived improvements in task-specific and global models.
  Cluster tasks together. 
  The approach clusters/groups related tasks by drawing the parameters for the related tasks from a mixture of Gaussians. 
  The grouping of related tasks ensures that unrelated tasks do not contaminate, as such contamination might lead to poorer understanding of the individual tasks.
  (24)
- [Interactive multi-task relationship learning [2016, ICDM]](https://ieeexplore.ieee.org/abstract/document/7837848):
  Try to learn the task relationships.
  Query the partial orders between tasks.
  For example, whether the task i and j are more related than task i and k is much easier than asking to which extent the task i and j are related to each other.
  MTRL (kMTRL) formulates after having the human-provided knowledge, which learns a task covariance matrix constrained by the partial order relationships in the domain knowledge.
  (10)
- Active Learning from Peers [2017, NeurIPS]:
  A stream-based scenario.
- Safe Active Learning for Multi-Output Gaussian Processes [2022]: 
  Multi-output regression.

### Heterogeneous MTAL

- [Multi-task active learning for linguistic annotations [2008, ACL Press]](https://www.aclweb.org/anthology/P08-1098.pdf):
  First MTAL work.
  A two-task annotation scenario: named entity and syntactic parse tree annotations on three different corpora.
  Rank the usefulness of each instance on one of the two tasks, then combine the ranks as the selection score.
  (81)
-  [Multi-Task Active Learning [2012, PhD Thesis]](https://www.lti.cs.cmu.edu/sites/default/files/research/thesis/2012/abhay_harpale_multi-task_active_learning.pdf):
  Abhay Harpale took MTAL as the topic of his PhD thesis.
  They focused on the Genre Classification and Collaborative Filtering and proposed G+CTR. 
  Two tasks shares a same parameter.
  The AL strategy selects the instance which would lead to the maximum expected change of the shared parameter.
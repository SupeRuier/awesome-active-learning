# At the beginning
**We assume the readers already have the basic knowledge of active learning.**
We will not cover all the details of the whole fields, but we will display the core information of active learning which you should know in each basic problem setting.
We are trying to make the taxonomy problem-oriented, so that you can find the relevant paper you need.
Our purpose is to help you easily find the appropriate methods to solve your problems.
And this site should work like a short cheat-sheet.

- [At the beginning](#at-the-beginning)
- [Dimensions in Taxonomy](#dimensions-in-taxonomy)
  - [Explanation for each dimension:](#explanation-for-each-dimension)
  - [The problem settings:](#the-problem-settings)
- [General dimensions](#general-dimensions)
  - [Pool-based](#pool-based)
    - [Classification](#classification)
    - [Regression](#regression)
  - [Stream-based](#stream-based)
    - [Classification](#classification-1)
    - [Regression](#regression-1)
  - [Query-synthesis](#query-synthesis)
- [Other Dimensions](#other-dimensions)
  - [Multi-label AL](#multi-label-al)
  - [Multi-task AL](#multi-task-al)
  - [Multi-domain AL](#multi-domain-al)
- [Problems in other Research Field](#problems-in-other-research-field)

# Dimensions in Taxonomy 

In this section, we summarize AL works into the following dimensions.
Each dimension is an assumption of the research problem.
We wouldn't consider all these dimensions all at once but select the basic problem settings.
The first four dimensions are the foundations of current AL works.
With these foundations, AL would be extend to several extension dimensions and some other combinations.

| General Dimensions                     | Value                     |
| ------------------------------------ | ------------------------- |
| Scenarios/Problem Settings           | pool-based/stream-based/query-synthesis   |
| Task                                 | classification/regression |
| Batch mode (For pool-based problems) | yes/no                    |

| Other Dimensions | Value           |
| ---------------- | --------------- |
| Multi-label      | single/multiple |
| Multi-task       | single/multiple |
| Multi-domain     | single/multiple |
| Multi-view/modal | single/multiple |

## Explanation for each dimension:
- Scenarios/Problem Settings:
  The way AL is basing on.
  It decides how the data reveal and how to get the instance to query.
  - Pool based: select instance from a pre-collected data pools to annotate.
  - Stream based: decide wether to annotate the comming instance in a data stream.
  - Query synthesis: the queried instances are synthezised by the algorithm.
- Task:
  The mission we are going to accomplish.
- Batch mode:
  In pool-based AL, it decides whether select more than one instance at once to annotate.
  There should be a specific selection criteria for batch selection.
- Multi-class (Classification):
  In a classification task, each instance has one label from several possible classes.
- Multi-label (Classification):
  In a classification task, each instance has multiple labels.
- Multi-task:
  The model or set of models handles multiple different tasks simultaneously.
  For instance, handle two classification tasks at the same time, or one classification and one regression.
- Multi-domain:
  Similar to multi-task, but the data are from different datasets(domains).
  The model or set of models handles multiple datasets simultaneously.
- Multi-view/modal:
  The instances might have different views (different sets of features).
  The model or set of models handles different views simultaneously.

So we would classify current AL works though the mentioned basic dimensions. 
We will list the representative works on each dimension (with a short introduction hopefully.)
Beside these dimensions, we will also introduce several combinations with AL and other problem settings.
Some real life applications also will be introduced.

## The problem settings:

We address that the first two dimensions are the most basic.
According to scenarios and tasks, almost all the AL works could be devided into the following sub-problem settings.

|                | Pool-based                     | Stream-based         |
| -------------- | ------------------------------ | -------------------- |
| Classification | PB-classification (most works) | SB-classification    |
| Regression     | PB-regression                  | SB-regression (rare) |

Under these problem settings, there are also other dimensions：

- Batch mode (under pool-based problems)
- Additional constrains
  - Multi-class (Classification)
  - Multi-label (Classification)
  - Multi-task
  - Multi-domain
  - Multi-view/modal

# General dimensions
In this chapter, we will talk about two problem settings, Pool-based setting and Stream-based setting.
For each problem, we only discuss classification and regression tasks here.

## Pool-based

In pool based setting, a bunch of unlabeled data could be collected in advance as a data pool.
The purpose of pool-based active learning is to learn a model on the current data pool with as less labeled instances as possible.
The instances need to be annotated are selected iteratively in the active learning loop with the corresponding query strategy.
Instead of selection of instances, some other works try to generalize new instances to query, which is called **Query synthesis**.
We won't talk about the query synthesis for now but focus on how to select instances.

The instances selection strategies evaluate how useful the instances are.
Different works evaluate instances in different ways.
We note that most of the methods are model-specific, which means they can't freely apply to all the kinds of models.
Some methods are partly model-free, and they are most applied on discriminative models (e.g. Uncertainty).
Here are the common used heuristics for selection.

| Evaluation          | Description                                              | Comments                             |
| ------------------- | -------------------------------------------------------- | ------------------------------------ |
| Informativeness     | Only use the output of the current model.                | Neglect the underlying distribution. |
| Representativeness  | Utilize the unlabeled instances distribution.            | Normally used with the first type.   |
| Future improvements | Evaluate how much the model's performance would improve. | The evaluations usually take time.   |
| A learnt evaluation | Learn a evaluation function directly.                    |                                      |

Batch mode selection is important in AL.
In real life cases, annotating a batch of instance would be more efficient.
It requires that the information overlap of instances in a single query batch should be small enough.
There are several types of batch selection techniques.
But the intuitions are same, which is make the selected instances be diverse enough.

| Intuition                                   | Techniques                                       |
| ------------------------------------------- | ------------------------------------------------ |
| Diverse the instances in the selected batch | Usually reselect from the preselected set.       |
|                                             | Some works treat this as a optimization problem. |

### Classification

Batch-mode and multi-class are two most important dimensions in pool-based classification.
Batch makes the query selection more efficient and avoids redundant information query.
And multiple class is common in real life.
We have to note that a large amount of works focus on non-batch mode binary classification (A).

|                 | Binary classification     | Multi-class classification |
| --------------- | ------------------------- | -------------------------- |
| Non-batch mode: | (A). Most of the AL works | (B). Generalize from (A)   |
| Batch mode:     | (C). Improve over (A)     | (D). Combine (B) and (C)   |

For more details, the list of works with short introductions could see [here](subfields/pb_classification.md).

### Regression

For active learning regression (ALR), there are two problem settings.
Supervised ALR is similar to the conventional pool based AL where the selection proceed interactively.
Unsupervised ALR (passive sampling sometimes) assume we don't have any labeled instances when we select data.
So in unsupervised ALR, the selection is only happened once at the beginning.
We list several representative methods in the following table.

| Active learning for Regression: | Supervised            | Unsupervised (Passive) |
| ------------------------------- | --------------------- | ---------------------- |
| Non-batch mode                  | QBC/EMCM/RSAL/GSy/iGS | P-ALICE/Gsx/iRDM       |
| Batch mode                      | EBMALR                | **N/A**                |

For more details, the list of works could see [here](subfields/pb_regression.md).

## Stream-based

In stream-based AL, the unlabeled data come with a stream manner, and the AL module decides whether to annotate the coming instance to update the model.
This setting is not as popular as pool-based active learning. 
In most times, it needs to consider data drift where the underlying distribution is varying over time.
The common methodology is to set a threshold and define a information measurement score, and the coming instance with a score above the threshold would be queried.

### Classification

The list of works could see [here](subfields/sb_classification.md).

### Regression

The list of works could see [here](subfields/sb_regression.md).

## Query-synthesis

# Other Dimensions

Multi-label/task/domain problems are usually considered in a pool based AL problem.

## Multi-label AL

The list of multi-label AL works could see [here](subfields/MLAL.md).

## Multi-task AL

The list of multi-task AL works could see [here](subfields/MTAL.md).

## Multi-domain AL

The list of multi-domain AL works could see [here](subfields/MDAL.md).

# Problems in other Research Field

Active learning is also been used with other Learning/Research paradigms.
Some of them are use AL to reduce the annotation cost.
Others try to improve the AL process with the knowledge in other fields.

| Type          | Combination with other research problems                |
| ------------- | ------------------------------------------------------- |
| Utilize AL    | Computer vision (CV)                                    |
|               | Natural Language Processing (NLP)                       |
|               | Domain adaptation/Transfer learning                     |
|               | Graph data                                              |
|               | Metric learning/Pairwise comparison/Similarity learning |
|               | One-shot learning                                       |
|               | Clustering                                              |
|               | Generative Adversarial Network                          |
|               | De-noise                                                |
|               | Causal Analysis                                         |
|               | Positive and unlabeled (PU) learning                    |
|               | Reinforcement Learning                                  |
| To improve AL | Reinforcement Learning                                  |
|               | Meta-learning                                           |
|               | Quantum computing                                       |
|               | GAN                                                     |

The list of works could see [here](subfields/AL_combinations.md) (Not finished yet.)



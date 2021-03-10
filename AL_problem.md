# Basic Problem Settings
**We assume the readers already have the basic knowledge of active learning.**
In this chapter, we try to identify and describe different type of basic active learning problem settings.

- [Basic Problem Settings](#basic-problem-settings)
- [Taxonomy](#taxonomy)
- [Pool-based Scenario](#pool-based-scenario)
  - [Classification](#classification)
  - [Regression](#regression)
- [Stream-based Scenario](#stream-based-scenario)
  - [Classification](#classification-1)
  - [Regression](#regression-1)
- [Query-Synthesis Scenario](#query-synthesis-scenario)

# Taxonomy 

In this basic problem settings, we only consider two dimensions in our taxonomy:

- Scenarios:
  The way AL is basing on.
  It decides how the data reveal and how to get the instance to query.
  - Pool based: select instance from a pre-collected data pools to annotate.
  - Stream based: decide wether to annotate the coming instance in a data stream.
  - Query synthesis: the queried instances are synthesized by the algorithm.
- Task: We are going to accomplish.
  - Classification
  - Regression

According to scenarios and tasks, almost all the AL works could be divided into the following sub-problem settings.

|                | Pool-based                     | Stream-based         | Query synthesis |
| -------------- | ------------------------------ | -------------------- | :-------------: |
| Classification | PB-classification (most works) | SB-classification    |        -        |
| Regression     | PB-regression                  | SB-regression (rare) |        -        |


# Pool-based Scenario

In pool-based setting, a bunch of unlabeled data could be collected in advance as a data pool.
The purpose of pool-based active learning is to learn a model on the current data pool with as less labeled instances as possible.
The instances need to be annotated are selected iteratively in the active learning loop with the corresponding query strategy.

The instances selection strategies evaluate how useful the instances are.
Different works evaluate instances in different ways.
Here are the common used heuristics for selection.

<!-- TODO fix the logics -->
| Evaluation          | Description                                              | Comments                             |
| ------------------- | -------------------------------------------------------- | ------------------------------------ |
| Informativeness     | Only use the output of the current model.                | Neglect the underlying distribution. |
| Representativeness  | Utilize the unlabeled instances distribution.            | Normally used with the first type.   |
| Future improvements | Evaluate how much the model's performance would improve. | The evaluations usually take time.   |
| A learnt evaluation | Learn a evaluation function directly.                    |                                      |

## Classification

Batch-mode and multi-class are two most important dimensions in pool-based classification.
Batch makes the query selection more efficient and avoids redundant information query.
And multiple class is common in real life.
We have to note that a large amount of works focus on non-batch mode binary classification (A).

|                 | Binary classification     | Multi-class classification |
| --------------- | ------------------------- | -------------------------- |
| Non-batch mode: | (A). Most of the AL works | (B). Generalize from (A)   |
| Batch mode:     | (C). Improve over (A)     | (D). Combine (B) and (C)   |

For more details, the list of works with short introductions could see [here](subfields/pb_classification.md).

## Regression

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

# Stream-based Scenario

In stream-based AL, the unlabeled data come with a stream manner, and the AL module decides whether to annotate the coming instance to update the model.
This setting is not as popular as pool-based active learning. 
In most times, it needs to consider data drift where the underlying distribution is varying over time.
The common methodology is to set a threshold and define a information measurement score, and the coming instance with a score above the threshold would be queried.

## Classification

The list of works could see [here](subfields/sb_classification.md).

## Regression

The list of works could see [here](subfields/sb_regression.md).

# Query-Synthesis Scenario

Instead of selection of instances, some other works try to generalize new instances to query, which is called **Query synthesis**.
We won't talk about the query synthesis for now but focus on how to select instances.

This field is not well developed in the past years.
TODO Fill this slot later.



# Basic Problem Settings
**We assume the readers already have the basic knowledge of active learning.**
In this chapter, we try to identify and describe different type of basic active learning problem settings.

- [Basic Problem Settings](#basic-problem-settings)
- [Taxonomy](#taxonomy)
- [Pool-based Scenario](#pool-based-scenario)
  - [Classification](#classification)
  - [Regression](#regression)
- [Stream-based Scenario](#stream-based-scenario)
- [Query Synthesis Scenario](#query-synthesis-scenario)

# Taxonomy 

In this chapter, we consider three types of scenarios and two types of tasks in the basic problem setting.

- Scenarios: (where the queried instances are from)
  - pool-based: select from a pre-collected data pool
  - stream-based: select from a steam of incoming data
  - query synthesis: generate query instead of selecting data
- Task: (what we are going to accomplish)
  - classification
  - regression

According to scenarios and tasks, the AL works could be divided into the following sub-problem settings.

|                | Pool-based                     | Stream-based      | Query synthesis |
| -------------- | ------------------------------ | ----------------- | :-------------: |
| Classification | PB-classification (most works) | SB-classification |        -        |
| Regression     | PB-regression                  | SB-regression     |        -        |

# Pool-based Scenario

In pool-based setting, a bunch of unlabeled data should be collected in advance as a data pool.
The purpose of pool-based active learning is to learn a model on the current data pool with as less labeled instances as possible.
The instances need to be annotated are selected iteratively in the active learning loop with the corresponding query strategy.

The instances selection strategies evaluate how useful the instances are.
So the AL strategies would give each instance a score.
The score usually imply how much information the instance contains in the corresponding task.
The instances with highest scores would be selected.
Different strategies calculate the scores in different ways.

In pool-based scenario, batch mode selection is also important, i.e. select a batch of instances with the maximum information.
Batch makes the query selection more efficient and avoids redundant information query.
We summarize the idea of **batch-mode** selection [**here**](contents/batch_mode.md).

## Classification

We have to note that a large amount of works focus on pool based classification.
We categorized the current pool-based classification strategies by how they calculate the scores.

| Score                         | Description                                       | Comments                                                                                              |
| ----------------------------- | ------------------------------------------------- | ----------------------------------------------------------------------------------------------------- |
| **Informativeness**           | Uncertainty by the model prediction               | Usually refers to how much information instances would bring to the model.                            |
| **Representativeness-impart** | Represent the underlying distribution             | Normally used with informativeness. This type of methods may have overlaps with batch-mode selection. |
| **Expected Improvements**     | The improvement of the model's performance        | The evaluations usually take more time.                                                               |
| **Learn to score**            | Learn a evaluation function directly.             |                                                                                                       |
| **Others**                    | Could not classified into the previous categories |                                                                                                       |

For more details, the list of works with short introductions could see [**here**](contents/pb_classification.md).

## Regression

For active learning regression (ALR), there are two problem settings.
Supervised ALR is similar to the conventional pool based AL where the selection proceed interactively.
Unsupervised ALR (passive sampling sometimes) assume we don't have any labeled instances when we select data.
So in unsupervised ALR, the selection is only happened once at the beginning.
In this case, the active refers the way to select.
We list several representative methods in the following table.

| Active learning for Regression | Supervised            | Unsupervised     |
| ------------------------------ | --------------------- | ---------------- |
| Non-batch mode                 | QBC/EMCM/RSAL/GSy/iGS | P-ALICE/Gsx/iRDM |
| Batch mode                     | EBMALR                | -                |

For more details, the list of works could see [**here**](contents/pb_regression.md).

# Stream-based Scenario

In stream-based AL, the unlabeled data come with a stream manner, and the AL module decides whether to annotate the coming instance to update the model.
This setting is also called **online AL**.
There won't be any comparisons between different instances.
This setting is not as popular as pool-based active learning. 
In most times, it needs to consider data drift where the underlying distribution is varying over time.

For both the classification and the regression tasks, the common methodology is to set a threshold and define a information measurement score, and the coming instance with a score above the threshold would be queried.
The corresponding works would be found in the following links:
- [**Stream-based Classification**](contents/sb_classification.md).
- [**Stream-based Regression**](contents/sb_regression.md).

# Query Synthesis Scenario

Instead of selecting instances, another type of works tries to generalize new instances to query, which is called **Query synthesis**.
This field is not well developed in the past years.
But there still are several works focus on it.
For more details, the list of works could see [**here**](contents/query-synthesis.md).
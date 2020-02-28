# At the beginning
In this file, we give our taxonomy of active learning.
We will not cover all the details of the whole fields, but we will display the core information of active learning which you should know in each basic problem setting.
We are trying to make the taxonomy problem-oriented.
Our purpose is to help you easily find the appropriate methods to solve your problem.

- [At the beginning](#at-the-beginning)
- [Taxonomy (table of content)](#taxonomy-table-of-content)
- [Representative works on different dimensions](#representative-works-on-different-dimensions)
  - [Pool-based](#pool-based)
    - [Classification](#classification)
    - [Regression](#regression)
  - [Stream-based](#stream-based)
    - [Classification](#classification-1)
  - [Query synthesis](#query-synthesis)
- [Representative works on other research problems and applications](#representative-works-on-other-research-problems-and-applications)
  - [Combinations with other research problems](#combinations-with-other-research-problems)
  - [Applications](#applications)

# Taxonomy (table of content)

In this summary, we will analysis AL in the following dimensions.
Each dimension is a assumption of the research problem.
We wouldn't consider all these dimensions all at once but select the common seen and basic problem settings.
The first four dimensions are the foundation of current AL works.
With these foundation, AL would be extend to several extension dimensions and some other combinations.

| Dimension                    | Value                                   |
| ---------------------------- | --------------------------------------- |
| Scenarios                    | pool-based/stream-based/query synthesis |
| Task                         | classification/regression               |
| **(For pool-based)**         |                                         |
| Batch mode                   | yes/no                                  |
| **(For stream-based)**       |                                         |
| Concept drift                | with/without                            |
| **(Other dimensions)**       |                                         |
| Multi-class (Classification) | binary/multiple                         |
| Multi-label (Classification) | single/multiple                         |
| Multi-task                   | single/multiple                         |
| Multi-domain                 | single/multiple                         |
| Multi-view/modal             | single/multiple                         |

EXPLANATION FOR EACH DIMENSION
- Scenarios:
  The way AL basing on.
  It decides how the data reveal and how to get the instance to query.
- Task:
  The mission we are going to accomplish.
- Batch mode:
  In pool-based AL, it decides whether select more than one instance all at once to annotate.
  There should be specific criteria for batch selection.
- Concept drift:
  In stream-based AL, the data distribution may change over time.
- Multi-class (Classification):
  In classification each instance has one label from several possible classes.
- Multi-label (Classification):
  In classification, each instance has multiple label.
- Multi-task:
  The model handles multiple different tasks simultaneously.
  For instance, handle two classification tasks at the same time, or one classification one regression.
- Multi-domain:
  Similar to multi-task, but the data from different datasets(domains).
  The model handles multiple datasets simultaneously.
- Multi-view/modal:
  The instances might have different views(different sets of features).
  The model handles different views simultaneously.

So we would classify current AL works though the mentioned basic dimensions and we focus on pool based classification problem, because it is the most **basic** part in AL.
Besides, we also introduce several combinations with other problem setting.
Some applications also will be introduced.

# Representative works on different dimensions

In this chapter, we will talk about the common considered dimensions in AL.
We will give the most famous/useful methods/frameworks under that dimensions.
For these works, we will also provide a short summary.

## Pool-based
Pool based setting means we can collect a bunch of unlabeled data as a data pool in advance.
And the mission is to annotate those data (i.e. build a model on it).

### Classification

Batch-mode and multi-class are two most important dimensions in pool-based classification.
Batch makes the query selection more efficient and avoids redundant information query.
And multiple class is common in real life.

| Classification: | Binary classification | Multi-class classification |
| --------------- | --------------------- | -------------------------- |
| Non-batch mode: |                       |                            |
| Batch mode:     |                       |                            |

Besides, multi-label/task/domain also could be take into account upon the multi-class dimension.

| Classification: | Multi-label | Multi-task | Multi-domain |
| --------------- | ----------- | ---------- | ------------ |
| Non-batch mode  |             |            |              |
| Batch mode      |             |            |              |

### Regression

| Regression:     |     |
| --------------- | --- |
| Non-batch mode: |     |
| Batch mode:     |     |

## Stream-based

In stream-based AL, the unlabeled data come with a stream manner, and the AL decides whether to annotate the coming instance to update the model.
This setting is not as popular as pool based active learning. 

### Classification
In stream-based AL, normally it is classification problem with incremental classifier.
Sometimes it also needs to consider data drift.

| Classification:     |
| ------------------- | --- |
| without data drift: |     |
| with data drift:    |     |

## Query synthesis
The quired items are not selected but synthesized from the distribution.
There are not too many works in this fields.

# Representative works on other research problems and applications

## Combinations with other research problems

| Combination with other research problems | Works |
| ---------------------------------------- | ----- |
| Transfer learning/Domain adaptation      |       |
| Semi-supervised learning                 |       |
| Reinforcement Learning                   |       |
| Meta learning                            |       |
| Generative Adversarial Network           |       |

## Applications
We also summarized the known AL applications from papers.

| Application                         | Works |
| ----------------------------------- | ----- |
| Remote Sensing Image Classification |       |
| Recommendation                      |       |

# Active Learning Summary
In this file, we give our taxonomy of active learning.
We will not cover all the details of the whole fields, but we will display the core information of active learning which you should know in each basic problem setting.
We are trying to make the taxonomy problem-oriented.

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

NEED AN EXPLANATION FOR EACH DIMENSION.

And most of active learning works focus on pool-based active learning in classification setting.
And compared to pool/stream-based AL, query synthesis is not really popular and we will not discuss it too much.
So we would classify current AL works though the mentioned basic dimensions and we focus on pool based classification problem, because it is the most **basic** part in AL.

Besides, we also introduce several combination with other problem setting.
Some applications also will be introduced.

# Representative works on different dimensions

In this chapter, we will talk about the common considered dimensions in AL.
We will give the most famous/useful methods/frameworks under that situation.
For these works, we will also provide a short summary.

## Pool-based
Pool based setting means we can collect a bunch of unlabeled data as a data pool in advance.
And the mission is to annotate those data.

### Classification

Batch-mode and multi-class are two most important dimensions in pool-based classification.

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

In stream-based AL, the unlabeled data come with a stream manner, and the AL decides whether to annotate the coming instance in order to update the model.
This setting is not as popular as pool based active learning. 

### Classification
In stream-based AL, normally it is classification problem with incremental classifier.
Sometimes it also needs to consider data drift.

| Classification:     |
| ------------------- | --- |
| without data drift: |     |
| with data drift:    |     |

## Query synthesis
The quired items are not selected but synthesized.
There are not too many works in this fields.

# Representative works on other problem setting and applications

## Combination with other research  problems
AL always used with other perspectives.
From my point of views, it is far from application, but is a new angle to see the technic.

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

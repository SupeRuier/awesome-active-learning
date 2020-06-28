## Without concept drift
Can't find works in this setting so far.
This setting is not as common and useful as the setting with data drift.

## With concept drift
1. [Active Mining of Data Streams [2004, SIAM]](https://epubs.siam.org/doi/abs/10.1137/1.9781611972740.46): 
   It estimates the error of the model on the new data stream without knowing the true class labels. 
   When significantly higher error is suspected, it investigates the true class labels of a selected number of examples in the most recent data stream to verify the suspected higher error.
   They took decision trees as the base model.
   The error is basing on the leaf statistic (probability distribution on each node) and the validation loss.
   (146)
2. [Active Learning from Data Streams [2007, ICDM]](https://ieeexplore.ieee.org/abstract/document/4470323):
   Use a classifier ensemble to predict the labels.
   A Minimal Variance principle of the ensemble classifiers is introduced to guide instance labeling from data streams. 
   In addition, a weight updating rule of ensemble classifiers is derived to ensure that our instance labeling process can adaptively adjust to dynamic drifting concepts in the data.
   (103)
3. [Unbiased Online Active Learning in Data Streams [2011, SIGKDD]](https://dl.acm.org/doi/abs/10.1145/2020408.2020444):
   Binary classification problem.
   Select the instance with a higher entropy above the threshold.
   (117)
4. [Active Learning With Drifting Streaming Data [2014, TNNLS]](https://ieeexplore.ieee.org/abstract/document/6414645):
   **R-VAR**.
   Address that many of the previous approaches apply static active learning to batches, which is not possible in data streams where historical instances cannot be stored in memory.
   This work proposed a framework, which not only select decide if to query the current instance, but also detect the corresponding concept drift.
   If the change is detect, switch the prediction function to the one trained while the change warning is signaled.
   They use a uncertainty sampling with a dynamic threshold and randomization.
   (258)
5. [Active and adaptive ensemble learning for online activity recognition from data streams [2017, Knowledge-Based Systems]](https://www.sciencedirect.com/science/article/pii/S0950705117304513):
   Multi-classification setting.
   Use ensemble classifiers and the weights are adjusted after each selection.
   Use uncertainty sampling with adaptive threshold.
   (25)
6. [Online Active Linear Regression via Thresholding [2017, AAAI]](https://www.aaai.org/ocs/index.php/AAAI/AAAI17/paper/viewPaper/14599):
   For linear regression.
   For each incoming observation X, they compute its weighted norm. 
   If the norm is above the threshold, then we select the observation, otherwise we ignore it.
   The threshold could be updated after querying.
7. [Active learning for classifying data streams with unknown number of classes [2018, Neural Networks]](https://www.sciencedirect.com/science/article/pii/S0893608017302435):
   **SAL**.
   Also take into account possible emergence and fading of classes, known as concept evolution.
   Dirichlet mixture models and the stick breaking process are adopted and adapted to meet the requirements of online learning.
   Only labels of samples that are expected to reduce the expected future error are queried.
   (25)
8. [Adaptive Ensemble Active Learning for Drifting Data Stream Mining [2019, IJCAI]](https://pdfs.semanticscholar.org/0a52/d3d3108b2a67ac7de2ab9de6275234b246d1.pdf):
   **EAL-MAB**.
   A plug-in solution, capable of working with most of existing streaming ensemble classiﬁers. 
   They this process as a Multi-Armed Bandit problem, obtaining an efficient and adaptive ensemble active learning procedure by selecting the most competent classiﬁer from the pool for each query.
   Guide the instance selection by measuring the generalization capabilities of the classifiers.
   The proposed approach selects dynamically the most competent classifier to be responsible for the query decision.
   Selecting instances for label query by measuring the increase in generalization capabilities of the classifier according to a metric m on a separate validation set V.
   (3)


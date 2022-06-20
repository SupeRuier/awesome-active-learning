# Pool-Based Active Learning for Regression

For active learning regression (ALR), there are two problem settings.
Supervised ALR is similar to the conventional pool based AL where the selection proceed interactively.
Unsupervised ALR (passive sampling sometimes) assume we don't have any labeled instances when we select data.

| Active learning for Regression: | Supervised            | Unsupervised     |
| ------------------------------- | --------------------- | ---------------- |
| Non-batch mode                  | QBC/EMCM/RSAL/GSy/iGS | P-ALICE/Gsx/iRDM |
| Batch mode                      | EBMALR                | -                |

## Papers (Non-batch mode):

### Supervised:
1. [Active learning for regression based on query by committee [2007, IDEAL]](https://link.springer.xilesou.top/chapter/10.1007/978-3-540-77226-2_22): 
  **QBC**.
  The learner attempts to collect data that reduces variance over both the output predictions and the parameters of the model itself. 
    (95)
2. [Maximizing Expected Model Change for Active Learning in Regression [2013, ICDE]](https://ieeexplore.ieee.org/abstract/document/6729489/): 
  Use expected model change from gradient descend (**EMCM**).
  Choose Gradient Boost Decision Tree (GBDT) as the learner for nonlinear regression.
  (60)
3. [Kernel ridge regression with active learning for wind speed prediction [2013, Applied Energy]](https://www.sciencedirect.com/science/article/pii/S0306261912006964):
  **RSAL**.
  Use a residual regressor to predict the error of each unlabeled sample and selects the one with the largest error to label.
 (69)
4. Pool-Based Sequential Active Learning for Regression [2018, IEEE transactions on neural networks and learning systems]: 
  Reduce **EBMALR** to the sequential selection (non-batch). 
  And take diversity into account when only query single instance.
  (12)
5. [Active learning for regression using greedy sampling (2019, Information Science)](https://www.sciencedirect.com/science/article/pii/S0020025518307680):
  The first approach (GSy) selects new samples to increase the diversity in the output space (the predicted value farthest from the values of annotated instances). 
  The second (iGS) selects new samples to increase the diversity in both input and output spaces (the predicted value farthest from the values of annotated instances,and the selected instance farthest from the labeled one.).
  (16)
6. Active Nearest Neighbor Regression Through Delaunay Refinement [2022, ICML]：
   Active Nearest Neighbor Regressor (ANNR) select novel query points in a way that takes the geometry of the function graph into account.
    
### Unsupervised:
1. [Pool-based Active Learning in Approximate Linear Regression (2009, Machine Learning)](https://idp.springer.com/authorize?response_type=cookie&client_id=springerlink&redirect_uri=http://link.springer.com/article/10.1007/s10994-009-5100-3): 
  Only for linear regression.
  **P-ALICE** (Pool-based Active Learning using the Importance-weighted least-squares learning based on Conditional Expectation of the generalization error).
  Estimate the label uncertainty as the weights while selecting the M samples, and builds a weighted linear regression model from them.
  The base learner used is an additive regression model, and the parameters are learned by importance-weighted least-squares minimization. 
  (66)
2. [Active learning for regression using greedy sampling (2019, Information Science)](https://www.sciencedirect.com/science/article/pii/S0020025518307680):
  The unsupervised approach (GSx) selects samples on the original space .
  (16)
3. [Pool-Based Unsupervised Active Learning for Regression Using Iterative Representativeness-Diversity Maximization (iRDM) [2020, Arxiv]](https://arxiv.org/abs/2003.07658):
  Unsupervised ALR is to actively select instances once for all.
  Select the samples to label without knowing any true label information at the beginning.
  **Very good comparison of previous works.**
  (0)

## Papers (Batch mode):
1. [Offline EEG-based driver drowsiness estimation using enhanced batch-mode active learning (EBMAL) for regression [2016, IEEE International Conference on Systems, Man, and Cybernetics (SMC)]](https://ieeexplore.ieee.org/abstract/document/7844328/): 
   Consider informativeness, representativeness and diversity. 
   The diversity was achieved by using k-means after the pre-selection with conventional AL strategy.
   QBC and EMCM(Expected Model Change Maximization) as based AL strategy.
   **EBMALR**.
   (23)
2. A Framework and Benchmark for Deep Batch Active Learning for Regression [2022]
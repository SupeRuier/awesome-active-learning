### Optimum performance boundary
- [Towards Understanding the Optimal Behaviors of Deep Active Learning Algorithms [2021]](https://arxiv.org/pdf/2101.00977.pdf):
  Point out that there is little study on what the optimal AL algorithm looks like, which would help researchers understand where their models fall short and iterate on the design.
  Present a simulated annealing algorithm to search for an optimal strategy and analyze it for several different tasks.
- [On Statistical Bias In Active Learning: How And When To Fix It [2021, ICLR]](https://arxiv.org/pdf/2101.11665.pdf): 
  Show that correcting for active learning bias with underparameterised models leads to improved downstream performance.
- [Feedback Coding for Active Learning](https://arxiv.org/pdf/2103.00654.pdf): Formulate general active learning problems in terms of a feedback coding system, and a demonstration of this approach through the application and analysis of active learning in logistic regression.

### Algorithm and Analysis
- Active learning using region-based sampling


### Adversarial Examples

- Exploring Adversarial Examples for Efficient Active Learning in Machine Learning Classifiers [2021]: Our theoretical proofs provide support to more efficient active learning methods with the help of adversarial examples, contrary to previous works where adversar- ial examples are often used as destructive solutions

### Agnostic Active Learning
- Agnostic Active Learning [2006, ICML]
- Beyond Disagreement-based Agnostic Active Learning [2014, NeurIPS]

### Adversarial Label Corruptions

- [Corruption Robust Active Learning [2021, NeurIPS]](https://arxiv.org/pdf/2106.11220.pdf): 
  Under a streaming-based active learning for binary classification setting.

### Active Class Selection

- Certification of Model Robustness in Active Class Selection [2021]

### Representative-Based Approach

- Robust active representation via $l_{2,p}$-norm constraints [2021, KBS]

### Label/Query Complexity

- Reduced Label Complexity For Tight l2 Regression [2023]
- Query Complexity of Active Learning for Function Family With Nearly Orthogonal Basis [2023]

### Sampling Bias

- [On Statistical Bias In Active Learning: How and When to Fix It [2021, ICLR]](https://openreview.net/pdf?id=JiYq3eqTKY):
  Active learning can be helpful not only as a mechanism to reduce variance as it was originally designed, but also because it introduces a bias that can be actively helpful by regularizing the model.

### Stream-Based AL

- Neural Active Learning with Performance Guarantees [2021, NeurIPS]:
  Non-parametric regimes.
- learning with Labeling Induced Abstentions [2021, NeurIPS]:
  The performance should only be evaluated on the rest unlabeled instances.

### Label Query and Seed Query

Seed query represents the items from specific classes.
For instance, "finding the image of a car".

- Active Learning of Classifiers with Label and Seed Queries [2022]

### Deep AL 

- Investigating the Consistency of Uncertainty Sampling in Deep Active Learning [2021, DAGM-GCPR]
- Active Learning with Neural Networks: Insights from Nonparametric Statistics [2022]

### Performance Improvement

- Boosting Active Learning via Improving Test Performance [2022, AAAI]: 
  Through expected loss or the output entropy of the output instances.
  This work is still in the form of gradient length, and it is similar to EGL.
  The difference is that this work calculates the gradients with the expectation of losses, while EGL take the expectations of gradients.

### Budget v.s. Strategy

- Active Learning on a Budget: Opposite Strategies Suit High and Low Budgets [2022, ICML]:
  This work shows a behavior reminiscent of phase transition: typical points (representativeness) should best be queried in the low budget regime, while atypical (or uncertain) points are best queried when the budget is large.

### Cold Start Selection

- Active Learning Through a Covering Lens [2022]

### AL v.s. Passive

- On The Effectiveness of Active Learning by Uncertainty Sampling in Classification of High-Dimensional Gaussian Mixture Data [2022, ICASSP]
- Uniform versus uncertainty sampling: When being active is less efficient than staying passive [2022, ICML]: 
  With high-dimensional logistic regression, passive learning often outperforms uncertainty-based active learning for low label budgets.
  This high-dimensional phenomenon happens primarily when the separation between the classes is small.

### Indirect Active Learning
The covariate X might be the result of a complex process that the learner can influence and measure but not completely control.

- Nonparametric Indirect Active Learning [2022, ICML workshop]

### Formalism for AL

- A Markovian Formalism for Active Querying [2023]

# To be classified

- Active Learning with Label Comparisons [2022]
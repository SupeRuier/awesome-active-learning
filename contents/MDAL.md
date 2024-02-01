# Multi-Domain Active Learning

Multi-domain active learning (MDAL) is using AL on multi-domain learning problems.
In multi-domain learning (MDL) problems, the data are from different domains (data sources).
MDL try to train classifiers on all these domain at the same time and try to utilize the shared information among these domains.
There are not many works on this field.

Works:
1. [Multi-domain active learning for text classification [2012, KDD]](https://dl.acm.org/doi/abs/10.1145/2339530.2339701):
   This is a general MDAL framework, but it is only applied on text dataset.
   The strategy select the instance which could most reduce the version space of the multi-domain SVM model.
2. [Multi-domain active learning for recommendation [2016, AAAI]](https://www.aaai.org/ocs/index.php/AAAI/AAAI16/paper/viewPaper/12369)
   Use the Rating-Matrix Generative Model (RMGM) for multi-domain recommendation problem.
   The strategy select the instance which would induce the maximum expected generalization error.
   The error was approximately divided into domain-shared part, user part and the item part.
3. [Active learning in multi-domain collaborative filtering recommender systems [2018, SAC]](https://dl.acm.org/doi/10.1145/3167132.3167277)
4. [Multi-Domain Active Learning: Literature Review and Comparative Study [2022, TETCI]](https://arxiv.org/abs/2106.13516)
5. Multi-domain Active Learning for Semi-supervised Anomaly Detection [2022]
6. Active Learning Over Multiple Domains in Natural Language Tasks [2022]
7. Investigating Multi-source Active Learning for Natural Language Inference [2023]
8. Multi-Domain Learning From Insufficient Annotations [2023, ECAI]
9. Perturbation-Based Two-Stage Multi-Domain Active Learning [2023, CIKM]
10. Benchmarking Multi-Domain Active Learning on Image Classification [2023]
11. Composite Active Learning: Towards Multi-Domain Active Learning with Theoretical Guarantees [2024]

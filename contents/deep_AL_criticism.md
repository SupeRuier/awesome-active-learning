
# Criticism/Discovery on Deep AL

Several works compare the current DeepAL methods, and state that their experiments are flawed.
We think these papers are very interesting.
So here we give the details of each paper.

## 1. Parting with Illusions about Deep Active Learning [2019]

This work state that current state-of-art DeepAL works doesn't consider the parallel setting such as "Semi-supervised learning", "data augmentation" etc.
So they hold a comparative study on several AL strategies with SL and SSL training paradigms.
They hold the experiments on two tasks: image classification and semantic segmentation.

Results from classification task:
- AL works well with data augmentation, but data augmentation blurs the differences between AL strategies: they all perform largely the same.
- Combining SSL and AL can be yields an improvement over raw SSL.
- Relative ranking of the AL methods changes completely on different datasets
- AL selection strategy is counter-productive in the low-budget regime, even worse than Random Sampling.
- SSL-AL method clearly outperforms fine-tuning of a pre-trained ImageNet network in both high-and low-budget settings.

Results from semantic segmentation task:
- Random selection with SSL performs best

Overall conclusion:
- Current evaluation protocol used in active learning is sub-optimal which in turn leads to wrong conclusions about the methods’ performance.
- Modern semi-supervised learning algorithms applied in the conventional active learning setting show a higher relative performance increase.
- State-of-the-art active learning approaches often fail to outperform simple random sampling, especially when the labeling budget is small.

## 2. Towards Robust and Reproducible Active Learning Using Neural Networks [2020]

This work state the performance of random sampling baseline and AL strategies vary significantly over different papers.
With the goal of improving the reproducibility and robustness of AL methods, in this study they evaluate the performance of these methods for image classification compared to a random sampling in a fair experimental environment.
They also state that most AL works overlooked the regularization which would reduce the generalization error.
They hold the comparative study on different regularization setting.
(parameter norm penalty, random augmentation (RA), stochastic weighted averaging (SWA), and shake-shake (SS))

Results from image classification task:
- The performance of RS is significantly better than what they state in the other works. And there is no strategy performs clearly better than RS.
- With different AL batch size, the performance of strategies is inconsistent.
- AL methods do not outperform RS, and it isn't robust on class imbalanced setting.
- Models trained with RA and SWA consistently achieve significant performance gains across all AL iterations and exhibit appreciably smaller variance across multiple runs of the experiments.
- Consider the selected instances from VGG16 to ResNet18 and WRN-28-2, the performance varies. RS still performs well.

## 3. Effective Evaluation of Deep Active Learning on Image Classification Tasks [2021, Open Review]

Point out four issues for current AL works:
1. Contradictory observations on the performance of different AL algorithms
2. Unintended exclusion of important generalization approaches such as data augmentation and SGD for optimization
3. A lack of study of evaluation facets like the labeling efficiency of AL
4. Little or no clarity on the scenarios in which AL outperforms random sampling (RS).

They presented a unified re-implementation of state-of-the-art AL algorithms in the context of image classification.
Besides, they did a careful analysis of how existing approaches perform with varying amounts of redundancy and number of examples per class.

Point out some important details are not clear:
> - Should model training be restarted after every round, or can one fine-tune the current model? 
> - What is the effect of using carefully crafted seed sets in AL? 
> - Is there an ideal AL batch size, or does the batch size actually matter? 
> - When is AL more effective than random sampling, and what are the factors that affect the labeling efficiency of AL algorithms? Lastly, 
> - How scalable are AL algorithms; specifically, what is the amount of time taken by model training versus that taken by AL selection, and can this be made more compute and energy-efficient?

Key finding and takeaways:
> - Data augmentation and other generalization approaches have a considerable impact on the test performance as well as on the labeling efficiency.
> - SGD performs and generalizes better than Adam consistently in AL.
> - In the presence of data augmentation and the SGD optimizer, there is no significant advantage of diversity over very simple uncertainty sampling (at least in most standard academic data sets).
> - When we make the data set artificially redundant (either by repeating data points or by using near repetitions through augmentations), we see that BADGE starts outperforming uncertainty sampling.
> - The number of instances per class has an important impact on the performance of AL algorithms: The fewer the examples there are per class, the smaller the room there is for AL to improve over random sampling
> - The initialization of the labeled seed set (e.g., a random versus a more diverse or representative seed set) has little to no impact on the performance of AL after a few rounds; 
> - Reasonably sized choices of the AL batch size also have little to no impact.
> - Updating the models from previous rounds (fine-tuning) versus retraining the models from scratch negatively impacts the performance of AL only in the early selection rounds
> - The most time-consuming and energy-inefficient part of the AL loop is the model (re)training.

## 4. Reducing Label Effort: Self-Supervised meets Active Learning [2021, ICCV]

It shows that the improvements brought from AL are far less than that from self-training.
Only when the number of labeled instances approaches 50%, the gap could disappear.

## 5. On the marginal benefit of active learning: Does self-supervision eat its cake? [2021, ICASSP]: 

Fail to observe any additional benefit of state-of-the-art active learning algorithms when combined with state-of-the-art S4L techniques.

## 6. Active Learning on a Budget: Opposite Strategies Suit High and Low Budgets [2022]

This work focuses on a **cold start** setting.
Empirically reveal the unsuccessfulness of conventional AL methods with NN on very small budget, under supervised, self-supervised embedding or semi-supervised training.

## 7. Toward Realistic Evaluation of Deep Active Learning Algorithms in Image Classification [2023]

Present an AL benchmarking suite and run extensive experiments on five datasets shedding light on the questions: when and how to apply AL?
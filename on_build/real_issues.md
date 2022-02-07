The issues of AL & the solutions.

### AL Bias

At the beginning, the labeled instances are few.
In this case, the models could easily be overfitted.

Analysis:
- [On Statistical Bias In Active Learning: How and When to Fix It [2021, ICLR]](https://openreview.net/pdf?id=JiYq3eqTKY):
  Active learning can be helpful not only as a mechanism to reduce variance as it was originally designed, but also because it introduces a bias that can be actively helpful by regularizing the model.
- Addressing Bias in Active Learning with Depth Uncertainty Networks... or Not [2021, Arxiv]:
  Reducing "AL bias" doesn't bring improvement on the low "overfitting bias" model DUN.
  When eliminate the "AL bias" with importance weights, we always pay the price of additional variance ("overfitting bias").

Thus, several models try to solve the overfitting at the beginning to improve the overall performance.
It could be done by designing an adaptive loss.

Adapted models:
- [On Statistical Bias In Active Learning: How and When to Fix It [2021, ICLR]](https://openreview.net/pdf?id=JiYq3eqTKY)
- Depth Uncertainty Networks for Active Learning [2021, NeurIPS]
- Towards Dynamic and Scalable Active Learning with Neural Architecture Adaption for Object Detection [2021, BMVC]:
  Add NAS into the AL loops.
# Intrinsic Issues of AL

Current **intrinsic issues** in the AL system and the **solutions**.
These issues are not from the changes in assumptions but from the nature of AL.

## AL Bias

The selection of AL introduces bias to the system.

### Analysis
- [On Statistical Bias In Active Learning: How and When to Fix It [2021, ICLR]](https://openreview.net/pdf?id=JiYq3eqTKY):
  Active learning can be helpful not only as a mechanism to reduce variance as it was originally designed, but also because it introduces a bias that can be actively helpful by regularizing the model.
- Addressing Bias in Active Learning with Depth Uncertainty Networks... or Not [2021, Arxiv]:
  Reducing "AL bias" doesn't bring improvement on the low "overfitting bias" model DUN.
  When eliminate the "AL bias" with importance weights, we always pay the price of additional variance ("overfitting bias").
- Critical Gap Between Generalization Error and Empirical Error in Active Learning [2023, WACV]:
  The assumption that a large amount of annotated data is available for evaluating model performance apart from the data selected by AL is not realistic.
  Therefore, in a realistic model construction using AL, generalization error in the actual production environment should be estimated by cross-validation only using the data selected by AL.
  There is a gap between the actual generalization error and the empirical error when conduct- ing cross-validation on the AL-selected data.

### Adaptive Loss
Thus, several models try to solve the overfitting at the beginning to improve the overall performance.
It could be done by designing an adaptive loss.

- [On Statistical Bias In Active Learning: How and When to Fix It [2021, ICLR]](https://openreview.net/pdf?id=JiYq3eqTKY)
- Depth Uncertainty Networks for Active Learning [2021, NeurIPS]
- Towards Dynamic and Scalable Active Learning with Neural Architecture Adaption for Object Detection [2021, BMVC]:
  Add NAS into the AL loops.

### Resampling

Resample the selected uncertainty data based on feature matching to alleviate the problem of data bias.

Resample:
- Unsupervised Fusion Feature Matching for Data Bias in Uncertainty Active Learning [2022, TNNLS]

## Out-of-Distribution (OOD) Detection:

- Energy-based Out-of-distribution Detection [2020, NeurIPS]
- Active Incremental Learning for Health State Assessment of Dynamic Systems With Unknown Scenarios [2022, IEEE Transactions on Industrial Informatics]
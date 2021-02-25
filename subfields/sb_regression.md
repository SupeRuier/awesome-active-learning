## Regression

- [Online Active Linear Regression via Thresholding [2017, AAAI]](https://www.aaai.org/ocs/index.php/AAAI/AAAI17/paper/viewPaper/14599):
   For linear regression.
   For each incoming observation X, they compute its weighted norm. 
   If the norm is above the threshold, then we select the observation, otherwise we ignore it.
   The threshold could be updated after querying.
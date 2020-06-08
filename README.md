# Active learning summary

In this repository, previous works of active learning were categorized. 
We try to summarize the current AL works from **problem-orientated approach** and **technique-orientated approach**.
We also summarized the applications of AL.
The software resources and the relevant scholars are listed.

# At the Beginning

Active learning is used to reduce the annotation cost in machine learning process.
There have been several surveys for this topic.
The main ideas and the scenarios are introduced in these surveys.

- Active learning: theory and applications [[2001]](https://ai.stanford.edu/~koller/Papers/Tong:2001.pdf.gz)
- **Active Learning Literature Survey (Recommend to read)**[[2009]](https://minds.wisconsin.edu/handle/1793/60660)
- A survey on instance selection for active learning [[2012]](https://link.springer.com/article/10.1007/s10115-012-0507-8)
- Active Learning: A Survey [[2014]](https://www.taylorfrancis.com/books/e/9780429102639/chapters/10.1201/b17320-27)

# Our summarization/Categorization

If you are pursuing use AL to reduce the cost heavily problems in a pre-defined problem setting, we summarized the previous works in a [problem-orientated order](AL_core.md).

If you are trying to improve the performance of the current AL or try to find a appropriate framework for the current model, we summarized the we summarized the previous works in a [technique-orientated order](AL_technique.md).

Besides, there are many real considerations when we implement AL to real life scenarios.
We summarize these works [here](AL_considerations.md).

# Real Applications of Active Learning

AL has already been used in many real life applications.
For many reasons, the implementations in many companies are confidential.
But we can still find some innovated applications from several published papers and websites.

If you are wondering how could AL be used in real life scenarios, we summarized a list of works [here](subfields/AL_applications.md).

# Resources:
## Software Packages/Libraries
There already are several python AL project:
- [Google's active learning playground](https://github.com/google/active-learning)
- [A modular active learning framework for Python](https://github.com/modAL-python/modAL)
- [libact: Pool-based Active Learning in Python](https://github.com/ntucllab/libact)
- [ALiPy](https://github.com/NUAA-AL/ALiPy): 
  An AL tool-box from NUAA. 
  The project is leaded by Shengjun Huang.
- [pytorch_active_learning](https://github.com/rmunro/pytorch_active_learning)
- [Deep-active-learning](https://github.com/ej0cl6/deep-active-learning)
- [active-learning-workshop](https://github.com/Azure/active-learning-workshop): 
  KDD 2018 Hands-on Tutorial: Active learning and transfer learning at scale with R and Python

# Groups/Scholars:
1. [Hsuan-Tien Lin](https://www.csie.ntu.edu.tw/~htlin/)
2. [Shengjun Huang](http://parnec.nuaa.edu.cn/huangsj/) (NUAA)
3. [Dongrui Wu](https://sites.google.com/site/drwuHUST/publications/completepubs) (Active Learning for Regression)

# Need to be finished

| Check list | Chapter                                                           |
| ---------- | ----------------------------------------------------------------- |
| 5          | [Taxonomy by techniques](AL_technique.md)                         |
| 1 done     | [Pool-based regression](subfields/pb_regression.md).              |
| 3          | [MLAL](subfields/MLAL.md).                                        |
| 3          | [MTAL](subfields/MTAL.md).                                        |
| 2          | [MDAL](subfields/MDAL.md).                                        |
| 4          | [Stream-based classification](subfields/sb_classification.md)     |
| 4          | [Stream-based regression](subfields/sb_regression.md)             |
| 6          | [Practical considerations](subfields/practical_considerations.md) |
| 7          | [Combination with other fields](subfields/AL_combinations.md)     |

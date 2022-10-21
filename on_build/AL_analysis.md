## Deep AL

### Instead of Softmax
- To Softmax, or not to Softmax: that is the question when applying Active Learning for Transformer Models [2022]:
  AL fails to work for very deep NN such as Transformer models, rarely beating pure random sampling.
  The common explanation is that AL methods favor hard-to-learn samples, often simply called outliers, which therefore neglects the potential benefits from AL.

### Margin
- Is margin all you need? An extensive empirical study of active learning on tabular data [2022]
# Discriminative Models

Includes discriminative models implementations:
- Conditional Random Field: `dm.get_CRF(corpus_filename, model_filename, squared_sigma)`
- Maximum Entropy Markov Model: `dm.get_MEMM(corpus_filename, model_filename, squared_sigma)` 
- Structured Preceptron: `dm.get_SP(corpus_filename, model_filename, squared_sigma)`
- Structured SVM: `dm.get_SSVM(corpus_filename, model_filename, squared_sigma)`

For training: 
```
python dm_train.py
```

For testing:
```
python dm_test.py
```

## Requirements

```
python>=3.6
numpy
```

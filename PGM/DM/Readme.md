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

## Acknowledge
some code are adapted from the following links:
1. https://github.com/timvieira/crf/
2. https://github.com/lancifollia/crf
3. https://github.com/kohjingyu/CRF-sentiment-analysis
4. https://github.com/kohjingyu/hidden-markov-model 
5. https://github.com/bplank/sp/blob/master/simpletagger.py

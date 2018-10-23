# EuroParl Language Detection Challenge

The goal of the challenge is to train a model to classify sentences as
belonging to one of 21 languages spoken in the EU, based on the provided
training and test sets are predefined and taken from the European
Parliament Proceedings Parallel Corpus (EuroParl).

We compare two possible ways of language identification based on lexical
features (words):

1. [baseline](baseline.py): a basic dictionary approach that chooses
the language label of the majority of constituent tokens;
2. [fastText](https://github.com/facebookresearch/fastText): multinomial
logistic regression using subword information.

Experiments and results are described in the notebook
[LanguageDetection](LanguageDetection.ipynb), preprocessing functions
are stored and documented in [utils.py](utils.py).
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from collections import Counter, defaultdict

from typing import List, DefaultDict, Tuple, Any, Union


class Baseline:
    def __init__(self, train_path: str = None, min_count: int = 1,
                 unk: Any = "UNK"):
        """A model assigning a language tag to sentences based on a vocabulary.

        Args:
            train_path: Path to train sentences in the format:
                label1 sentence1
                label2 sentence2
                ...
            min_count: Min frequency of tokens to be included in the vocab.
            unk: Tag for sentences with a majority of unseen tokens.
        """
        self._vocab = self._baseline_train(train_path, min_count)
        self._unk = unk

    def __len__(self):
        return len(self._vocab)

    def predict(self, test_sents: List[List[str]]) -> List[str]:
        """Tags tokenized sentences from the `test_sents` list.

        Args:
            test_sents: A list of tokenized sentences.

        Returns: A list of tags corresponding to the input sentences.

        """
        return [self._predict_sent(sent) for sent in test_sents]

    def _predict_sent(self, sent: List[str]) -> Union[Counter, str]:
        argmax_labels = [self._vocab[word] for word in sent if
                         word in self._vocab]
        return (Counter(argmax_labels).most_common(1)[0][0]
                if argmax_labels else self._unk)

    def _baseline_train(self, path: str,
                        min_count: int) -> DefaultDict[str, Counter]:
        if not path:
            return defaultdict(Counter)
        return self._word_to_lang(path, min_count)

    @staticmethod
    def _word_to_lang(train_path: str,
                      min_count: int) -> DefaultDict[str, Counter]:
        word_to_label = defaultdict(Counter)
        with open(train_path, "r", encoding="utf-8") as f:
            for line in f:
                if not line:
                    continue
                label, sent = line.strip().split(" ", 1)
                for word in sent.split():
                    word_to_label[word][label] += 1
        # leave only argmax labels
        for word in list(word_to_label):
            label_cnt = word_to_label[word]
            if sum(label_cnt.values()) < min_count:
                del word_to_label[word]
            else:
                word_to_label[word] = label_cnt.most_common(1)[0][0]
        return word_to_label


def read_labeled_sents(path: str) -> List[Tuple[Any]]:
    """Reads lines from the file `path` and returns labels and sentences."""
    with open(path, "r", encoding="utf-8") as f:
        labels_and_sents = [(split[0], split[1].split())
                            for line in f if line.strip()
                            for split in (line.strip().split(" ", 1),)]
    return list(zip(*labels_and_sents))

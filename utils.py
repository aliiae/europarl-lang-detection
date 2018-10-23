#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import random
import re
import string
from multiprocessing import Pool
from typing import List, Set, Tuple, Optional, Generator

from mosestokenizer import MosesSentenceSplitter, MosesTokenizer


def write_sentences(corpus_folder: str, output_folder: str, max_files: int,
                    test_set: Optional[str] = None,
                    test_output: Optional[str] = None) -> List[str]:
    """Extracts sentences from `corpus_folder` and saves them in `output_folder`.

    Sentences are extracted from the files available for all the languages
    (i.e., from parallel docs). The number of files for each language
    is restricted by the global `max_files` variable and/or the number
    of available files.

    If a test set is provided, each cleaned sentence will be checked against
    its presence in the test set, to avoid overlaps.

    Args:
        corpus_folder: Path to the corpus folder with the following structure:
        |> /corpus_folder
        |--> /txt
        |----> /lang1
        |------> /text1_in_lang1
        output_folder: Path to the output folder to write extracted sentences,
        with the following structure:
        |> /output_folder
        |--> /lang1.txt
        max_files: Maximum number of files per language.
        test_set: Optional path to a test set; if provided, training sentences
        will be checked against an overlap with test sentences.
        test_output: Optional path to the output file to re-format the test set.

    Returns: A list of language ids in the corpus.

    """
    splitter, tokenizer = MosesSentenceSplitter(), MosesTokenizer()
    files_overlap = _get_files_overlap(corpus_folder)
    if len(files_overlap) > max_files:
        random.seed = 1004
        files_overlap = random.sample(files_overlap, max_files)
    walk = os.walk(corpus_folder)
    test_sents = (process_test_set(test_set, test_output, tokenizer, splitter)
                  if test_set else None)
    langs = next(walk)[1]
    print(f"{len(langs)} languages:", langs)

    pool = Pool()
    w, n_threads = list(walk), 4
    walks = [w[i:i + n_threads] for i in range(0, len(w), n_threads)]
    for walk in walks:
        for root, dirs, filenames in walk:
            pool.apply_async(process_lang, [filenames, files_overlap,
                                            output_folder, root, test_sents])
    pool.close()
    pool.join()
    return langs


def process_lang(filenames: List[str], files_overlap: Set[str],
                 output_folder: str, root: str, test_sents: Set[str]):
    """Process files in the `lang` directory (see `write_sentences`)."""
    splitter, tokenizer = MosesSentenceSplitter(), MosesTokenizer()
    lang = os.path.basename(root)
    with open(os.path.join(output_folder, lang) + ".txt", "w",
              encoding="utf-8") as out:
        for filename in filenames:
            if filename not in files_overlap:
                continue
            full_path = os.path.join(root, filename)
            try:
                sents, skipped = _extract_text_from_file(full_path,
                                                         tokenizer,
                                                         splitter,
                                                         test_sents)
                out.writelines(_clean_sentences(sents))
            except UnicodeDecodeError as e:
                print(e, lang, filename)


def _get_files_overlap(corpus_folder: str) -> Set[str]:
    """Gathers filenames present across all the languages."""
    walk = os.walk(corpus_folder)
    next(walk)
    lang_to_files = {os.path.basename(root): set(files) for root, _, files in
                     walk}
    return set.intersection(*lang_to_files.values())


def process_test_set(test_set_path: str, output: str, tokenizer, splitter,
                     label_prefix: str = "__label__") -> Set[str]:
    """Preprocess the test set (split sentences, tokenize, clean).

    Args:
        test_set_path: Path to the test set file.
        output: Path to the output file with test lines in the format:
        `label_prefix`lang preprocessed_test_line
        label_prefix: Prefix for fastText to recognize a label.

    Returns: A set of preprocessed sentences (=split lines).

    """
    test_set, test_sentences = [], set()
    with open(test_set_path, "r", encoding="utf-8") as f:
        orig_test_set = [line.split("\t") for line in f]
    for lang, sent in orig_test_set:
        tokenized = " ".join(tokenizer(sent)).strip()
        test_sentences.update(
            " ".join(tokenizer(s)) for s in _clean_sentences(splitter([sent])))
        test_set.append((lang, tokenized))
    _write_test_lines(output, test_set, label_prefix)
    return test_sentences


def _write_test_lines(output: str, test_set: List[Tuple[str, str]],
                      label_prefix: str):
    """Writes preprocessed test lines into `output`."""
    labels, sents = zip(*test_set)
    with open(output, "w", encoding="utf-8") as out:
        for lang, clean_sent in zip(labels, _clean_sentences(sents)):
            out.write("{}{} {}".format(label_prefix, lang, clean_sent))


def _extract_text_from_file(full_path: str, tokenizer: MosesTokenizer,
                            sent_splitter: MosesSentenceSplitter, test_sents:
        Optional[Set[str]]) -> Tuple[List[str], int]:
    """Extracts, splits, and tokenizes sentences from the given file.

    Args:
        full_path: Path to the file.
        sent_splitter: Sentence splitter object for a default language (en).
        tokenizer: Tokenizer object for a default language (en).
        test_sents: Path to a test set; if provided, training sentences will be
        checked against an overlap with test sentences.

    Returns:
        A list of tokenized sentences joined via whitespace and the number of
        skipped sentences overlapping with the test set (if applicable).

    """
    result = []
    skipped_count = 0
    with open(full_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("<"):  # skip tags
                continue
            for sentence in sent_splitter([line]):
                if test_sents and sentence in test_sents:
                    skipped_count += 1
                    continue
                result.append(" ".join(tokenizer(sentence)))
    return result, skipped_count


def _clean_sentences(sentences: List[str]) -> Generator[str, None, None]:
    """Yields a lowercase sentence with digits and punctuation removed."""
    for sentence in sentences:
        tmp = "".join(
            re.findall("[^\n\t\d–{}]+".format(string.punctuation),
                       sentence.lower())).strip()
        tmp = re.sub("\s+", " ", tmp)
        if len(tmp) > 1:
            yield tmp + "\n"

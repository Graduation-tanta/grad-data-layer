#!/usr/bin/env python3
# Copyright 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Various retriever utilities."""

import regex
import unicodedata
import numpy as np
import scipy.sparse as sp
from sklearn.utils import murmurhash3_32

# Sparse matrix(Compressed Sparse Row matrix csr) saving/loading helpers.


def save_sparse_csr(filename, matrix, metadata=None):
    
    
    """
    indptr points to row starts (tells where each row begins).
    data is an array which contains all non-zero entries in the row-major order.
    indices is array of column indices (tells us which cells have non-zero values)
    """
    # https://rushter.com/blog/scipy-sparse-matrices/
    data = {
        'data': matrix.data,
        'indices': matrix.indices,
        'indptr': matrix.indptr,
        'shape': matrix.shape,
        'metadata': metadata,
    }
    
    """Save several arrays into a single file in uncompressed .npz format"""
    # https://docs.scipy.org/doc/numpy-1.10.0/reference/generated/numpy.savez.html
    
    np.savez(filename, **data)


def load_sparse_csr(filename):
    loader = np.load(filename)
    matrix = sp.csr_matrix((loader['data'], loader['indices'],
                            loader['indptr']), shape=loader['shape'])
    return matrix, loader['metadata'].item(0) if 'metadata' in loader else None


# Token hashing.
"""Unsigned 32 bit murmurhash for feature hashing."""


def hash(token, num_buckets):
    
    return murmurhash3_32(token, positive=True) % num_buckets

# Text cleaning.


STOPWORDS = {
    'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', 'your',
    'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she',
    'her', 'hers', 'herself', 'it', 'its', 'itself', 'they', 'them', 'their',
    'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that',
    'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
    'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an',
    'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of',
    'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through',
    'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down',
    'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then',
    'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any',
    'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor',
    'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 'can',
    'will', 'just', 'don', 'should', 'now', 'd', 'll', 'm', 'o', 're', 've',
    'y', 'ain', 'aren', 'couldn', 'didn', 'doesn', 'hadn', 'hasn', 'haven',
    'isn', 'ma', 'mightn', 'mustn', 'needn', 'shan', 'shouldn', 'wasn', 'weren',
    'won', 'wouldn', "'ll", "'re", "'ve", "n't", "'s", "'d", "'m", "''", "``"
}

"""Resolve different type of unicode encodings."""
# http://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.normalize.html


def normalize(text):
    # https://docs.python.org/3/library/unicodedata.html
    return unicodedata.normalize('NFD', text)


"""Take out english stopwords, punctuation, and compound endings."""


def filter_word(text):

    text = normalize(text)
    """
    \p{Punctuation}: any kind of punctuation character.
    https://www.regular-expressions.info/unicode.html#prop
    """

    if regex.match(r'^\p{P}+$', text):
        return True
    """The method lower() returns a copy of the string 
    in which all case-based characters have been lower cased
    """
    # https://www.tutorialspoint.com/python/string_lower.htm
    
    if text.lower() in STOPWORDS:
        return True
    return False


"""Decide whether to keep or discard an n-gram.

    Args:
        gram: list of tokens (length N)
        mode: Option to throw out ngram if
          'any': any single token passes filter_word
          'all': all tokens pass filter_word
          'ends': book-ended by filterable tokens
"""


def filter_ngram(gram, mode='any'):
   
    filtered = [filter_word(w) for w in gram]
    if mode == 'any':
        return any(filtered)
    elif mode == 'all':
        return all(filtered)
    elif mode == 'ends':
        return filtered[0] or filtered[-1]
    else:
        raise ValueError('Invalid mode: %s' % mode)

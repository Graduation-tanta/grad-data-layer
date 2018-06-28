#!/usr/bin/env python3
# Copyright 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
"""Rank documents with TF-IDF scores"""
import os
import logging
import numpy as np
import scipy.sparse as sp

from multiprocessing.pool import ThreadPool
from functools import partial

from . import utils
from .. import tokenizers

DEFAULTS = {
    'corenlp_classpath': os.getenv('CLASSPATH')
}


def get_corenlp(name):
    if name == 'corenlp':
        return tokenizers.CoreNLPTokenizer
    raise RuntimeError('Invalid tokenizer: %s' % name)

# logging.getLogger(): implement a flexible event logging system for applications and libraries.


logger = logging.getLogger(__name__)


class TfidfDocRanker(object):
    """Loads a pre-weighted inverted index of token/document terms.
    Scores new queries by taking sparse dot products.
    """

    def __init__(self, tfidf_path=None, strict=True):
        """
        Args:
            tfidf_path: path to saved model file
            strict: fail on empty queries or continue (and return empty result)
        """
        # Load from disk
        tfidf_path = tfidf_path or DEFAULTS['tfidf_path']
        # Logs a message with level INFO on this logger.
        # Here, we log the path to saved model file
        logger.info('Loading %s' % tfidf_path)
        """return matrix with metadata by using load_sparse 
        matrix stored in doc_mat variable, and this is data
        metadata contains many columns
        ngrams, hash_size, tokenizers, doc_frequencies, doc_dict
        
        """
        matrix, metadata = utils.load_sparse_csr(tfidf_path)
        self.doc_mat = matrix
        """gram: list of tokens (length N)
        ngram: Use up to N-size n-grams (e.g. 2 = unigrams + bigrams). 
        By default only ngrams without stopwords or punctuation are kept.
        """
        self.ngrams = metadata['ngram']
        # hash-size: Number of buckets to use for hashing ngrams.
        self.hash_size = metadata['hash_size']
        # tokenizer: String option specifying tokenizer type to use (e.g. 'corenlp').
        self.tokenizer = get_corenlp(metadata['tokenizer'])()
        # squeeze(): Remove single-dimensional entries from the shape of an array
        """
        we will remove single dimensional entries from the doc_freq 
        doc_freqs : word --> # of docs it appears in. 
        """
        self.doc_freqs = metadata['doc_freqs'].squeeze()
        self.doc_dict = metadata['doc_dict']
        # Get the length of  doc_dict[0] in which is the number of documents
        self.num_docs = len(self.doc_dict[0])
        self.strict = strict

    def get_doc_index(self, doc_id):
        """Convert doc_id --> doc_index"""
        """from metadata of data, get doc_index 
        from doc_id 
        """
        return self.doc_dict[0][doc_id]
    """convert the tex(query) to sparse vector
     https://stackoverflow.com/questions/31732632/sparse-vector-in-python
    """

    def get_doc_id(self, doc_index):
        """Convert doc_index --> doc_id"""
        """from metadata of data, get doc_id
        from doc_index
        """
        return self.doc_dict[1][doc_index]

    def closest_docs(self, query, k=1):
        """Closest docs by dot product between query and documents
        in tfidf weighted word vector space.
        """
        spvec = self.text2spvec(query)
        res = spvec * self.doc_mat

        if len(res.data) <= k:
            # https://stackoverflow.com/questions/17901218/numpy-argsort-what-is-it-doing
            """argsort(): Returns the indices that would sort an array.
            """
            o_sort = np.argsort(-res.data)
        else:
            """argpartition split the operand into the bottom k elements and the rest.
            using argsort over argpartiton to do the same task would only be slower,
            but that the order would be guaranteed
            https://stackoverflow.com/questions/42184499/cannot understand-numpy-argpartition-output
            """
            # we will split the data into k and sort it.
            o = np.argpartition(-res.data, k)[0:k]
            o_sort = o[np.argsort(-res.data[o])]

        doc_scores = res.data[o_sort]
        doc_ids = [self.get_doc_id(i) for i in res.indices[o_sort]]
        return doc_ids, doc_scores

    def batch_closest_docs(self, queries, k=1, num_workers=None):
        """Process a batch of closest_docs requests multithreaded.
        Note: we can use plain threads here as scipy is outside of the GIL.
        """
        """Partial(): allow us to fix a certain number of arguments of a function and generate a new function.
        The Pool class is used to represent a pool of worker processes. 
        It has methods which can allow you to offload tasks to the worker processes.
        map(): it supports only one iterable argument though. It blocks until the result is ready.
        """
        # it chops closest_docs into chunk and submits every chunk with queries as a task.
        with ThreadPool(num_workers) as threads:
            closest_docs = partial(self.closest_docs, k=k)
            results = threads.map(closest_docs, queries)
        return results

    def parse(self, query):
        """Parse the query into tokens (either ngrams or tokens)."""
        tokens = self.tokenizer.tokenize(query)
        return tokens.ngrams(n=self.ngrams, uncased=True,
                             filter_fn=utils.filter_ngram)

    def text2spvec(self, query):
        """Create a sparse tfidf-weighted word vector from query.
        tfidf = log(tf + 1) * log((N - Nt + 0.5) / (Nt + 0.5))
        """
        # Get hashed ngrams
        # normalize(query): get unicode in NFD of query
        words = self.parse(utils.normalize(query))
        # get id of each word after hashing and stored it in wids.
        wids = [utils.hash(w, self.hash_size) for w in words]

        if len(wids) == 0:
            if self.strict:
                """if we have encoding error,
                the exception will be raised"""
                """The statements used to deal with exceptions are raise and
                except. Both are language keywords. The most common form of
                throwing an exception with raise uses an instance of an exception
                class.
                """

                raise RuntimeError('No valid word in: %s' % query)
            else:
                # logging.warning() if there is nothing the client application can do about the situation,
                #  but the event should still be noted
                # https://stackoverflow.com/questions/9595009/python-warnings-warn-vs-logging-warning
                logger.warning('No valid word in: %s' % query)
                return sp.csr_matrix((1, self.hash_size))

        # Count TF
        # I need to find unique rows in a numpy.array.
        # Ensure that all word ids are not repeated.
        # https://stackoverflow.com/questions/16970982/find-unique-rows-in-numpy-array

        wids_unique, wids_counts = np.unique(wids, return_counts=True)
        tfs = np.log1p(wids_counts)

        # Count IDF
        # An array with natural logarithmic value of x + 1;
        # where x belongs to all elements of input array.
        # https://stackoverflow.com/questions/49538185/what-is-the-purpose-of-numpy-log1p
        """
         tfidf = log(tf + 1) * log((N - Nt + 0.5) / (Nt + 0.5))
         Ns => Nt, num_docs => N
         Ns: word --> # of docs it appears in.
         tfs= log(tf + 1)
         idfs= log((N - Nt + 0.5) / (Nt + 0.5))
         data= tfidf
        """

        Ns = self.doc_freqs[wids_unique]
        idfs = np.log((self.num_docs - Ns + 0.5) / (Ns + 0.5))
        idfs[idfs < 0] = 0

        # TF-IDF
        data = np.multiply(tfs, idfs)

        # One row, sparse csr matrix
        indptr = np.array([0, len(wids_unique)])

        """scipy.sparse.csr_matrix is a Compressed Sparse Row matrix
        https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.sparse.csr_matrix.html
        """
        spvec = sp.csr_matrix(
            (data, wids_unique, indptr), shape=(1, self.hash_size)
        )

        return spvec

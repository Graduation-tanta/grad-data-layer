"""Rank documents with TF-IDF scores"""

import logging
import numpy as np
import scipy.sparse as sp

from multiprocessing.pool import ThreadPool
from functools import partial

from . import utils
from . import DEFAULTS
from .. import tokenizers

logger = logging.getLogger(__name__)

<<<<<<< HEAD
"""Loads a pre-weighted inverted index of token/document terms.
    Scores new queries by taking sparse dot products.
"""
class TfidfDocRanker(object):
   
    """
        Args:
            tfidf_path: path to saved model file
            strict: fail on empty queries or continue (and return empty result)
    """
    def __init__(self, tfidf_path=None, strict=True):
      
=======

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
>>>>>>> 1794588285e16f3b3a17469dc1b4784543dcd824
        # Load from disk
        tfidf_path = tfidf_path or DEFAULTS['tfidf_path']
        logger.info('Loading %s' % tfidf_path)
        matrix, metadata = utils.load_sparse_csr(tfidf_path)
        self.doc_mat = matrix
        self.ngrams = metadata['ngram']
        self.hash_size = metadata['hash_size']
        self.tokenizer = tokenizers.get_class(metadata['tokenizer'])()
        self.doc_freqs = metadata['doc_freqs'].squeeze()
        self.doc_dict = metadata['doc_dict']
        self.num_docs = len(self.doc_dict[0])
        self.strict = strict

    def get_doc_index(self, doc_id):
<<<<<<< HEAD
         return self.doc_dict[0][doc_id]
=======
        """Convert doc_id --> doc_index"""
        return self.doc_dict[0][doc_id]
>>>>>>> 1794588285e16f3b3a17469dc1b4784543dcd824

    def get_doc_id(self, doc_index):
<<<<<<< HEAD
         return self.doc_dict[1][doc_index]
     
        
    """Closest docs by dot product between query and documents
            in tfidf weighted word vector space.
    """
    def closest_docs(self, query, k=1):

        #convert the tex(query) to sparse vector
        #https://stackoverflow.com/questions/31732632/sparse-vector-in-    python
=======
        """Convert doc_index --> doc_id"""
        return self.doc_dict[1][doc_index]

    def closest_docs(self, query, k=1):
        """Closest docs by dot product between query and documents
        in tfidf weighted word vector space.
        """
>>>>>>> 1794588285e16f3b3a17469dc1b4784543dcd824
        spvec = self.text2spvec(query)
        res = spvec * self.doc_mat

        if len(res.data) <= k:
<<<<<<< HEAD

           #https://stackoverflow.com/questions/17901218/numpy-argsort-what-is-it-doing
            o_sort = np.argsort(-res.data)
        else:
            #argpartition split the operand into the bottom k elements and the rest.
            #using argsort over argpartiton to do the same task would only be slower, but that the order would be guaranteed
 
            #https://stackoverflow.com/questions/42184499/cannot understand-numpy-argpartition-output
=======
            o_sort = np.argsort(-res.data)
        else:
>>>>>>> 1794588285e16f3b3a17469dc1b4784543dcd824
            o = np.argpartition(-res.data, k)[0:k]
            o_sort = o[np.argsort(-res.data[o])]

        doc_scores = res.data[o_sort]
        doc_ids = [self.get_doc_id(i) for i in res.indices[o_sort]]
        return doc_ids, doc_scores
<<<<<<< HEAD
 
    """Process a batch of closest_docs requests multithreaded.
           Note: we can use plain threads here as scipy is outside of the GIL.
    """
    def batch_closest_docs(self, queries, k=1, num_workers=None):
       
        with ThreadPool(num_workers) as threads:

            #The partial() is used for partial function application which “freezes” some portion of a function’s arguments and/or keywords resulting in a new object with a simplified signature
            #https://docs.python.org/2/library/functools.html
=======

    def batch_closest_docs(self, queries, k=1, num_workers=None):
        """Process a batch of closest_docs requests multithreaded.
        Note: we can use plain threads here as scipy is outside of the GIL.
        """
        with ThreadPool(num_workers) as threads:
>>>>>>> 1794588285e16f3b3a17469dc1b4784543dcd824
            closest_docs = partial(self.closest_docs, k=k)
            results = threads.map(closest_docs, queries)
        return results

<<<<<<< HEAD
    """Parse the query into tokens (either ngrams or tokens)."""
    def parse(self, query):
         tokens = self.tokenizer.tokenize(query)
         return tokens.ngrams(n=self.ngrams, uncased=True,
                             filter_fn=utils.filter_ngram)

    """Create a sparse tfidf-weighted word vector from query. """
    def text2spvec(self, query):
        
        tfidf = log(tf + 1) * log((N - Nt + 0.5) / (Nt + 0.5))
=======
    def parse(self, query):
        """Parse the query into tokens (either ngrams or tokens)."""
        tokens = self.tokenizer.tokenize(query)
        return tokens.ngrams(n=self.ngrams, uncased=True,
                             filter_fn=utils.filter_ngram)

    def text2spvec(self, query):
        """Create a sparse tfidf-weighted word vector from query.

        tfidf = log(tf + 1) * log((N - Nt + 0.5) / (Nt + 0.5))
        """
>>>>>>> 1794588285e16f3b3a17469dc1b4784543dcd824
        # Get hashed ngrams
        words = self.parse(utils.normalize(query))
        wids = [utils.hash(w, self.hash_size) for w in words]

        if len(wids) == 0:
            if self.strict:
<<<<<<< HEAD
                
                """The statements used to deal with exceptions are raise and
                   except. Both are language keywords. The most common form of
                   throwing an exception with raise uses an instance of an exception
                  class.
                """
                #https://doughellmann.com/blog/2009/06/19/python-exception-handling-techniques/
               
=======
>>>>>>> 1794588285e16f3b3a17469dc1b4784543dcd824
                raise RuntimeError('No valid word in: %s' % query)
            
            else:
<<<<<<< HEAD
            
                #logging.warning() if there is nothing the client application can do about the situation, but the event should still be noted
                #https://stackoverflow.com/questions/9595009/python-warnings-warn-vs-logging-warning
=======
>>>>>>> 1794588285e16f3b3a17469dc1b4784543dcd824
                logger.warning('No valid word in: %s' % query)
                return sp.csr_matrix((1, self.hash_size))

        # Count TF
<<<<<<< HEAD
        #I need to find unique rows in a numpy.array.
        #https://stackoverflow.com/questions/16970982/find-unique-rows-in-numpy-array
        wids_unique, wids_counts = np.unique(wids, return_counts=True)

        #An array with natural logarithmic value of x + 1;where x belongs to all elements of input array.
        #https://stackoverflow.com/questions/49538185/what-is-the-purpose-of-numpy-log1p
=======
        wids_unique, wids_counts = np.unique(wids, return_counts=True)
>>>>>>> 1794588285e16f3b3a17469dc1b4784543dcd824
        tfs = np.log1p(wids_counts)

        # Count IDF
        Ns = self.doc_freqs[wids_unique]
        idfs = np.log((self.num_docs - Ns + 0.5) / (Ns + 0.5))
        idfs[idfs < 0] = 0

        # TF-IDF
        data = np.multiply(tfs, idfs)

        # One row, sparse csr matrix
        indptr = np.array([0, len(wids_unique)])
<<<<<<< HEAD

        #scipy.sparse.csr_matrix is a Compressed Sparse Row matrix
        #https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.sparse.csr_matrix.html
        spvec = sp.csr_matrix((data, wids_unique, indptr), shape=(1, self.hash_size))
        
=======
        spvec = sp.csr_matrix(
            (data, wids_unique, indptr), shape=(1, self.hash_size)
        )

>>>>>>> 1794588285e16f3b3a17469dc1b4784543dcd824
        return spvec

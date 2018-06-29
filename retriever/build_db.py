#!/usr/bin/env python3
# Copyright 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
"""A script to read in and store documents in a sqlite database."""


import argparse
import sqlite3
import json
import os
import logging
import importlib.util

from multiprocessing import Pool as ProcessPool
from tqdm import tqdm
from . import utils

"""To efficiently store and access our documents, we store them in a sqlite database. 
The key is the doc_id and the value is the text.
"""

# logging.getLogger(): implement a flexible event logging system for applications and libraries.
logger = logging.getLogger()

"""logger.setLevel: Sets the threshold for this logger to level.
    Logging messages which are less severe than level will be ignored
    The level parameter now accepts a string representation of the level
    such as ‘INFO’ as an alternative to the integer constants such as INFO.
"""
logger.setLevel(logging.INFO)

"""asctime() converts a tuple or struct_time representing a time as returned by gmtime() 
or localtime() to a 24-character string of the following form: 'Tue Feb 17 23:21:05 2009'.
"""
fmt = logging.Formatter('%(asctime)s: [ %(message)s ]', '%m/%d/%Y %I:%M:%S %p')

"""logging.StreamHandler(): Returns a new instance of the StreamHandler class. If stream is specified,
the instance will use it for logging output; otherwise, sys.stderr will be used.    
"""
console = logging.StreamHandler()
console.setFormatter(fmt)
logger.addHandler(console)


# ------------------------------------------------------------------------------
# Import helper
# ------------------------------------------------------------------------------


PREPROCESS_FN = None


def init(filename):
    global PREPROCESS_FN
    if filename:
        # if we have filename, we will executes module in filename.
        PREPROCESS_FN = import_module(filename).preprocess


def import_module(filename):
    # Import a module given a full path to the file.

    """importlib.util.spec_from_file_location()
    A factory function for creating a ModuleSpec instance based on the path to a file.
    Missing information will be filled in on the spec by making use of loader APIs 
    and by the implication that the module will be file-based.
    https://docs.python.org/3/library/importlib.html#importlib.util.spec_from_file_location
    """
    # doc_filter: file_name, filename: is the path to doc_filter
    spec = importlib.util.spec_from_file_location('doc_filter', filename)

    """importlib.util.module_from_spec():
    Create a new module based on spec and spec.loader.create_module.
    https://docs.python.org/3/library/importlib.html#importlib.util.module_from_spec
    """
    # we get module from the file doc_filter
    module = importlib.util.module_from_spec(spec)

    """An abstract method that executes the module in its own namespace when a module is imported or reloaded.
     The module should already be initialized when exec_module() is called.
    https://docs.python.org/3.4/library/importlib.html#importlib.abc.Loader.exec_module
    """
    # module executes
    spec.loader.exec_module(module)
    return module


# ------------------------------------------------------------------------------
# Store corpus.
# ------------------------------------------------------------------------------


def iter_files(path):
    """Walk through all files located under a root path."""
    if os.path.isfile(path):
        yield path
    elif os.path.isdir(path):
        for dirpath, _, filenames in os.walk(path):
            for f in filenames:
                yield os.path.join(dirpath, f)
    else:
        raise RuntimeError('Path %s is invalid' % path)


def get_contents(filename):
    """Parse the contents of a file. Each line is a JSON encoded document."""
    global PREPROCESS_FN
    documents = []
    with open(filename) as f:
        for line in f:
            # Parse document
            # json.loads: load string by string
            # so, here we load line by line until reach \n
            doc = json.loads(line)

            # Maybe preprocess the document with custom function
            # if we have module to preprocess document we will do this.
            if PREPROCESS_FN:
                doc = PREPROCESS_FN(doc)
            # Skip if it is empty or None, we have module but doc is empty
            if not doc:
                continue
            # Add the document
            # we will get normal form of doc['id'] and doc['text']
            # and add it in JSON encoded document
            documents.append((utils.normalize(doc['id']), doc['text']))
    return documents


def store_contents(data_path, save_path, preprocess, num_workers=None):
    """Preprocess and store a corpus of documents in sqlite.

    Args:
        data_path: Root path to directory (or directory of directories) of files
          containing json encoded documents (must have `id` and `text` fields).
        save_path: Path to output sqlite db.
        preprocess: Path to file defining a custom `preprocess` function. Takes
          in and outputs a structured doc.
        num_workers: Number of parallel processes to use when reading docs.
    """
    if os.path.isfile(save_path):
        raise RuntimeError('%s already exists! Not overwriting.' % save_path)

    logger.info('Reading into database...')
    conn = sqlite3.connect(save_path)
    c = conn.cursor()
    c.execute("CREATE TABLE documents (id PRIMARY KEY, text);")

    workers = ProcessPool(num_workers, initializer=init, initargs=(preprocess,))
    files = [f for f in iter_files(data_path)]
    count = 0
    with tqdm(total=len(files)) as pbar:
        for pairs in tqdm(workers.imap_unordered(get_contents, files)):
            count += len(pairs)
            c.executemany("INSERT INTO documents VALUES (?,?)", pairs)
            pbar.update()
    logger.info('Read %d docs.' % count)
    logger.info('Committing...')
    conn.commit()
    conn.close()


# ------------------------------------------------------------------------------
# Main.
# ------------------------------------------------------------------------------


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('data_path', type=str, help='/path/to/data')
    parser.add_argument('save_path', type=str, help='/path/to/saved/db.db')
    parser.add_argument('--preprocess', type=str, default=None,
                        help=('File path to a python module that defines '
                              'a `preprocess` function'))
    parser.add_argument('--num-workers', type=int, default=None,
                        help='Number of CPU processes (for tokenizing, etc)')
    args = parser.parse_args()

    store_contents(
        args.data_path, args.save_path, args.preprocess, args.num_workers
    )

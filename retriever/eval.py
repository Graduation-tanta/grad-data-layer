
"""Evaluate the accuracy of our retriever module."""

import regex as re
import logging
import argparse
import json
import time
import os

from multiprocessing import Pool as ProcessPool
from multiprocessing.util import Finalize
from functools import partial
from .. import retriever, tokenizers
from . import utils


def get_class_re(name):
    if name == 'tfidf':
        return retriever.TfidfDocRanker
    if name == 'sqlite':
        return retriever.DocDB
    raise RuntimeError('Invalid retriever class: %s' % name)

# ------------------------------------------------------------------------------
# Multiprocessing target functions.
# ------------------------------------------------------------------------------

PROCESS_TOK = None
PROCESS_DB = None

#init functions to initiate our objects
""" 
in tokenizer, we have only one class which is corenlp.
and we can determine operation depends on this class.
in database in retriever, the  determination of classes is 
dependant on requirements either tfidf or sqlite.
So we have 2 process tok and db
"""

def init(tokenizer_class, tokenizer_opts, db_class, db_opts):
    global PROCESS_TOK, PROCESS_DB
    PROCESS_TOK = tokenizer_class(**tokenizer_opts)
    #Finalize: is responsible for adding a callback to the registry.
    Finalize(PROCESS_TOK, PROCESS_TOK.shutdown, exitpriority=100)
    PROCESS_DB = db_class(**db_opts)
    Finalize(PROCESS_DB, PROCESS_DB.close, exitpriority=100)


def regex_match(text, pattern):
    """Test if a regex pattern is contained within a text."""
    #https://docs.python.org/3/howto/regex.html
    """
       re.compile():  accepts an optional flags argument, used to enable various special features and syntax variations.
       re.IGNORECASE: Perform case-insensitive matching; character class and literal strings will match letters by ignoring case.
       re.MULTILINE: Usually (^) matches only at the beginning of the string,
       and $ matches only at the end of the string and immediately before the newline (if any) at the end of the string.
       When this flag is specified, (^) matches at the beginning of the string and at the beginning of each line within the string,
       immediately following each newline. Similarly, the $ metacharacter matches either at the end of the string and at the end of each line 
       (immediately preceding each newline).
       re.UNICODE: match all the unicode variants(non english letters).
    """
    try:
        pattern = re.compile(
            pattern,
            flags=re.IGNORECASE + re.UNICODE + re.MULTILINE,
        )
    except BaseException:
        return False
    return pattern.search(text) is not None


def has_answer(answer, doc_id, match):
    #Check if a document contains an answer string.

    global PROCESS_DB, PROCESS_TOK
    #by using get_text function in doc_db we will get text of document by its id
    text = PROCESS_DB.get_text(doc_id)
    #get unicode of text and store it in text variable again
    text = utils.normalize(text)
    #If `match` is string, token matching is done between the text and answer.
    if match == 'string':
        """ tokenize: is a function in a base class in tokenizer which raise NotImplementedError.
        https://docs.python.org/3/library/exceptions.html#NotImplementedError 
        NotImplementedError: This exception is derived from RuntimeError.
        In user defined base classes, abstract methods should raise this exception
        when they require derived classes to override the method, 
        or while the class is being developed to indicate that the real implementation still needs to be added.
        words(): Returns a list of the text of each token
        Args:uncased: lower cases text
        """
        text = PROCESS_TOK.tokenize(text).words(uncased=True)
        # Answer is a list of possible strings
        for single_answer in answer:
            single_answer = utils.normalize(single_answer)
            single_answer = PROCESS_TOK.tokenize(single_answer)
            single_answer = single_answer.words(uncased=True)
            for i in range(0, len(text) - len(single_answer) + 1):
                if single_answer == text[i: i + len(single_answer)]:
                    return True

    # If `match` is regex, we search the whole text with the regex.
    elif match == 'regex':
        # Answer is a regex
        single_answer = utils.normalize(answer[0])
        if regex_match(text, single_answer):
            return True
    #or return False if answer in this document
    return False


def get_score(answer_doc, match):
    """Search through all the top docs to see if they have the answer."""
    answer, (doc_ids, doc_scores) = answer_doc
    for doc_id in doc_ids:
        if has_answer(answer, doc_id, match):
            return 1
    return 0


# ------------------------------------------------------------------------------
# Main
# ------------------------------------------------------------------------------


if __name__ == '__main__':

    #logging.getLogger(): implement a flexible event logging system for applications and libraries.
    logger = logging.getLogger()
    """logger.setLevel: Sets the threshold for this logger to level.
    Logging messages which are less severe than level will be ignored
    The level parameter now accepts a string representation of the level
    such as ‘INFO’ as an alternative to the integer constants such as INFO.
    """
    logger.setLevel(logging.INFO)
    """
    logging.Formatter: Sets the Formatter for this handler to fmt.
    Returns a new instance of the Formatter class.
    The instance is initialized with a format string for the message as a whole
    https://docs.python.org/3/library/logging.html#logging.Formatter
    """
    fmt = logging.Formatter('%(asctime)s: [ %(message)s ]',
                            '%m/%d/%Y %I:%M:%S %p')
    """
   logging.StreamHandler(): Returns a new instance of the StreamHandler class. If stream is specified,
    the instance will use it for logging output; otherwise, sys.stderr will be used.
    """
    console = logging.StreamHandler()
    console.setFormatter(fmt)
    logger.addHandler(console)

    parser = argparse.ArgumentParser()
    parser.add_argument('dataset', type=str, default=None)
    parser.add_argument('--model', type=str, default=None)
    parser.add_argument('--doc-db', type=str, default=None,
                        help='Path to Document DB')
    parser.add_argument('--tokenizer', type=str, default='regexp')
    parser.add_argument('--n-docs', type=int, default=5)
    parser.add_argument('--num-workers', type=int, default=None)
    parser.add_argument('--match', type=str, default='string',
                        choices=['regex', 'string'])
    args = parser.parse_args()

    # start time
    start = time.time()

    # read all the data and store it
    logger.info('Reading data ...')
    questions = []
    answers = []
    for line in open(args.dataset):
        data = json.loads(line)
        question = data['question']
        answer = data['answer']
        questions.append(question)
        answers.append(answer)

    # get the closest docs for each question.
    logger.info('Initializing ranker...')
    ranker = get_class_re('tfidf')(tfidf_path=args.model)

    logger.info('Ranking...')
    closest_docs = ranker.batch_closest_docs(
        questions, k=args.n_docs, num_workers=args.num_workers
    )
    answers_docs = zip(answers, closest_docs)

    # define processes
    tok_class = tokenizers.get_class(args.tokenizer)
    tok_opts = {}
    #again
    db_class = retriever.DocDB
    db_opts = {'db_path': args.doc_db}
    processes = ProcessPool(
        processes=args.num_workers,
        initializer=init,
        initargs=(tok_class, tok_opts, db_class, db_opts)
    )

    # compute the scores for each pair, and print the statistics
    logger.info('Retrieving and computing scores...')
    get_score_partial = partial(get_score, match=args.match)
    scores = processes.map(get_score_partial, answers_docs)

    filename = os.path.basename(args.dataset)
    stats = (
        "\n" + "-" * 50 + "\n" +
        "{filename}\n" +
        "Examples:\t\t\t{total}\n" +
        "Matches in top {k}:\t\t{m}\n" +
        "Match % in top {k}:\t\t{p:2.2f}\n" +
        "Total time:\t\t\t{t:2.4f} (s)\n"
    ).format(
        filename=filename,
        total=len(scores),
        k=args.n_docs,
        m=sum(scores),
        p=(sum(scores) / len(scores) * 100),
        t=time.time() - start,
    )

    print(stats)

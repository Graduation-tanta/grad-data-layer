#!/usr/bin/env python3
# Copyright 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""A script to generate distantly supervised training data.

Using Wikipedia and available QA datasets(squad, webquestion,..etc), we search for a paragraph
that can be used as a supporting context.
"""

import argparse
import uuid
import heapq
import logging
import regex as re
import os
import json
import random
from functools import partial
from collections import Counter
from multiprocessing import Pool, cpu_count
from multiprocessing.util import Finalize
from nltk.tokenize import word_tokenize
from nltk.chunk import ne_chunk
from nltk.tag import pos_tag
from .. import retriever, tokenizers


def get_class_re(name):
    if name == 'tfidf':
        return retriever.TfidfDocRanker
    if name == 'sqlite':
        return retriever.DocDB
    raise RuntimeError('Invalid retriever class: %s' % name)


def get_corenlp(name):
    if name == 'corenlp':
        return tokenizers.CoreNLPTokenizer
    raise RuntimeError('Invalid tokenizer: %s' % name)


logger = logging.getLogger()


# ------------------------------------------------------------------------------
# Fetch text, tokenize + annotate
# ------------------------------------------------------------------------------

PROCESS_TOK = None
PROCESS_DB = None
# init functions to initiate our objects
""" 
in tokenizer, we have only one class which is corenlp.
and we can determine operation depends on this class.
in database in retriever, the  determination of classes is 
dependant on requirements either tfidf or sqlite.
So we have 2 process tok and db
"""


def init(tokenizer_class, tokenizer_opts, db_class=None, db_opts=None):
    global PROCESS_TOK, PROCESS_DB
    PROCESS_TOK = tokenizer_class(**tokenizer_opts)
    # Finalize: is responsible for adding a callback to the registry.
    Finalize(PROCESS_TOK, PROCESS_TOK.shutdown, exitpriority=100)

    # optionally open a db connection
    if db_class:
        PROCESS_DB = db_class(**db_opts)
        Finalize(PROCESS_DB, PROCESS_DB.close, exitpriority=100)


# fetch text by using doc_id in get_text function in DOCDB class
def fetch_text(doc_id):
    global PROCESS_DB
    return PROCESS_DB.get_text(doc_id)

# which should return a Tokens class with its function


def tokenize_text(text):
    global PROCESS_TOK
    # return a Tokens class and access its functions.
    return PROCESS_TOK.tokenize(text)


def nltk_entity_groups(text):
    # Return all contiguous NER tagged chunks by NLTK.
    # https://www.nltk.org/book/ch07.html
    """nltk.ne_chunk: returns a nested nltk.tree.Tree object 
    so you would have to traverse the Tree object to get to the NEs.
    POS(part-of speech)-tagger: processes a sequence of words,
    and attaches a part of speech tag to each word
    """
    parse_tree = ne_chunk(pos_tag(word_tokenize(text)))
    ner_chunks = [' '.join([l[0] for l in t.leaves()])
                  for t in parse_tree.subtrees() if t.label() != 'S']
    return ner_chunks


# ------------------------------------------------------------------------------
# Find answer candidates.
# ------------------------------------------------------------------------------


def find_answer(paragraph, q_tokens, answer, opts):
    """Return the best matching answer offsets from a paragraph.

    The paragraph is skipped if:
    * It is too long or short.
    * It doesn't contain the answer at all.
    * It doesn't contain named entities found in the question.
    * The answer context match score is too low.
      - This is the unigram + bigram overlap within +/- window_sz.
    """
    # Length check
    """char-max            Maximum allowed context length
       char-min            Minimum allowed context length
       this arguments determined by user
    """
    if len(paragraph) > opts['char_max'] or len(paragraph) < opts['char_min']:
        return

    # Answer check
    """
    regex: Flag if answers are expressed as regexps
    """
    if opts['regex']:
        # Add group around the whole answer
        answer = '(%s)' % answer[0]
        # determine start and end of line with unicode of non-english chars
        ans_regex = re.compile(answer, flags=re.IGNORECASE + re.UNICODE)

        # Search and find all answers matches with ans_regex
        answers = ans_regex.findall(paragraph)

        # answers a[0] if a is tuple else return list comprehension of a in answers
        """https://docs.python.org/3/library/functions.html#isinstance
        isinstance(): instance(object, classinfo) Return true if the object argument is an instance 
        of the classinfo argument, or of a (direct, indirect or virtual) subclass thereof. 
        If object is not an object of the given type, the function always returns false.
        """
        answers = {a[0] if isinstance(a, tuple) else a for a in answers}

        # answers we will get in matching regex we will return a copy from it and
        # remove whitespaces from it  if length of a (single answer) in answers
        """https://docs.python.org/3/library/stdtypes.html#str.strip
        strip(chars): Return a copy of the string with the leading and trailing characters removed.
         The chars argument is a string specifying the set of characters to be removed.
          If omitted or None, the chars argument defaults to removing whitespace.
        """
        answers = {a.strip() for a in answers if len(a.strip()) > 0}

    else:
        """if regex of answers isn't found in paragraph we will return 
        list comprehension of answers if the some these in paragraph
        """
        answers = {a for a in answer if a in paragraph}

    if len(answers) == 0:
        return
    """
         if there is no answer we return null
    """
    # Entity check. Default tokenizer + NLTK to minimize falling through cracks
    q_tokens, q_nltk_ner = q_tokens
    for ne in q_tokens.entity_groups():
        # if chunk isn't in paragraph return null
        if ne[0] not in paragraph:
            return
    # if chunk isn't in q_nltk_ner return null
    for ne in q_nltk_ner:
        if ne not in paragraph:
            return

    # Search...
    # object of tokenizer's method
    p_tokens = tokenize_text(paragraph)
    """Returns a list of the text of each token
    Args:
        uncased: lower cases text
    """
    p_words = p_tokens.words(uncased=True)
    """Returns a list of all ngrams from length 1 to n.
        Args:
            n: upper limit of ngram length
            uncased: lower cases text
            filter_fn: user function that takes in an ngram list and returns
              True or False to keep or not keep the ngram
            as_string: return the ngram as a string vs list
    """
    """https://pymotw.com/2/collections/counter.html#counter
    Counter(): is a container that keeps track of how many times equivalent values are added.
    It can be used to implement the same algorithms for which
    bag or multiset data structures are commonly used in other languages."""
    # q_grams: is container to keep track how many(list of words, you can put them back together into a single string)
    q_grams = Counter(q_tokens.ngrams(
        n=2, uncased=True, filter_fn=retriever.utils.filter_ngram
    ))

    best_score = 0
    best_ex = None
    for ans in answers:
        try:
            """Returns a list of the text of each token"""
            a_words = tokenize_text(ans).words(uncased=True)
        except RuntimeError:
            # logger.warn(): Logs a message with level WARNING on this logger
            logger.warn('Failed to tokenize answer: %s' % ans)
            continue
        for idx in range(len(p_words)):
            if p_words[idx:idx + len(a_words)] == a_words:
                # Overlap check
                # window-sz: Use context on +/- window_sz for overlap measure
                w_s = max(idx - opts['window_sz'], 0)
                w_e = min(idx + opts['window_sz'] + len(a_words), len(p_words))
                """Return a view of the list of tokens from [i, j).
                i: w_s
                j:w_e
                w_tokens: is a list of data[w_s, w_e]"""
                w_tokens = p_tokens.slice(w_s, w_e)
                w_grams = Counter(w_tokens.ngrams(
                    n=2, uncased=True, filter_fn=retriever.utils.filter_ngram
                ))
                # score is sum of counter of list of words without stopwords
                score = sum((w_grams & q_grams).values())
                if score > best_score:
                    # Success! Set new score + formatted example
                    best_score = score
                    best_ex = {
                        'id': uuid.uuid4().hex,
                        'question': q_tokens.words(),
                        'document': p_tokens.words(),
                        'offsets': p_tokens.offsets(),
                        'answers': [(idx, idx + len(a_words) - 1)],
                        'qlemma': q_tokens.lemmas(),
                        'lemma': p_tokens.lemmas(),
                        'pos': p_tokens.pos(),
                        'ner': p_tokens.entities(),
                    }
    if best_score >= opts['match_threshold']:
        return best_score, best_ex


def search_docs(inputs, max_ex=5, opts=None):
    """Given a set of document ids (returned by ranking for a question), search
    for top N best matching (by heuristic) paragraphs that contain the answer.
    """
    if not opts:
        raise RuntimeError('Options dict must be supplied.')

    doc_ids, q_tokens, answer = inputs
    examples = []
    for i, doc_id in enumerate(doc_ids):
        """re.split(): Split string by the occurrences of pattern.
        https://docs.python.org/3.1/library/re.html#re.split
        """
        """we will split text fetched by doc_id by new line 
        """
        for j, paragraph in enumerate(re.split(r'\n+', fetch_text(doc_id))):
            # Return the best matching answer offsets from a paragraph.
            # and storing it in found variable
            found = find_answer(paragraph, q_tokens, answer, opts)
            # if found have value not null ,Reverse ranking, giving priority to early docs + paragraphs
            if found:
                score = (found[0], -i, -j, random.random())
                # max-ex:  Maximum matches generated per question.
                # general options
                if len(examples) < max_ex:
                    """https://docs.python.org/3/library/heapq.html#heapq.heappush
                    heapq.heappush(heap, item): Push the value item onto the heap, maintaining the heap invariant.
                    """
                    heapq.heappush(examples, (score, found[1]))
                else:
                    """heapq.heappushpop(heap, item): Push item on the heap, then pop 
                    and return the smallest item from the heap.
                    """
                    heapq.heappushpop(examples, (score, found[1]))
    return [e[1] for e in examples]


def process(questions, answers, outfile, opts):
    """Generate examples for all questions."""
    # Logs a message with level INFO on this logger.
    logger.info('Processing %d question answer pairs...' % len(questions))
    # Save two files
    logger.info('Will save to %s.dstrain and %s.dsdev' % (outfile, outfile))

    # Load ranker
    """ranker: Ranking method for retrieving documents (e.g. 'tfidf')
    with strict: 
    """
    ranker = opts['ranker_class'](strict=False)
    logger.info('Ranking documents (top %d per question)...' % opts['n_docs'])
    ranked = ranker.batch_closest_docs(questions, k=opts['n_docs'])
    ranked = [r[0] for r in ranked]

    # Start pool of tokenizers with ner enabled
    workers = Pool(opts['workers'], initializer=init,
                   initargs=(opts['tokenizer_class'], {'annotators': {'ner'}}))

    logger.info('Pre-tokenizing questions...')
    q_tokens = workers.map(tokenize_text, questions)
    q_ner = workers.map(nltk_entity_groups, questions)
    q_tokens = list(zip(q_tokens, q_ner))
    workers.close()
    workers.join()

    # Start pool of simple tokenizers + db connections
    workers = Pool(opts['workers'], initializer=init,
                   initargs=(opts['tokenizer_class'], {},
                             opts['db_class'], {}))

    logger.info('Searching documents...')
    cnt = 0
    inputs = [(ranked[i], q_tokens[i], answers[i]) for i in range(len(ranked))]
    search_fn = partial(search_docs, max_ex=opts['max_ex'], opts=opts['search'])
    with open(outfile + '.dstrain', 'w') as f_train, \
         open(outfile + '.dsdev', 'w') as f_dev:
        for res in workers.imap_unordered(search_fn, inputs):
            for ex in res:
                cnt += 1
                f = f_dev if random.random() < opts['dev_split'] else f_train
                f.write(json.dumps(ex))
                f.write('\n')
                if cnt % 1000 == 0:
                    logging.info('%d results so far...' % cnt)
    workers.close()
    workers.join()
    logging.info('Finished. Total = %d' % cnt)


# ------------------------------------------------------------------------------
# Main & commandline options
# ------------------------------------------------------------------------------


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('data_dir', type=str, help='Dataset directory')
    parser.add_argument('data_name', type=str, help='Dataset name')
    parser.add_argument('out_dir', type=str, help='Output directory')

    dataset = parser.add_argument_group('Dataset')
    dataset.add_argument('--regex', action='store_true',
                         help='Flag if answers are expressed as regexps')
    dataset.add_argument('--dev-split', type=float, default=0,
                         help='Hold out for ds dev set (0.X)')

    search = parser.add_argument_group('Search Heuristic')
    search.add_argument('--match-threshold', type=int, default=1,
                        help='Minimum context overlap with question')
    search.add_argument('--char-max', type=int, default=1500,
                        help='Maximum allowed context length')
    search.add_argument('--char-min', type=int, default=25,
                        help='Minimum allowed context length')
    search.add_argument('--window-sz', type=int, default=20,
                        help='Use context on +/- window_sz for overlap measure')

    general = parser.add_argument_group('General')
    general.add_argument('--max-ex', type=int, default=5,
                         help='Maximum matches generated per question')
    general.add_argument('--n-docs', type=int, default=5,
                         help='Number of docs retrieved per question')
    general.add_argument('--tokenizer', type=str, default='corenlp')
    general.add_argument('--ranker', type=str, default='tfidf')
    general.add_argument('--db', type=str, default='sqlite')
    general.add_argument('--workers', type=int, default=cpu_count())
    args = parser.parse_args()

    # Logging
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter('%(asctime)s: [ %(message)s ]',
                            '%m/%d/%Y %I:%M:%S %p')
    console = logging.StreamHandler()
    console.setFormatter(fmt)
    logger.addHandler(console)

    # Read dataset
    dataset = os.path.join(args.data_dir, args.data_name)
    questions = []
    answers = []
    for line in open(dataset):
        data = json.loads(line)
        question = data['question']
        answer = data['answer']

        # Make sure the regex compiles
        if args.regex:
            try:
                re.compile(answer[0])
            except BaseException:
                logger.warning('Regex failed to compile: %s' % answer)
                continue

        questions.append(question)
        answers.append(answer)

    # Get classes
    ranker_class = get_class_re(args.ranker)
    db_class = get_class_re(args.db)
    tokenizer_class = get_corenlp('corenlp')

    # Form options
    search_keys = ('regex', 'match_threshold', 'char_max',
                   'char_min', 'window_sz')
    opts = {
        'ranker_class': get_class_re(args.ranker),
        'tokenizer_class': get_corenlp('corenlp'),
        'db_class': get_class_re(args.db),
        'search': {k: vars(args)[k] for k in search_keys},
    }
    opts.update(vars(args))

    # Process!
    outname = os.path.splitext(args.data_name)[0]
    outfile = os.path.join(args.out_dir, outname)
    process(questions, answers, outfile, opts)

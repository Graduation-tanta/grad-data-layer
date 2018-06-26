"""Preprocess the SQuAD dataset for training."""
#https://rajpurkar.github.io/SQuAD-explorer/dataset/train-v1.1.json

import argparse
import os
import sys
import json
import time

from multiprocessing import Pool
from multiprocessing.util import Finalize
from functools import partial
from .. import tokenizers


def get_corenlp(name):
    if name == 'corenlp':
        return tokenizers.CoreNLPTokenizer
    raise RuntimeError('Invalid tokenizer: %s' % name)

# ------------------------------------------------------------------------------
# Tokenize + annotate.
# ------------------------------------------------------------------------------


""" 
in tokenizer, we have only one class which is corenlp.
and we can determine operation depends on this class."""

TOK = None


# init functions to initiate our variables
def init(tokenizer_class, options):
    global TOK
    # Finalize: is responsible for adding a callback to the registry.
    TOK = tokenizer_class(**options)
    Finalize(TOK, TOK.shutdown, exitpriority=100)


def tokenize(text):
    """Call the global process tokenizer on the input text."""
    global TOK
    tokens = TOK.tokenize(text)
    output = {
        # https://datascience.stackexchange.com/questions/586/what-are-the-main-types-of-nlp-annotators
        # Returns a list of the text of each token
        # Args:uncased: lower cases text
        'words': tokens.words(),

        # Returns a list of [start, end] character offsets of each token.
        'offsets': tokens.offsets(),

        # Returns a list of part-of-speech tags of each token.
        # Returns None if this annotation was not included.
        'pos': tokens.pos(),

        # Returns a list of the lemmatized text of each token.
        # Returns None if this annotation was not included.
        'lemma': tokens.lemmas(),

        # Returns a list of named-entity-recognition tags of each token.
        # Returns None if this annotation was not included.
        'ner': tokens.entities(),
    }
    return output


# ------------------------------------------------------------------------------
# Process dataset examples
# ------------------------------------------------------------------------------


def load_dataset(path):
    """Load json file and store fields separately."""
    with open(path) as f:
        # Load the file of squad data only data index.
        data = json.load(f)['data']

    """qids: question id in squad
       questions : in squad
       answers : in squad, answers contain start answer and text contains these answers.
       qid2cid: question id to context id.
    """
    output = {'qids': [], 'questions': [], 'answers': [],
              'contexts': [], 'qid2cid': []}
    """squad contains article in data,
       where article contains column which contains paragraph.
       paragraph contains context column. so, we will add context to column contexts of output and so on.
    """
    
    for article in data:
        for paragraph in article['paragraphs']:
            output['contexts'].append(paragraph['context'])
            for qa in paragraph['qas']:
                output['qids'].append(qa['id'])
                output['questions'].append(qa['question'])
                output['qid2cid'].append(len(output['contexts']) - 1)
                if 'answers' in qa:
                    output['answers'].append(qa['answers'])
    return output


def find_answer(offsets, begin_offset, end_offset):
    """Match token offsets with the char begin/end offsets of the answer."""

    """i for i : is a list comprehension syntax ([expression for loop]),
    which is a shorthand loop syntax for producing a list.
    if tok[0] == begin_offset: (start for answer),tok in enumerate(offsets).
    tok will takes the offset from offsets and becomes the start of answer.
    if tok[1] == end_offset: (end for answer), tok in enumerate(offsets).
    tok will takes the offset from offsets and becomes the end of answer.
    """
    """https://docs.python.org/3/library/functions.html#enumerate
     The enumerate(): Return an enumerate object. iterable must be a sequence, 
     an iterator, or some other object which supports iteration
    """

    start = [i for i, tok in enumerate(offsets) if tok[0] == begin_offset]
    end = [i for i, tok in enumerate(offsets) if tok[1] == end_offset]
    """assert(): is systematic way to check that the internal state of a program 
    which is as the programmer expected, with the goal of catching bugs.
    """
    assert(len(start) <= 1)
    assert(len(end) <= 1)
    if len(start) == 1 and len(end) == 1:
        # first answer
        return start[0], end[0]


def process_dataset(data, tokenizer, workers=None):

    tokenizer_class = get_corenlp(tokenizer)

    """Iterate processing (tokenize, parse, etc) dataset multithreaded."""
    """Partial(): allow us to fix a certain number of arguments of 
    a function and generate a new function.
    The Pool class is used to represent a pool of worker processes. 
    It has methods which can allow you to offload tasks to the worker processes.
    initiate process
    https://docs.python.org/2/library/multiprocessing.html#module-multiprocessing.pool
    """
    make_pool = partial(Pool, workers, initializer=init)

    """https://datascience.stackexchange.com/questions/586/what-are-the-main-types-of-nlp-annotators
    annotators: are the basic Natural Language Processing capabilities  that are usually necessary to 
    extract language units from textual data for sake of search and other applications
    lemma: to convert a given word into its canonical form
    """
    workers = make_pool(initargs=(tokenizer_class, {'annotators': {'lemma'}}))

    """map(): it supports only one iterable argument though. 
    It blocks until the result is ready.
    #it chops data['questions'] column in chunk and submits 
    every chunk with process as a task.
    """
    q_tokens = workers.map(tokenize, data['questions'])

    """close(): Indicate that no more data will be put on this queue by the current process.
    The background thread will quit once it has flushed all buffered data to the pipe.
    This is called automatically when the queue is garbage collected.
    """
    workers.close()

    """join(): Block the calling thread until the process whose 
    join() method is called terminates or until the optional timeout occurs.
    If timeout is None then there is no timeout.
    """
    workers.join()
    """Part-of-speech Tagger - to guess part of speech of each word in the context of sentence; 
    usually each word is assigned a so-called POS-tag from 
    a tagset developed in advance to serve your final task (for example, parsing)
    Named Entity Recognition - to extract so-called named entities from the text. 
    Named entities are the chunks of words from text, which refer to an entity of certain type. 
    The types may include: geographic locations (countries, cities, rivers, ...), person names, etc.
    """

    workers = make_pool(
        initargs=(tokenizer_class, {'annotators': {'lemma', 'pos', 'ner'}})
    )
    """#it chops data['contexts'] column in chunk and submits 
    every chunk with process as a task.
    """
    c_tokens = workers.map(tokenize, data['contexts'])
    workers.close()
    workers.join()

    for idx in range(len(data['qids'])):
        question = q_tokens[idx]['words']
        qlemma = q_tokens[idx]['lemma']
        document = c_tokens[data['qid2cid'][idx]]['words']
        offsets = c_tokens[data['qid2cid'][idx]]['offsets']
        lemma = c_tokens[data['qid2cid'][idx]]['lemma']
        pos = c_tokens[data['qid2cid'][idx]]['pos']
        ner = c_tokens[data['qid2cid'][idx]]['ner']
        ans_tokens = []
        if len(data['answers']) > 0:
            for ans in data['answers'][idx]:
                found = find_answer(offsets,
                                    ans['answer_start'],
                                    ans['answer_start'] + len(ans['text']))
                if found:
                    ans_tokens.append(found)
        yield {
            'id': data['qids'][idx],
            'question': question,
            'document': document,
            'offsets': offsets,
            'answers': ans_tokens,
            'qlemma': qlemma,
            'lemma': lemma,
            'pos': pos,
            'ner': ner,
        }


# -----------------------------------------------------------------------------
# Commandline options
# -----------------------------------------------------------------------------


parser = argparse.ArgumentParser()
parser.add_argument('data_dir', type=str, help='Path to SQuAD data directory')
parser.add_argument('out_dir', type=str, help='Path to output file dir')
parser.add_argument('--split', type=str, help='Filename for train/dev split',
                    default='SQuAD-v1.1-train')
parser.add_argument('--workers', type=int, default=None)
parser.add_argument('--tokenizer', type=str, default='corenlp')
args = parser.parse_args()

# https://docs.python.org/2/library/time.html#time.time
t0 = time.time()

in_file = os.path.join(args.data_dir, args.split + '.json')
print('Loading dataset %s' % in_file, file=sys.stderr)
dataset = load_dataset(in_file)

out_file = os.path.join(
    args.out_dir, '%s-processed-%s.txt' % (args.split, args.tokenizer)
)
print('Will write to file %s' % out_file, file=sys.stderr)
with open(out_file, 'w') as f:
    for ex in process_dataset(dataset, args.tokenizer, args.workers):
        f.write(json.dumps(ex) + '\n')
print('Total time: %.4f (s)' % (time.time() - t0))

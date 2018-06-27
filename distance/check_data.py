#!/usr/bin/env python3
# Copyright 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
"""A script to visually inspect generated data."""

import argparse
import json
from termcolor import colored

# Add class ArgumentParser to trigger actions
parser = argparse.ArgumentParser()

# Specify actions by using add_argument()
parser.add_argument('file', type=str)

# relevant variable
args = parser.parse_args()

# open file to input the generated data line by line
with open(args.file) as f:
    lines = f.readlines()
    for line in lines:
        # json.loads(): load string, but load(): to load file
        data = json.loads(line)
        """join() method: is a string method, 
        and takes a list of things to join with the string
        so here, we want to add single quotation to question 
        """
        question = ' '.join(data['question'])
        # there are many answers but one question. so, we take first answer.
        # we determine start and end of first answer.
        start, end = data['answers'][0]
        doc = data['document']
        # pre: to determine start of answer in document and put it in single quotation.
        pre = ' '.join(doc[:start])
        # determine the answer by color red and put it in single quotation and bold font
        ans = colored(' '.join(doc[start: end + 1]), 'red', attrs=['bold'])
        post = ' '.join(doc[end + 1:])
        print('-' * 50)
        print('Question: %s' % question)
        print('')
        print('Document: %s' % (' '.join([pre, ans, post])))
        input()

# WebQuestions' url: http://nlp.stanford.edu/static/software/sempre/release-emnlp2013/lib/data/webquestions/dataset_11/webquestions.examples.train.json.bz2
"""A script to convert the default WebQuestions dataset to the format:
WebQuestions' raw: {"url": "http://www.freebase.com/view/en/justin_bieber", "targetValue": "(list (description \"Jazmyn Bieber\") (description \"Jaxon Bieber\"))", "utterance": "what is the name of justin bieber brother?"}

 new format:
'{"question": "q1", "answer": ["a11", ..., "a1i"]}'
...
'{"question": "qN", "answer": ["aN1", ..., "aNi"]}'

"""

import argparse
import re
import json

# Add class ArgumentParser to trigger actions 
parser = argparse.ArgumentParser()

# Specify actions by using add_argument()
parser.add_argument('input', type=str)
parser.add_argument('output', type=str)

# relevant variable
args = parser.parse_args()

# Read dataset and  specifying it to input action "run time"
"""
json.load: takes a file-like object parameter.
"""
with open(args.input) as f:
    dataset = json.load(f)

# Iterate and write question-answer pairs
"""
re.findall(): module is used when you want to iterate over the lines of the file(dataset)
 it will return a list of all the matches in a single step.
$: Matches end of line.
.: Matches any single character except newline.
description|\: Matches either description or \.
?: Specifies position using a pattern. Doesn't have a range.
json.dumps: takes an object and produces a string.
"""
with open(args.output, 'w') as f:
    for ex in dataset:
        question = ex['utterance']
        answer = ex['targetValue']
"""
for our example:
question <= "utterance": "what is the name of justin bieber brother?"
answers <= "(list (description \"Jazmyn Bieber\") (description \"Jaxon Bieber\"))"
"""
        answer = re.findall(
            r'(?<=\(description )(.+?)(?=\) \(description|\)\)$)', answer
        )
        answer = [a.replace('"', '') for a in answer]
"""
"answer": ["Jazmyn Bieber", "Jaxon Bieber"].
"""
        f.write(json.dumps({'question': question, 'answer': answer}))
        f.write('\n')

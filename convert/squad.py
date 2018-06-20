#SQuAD's url: https://rajpurkar.github.io/SQuAD-explorer/dataset/train-v1.1.json
"""A script to convert the default SQuAD dataset to the format:

'{"question": "q1", "answer": ["a11", ..., "a1i"]}'
...
'{"question": "qN", "answer": ["aN1", ..., "aNi"]}'

"""

import argparse
import json

# Add class ArgumentParser to trigger actions 
parser = argparse.ArgumentParser()

# Specify actions by using add_argument()
parser.add_argument('input', type=str)
parser.add_argument('output', type=str)
args = parser.parse_args()

# Read dataset 
with open(args.input) as f:
    dataset = json.load(f)
             

# Iterate and write question-answer pairs
with open(args.output, 'w') as f:
    for article in dataset['data']:
        for paragraph in article['paragraphs']:
            for qa in paragraph['qas']:
                question = qa['question']
                answer = [a['text'] for a in qa['answers']]
                f.write(json.dumps({'question': question, 'answer': answer}))
                f.write('\n')

"""A script to convert the default SQuAD dataset to the format:

'{"question": "q1", "answer": ["a11", ..., "a1i"]}'
...
'{"question": "qN", "answer": ["aN1", ..., "aNi"]}'

"""

import argparse
import json

# Add class ArgumentParser to can trigger actions 
parser = argparse.ArgumentParser()

# Specify actions by using add_argument()
parser.add_argument('input', type=str)
parser.add_argument('output', type=str)
args = parser.parse_args()

# Read dataset 
with open(args.input) as f:
    dataset = json.load(f)
"""
squad is a json file contains |=>data=========>contains|==>paragraphs==============|=>paragraph
                              |=>version :1.1          |==>documents_id            |=>qas==========|=>answers====|=>answer_start: "N" 
					    		                                                                   |=>id         |=>text: "answer"
																								   |=>question: "paragraph question"
"""                

# Iterate and write question-answer pairs
with open(args.output, 'w') as f:
    for article in dataset['data']:
        for paragraph in article['paragraphs']:
            for qa in paragraph['qas']:
                question = qa['question']
                answer = [a['text'] for a in qa['answers']]
                f.write(json.dumps({'question': question, 'answer': answer}))
                f.write('\n')

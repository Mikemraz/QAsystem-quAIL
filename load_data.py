import json
from util import preprocess

with open('../data/quAIL/train_questions.json', 'rb') as f:
  data = json.load(f)

artical_list =  list(data['data'].keys())

print(data['data']['u001']['questions']['u001_0']['answers'].keys())

q = data['data']['u001']['questions']['u001_0']['question']
ans = data['data']['u001']['questions']['u001_0']['answers']['0']
print(q)
res = preprocess(q)
print(res)

"""
The structure of quAIL training dataset:

{'version': <str>,
'data': 
    'u001':
        'author': <str>
        'title': <str>
        'context': <str>
        'questions':
            'u001_0':
                'question': <str>
                'answers':
                    '0': <str>
                    '1': <str>
                    '2': <str>
                    '3': <str>}
"""

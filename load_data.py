import os
from util import make_stitched_data, QADataset
import pickle
from torch.utils.data import DataLoader


train_stitched_data_file_name = 'train_stitched_data.txt'
dev_stitched_data_file_name = 'dev_stitched_data.txt'

f_list = os.listdir()
if train_stitched_data_file_name not in f_list:
    make_stitched_data(file_name=train_stitched_data_file_name)
if dev_stitched_data_file_name not in f_list:
    make_stitched_data(file_name=dev_stitched_data_file_name)

with open(train_stitched_data_file_name, 'rb') as f:
    train_data = pickle.load(f)
with open(dev_stitched_data_file_name, 'rb') as f:
    dev_data = pickle.load(f)

quiz_length_list = [len(quiz) for quiz,label in train_data]
max_quiz_length = max(quiz_length_list)

train_texts =[quiz for quiz,label in train_data]
train_labels = [label for quiz,label in train_data]
dev_texts =[quiz for quiz,label in dev_data]
dev_labels = [label for quiz,label in dev_data]

dataset_train = QADataset(train_texts,train_labels,max_len=max_quiz_length)
dataset_dev = QADataset(dev_texts,dev_labels,vocab=dataset_train.vocab,labels_vocab=dataset_train.labels_vocab,max_len=max_quiz_length)

dataloader_train = DataLoader(dataset_train,shuffle=True,batch_size=64)
dataloader_dev = DataLoader(dataset_dev,shuffle=True,batch_size=64)

print(dataset_dev[100])
for batch in dataloader_dev:
    print(batch)
    break

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

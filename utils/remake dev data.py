import json

save_to_file = '../data/quAIL/new_dev_key.json'
dev_key_file = '../data/quAIL/dev_key.json'
with open(dev_key_file, 'rb') as f:
    data = json.load(f)

new_data = {}
new_data['data'] = {}
all_type_list = data['data'].keys()
for type in all_type_list:
    type_data_dict = data['data'][type]
    quiz_idx_list = data['data'][type].keys()
    for quiz_idx in quiz_idx_list:
        new_data['data'][quiz_idx] = type_data_dict[quiz_idx]

with open(save_to_file, 'w') as outfile:
    json.dump(new_data, outfile)

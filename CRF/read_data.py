

def read_data(file_path):
    with open(file_path,'r') as f:
        data = f.readlines()
    data = [item.rstrip('\n').split(' ') for item in data]
    all_sentences = []
    new_sentence = []
    for d in data:
        if d[0]:
            new_sentence.append(d)
        else:
            all_sentences.append(new_sentence)
            new_sentence = []
    return all_sentences
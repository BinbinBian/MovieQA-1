import data_loader
from nltk.tokenize import sent_tokenize
from nltk.tokenize import word_tokenize
import numpy as np
import collections
import random
ignore_word_list = ['.', ',',':']

mqa = data_loader.DataLoader()
story, qa = mqa.get_story_qa_data('train', 'plot')

dictionary = dict()
data = []
words = []

def build_dataset():
    for key_val in story.keys():
        each_story = story[key_val]

        for paragraph in each_story:
            sentence_tokenize_list = sent_tokenize(paragraph)
            for sentences in sentence_tokenize_list:
                word_list = word_tokenize(sentences)
                for word in word_list:
                    words.append(word)
                    if (word in ignore_word_list) == False and (word in dictionary.keys()) == False:
                        dictionary[word] = len(dictionary)

    for word in words:
        index = dictionary[word]
        data.append(index)

    return data, dictionary

del words

data, dictionary = build_dataset()

# batch of skip-gram model
def next_batch(batch_size, num_skips, skip_windows):
    global data_index
    assert batch_size % num_skips == 0
    assert num_skips <= 2*skip_windows

    batch = np.zeros(batch_size)
    label = np.zeros((batch_size,1))
    span = 2*skip_windows + 1
    buffer = collections.deque(maxlen=span)
    for _ in xrange(span):
        buffer.append(data[data_index])
        data_index = (data_index+1)%len(data)

    for i in xrange(batch_size/num_skips):
        target = skip_windows
        avoid = [skip_windows]

        for j in xrange(num_skips):
            while target in avoid : target = random.randint(0, span-1)
            avoid.append(target)
            batch[i*num_skips+j] = buffer[skip_windows]
            label[i*num_skips+j] = buffer[target]

        buffer.append(data[data_index])
        data_index = (ddata_index+1)%len(data)
    return batch, label



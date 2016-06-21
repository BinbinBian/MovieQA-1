import data_loader
from nltk.tokenize import sent_tokenize
from nltk.tokenize import word_tokenize
import numpy as np
import collections
import random
from IPython import embed
import tensorflow as tf
import math

TRAIN_FLAG = True
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
                    if (word in ignore_word_list) == False : words.append(word)
                    if (word in ignore_word_list) == False and (word in dictionary) == False:
                        dictionary[word] = len(dictionary)

    for qa_info in qa:
        question = str(qa_info.question)
        answers = [str(answer) for answer in qa_info.answers]

        question_tokenize = word_tokenize(question)
        for word in question_tokenize:
            if (word in ignore_word_list) == False: words.append(word)
            if (word in ignore_word_list) == False and (word in dictionary) == False:
                dictionary[word] = len(dictionary)

        for ans in answers:
            ans_tokenize = word_tokenize(ans)
            for ans in ans_tokenize:
                if (word in ignore_word_list) == False: words.append(word)
                if (word in ignore_word_list) == False and (word in dictionary) == False:
                    dictionary[word] = len(dictionary)




    dictionary['UKN'] = len(dictionary)

    print 'vocabuluary size >> %d' % len(dictionary)
    for word in words:
        index = dictionary[word]
        data.append(index)

    return data, dictionary

data, dictionary = build_dataset()

del words
data_index = 0
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
        data_index = (data_index+1)%len(data)
    return batch, label

batch_size = 128
embedding_size = 300
skip_window = 4
num_skips = 2

valid_size = 16
valid_window = 100
valid_examples = np.random.choice(valid_window, valid_size, replace=False)

num_sampled = 64    # Number of negative examples to sample.
vocabulary_size = len(dictionary)
graph = tf.Graph()

with graph.as_default():

    # Input data
    train_inputs = tf.placeholder(tf.int32, shape=[None])
    train_labels = tf.placeholder(tf.int32, shape=[None,1])
    valid_dataset = tf.constant(valid_examples, dtype=tf.int32)

    # Ops and variables pinned to the CPU
    with tf.device('/gpu:0'):
        # Look up embeddings for inputs.
        embeddings = tf.Variable(tf.random_uniform([vocabulary_size, embedding_size], -1.0, 1.0))
        embed = tf.nn.embedding_lookup(embeddings, train_inputs)

        # construct the variables for the NCE loss
        nce_weights = tf.Variable(tf.truncated_normal([vocabulary_size, embedding_size], stddev=1.0 / math.sqrt(embedding_size)))
        nce_biases = tf.Variable(tf.zeros([vocabulary_size]))

    # Compute the average NCE loss for the batch
    # tf.nce_loss automatically draws a new sample of the negative labels each
    # time we evaluate the loss.

    loss = tf.reduce_mean(tf.nn.nce_loss(nce_weights, nce_biases, embed, train_labels, num_sampled, vocabulary_size))

    # Construct the SGD optimizer using a learning rate of 1.0
    optimizer = tf.train.AdamOptimizer(0.001).minimize(loss)

    init = tf.initialize_all_variables()

# begin training
num_steps = 400000


with tf.Session(graph=graph) as session:
    saver = tf.train.Saver()
    save_path = './word2vec_shortcut/'

    if TRAIN_FLAG == True:
        # We must initialize all variables before we use them
        init.run()
        print "Initialized!"

        average_loss = 0
        for step in xrange(num_steps):
            batch_inputs, batch_labels = next_batch(batch_size, num_skips, skip_window)
            feed_dict = {train_inputs : batch_inputs, train_labels : batch_labels}

            _, loss_val = session.run([optimizer, loss], feed_dict=feed_dict)
            average_loss += loss_val

            if step % 2000 == 0:
                if step > 0:
                    average_loss /= 2000
                print "Average loss at step %d : %lf" % (step, average_loss)

            if step % 30000 == 0:
                print "shortcut of model saved...at %d iteration...!" % step
                saver.save(session, save_path + str(step) + '.ckpt')

    def encode(model, word):
        """ given trained word2vec model and target words(sentecne),
        get the embedding of that word.

        Args:
            param1 (string) : trained word2vec checkpoint. it might be 'word2vec.ckpt' and
            it is located in ./word2vec_shortcut/

            param2 (list) : target word list formatting like ['I', 'like', 'apple']

        Return:
            mean-pooled embeddings of word list
        """
        model = save_path + model
        saver.restore(session, model)
        preprocessed_word = []
        for w in word:
            if w in dictionary: word.append(w)
            else: word.append('UKN')
        word_index = np.array([dictionary[w] for w in preprocessed_word])
        return session.run([embed], feed_dict={train_inputs:word_index})







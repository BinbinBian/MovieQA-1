import sys
import numpy as np
sys.path.append('../skip-thoughts')
import skipthoughts
from tqdm import tqdm
import hickle
import threading
import pickle
import word2vec
from nltk.tokenize import sent_tokenize
from nltk.tokenize import word_tokenize


model = skipthoughts.load_model()


class Dataset(object):
    """This is a dataset ADT that contains story, QA.

    Args:
        param1 (dictionay) : (IMdb key, video clip name value) pair dictionary
        param2 (list) : QAInfo type list.

        We are able to get param1 and param2 by mqa.get_story_qa_data() function.
    """

    def __init__(self, story, qa):
        self.story = story
        self.qa = qa

        # embedding matrix Z = Word2Vec or Skipthoughts
        self.zq = [] # embedding matrix Z * questino q
        self.zsl = [] # embedding matrix Z * story sl
        self.zaj = [] # embedding matrix Z * answer aj
        self.ground_truth = [] # correct answer index

        self.zq_val = []
        self.zsl_val = []
        self.zaj_val = []
        self.ground_truth_val = []

        self.index_in_epoch_train = 0
        self.index_in_epoch_val = 0
        self.num_train_examples = 0
        self.num_val_examples = 0

        #self.embedding() # for generating hickle dump file.

    def embedding(self, embedding_method='skipthoughts'):
        """ getting Zq and Zsl by using (Word2Vec or Skipthoughts).
        """

        class DataSets(object):
            pass

        data_sets = DataSets()

        if embedding_method == 'word2vec':
            model = 'word2vec.ckpt'
            for qa_info in qa:
                """ ==================================================
                getting basic factor of dataset."""
                question = str(qa_info.question).split()
                answers = [str(answer) for answer in qa_info.answers]
                correct_index = qa_info.correct_index
                imdb_key = str(qa_info.imdb_key)
                stories = self.story[imdb_key]
                error = False
                imdb_key_check = dict()
                validation_flag = str(qa_info.qid)
                """================================================="""

                question_embedding = word2vec.encode(model, quesiton) # word2vec embedding : question
                for answer in answers:
                    if len(answer) == 0: error = True
                if error == True: continue
                local_answers = [word2vec.encode(model,answer) for answer in answers] # word2vec embedding : answer
                gt = [0.0] * 5
                gt[correct_index] = 1.0

                local_stories = []
                if imdb_key in imdb_key_check: last_stories
                else:
                    imdb_key_check[imdb_key] = 1
                    for paragraph in stories:
                        paragraph_tokenize = sent_tokenize(paragraph)
                        for sentences in paragraph_tokenize:
                            local_stories.append(word2vec.encode(model, sentences)) # word2vec embedding : story

            w2v_dim = 300
            if validation_flag.find('train') != -1:
                self.zq.append(question_embedding.reshape((w2v_dim)))
                self.zaj.append(np.transpose(np.array(local_answers).reshape(5, w2v_dim)))
                self.ground_truth.append(np.array(gt))
                zsl_row = np.array(local_stories).shape[0]
                print "zsl shape >> ",
                print np.array(local_stories).shape
                self.zsl.append(np.transpose(np.array(local_stories).reshpae(zsl_row, w2v_dim)))

            if validation_flag.find('val') != -1:
                self.zq_val.append(question_embedding.reshape((w2v_dim)))
                self.zaj_val.append(np.transpose(np.array(local_answers).reshape(5, w2v_dim)))
                self.ground_truth_val.append(np.array(gt))
                zsl_row = np.array(local_stories).shape[0]
                print "zsl shape >> ",
                print np.array(local_stories).shape
                self.zsl_val.append(np.transpose(np.array(local_stories).reshape(zsl_row, w2v_dim)))



            print "==============================================="
            print "each QAInfo status >> "
            print "question embedding shape >> ",
            print np.array(self.zq_val).shape
            print "answer embedding shape >> ",
            print np.array(self.zaj_val).shape
            print "stories embedding shape >> ",
            try:
                print np.array(self.zsl_val).shape
            except:
                print "warning : dimension error."

            print "ground truth shape >> ",
            print np.array(self.ground_truth_val).shape
            print "================================================"






        if embedding_method == 'skipthoughts':
            def embedding_thread(a, b):
                imdb_key_check = {}
                last_stories = []
                for i in tqdm(xrange(a,b)):
                    error = False
                    #if i == 100 : break

                    qa_info = self.qa[i]
                    question = str(qa_info.question)
                    answers = qa_info.answers
                    correct_index = qa_info.correct_index
                    imdb_key = str(qa_info.imdb_key)
                    validation_flag = str(qa_info.qid)

                    question_embedding = skipthoughts.encode(model, [question])
                    assert question_embedding.shape == (1,4800)

                    for answer in answers:
                        if len(answer) == 0 : error = True
                    if error == True :continue
                    local_answers = [skipthoughts.encode(model, [str(answer)]) for answer in answers]

                    gt = [0.0] * 5
                    gt[correct_index] = 1.0

                    stories = self.story[imdb_key]


                    local_stories = []
                    #for s in stories : print [str(s)]
                    if imdb_key in imdb_key_check: local_stories = last_stories
                    else:
                        imdb_key_check[imdb_key] = 1
                        local_stories = [skipthoughts.encode(model, [str(s)])  for s in stories]
                        last_stories = local_stories

                    skip_dim = 4800
                    if validation_flag.find('train') != - 1:
                        self.zq.append(question_embedding.reshape((skip_dim)))
                        self.zaj.append(np.transpose(np.array(local_answers).reshape(5,skip_dim)))
                        self.ground_truth.append(np.array(gt))
                        zsl_row = np.array(local_stories).shape[0]
                        print "zsl shape >> ",
                        print np.array(local_stories).shape
                        self.zsl.append(np.transpose(np.array(local_stories).reshape(zsl_row, skip_dim)))

                    elif validation_flag.find('val') != -1:
                        self.zq_val.append(question_embedding.reshape((skip_dim)))
                        self.zaj_val.append(np.transpose(np.array(local_answers).reshape(5,skip_dim)))
                        self.ground_truth_val.append(np.array(gt))
                        zsl_row = np.array(local_stories).shape[0]
                        self.zsl_val.append(np.transpose(np.array(local_stories).reshape(zsl_row,skip_dim)))




                    print "==========================="
                    print "each QAInfo status >> "
                    print "question embedding shape >> ",
                    print np.array(self.zq_val).shape
                    print "answer embedding shape >> ",
                    print np.array(self.zaj_val).shape
                    print "stories embedding shape >> ",
                    try:
                        print np.array(self.zsl_val).shape
                    except:
                        print "warning : dimension error."

                    print "ground truth shape >> ",
                    print np.array(self.ground_truth_val).shape
                    print "=========================="

            # This code is run by multithreading, but do not scale well..
            """
            qa_length = len(self.qa)
            ts = qa_length
            th = [threading.Thread(target=embedding_thread, args=(i*ts,(i+1)*ts)) for i in xrange(qa_length/ts)]
            print "load dataset by multithreading."
            for i in xrange(len(th)): th[i].start()
            for i in xrange(len(th)): th[i].join()
            """

            embedding_thread(0, len(self.qa))


            skipthoughts_dict = dict()
            skipthoughts_dict['zq_train'] = np.array(self.zq)
            skipthoughts_dict['zaj_train'] = np.array(self.zaj)
            skipthoughts_dict['zsl_train'] = self.zsl
            skipthoughts_dict['ground_truth_train'] = np.array(self.ground_truth)

            skipthoughts_dict['zq_val'] = np.array(self.zq_val)
            skipthoughts_dict['zaj_val'] = np.array(self.zaj_val)
            skipthoughts_dict['zsl_val'] = self.zsl_val
            skipthoughts_dict['zsl_ground_truth'] = np.array(self.ground_truth_val)


            self.num_train_examples = np.array(self.zq).shape[0]
            self.num_val_examples = np.array(self.zq_val).shape[0]

            return skipthoughts_dict


    def next_batch(self, batch_size, type = 'train'):
        """ at training phase, getting training(or validation) data of predefined batch size.

        Args:
            param1 (int) : batch size
            param2 (string) : type of the data you want to get. You might choose between 'train' or 'val'

        Return:
            batch size of (zq, zaj, zsl, ground_truth) pair value would be returned.
        """

        if type == 'train':
            assert batch_size <= self.num_train_examples

            start = self.index_in_epoch_train
            self.index_in_epoch_train += batch_size

            if self.index_in_epoch_train > self.num_train_examples:
                """
                if batch index touch the # of exmaples,
                shuffle the training dataset and start next new batch
                """
                perm = np.arange(self.num_train_examples)
                np.random.shuffle(perm)
                self.zq = self.zq[perm]
                self.zsl = self.zsl[perm]
                self.ground_truth = self.ground_truth[perm]
                self.zaj = self.zaj[perm]

                # start the next batch
                start = 0
                self.index_in_epoch = batch_size
            end = self.index_in_epoch_train
            return self.zq[start:end], self.zaj[start:end], self.zsl[start:end], self.ground_truth[start:end]

        elif type == 'val':
            assert batch_size <= self.num_val_examples

            start = self.index_in_epoch_val
            self.index_in_epoch_val += batch_size

            if self.index_in_epoch_val > self.num_val_examples:
                perm = np.arange(self.num_val_examples)
                np.random.shuffle(perm)
                self.zq_val = self.zq_val[perm]
                self.zsl_val = self.zsl_val[perm]
                self.ground_truth_val = self.ground_truth_val[perm]
                self.zaj_val = self.zaj_val[perm]

                start = 0
                self.index_in_epoch_val = batch_size
            end = self.index_in_epoch_train
            return self.zq_val[start:end], self.zaj_val[start:end], self.zsl_val[start:end], self.ground_truth_val[start:end]




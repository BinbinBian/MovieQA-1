import sys
import numpy as np
sys.path.append('../skip-thoughts')
import skipthoughts

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
        self.ground_truth = [] # correct answer index

    def embedding(self, embedding_method='skipthoughts'):
        """ getting Zq and Zsl by using (Word2Vec or Skipthoughts).
        """

        class DataSets(object):
            pass

        data_sets = DataSets()
        #for debugging

        if embedding_method == 'skipthoughts':

            for qa_info in self.qa:
                question = str(qa_info.question)
                answers = qa_info.answers
                correct_index = qa_info.correct_index
                imdb_key = str(qa_info.imdb_key)

                print "question >> " + question
                question_embedding = skipthoughts.encode(model, question.split())
                self.zq.append(question_embedding)
                print "shape >> ",
                print question_embedding.shape
                assert question_embedding.shape == (1,4800)

                local_answers = []
                for answer in answers:
                    print "answer >> "  + str(answer)
                    local_answers.append(skipthoughts.encode(model, str(answer).split()))

                #local_answers = [skipthoughts.encode(model, str(answer)) for answer in answers]
                self.answers.append(np.array(local_answers))

                gt = [0.0] * 5
                gt[correct_index] = 1.0

                local_stories = []
                for s in stories:
                    print "story >> " + str(s)
                    local_stories.append(skipthoughts.encode(model, str(s).split()))
                stories = self.story(imdb_key)
                #local_stories = [skipthoughts.encode(model, str(s).split()) for s in stories]

                self.zsl.append(np.array(local_stories))






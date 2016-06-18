import theano
import os
import data_loader

mqa = data_loader.DataLoader()
story, qa = mqa.get_story_qa_data('train', 'plot')




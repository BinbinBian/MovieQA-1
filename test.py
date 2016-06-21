from data_unit import *
import data_loader
import hickle
import pickle

mqa = data_loader.DataLoader()
story, qa = mqa.get_story_qa_data('val', 'plot')

a = Dataset(story, qa)
a_dict = a.embedding()
filename = './hickle_dump/skipthoughts_large_val'
f = open(filename, 'w')
pickle.dump(a_dict, f)
f.close()

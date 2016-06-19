import data_loader

mqa = data_loader.DataLoader()
story, qa = mqa.get_story_qa_data('train', 'plot')


for key_val in story.keys():
    each_story = story[key_val]

    for paragraph in each_story:
        print paragraph.split('.')

    print "========================================"
    a = raw_input()

"""
for each_qa in qa:
    print each_qa.question
    each_story = story[each_qa.imdb_key]
    print each_story
    a = raw_input()
"""

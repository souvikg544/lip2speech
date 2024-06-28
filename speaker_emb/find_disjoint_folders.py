import pickle, os
from shutil import copy
# train = pickle.load(open('logs/filenames_train.pkl', 'rb'))
test = pickle.load(open('logs/filenames_test.pkl', 'rb'))

# train_vids = {}
# for x in train:
# 	train_vids[x[:x.rfind('/')]] = True

test_vids = {}
for x in test:
	test_vids[x[:x.rfind('/')]] = True

# for t in train_vids:
# 	if t in test_vids:
# 		del test_vids[t]

SAVED_TEST_FOLDER = '../test_data/'

to_copy = 'n'
for t in test_vids:
	to_copy = input(t)
	if to_copy == 'y':
		d = os.mkdir(SAVED_TEST_FOLDER + os.path.basename(t))
		for f in os.listdir(t):
			src = os.path.join(t, f)
			dest = SAVED_TEST_FOLDER + os.path.basename(t) + '/' + f
			copy(src, dest)
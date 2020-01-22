import pickle
import sys

test_pickle_path = sys.argv[1]

with open(test_pickle_path, 'rb') as f:
   mynewlist = pickle.load(f)

print(mynewlist)

import numpy as np

train_suffix = '_train_1.csv'
test_suffix = '_train_2.csv'

train_labels = np.loadtxt('labels_train_1.csv', dtype='int')
test_labels = np.loadtxt('labels_train_2.csv', dtype='int')

ctr = 0
prev = -1
for i in train_labels:
    if prev != i:
        ctr+=1
        prev = i

print(ctr)

ctr = 0
prev = -1
for i in test_labels:
    if prev != i:
        ctr+=1
        prev = i

print(ctr)
# train changes 212 times
# test changes 112 times
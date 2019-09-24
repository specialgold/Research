# import os
# import numpy as np
# type = 'test'
# path = 'data/RCPSP/'
# if type == 'train':
#     path = path + 'train/'
#
# else:
#     path = path + 'test/'
#
# dirList = os.listdir(path)
# print(dirList)
#
#
#
# a = np.random.choice(dirList, 1)
# print(a[0])
#
# memory = np.ones((12), dtype=int) * -1
# print(memory[1:-1])
# from theano import function, config, shared, tensor
# import numpy
# import time
#
# vlen = 10 * 30 * 768  # 10 x #cores x # threads per core
# iters = 1000
#
# rng = numpy.random.RandomState(22)
# x = shared(numpy.asarray(rng.rand(vlen), config.floatX))
# f = function([], tensor.exp(x))
# print(f.maker.fgraph.toposort())
# t0 = time.time()
# for i in range(iters):
#     r = f()
# t1 = time.time()
# print("Looping %d times took %f seconds" % (iters, t1 - t0))
# print("Result is %s" % (r,))
# if numpy.any([isinstance(x.op, tensor.Elemwise) and
#               ('Gpu' not in type(x.op).__name__)
#               for x in f.maker.fgraph.toposort()]):
#     print('Used the cpu')
# else:
#     print('Used the gpu')

# list = [1,2]
# print(list.pop())
# print(list.pop())
# print(len(list))

# import sys
# print(len(sys.argv))
# print(str(sys.argv))
# print(sys.argv[0])
# print(sys.argv[1])
# print(sys.argv[2])
# print(sys.argv[3])
# import os
# import random
# path = 'data/30/'
# seq = 0
#
# dirList = os.listdir(path)
# random.shuffle(dirList)
# random.shuffle(dirList)
# random.shuffle(dirList)
# total = len(dirList)
# train = dirList[:int(total*0.8)]
# train_len = len(train)
# test = dirList[int(total*0.8):]
# fp = open('data/opt30/trainlist.txt', 'w')
# for file in train:
#     fp.write(file+'\n')
#
# fp.close()
# fp = open('data/opt30/testlist.txt', 'w')
# for file in test:
#     fp.write(file+'\n')
#
# fp.close()
import numpy as np
from pyTsetlinMachine.tm import MultiClassTsetlinMachine
from time import time
from pyTsetlinMachine.tools import Binarizer

X = np.load("statesCartPoleV4.npy")
Y = np.loadtxt("actionsCartPoleV4.txt", dtype=int)
splitratio = 0.7
for i in range(20):
    bits = i + 1
    print("XXX " + str(bits))
    print(X[0])
    b = Binarizer(max_bits_per_feature = bits)
    b.fit(X)
    X_transformed = b.transform(X)
    print(X_transformed[0])
    X_train = X_transformed[:int(len(X)*splitratio)]
    X_test = X_transformed[int(len(X)*splitratio):]
    Y_train = Y[:int(len(Y)*splitratio)]
    Y_test = Y[int(len(Y)*splitratio):]
    clauses = 3000
    s = 3.9
    thresh = 0.4
    #tm = MultiClassTsetlinMachine(clauses, clauses*0.4, 3.9)

    tm = MultiClassTsetlinMachine(clauses, clauses * thresh, s)
    for i in range(3):
        start = time()
        tm.fit(X_train, Y_train, epochs=1)
        stop = time()
        result = 100*(tm.predict(X_test) == Y_test).mean()
        #plotArray = np.append(plotArray, result)
        print("#%d Accuracy: %.2f%% (%.2fs)" % (i+1, result, stop-start))
        #with open('tsetlinAnimals1', 'wb') as tsetlin_file:
        #        pickle.dump(tm, tsetlin_file)
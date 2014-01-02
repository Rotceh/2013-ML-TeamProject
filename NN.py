__author__ = 'Yhchou'
import pdb
import pandas as pd

from pybrain.datasets import ClassificationDataSet
from pybrain.utilities import percentError
from pybrain.supervised.trainers import BackpropTrainer
from pybrain.tools.shortcuts     import buildNetwork
from pybrain.structure.modules   import SoftmaxLayer
#############################################################################
# [set Data]

#CSV_TRAIN = "dataset/train_na2zero.csv"
#CSV_TEST = "dataset/test_na2zero.csv"
CSV_TRAIN = "dataset/train_zero_60x60.csv"
CSV_TEST = "dataset/test_zero_60x60.csv"

df_train = pd.read_csv(CSV_TRAIN)
Y = df_train.y
Y = Y -1 # in order to make target in the range of [0, 1, 2, 3, ...., 11]
X = df_train.iloc[:, 1:].values

alldata = ClassificationDataSet(inp=X.shape[1], target=1, nb_classes=12)
for i in range(X.shape[0]):
    alldata.addSample(X[i, :], [Y[i]])
alldata._convertToOneOfMany()

df_test = pd.read_csv(CSV_TEST)
test_X = df_test.iloc[:, 1:].values



print "Number of training patterns: ", len(alldata)
print "Input and output dimensions: ", alldata.indim, alldata.outdim
print "First sample (input, target, class):"
print alldata['input'][0], alldata['target'][0], alldata['class'][0]

#############################################################################
# fnn
n = buildNetwork(alldata.indim, 10, 10, 10,  alldata.outdim, outclass=SoftmaxLayer, bias=True)
print(n)

#############################################################################

trainer = BackpropTrainer(n, dataset=alldata, learningrate=0.01, momentum=0.1, verbose=True, weightdecay=0.01)
model = trainer.trainUntilConvergence(maxEpochs=10, validationProportion=0.25)
print("[ the best parameter for minimal validation error]", n.params)
print( trainer.testOnClassData(alldata) )
allresult = percentError(trainer.testOnClassData(dataset = alldata) ,alldata['class'])
print("  train error: %5.2f%%" % allresult)

##
import numpy
print "Predicting with the neural network"
answerlist = []
for row in range(test_X.shape[0]):
    answer = numpy.argmax(n.activate(test_X[row, :]))
    answerlist.append(answer)
print(answerlist)

with open("NN_result.txt","w+") as f:
    f.write(str(answerlist))






####################################################
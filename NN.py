__author__ = 'Yhchou'
import pdb
import pandas as pd
import numpy
import time
import datetime

from pybrain.datasets import ClassificationDataSet
from pybrain.utilities import percentError
from pybrain.supervised.trainers import BackpropTrainer
from pybrain.tools.shortcuts     import buildNetwork
from pybrain.structure.modules   import SoftmaxLayer
from pybrain.tools.customxml.networkwriter import NetworkWriter
from sklearn.metrics import precision_score,recall_score,confusion_matrix

def makeClassificationDataSet(X, Y, nb_classes=12):
    """ dim(X) = c(n,m)
             dim(Y) = c(n,1)
             the class of Y must be 0 ,1, 2 ..., where its label starts with 0
        """
    alldata = ClassificationDataSet(inp=X.shape[1], target=1, nb_classes=nb_classes)
    [alldata.addSample(X[row, :], [Y[row]]) for row in range(X.shape[0])]
    alldata._convertToOneOfMany()
    return alldata

def predictOnData(test_X):
    """
        dim(test) = c(n2, m)
        this fn will return a list of predicted class
        """
    answerlist = list()
    for row in range(test_X.shape[0]):
        answer = numpy.argmax(n.activate(test_X[row, :]))
        answerlist.append(answer)
    pdb.set_trace()
    return answerlist

def NN_Report():
    line = ["\n","[START_TIME]" , START_TIME,"[END_TIME]",END_TIME,
    "---------------------------------------",
    "#1 Data Description",
    "[Number of training patterns] ", len(alldata),
    "[Input and output dimensions] ", alldata.indim, alldata.outdim,
    "[First sample (input, target, class)]", alldata['input'][0], alldata['target'][0], alldata['class'][0],
    "---------------------------------------",
    "#2 The structure of Network",
    n,
    "---------------------------------------",
    "#3 Training Info",
    "[N_LAYER]", N_LAYER,
    "[N_NEURAL]", N_NEURAL,
    "[LEARNING_RATE]", LEARNING_RATE,
    "[MOMENTUM]",MOMENTUM,
    "[WEIGHTDECAY]",WEIGHTDECAY,
    "[MAX_EPOCHS]",MAX_EPOCHS,
    "[VALIDATION_PROPORTION]",VALIDATION_PROPORTION,
    "[ the best parameter for minimal validation error]", n.params,
    "---------------------------------------",
    "#4 Validation",
    "[ predicted value for train data]",predictedVals,
    " [train error] %5.2f%%" % trainerror,
    "[The precision ]"+str(precision_score(alldata['class'],predictedVals)),
    "[The recall ] "+ str(recall_score(alldata['class'],predictedVals)),
    "[confusion matrix]", str(confusion_matrix(alldata['class'],predictedVals)),
    "---------------------------------------",
    "#5 Prediction",
    "[ predicted value for test data]", answerlist ]

    s = [ str(li) for li in line]
    return "\n".join(s)

t0 = time.time()
##################################################################
START_TIME = "-".join(str(datetime.datetime.now()).split(":"))
CSV_TRAIN = "dataset/train_zero_60x60.csv"
CSV_TEST = "dataset/test_zero_60x60.csv"
NROWS = 12

N_LAYER = 2 # hand-defined
N_NEURAL = str([10,10]) # hand-defined
LEARNING_RATE = 0.005
MOMENTUM = 0.1
WEIGHTDECAY = 0.01
MAX_EPOCHS = 500
VALIDATION_PROPORTION = 0.1
####################################################################
#  data preprocessed

df_train = pd.read_csv(CSV_TRAIN, nrows=NROWS)
X = df_train.iloc[:, 1:].values
Y = df_train.y
Y = Y -1            # in order to make target in the range of [0, 1, 2, 3, ...., 11]

df_test = pd.read_csv(CSV_TEST)
test_X = df_test.iloc[:, 1:].values

alldata = makeClassificationDataSet(X,Y,nb_classes=12)# make dataset
n = buildNetwork(alldata.indim, 10, 10,  alldata.outdim, outclass=SoftmaxLayer, bias=True)# set Neural Network
trainer = BackpropTrainer(n, dataset=alldata, learningrate=LEARNING_RATE, momentum=MOMENTUM, verbose=True, weightdecay=WEIGHTDECAY)# train- set error  mode
trainer.trainUntilConvergence(maxEpochs=MAX_EPOCHS, validationProportion=VALIDATION_PROPORTION)# train
predictedVals = trainer.testOnClassData(dataset=alldata)# set
trainerror = percentError(predictedVals ,alldata['class'])# validation

# prediction
answerlist = predictOnData(test_X)

####################################################################
END_TIME = "-".join(str(datetime.datetime.now()).split(":"))
t1=time.time()
print("[Total Time]")
print(t1-t0)

# report
report = NN_Report()
print(report)
with open("NN_result(%s).txt" % START_TIME,"w+") as f:
    f.writelines("[predicted y]")
    f.write(str([y+1 for y in answerlist])) # because of Y=Y-1 before
    f.write("\n#############################\n")
    f.writelines(report)
NetworkWriter.writeToFile(n, "NN_model(%s).xml" % START_TIME)



####################################################
# def NN_Report():
#     print("[Time]")
#     print(time_info)
#     print("---------------------------------------")
#     print("#1 Data Description")
#     print("[Number of training patterns] ", len(alldata))
#     print("[Input and output dimensions] ", alldata.indim, alldata.outdim)
#     print("[First sample (input, target, class)]")
#     print(alldata['input'][0], alldata['target'][0], alldata['class'][0])
#
#     print("---------------------------------------")
#     print("#2 The structure of Network")
#     print(n)
#
#     print("---------------------------------------")
#     print("#3 Training Info")
#     print("[ the best parameter for minimal validation error]", n.params)
#
#     print("---------------------------------------")
#     print("#4 Validation")
#     print("[ predicted value for train data]")
#     print(predictedVals)
#     print(" [train error] %5.2f%%" % trainerror)
#
#     print("---------------------------------------")
#     print("#5 Prediction")
#     print("[ predicted value for test data]")
#     print(answerlist)
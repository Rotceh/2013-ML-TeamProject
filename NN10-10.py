__author__ = 'Yhchou'
import pdb, sys
import pandas as pd
import numpy
import time
import datetime

from pybrain.datasets import ClassificationDataSet
from pybrain.utilities import percentError
from pybrain.supervised.trainers import BackpropTrainer
from pybrain.tools.shortcuts     import buildNetwork
from pybrain.structure.modules   import SoftmaxLayer
try:
    from pybrain.tools.customxml.networkwriter import NetworkWriter
except ImportError as e:
    print("chang xml module")
    from pybrain.tools.xml.networkwriter import NetworkWriter

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
    return answerlist


def transdata(data):
    return map(lambda x: x[0], data)

def NN_Report():
    print("[START_TIME]" , START_TIME)
    print("[END_TIME]",END_TIME)
    print("[Total time]", t1-t0)

    print("---------------------------------------")
    print("#1 Data Description")
    print("[Number of training patterns] ", len(alldata))
    print("[Input and output dimensions] ", alldata.indim, alldata.outdim)
    print("[First sample (input, target, class)]", alldata['input'][0], alldata['target'][0], alldata['class'][0])

    print("---------------------------------------")
    print("#2 The structure of Network")
    print(n)

    print("---------------------------------------")
    print("#3 Training Info")
    print("[N_HIDDEN_LAYER]", N_HIDDEN_LAYER)
    print("[N_NEURAL]", N_NEURAL)
    print("[LEARNING_RATE]", LEARNING_RATE)
    print("[MOMENTUM]",MOMENTUM)
    print("[WEIGHTDECAY]",WEIGHTDECAY)
    print("[MAX_EPOCHS]",MAX_EPOCHS)
    print("[VALIDATION_PROPORTION]",VALIDATION_PROPORTION)
    print("[ the best parameter for minimal validation error]", n.params)

    print("---------------------------------------")
    print("#4 Validation")
    print("[ predicted value for train data]", predictedVals)
    print(" [train error] %5.2f%%" % trainerror)
    print("[The precision ]", precision_score(transdata(alldata['class']),predictedVals))
    print("[The recall ] ", recall_score(transdata(alldata['class']),predictedVals))
    print("[confusion matrix]", confusion_matrix(transdata(alldata['class']),predictedVals))

    print("---------------------------------------")
    print("#5 Prediction")
    print("[ predicted value for test data]", predictedVals2Raw)



t0 = time.time()
START_TIME = "-".join(str(datetime.datetime.now()).split(":"))
#####################################
# parse cmd
container = [int(char) for char in sys.argv[1:]]
NROWS = container[0]
N_NEURAL = container[1:]

##################################################################
CSV_TRAIN = "dataset/train_zero_60x60.csv"
CSV_TEST = "dataset/test_zero_60x60.csv"
#  data preprocessed
df_train = pd.read_csv(CSV_TRAIN, nrows=NROWS)
X = df_train.iloc[:, 1:].values
Y = df_train.y
Y = Y -1            # in order to make target in the range of [0, 1, 2, 3, ...., 11]

df_test = pd.read_csv(CSV_TEST)
test_X = df_test.iloc[:, 1:].values

alldata = makeClassificationDataSet(X,Y,nb_classes=12)# make dataset

#####################################
#Model settings
DIM_LAYER = [alldata.indim] + N_NEURAL + [alldata.outdim]
NETWORK_SETTINGS = {"outclass": SoftmaxLayer, "bias":True}
N_HIDDEN_LAYER = len(N_NEURAL)

LEARNING_RATE = 0.005
MOMENTUM = 0.5
WEIGHTDECAY = 0.01
MAX_EPOCHS = 1000
VALIDATION_PROPORTION = 0.1
#####################################
n = buildNetwork(*DIM_LAYER, **NETWORK_SETTINGS)# set Neural Network
trainer = BackpropTrainer(n, dataset=alldata, learningrate=LEARNING_RATE, momentum=MOMENTUM, verbose=True, weightdecay=WEIGHTDECAY)# train- set error  mode
trainer.trainUntilConvergence(maxEpochs=MAX_EPOCHS, validationProportion=VALIDATION_PROPORTION)# train
predictedVals = trainer.testOnClassData(dataset=alldata)# set
trainerror = percentError(predictedVals ,alldata['class'])# validation

# prediction
answerlist = predictOnData(test_X)
predictedVals2Raw = [y+1 for y in answerlist]

####################################################################
END_TIME = "-".join(str(datetime.datetime.now()).split(":"))
t1 = time.time()

# report
filename = "NN%sn%s" % (str(NROWS), "x".join([str(nn) for nn in N_NEURAL]))
NetworkWriter.writeToFile(n, filename+".xml")
NN_Report()


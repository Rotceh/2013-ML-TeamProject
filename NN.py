__author__ = 'Yhchou'
import pdb
import pandas as pd

from pybrain.datasets            import ClassificationDataSet
from pybrain.utilities           import percentError
from pybrain.supervised.trainers import BackpropTrainer


#############################################################################
# [set Data]

#CSV_TRAIN = "dataset/train_na2zero.csv"
#CSV_TEST = "dataset/test_na2zero.csv"
CSV_TRAIN = "dataset/train_zero_60x60.csv"

df_train = pd.read_csv(CSV_TRAIN)
Y = df_train.y
Y = Y -1 # in order to make target in the range of [0, 1, 2, 3, ...., 11]
X = df_train.iloc[:, 1:].values


alldata = ClassificationDataSet(inp=X.shape[1], target=1, nb_classes=12)
for i in range(len(Y)):
    alldata.addSample(X[i, :], [Y[i]])

alldata._convertToOneOfMany()


print "Number of training patterns: ", len(alldata)
print "Input and output dimensions: ", alldata.indim, alldata.outdim
print "First sample (input, target, class):"
print alldata['input'][0], alldata['target'][0], alldata['class'][0]

#############################################################################
# [set NN ]
from pybrain.structure import FeedForwardNetwork, LinearLayer, SigmoidLayer, FullConnection
n = FeedForwardNetwork()

#  set layer type
inLayer = LinearLayer(alldata.indim, name="input")
outLayer = LinearLayer(alldata.outdim, name="output")
N_LAYER = 3
hiddenLayers = list()
for ii in range(N_LAYER):
    node = 1000
    hiddenLayers.append(SigmoidLayer(node, name="hidden%s" % (ii+1)))

#  add neural
n.addInputModule(inLayer)
n.addOutputModule(outLayer)
[n.addModule(hiddenLayer) for hiddenLayer in hiddenLayers]

#  set the the the way how we connect neurals
connection_Map = list()
connection_Map.append(FullConnection(inLayer, hiddenLayers[0]))
for ii in range(N_LAYER-1): # -1 because the last one doesn't link to any hidden layer but connect to output layer
    connection_Map.append(FullConnection(hiddenLayers[ii],hiddenLayers[ii+1]))
connection_Map.append(FullConnection(hiddenLayers[N_LAYER-1], outLayer))

# build Network: set connection
[n.addConnection(layer_connection) for layer_connection in connection_Map]

# deploy: make MLP useful
n.sortModules()
#pdb.set_trace()

#############################################################################

trainer = BackpropTrainer(n, dataset=alldata, learningrate=0.01, momentum=0.1, verbose=True, weightdecay=1)
trainer.trainEpochs(3)
allresult = percentError(trainer.trainUntilConvergence(),
                         alldata['class'])
print("epoch: %4d" % trainer.totalepochs,
      "  train error: %5.2f%%" % allresult)

#with open("NN_result.txt","w+"):
#    pass


####################################################
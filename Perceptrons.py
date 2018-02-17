import pandas as pd
import numpy as np

def normalizeData(dataset, categories):
	normData = dataset.copy()
	col = dataset[categories]
	col_norm = (col - col.min()) / (col.max() - col.min())
	normData[categories] = col_norm
	return normData

def trainingTestData(dataset, ratio):
	tr = int(len(dataset)*ratio)
	return dataset[:tr], dataset[tr:]

def getRandomWeights(length):
	#change weights to give values between -0.1 to 0.1
	weights = []
	weights.append(0)
	#length + 1 for bias
	for i in range(0,length):
		weights.append(np.random.random())
	return weights

def activation(inputs, weights, theta):
	inp = inputs.tolist()
	inp.insert(0,0)
	sumq = 0

	sumq = np.dot(inp, weights)
	if float(1 / (1 + np.exp(-sumq)))>= 0.5:
		return 1
	return 0

def updateWeights(inputs, learningRate, weights, localError):
	inp = inputs.tolist()
	inp.append(1)
	delta = learningRate * localError
	change = np.array(np.multiply(inp,delta))
	weights = np.add(weights, change)
	#for i in range(0, len(weights)):
	#    weights[i] += float(learningRate * localError*  inp[i])
	return weights
def evaluate(solutions, real):
	predictions = np.array(solutions)
	labels = np.array(real)
	return (predictions == labels).sum() / float(labels.size)

def train(features, labels):
	columns=dataset.columns.tolist()
	features = features.as_matrix()
	error = 0
	threshold = 0.5
	weights = []
	learningRate = 0.01
	weights = getRandomWeights(len(features[0]))
	i=0
	while i <= 150:
		for i in range(len(features)):
			output = activation(features[i],weights, threshold)
			error = labels[i] - output
			weights = updateWeights(features[i], learningRate, weights, error)

		i = i+1
	return weights

#intialization n hot Encoding
dataset = pd.read_csv("/Users/bhavanarama/desktop/assignment3/data.csv")
candidateOffice = pd.get_dummies(dataset.can_off, prefix = 'Office')
ratio = 0.80
tr = int(len(dataset)*ratio)
dataset = pd.concat([dataset, candidateOffice], axis = 1)
candidate = pd.get_dummies(dataset.can_inc_cha_ope_sea, prefix = 'Candidate')
dataset = pd.concat([dataset, candidate], axis = 1)
totalResult = dataset['winner'].astype(int)
trainResult = totalResult[:tr]
testResult = totalResult[tr:]
dataset = dataset.drop(["can_id", "can_nam","winner","can_off", "can_inc_cha_ope_sea"], axis=1)
dataset = normalizeData(dataset, 'net_ope_exp')
dataset = normalizeData(dataset, 'net_con')
dataset = normalizeData(dataset, 'tot_loa')

def sigmoid(x):
	if (1 / (1 + np.exp(-x))) >= 0.5:
		return 1
	else:
		return 0
#get training data
trainingData, testingData = trainingTestData(dataset, ratio)
weights = train(trainingData, trainResult)
testingData = dataset.as_matrix()
result = []
for i in range(0,len(testingData)):
	inp = testingData[i].tolist()
	inp.insert(0,0)
	result.append(sigmoid(np.dot(inp, weights)))
print result
print evaluate(result, totalResult.tolist())

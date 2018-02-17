import numpy as np
import pandas as pd

dataset = pd.read_csv("/Users/bhavanarama/desktop/assignment3/data.csv")

def evaluate(solutions, real):
	predictions = np.array(solutions)
	labels = np.array(real)
	return (predictions == labels).sum() / float(labels.size)

def normalizeData(dataset, categories):
	normData = dataset.copy()
	col = dataset[categories]
	col_norm = (col - col.min()) / (col.max() - col.min())
	normData[categories] = col_norm
	return normData

def trainingTestData(dataset, ratio):
	tr = int(len(dataset)*ratio)
	return dataset[:tr], dataset[tr:]

def preprocess(dataset):
	candidateOffice = pd.get_dummies(dataset.can_off, prefix = 'Office')
	dataset = pd.concat([dataset, candidateOffice], axis = 1)
	candidate = pd.get_dummies(dataset.can_inc_cha_ope_sea, prefix = 'Candidate')
	dataset = pd.concat([dataset, candidate], axis = 1)
	labels = dataset['winner'].astype(int)
	dataset = dataset.drop(["can_id", "can_nam","winner","can_off", "can_inc_cha_ope_sea"], axis=1)
	dataset = normalizeData(dataset, 'net_ope_exp')
	dataset = normalizeData(dataset, 'net_con')
	dataset = normalizeData(dataset, 'tot_loa')
	return dataset, labels

def derivative(x):
	return sigmoid(x)*(1.0 - sigmoid(x))

def sigmoid(x):
	return 1.0/(1.0 + np.exp(-x))

def getBackwards(x):
	return x.reverse()
class MLP:

	def __init__(self):
		self.nodes = [9, 4, 1]
		self.sigmoid = sigmoid
		self.derivative = derivative
		self.weights = []
		self.learningRate = 0.01
		#get random weights
		for i in range(0, len(self.nodes) - 1):
			if i == 0:
				continue
			self.weights.append(2*np.random.random((self.nodes[i-1] + 1, self.nodes[i] + 1)) -1)
		# output layer - random((2+1, 1)) : 3 x 1
		self.weights.append(2*np.random.random( (self.nodes[i] + 1, self.nodes[i+1])) - 1)
		print self.weights

	def train(self, features, labels):
		features = features.values
		labels = labels.tolist()
		#bias the inputs
		bias = np.atleast_2d(np.ones(features.shape[0]))
		features = np.concatenate((bias.T, features), axis=1)
		time = pd.datetime.now()
		time2 = time + pd.Timedelta('1 min')
		while True:
			if pd.datetime.now() >= time2:
				break
			rand = np.random.randint(features.shape[0])
			output = [features[rand]]
			for row in range(len(self.weights)):
				output.append(self.sigmoid(np.dot(output[row], self.weights[row])))

			#last layer
			difference = labels[rand] - output[-1]
			delta = [difference * self.derivative(output[-1])]

			#second to last but one layer
			for row in range(len(output) - 2, 0, -1):
				delta.append(delta[-1].dot(self.weights[row].T)*self.derivative(output[row]))

			#to backpropogate
			delta.reverse()

			#backpropogation
			for i in range(len(self.weights)):
				layer = np.atleast_2d(output[i])
				delta1 = np.atleast_2d(delta[i])
				self.weights[i] += self.learningRate * layer.T.dot(delta1)


	def predict(self, features):
		features = features.values
		output = []
		#predict output for each row in features
		for row in features:
			individualOutput = np.concatenate((np.ones(1).T, np.array(row)))
			for i in range(0, len(self.weights)):
				dotProduct = np.dot(individualOutput, self.weights[i])
				individualOutput = self.sigmoid(dotProduct)
			if individualOutput >= 0.5:
				output.append(1)
			else:
				output.append(0)
		return output

mlp = MLP()
train_ratio = 0.80
train_dataset, test_dataset = trainingTestData(dataset, train_ratio)
train_features, train_labels = preprocess(train_dataset)
mlp.train(train_features, train_labels)
test_features, test_labels = preprocess(test_dataset)
predictions = mlp.predict(test_features)
accuracy = evaluate(predictions, test_labels)
print accuracy

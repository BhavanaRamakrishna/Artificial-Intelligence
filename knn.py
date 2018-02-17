import pandas as pd
import numpy as np
dataset = pd.read_csv("data.csv")
def trainingTestData(dataset, ratio):
	tr = int(len(dataset)*ratio)
	return dataset[:tr], dataset[tr:]
def normalizeData(dataset, categories):
	normData = dataset.copy()
	col = dataset[categories]
	col_norm = (col - col.min()) / (col.max() - col.min())
	normData[categories] = col_norm
	return normData
def preprocessKNN(dataset):
	#one hot encoding
	candidateOffice = pd.get_dummies(dataset.can_off, prefix = 'Office')
	dataset = pd.concat([dataset, candidateOffice], axis = 1)
	candidate = pd.get_dummies(dataset.can_inc_cha_ope_sea, prefix = 'Candidate')
	dataset = pd.concat([dataset, candidate], axis = 1)
	labels = dataset['winner'].astype(int)
	dataset = dataset.drop(["can_id", "can_nam","winner","can_off", "can_inc_cha_ope_sea"], axis=1)
	#normalize attributes
	dataset = normalizeData(dataset, 'net_ope_exp')
	dataset = normalizeData(dataset, 'net_con')
	dataset = normalizeData(dataset, 'tot_loa')
	return dataset, labels
def evaluate(solutions, real):
	predictions = np.array(solutions)
	labels = np.array(real)
	return (predictions == labels).sum() / float(labels.size)
class KNN:
	def __init__(self):
		self.dataGroups = { 0: [], 1 :[] }

	def train(self, features, labels):
		features = features.as_matrix()
		#cluster features into 2 groups
		for i in range(0,len(features)):
			if labels[i] == 0:
				self.dataGroups[0].append(features[i])
			else:
				self.dataGroups[1].append(features[i])

	def predict(self, features):
		k = 9
		labels = []
		features = features.as_matrix()
		#for each row in features
		for rows in features:
			distances = []
			#determine the closest cluster
			for group in self.dataGroups:
				for variable in self.dataGroups[group]:
					euclidean_distance = np.sqrt(np.sum((np.array(variable)-np.array(rows))**2))
					distances.append([euclidean_distance, group])
			votes = [i[1] for i in sorted(distances)[:k]]
			#obtain the target of closest cluster
			oneCount = 0
			zeroCount = 0
			for i in range(0,len(votes)):
				if votes[i] == 1:
					oneCount = oneCount +1
				else:
					zeroCount = zeroCount + 1
			if oneCount > zeroCount:
				result = 1
			else:
				result = 0
			labels.append(result)
		return labels


train_ratio = 0.80
kNN = KNN()
train_dataset, test_dataset = trainingTestData(dataset, train_ratio)
train_features, train_labels = preprocessKNN(train_dataset)
kNN.train(train_features, train_labels)
test_features, test_labels = preprocessKNN(test_dataset)
predictions = kNN.predict(test_features)
accuracy = evaluate(predictions, test_labels.tolist())
print accuracy

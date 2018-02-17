import pandas as pd
import numpy as np
import collections
dataset = pd.read_csv("/Users/bhavanarama/Desktop/Assignment3/data.csv")
#dataset = pd.read_csv("/Users/bhavanarama/Desktop/Assignment3/newdata.csv")

def normalizeData(dataset, categories):
	normData = dataset.copy()
	col = dataset[categories]
	col_norm = (col - col.min()) / (col.max() - col.min())
	normData[categories] = col_norm
	return normData
def trainingTestData(dataset, ratio):
	tr = int(len(dataset)*ratio)
	return dataset[:tr], dataset[tr:]
def evaluate(solutions, real):
	predictions = np.array(solutions)
	labels = np.array(real)
	return (predictions == labels).sum() / float(labels.size)
def splitData(dataset, category):
	#range for each bucket
	bins = [-1, 0.2, 0.4, 0.6, 0.8, 1.1]
	#create 5 buckets
	buckets = [1, 2, 3, 4, 5]
	categories = pd.cut(dataset[category], bins, labels=buckets)
	dataset = dataset.drop([category], axis=1)
	dataset = pd.concat([dataset, categories], axis = 1)
	return dataset
def preprocessDecisionTree(dataset):
	dataset = normalizeData(dataset, 'net_ope_exp')
	dataset = normalizeData(dataset, 'net_con')
	dataset = normalizeData(dataset, 'tot_loa')
	testResult = dataset['winner'].astype(int)
	dataset = dataset.drop(["can_id", "can_nam","winner"], axis=1)
	#dataset = dataset.drop(["winner"], axis=1)
	dataset = splitData(dataset, 'net_ope_exp')
	dataset = splitData(dataset, 'net_con')
	dataset = splitData(dataset, 'tot_loa')
	return dataset, testResult
def getEntropy(features, labels, columns):
	entropy = 0.0
	hashmap = {}
	for val in labels:
		if (hashmap.has_key(val)):
			hashmap[val] = hashmap[val] + 1.0
		else:
			hashmap[val] = 1.0

	#obtain the frequency of each value in labels
	#count = labels.value_counts()
	"""entropy = (-count[0]/len(features)) * np.log2(count[0]/len(features))
	entropy += (-count[1]/len(features)) * np.log2(count[1]/len(features))"""
	for key in hashmap.values():

		entropy += ((-key/len(features)) * np.log2(key/len(features)))

	return entropy

def informationGain(features, labels, columns, maximum):
	#count = labels.value_counts()
	index = columns.index(maximum)

	hashmap = {}
	dataEntropy = 0.0
	#obtain the frequency of each of the values in the given attribute
	for row in features:
			if (hashmap.has_key(row[index])):
				hashmap[row[index]] = hashmap[row[index]] + 1.0
			else:
				hashmap[row[index]] = 1.0

	for val in hashmap.keys():
		frequency = hashmap[val] / sum(hashmap.values())

		i = 0
		#data = [row for row in features if row != [] and row[index] == val]
		data = []
		label = []

		for row in features:
			if row[index] == val:
				data.append(row)
				label.append(labels[i])
			i = i+1

		entropy = getEntropy(data, label, columns)

		dataEntropy += frequency * entropy



	gain = getEntropy(features, labels, maximum)
	gain = gain - dataEntropy
	return gain



def getMaxImportance(features, labels, columns):
	maximum = columns[0]
	gain = 0
	for col in columns:
		attributeInfo = informationGain(features, labels, columns, col)
		if attributeInfo > gain:

			gain = attributeInfo

			maximum = col
	return maximum
def decisionTree(features, labels, columns):
		frequentOutcome = getFrequentOutcome(labels)

		hashmap = {}
		for val in labels:
			if hashmap.has_key(val):
				hashmap[val] = hashmap[val] + 1
			else:
				hashmap[val] = 1
		v = list(hashmap.values())
		k = list(hashmap.keys())
		if len(features)-1 <= 0 or len(columns)-1 <= 0:
			return frequentOutcome
		elif k[v.index(max(v))] == len(labels):
			return k[v.index(max(v))]
		else:
			bestAttribute = getMaxImportance(features, labels, columns)
			tree = {bestAttribute : {}}
			values = []
			index = columns.index(bestAttribute)
			for row in features:
				if row[index] not in values:
					values.append(row[index])

			for value in values:
				dataSubset = []
				label = []
				i = 0
				for row in features:
					if (row[index] == value):
						tempRow = []
						label.append(labels[i])
						for j in range(0,len(row)):
							if j != index:
								tempRow.append(row[j])
								#might wanna add remove [[]]
						dataSubset.append(tempRow)
					i = i+1
				newColumn = columns[:]
				newColumn.remove(bestAttribute)
				subset = decisionTree(dataSubset,label, newColumn)
				tree[bestAttribute][value] = subset
		return tree
def getFrequentOutcome(target):
		hashmap = {}
		for val in target:
			if hashmap.has_key(val):
				hashmap[val] = hashmap[val] + 1
			else:
				hashmap[val] = 1
		"""if hashmap[0] > hashmap[1]:
			return 0
		else:
			return 1"""
		v = list(hashmap.values())
		k = list(hashmap.keys())
		return k[v.index(max(v))]
def test(tree, originalColumns, data, parent=None):
	if tree == 0:
		num = 0
	elif tree == 1:
		num = 1
	else:
		for val in tree.keys():
		#get index of val
			if val not in originalColumns:
				num =  parent[val]
				return num
			else:
				index = originalColumns.index(val)
				if tree[val][data[index]] != 0 or tree[val][data[index]] != 1:
					return test(tree[val][data[index]],originalColumns, data, tree)
				else:
					num = tree[val][data[index]]
					return num
	return num

class ID3:
	def __init__(self):
		#Decision tree state here
		#Feel free to add methods
			self.finalTree = {}
			self.columns = []
			self.decisionTree = decisionTree
			self.test = test

	def train(self, features, labels):
		#training logic here
		#input is list/array of features and labels
			columns=features.columns.tolist()
			self.columns = columns
			features = features.as_matrix()
			labels = labels.tolist()
			self.finalTree = self.decisionTree(features, labels, columns)
			return self.finalTree

	def predict(self, features):
		#Run model here
		#Return list/array of predictions where there is one prediction for each set of features
			features = features.as_matrix()
			originalColumns = ['can_off','can_inc_cha_ope_sea','net_ope_exp','net_con','tot_loa']
			predictions = []
			for row in features:
				temp = self.test(self.finalTree, originalColumns,row)
				predictions.append(temp)
			return predictions



decide = ID3()
tree = {}
train_ratio = 0.75
train_dataset, test_dataset = trainingTestData(dataset, train_ratio)
train_features, train_labels = preprocessDecisionTree(dataset)
tree = decide.train(train_features, train_labels)
print tree
test_features, test_labels = preprocessDecisionTree(dataset)
predictions = decide.predict(test_features)
accuracy = evaluate(predictions, test_labels)
print accuracy

from math import log
import operator


#计算给定数据集的香农熵
def calcShannonEnt(dataSet):
	numEntries = len(dataSet) #实例总数
	labelCounts = {}
	#为所有可能分类创建字典
	for featVec in dataSet:
		currentLabel = featVec[-1]
		if currentLabel not in labelCounts.keys():
			labelCounts[currentLabel] = 0
		labelCounts[currentLabel] += 1
	shannonEnt = 0.0
	for key in labelCounts:
		prob = float(labelCounts[key])/numEntries
		shannonEnt -= prob * log(prob, 2)
	return shannonEnt


#创建数据集
def createDataSet():
	dataSet = [[1, 1, 'yes'],
			[1, 1, 'yes'],
			[1, 0, 'no'],
			[0, 1, 'no'],
			[0, 1, 'no']]
	labels = ['no surfacing', 'flippers']
	return dataSet, labels


#划分数据集
def splitDataSet(dataSet, axis, value):
	retDataSet = []
	for featVec in dataSet:
		#print(featVec)
		if featVec[axis] == value:
			reducedFeatVec = featVec[:axis]
			#print("reducedFeatVec") 
			#print(reducedFeatVec)
			reducedFeatVec.extend(featVec[axis+1:])
			#print("reducedFeatVec")
			#print(reducedFeatVec)
			retDataSet.append(reducedFeatVec)
			#print("retDataSet")
			#print(retDataSet)
	return retDataSet


#选择最好的数据集划分方式
def chooseBestFeatureToSplit(dataSet):
	numFeatures = len(dataSet[0]) - 1 #特征总数
	baseEntropy = calcShannonEnt(dataSet) #计算香农熵
	#print(baseEntropy) #0.9709505944546686
	bestInfoGain = 0.0
	bestFeature = -1
	for i in range(numFeatures):
		featList = [example[i] for example in dataSet] 
		#print(featList)
		#[1, 1, 1, 0, 0]
		#[1, 1, 0, 1, 1]
		uniqueVals = set(featList) #创建分类标签列表 无重复元素
		#{0, 1}
		#{0, 1}
		#print(uniqueVals)
		newEntropy = 0.0
		for value in uniqueVals: #计算每种划分方式的信息熵
			subDataSet = splitDataSet(dataSet, i, value) #划分数据集
			#print(subDataSet)
			prob = len(subDataSet)/float(len(dataSet)) 
			#print("prob")
			#print(prob)
			newEntropy += prob * calcShannonEnt(subDataSet)
		infoGain = baseEntropy - newEntropy
		#print(infoGain)
		#print(newEntropy)
		if(infoGain > bestInfoGain): #增益大的是好的数据集划分
			bestInfoGain = infoGain
			bestFeature = i		
	return bestFeature


#返回出现频率最高的标签
def majorityCnt(classList):
	classCount = {}
	for vote in classList:
		if vote not in classCount.keys():
			classCount[vote] = 0
		classCount[vote] += 1
	sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True) #key=operator.itemgetter(1)按第二个域排序
	return sortedClassCount[0][0]


#递归创建决策树
def createTree(dataSet, labels):
	classList = [example[-1] for example in dataSet] #记录分类标签
	#print(classList)
	if classList.count(classList[0]) == len(classList): 
		#print("classList.count(classList[0])")
		#print(classList.count(classList[0]))
		return classList[0]
	if len(dataSet[0]) == 1:
		return majorityCnt(classList)
	bestFeat = chooseBestFeatureToSplit(dataSet)
	#print(bestFeat)
	bestFeatLabel = labels[bestFeat]
	#print(bestFeatLabel)
	myTree = {bestFeatLabel:{}} #使用字典存储树的信息
	del(labels[bestFeat]) #将划分最好的特征删掉
	#print(labels)
	featValues = [example[bestFeat] for example in dataSet]
	#print("featValues")
	#print(featValues)
	uniqueVals = set(featValues)
	for value in uniqueVals:
		subLabels = labels[:]
		myTree[bestFeatLabel][value] = createTree(splitDataSet(dataSet, bestFeat, value), subLabels)
	return myTree


#使用决策树的分类函数
def classify(inputTree, featLabels, testVec):
	firstStr = list(inputTree.keys())[0]
	#print(firstStr#)
	secondDict = inputTree[firstStr]
	#print(secondDict)
	featIndex = featLabels.index(firstStr) #返回firstStr在列表中的下标
	#print(featIndex)
	for key in secondDict.keys():
		if testVec[featIndex] == key:
			if type(secondDict[key]) == dict:
				classLabel = classify(secondDict[key], featLabels, testVec)
			else:
				classLabel = secondDict[key]
	return classLabel


def storeTree(inputTree, filename):
	import pickle
	fw = open(filename, 'wb')
	pickle.dump(inputTree, fw)
	fw.close()


def grabTree(filename):
	import pickle
	fr = open(filename, 'rb')
	return pickle.load(fr)















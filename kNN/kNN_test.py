from numpy import *
import operator
from kNN import classify0

#将文本记录转换为NumPy的解析程序
def file2matrix(filename):
    fr = open(filename)
    arrayOfLines = fr.readlines()
    numberOfLines = len(arrayOfLines)
    returnMat = zeros((numberOfLines, 3))
    classLabelVector = []
    index = 0
    for line in arrayOfLines:
        line = line.strip()
        listFromLine = line.split('\t')
        returnMat[index, :] = listFromLine[0:3]
        labels = {'didntLike':1, 'smallDoses':2, 'largeDoses':3}
        classLabelVector.append(labels[listFromLine[-1]])
        index += 1
    return returnMat,classLabelVector

#归一化特征值
def autoNorm(dataset):
	minVals = dataset.min(0)
	#print(minVals)
	maxVals = dataset.max(0)
	#print(maxVals)
	ranges = maxVals - minVals
	#print(ranges)
	normDataSet = zeros(shape(dataset))
	#print(normDataSet)
	m = dataset.shape[0]
	#print(m)
	normDataSet = dataset - tile(minVals, (m,1))
	#print(tile(minVals, (m,1)))
	#print(normDataSet)
	normDataSet = normDataSet/tile(ranges, (m,1))
	#print(normDataSet)
	return normDataSet, ranges, minVals

#分类器针对约会网站的测试代码
def datingClassTest():
	hoRatio = 0.10
	datingDataMat, datingLabels = file2matrix('datingTestSet.txt')
	normMat, ranges, minVals = autoNorm(datingDataMat)
	m = normMat.shape[0]
	print(m)
	numTestVecs = int(m*hoRatio)
	print(numTestVecs)
	errorCount = 0.0
	for i in range(numTestVecs):
		classifierResult = classify0(normMat[i,:], normMat[numTestVecs:m,:], datingLabels[numTestVecs:m], 3)
		print("the classifier came back with: %d, the real answer is: %d" % (classifierResult, datingLabels[i]))
		if(classifierResult != datingLabels[i]):
			errorCount += 1.0
	print("the total error rate is: %f" % (errorCount/float(numTestVecs)))

#约会网站预测函数
def classifyPerson():
	resultList = ['not at all', 'in small doses', 'in large doses']
	percentTats = float(input("percentage of time spent playing video games?"))
	ffMiles = float(input("frequent flier miles earned per year?"))
	iceCream = float(input("liters of ice cream consumed per year"))
	datingDataMat, datingLabels = file2matrix('datingTestSet.txt')
	normMat, ranges, minVals = autoNorm(datingDataMat)
	inArr = array([ffMiles, percentTats, iceCream])
	classifierResult = classify0((inArr-minVals)/ranges, normMat, datingLabels, 3)
	print("You will probably like this person: ", resultList[classifierResult - 1])


















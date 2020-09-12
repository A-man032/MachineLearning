from numpy import *
import operator
from kNN import classify0
import os

#将图像转换成向量
def img2vector(filename):
	returnVect = zeros((1, 1024))
	fr = open(filename)
	for i in range(32):
		lineStr = fr.readline()
		for j in range(32):
			returnVect[0, 32*i+j] = int(lineStr[j])
	return returnVect

#数字识别系统的测试代码
def handwritingClassTest():
	hwLabels = []
	trainingFileList = os.listdir('trainingDigits')
	m = len(trainingFileList)
	trainingMat = zeros((m,1024))
	for i in range(m):
		fileNameStr = trainingFileList[i]
		fileStr = fileNameStr.split('.')[0]
		classNumStr = int(fileStr.split('_')[0]) #从文件名解析分类数字
		hwLabels.append(classNumStr)
		trainingMat[i,:] = img2vector('trainingDigits/%s' % fileNameStr)
	testFileList = os.listdir('testDigits')
	errorCount = 0.0
	mTest = len(testFileList)
	for i in range(mTest):
		fileNameStr = testFileList[i]
		fileStr = fileNameStr.split('.')[0]
		classNumStr = int(fileStr.split('_')[0])
		vectorUnderTest = img2vector('testDigits/%s' % fileNameStr)
		classifierResult = classify0(vectorUnderTest, trainingMat, hwLabels, 3)
		print("classfier came back with: %d, the real answer is: %d" % (classifierResult, classNumStr))
		if(classifierResult != classNumStr):
			errorCount += 1.0
	print("\nthe total number of errors is: %d" % errorCount)
	print("\nthe total error rate is: %f" % (errorCount/float(mTest)))

from bayes import *

'''
***************使用朴素贝叶斯过滤垃圾邮件***************
'''

'''
函数描述：输入一个大字符串，将其解析成字符串列表，且列表去掉小于等于2个字符的字符串

Parameters:
	bigString - 大字符串
Returns:
	[tok.lower() for tok in listOfTokens if len(tok) > 2] - 字符串列表，去掉长度小于等于2个字符的字符串
Author:
	YM
Date:
	2020-9-10
'''
def textParse(bigString):
	import re
	listOfTokens = re.split(r'\W', bigString)
	return [tok.lower() for tok in listOfTokens if len(tok) > 2]


'''
函数描述：对贝叶斯垃圾邮件分类器进行自动化处理，使用朴素贝叶斯进行交叉验证

Author:
	YM
Date:
	2020-9-10
'''
def spamTest():
	docList = []  #文档数据集
	classList = []  #类别列表
	fullText = []  
	for i in range(1,26):
		#导入文件夹spam和ham下的文本文件，并将其解析成词列表
		wordList = textParse(open('email/spam/%d.txt' % i).read())
		docList.append(wordList)  #追加
		fullText.extend(wordList)  #合并
		classList.append(1)  #分类是垃圾邮件
		wordList = textParse(open('email/ham/%d.txt' % i).read())
		docList.append(wordList)
		fullText.extend(wordList)
		classList.append(0)  #分类不是垃圾邮件
	vocabList = createVocabList(docList)  #构建词汇表
	trainingSet = list(range(50))  #整数列表，值从0到49（一共有50个样本）
	testSet = []
	for i in range(10):  #随机选取10个文件，随机数字对应的文档添加到测试集，同时从训练集中剔除
		randIndex = int(random.uniform(0, len(trainingSet)))
		testSet.append(trainingSet[randIndex])
		del(trainingSet[randIndex])
	trainMat = []  #训练集词向量矩阵
	trainClasses = []  #训练集分类结果向量
	for docIndex in trainingSet:
		trainMat.append(setOfWords2Vec(vocabList, docList[docIndex]))
		trainClasses.append(classList[docIndex])
	p0V, p1V, pSpam = trainNB0(array(trainMat), array(trainClasses))  #计算概率
	errorCount = 0
	for docIndex in testSet:  #遍历测试集，对每封邮件分类。如果邮件分类错误，错误数+1
		wordVector = setOfWords2Vec(vocabList, docList[docIndex])
		if classifyNB(array(wordVector), p0V, p1V, pSpam) != classList[docIndex]:  #分类错误
			errorCount += 1
	print("the error rate is: ", float(errorCount)/len(testSet))






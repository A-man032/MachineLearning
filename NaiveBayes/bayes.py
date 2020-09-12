from numpy import *


'''
*********使用朴素贝叶斯过滤网站的恶意留言*********
'''

'''
函数描述：一些实验样本
'''
def loadDataSet():
	postingList=[['my', 'dog', 'has', 'flea', 'problem', 'help', 'please'], \
				['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'], \
				['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'], \
				['stop', 'posting', 'stupid', 'worthless', 'garbage'], \
				['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'], \
				['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']]
	classVec = [0, 1, 0, 1, 0, 1] #1代表侮辱性文字，0代表正常言论
	return postingList,classVec

'''
函数描述：创建一个包含在所有文档中出现的不重复词的列表

Parameters:
	dataSet - 数据集
Returns:
	list(vocabSet) - 数据集中所有文档的全部不重复词汇列表
Author:
	YM
Date:
	2020-9-8
'''
def createVocabList(dataSet):
	vocabSet = set([])
	for document in dataSet:
		vocabSet = vocabSet | set(document) #创建两个集合的并集
	return list(vocabSet)

"""
函数描述：标记词汇表中的单词是否在文档中出现
		词集模型 - 每个词只能出现一次

Parameters:
	vocabList - 词汇表 
	inputSet - 某个文档
Returns:
	returnVec - 文档向量，向量的每一个元素为1或0，分别表示词汇表中的单词在输入文档中是否出现
Author:
	YM
Date:
	2020-9-8
"""
def setOfWords2Vec(vocabList, inputSet):
	returnVec = [0]*len(vocabList) #创建一个和词汇表等长的向量，全设为0
	for word in inputSet:
		if word in vocabList: #遍历文档中出现词汇表中的词，标记为1
			returnVec[vocabList.index(word)] = 1
		else:
			print("the word: %s is not in my Vocabulary!" % word)
	return returnVec


'''
函数描述：词袋模型 - 每个词可以出现多次(词集模型的改进版)
Parameters:
	vocabList - 词汇表 
	inputSet - 某个文档
'''
def bagOfWords2VecMN(vocabList, inputSet):
	returnVec = [0]*len(vocabList) #创建一个和词汇表等长的向量，全设为0
	for word in inputSet:
		if word in vocabList: #每遇到一个词，增加词向量中的对应值
			returnVec[vocabList.index(word)] += 1
		else:
			print("the word: %s is not in my Vocabulary!" % word)
	return returnVec


'''
函数描述：朴素贝叶斯分类器训练函数

Parameters:
	trainMatrix - 文档矩阵，其中每个向量标记词汇库中的词汇是否在文章中出现，出现标为1
	trainCategory - 向量，由每篇文档类别标签所构成
Returns:
	p0Vect - 非侮辱性文章中每个词出现的频率
	p1Vect - 侮辱性文章中每个词出现的频率
	pAbusive - 出现侮辱性文章的概率
Author:
	YM
Date:
	2020-9-8
'''
def trainNB0(trainMatrix, trainCategory):
	numTrainDocs = len(trainMatrix)  #文档矩阵的长度 = 文档数
	numWords = len(trainMatrix[0])  #词汇表长度
	pAbusive = sum(trainCategory)/float(numTrainDocs)  #文章具有侮辱性语义的概率 p(ci)

	#如果出现0概率则连乘=0，为降低这种影响，将分子初始化为1，分母初始化为2
	#p0Num = zeros(numWords); p1Num = zeros(numWords)  #初始化分子
	p0Num = ones(numWords); p1Num = ones(numWords)  #将所有词的出现数初始化为1
	#p0Denom = 0.0; p1Denom = 0.0  #初始化分母	
	p0Denom = 2.0; p1Denom = 2.0   #分母初始化为2

	for i in range(numTrainDocs):  #遍历训练集中的所有文档
		if trainCategory[i] == 1:  #文章中出现侮辱性语义
			p1Num += trainMatrix[i]  #侮辱性文章所有词出现的次数
			p1Denom += sum(trainMatrix[i])  #侮辱性文章中的总词数
		else:
			p0Num += trainMatrix[i]
			p0Denom += sum(trainMatrix[i])

	#为防止下溢出（太多太小的数相乘，四舍五入得0），对结果取自然对数，可以避免下溢出或浮点数舍入导致的错误
	#p1Vect = p1Num/p1Denom  #侮辱性文章每个词出现的概率
	#p0Vect = p0Num/p0Denom
	p1Vect = log(p1Num/p1Denom)  #词汇表中所有词在侮辱性文章中出现的概率 p(w|ci)
	p0Vect = log(p0Num/p0Denom)  #词汇表中所有词在非侮辱性文章中出现的概率

	return p0Vect,p1Vect,pAbusive


'''
函数描述：朴素贝叶斯分类函数

Parameters:
	vec2Classify - 要分类的向量(词汇库中标记测试文章中出现的词汇)
	p0Vect - 非侮辱性文章中每个词出现的频率
	p1Vect - 侮辱性文章中每个词出现的频率
	pClass1 - 出现侮辱性文章的概率
Returns:
	1 - 该文章具有侮辱性语义
	0 - 该文章不具有侮辱性语义
Author:
	YM
Date:
	2020-9-9
'''
def classifyNB(vec2Classify, p0Vect, p1Vect, pClass1):
	#sum(vec2Classify * p1Vect)在之前的函数中已经取过对数了，这里不用再取了
	p1 = sum(vec2Classify * p1Vect) + log(pClass1)  #numpy中*是两个向量对应元素相乘 p(w|ci)*p(ci)取对数后就是加和
	p0 = sum(vec2Classify * p0Vect) + log(1.0 - pClass1)
	if p1 > p0:  #具有侮辱语义
		return 1
	else:
		return 0


'''
函数描述：测试函数

Author:
	YM
Date:
	2020-9-9
'''
def testingNB():
	listOPosts, listClasses = loadDataSet()  #从函数中调入数据
	myVocabList = createVocabList(listOPosts)  #生成无重复的所有词汇列表
	trainMat = []
	for postinDoc in listOPosts:
		trainMat.append(setOfWords2Vec(myVocabList, postinDoc))  #trainMat - 矩阵，每个向量标记词汇库中的词汇是否在文章中出现，出现标为1
	p0V, p1V, pAb = trainNB0(array(trainMat), array(listClasses))  

	testEntry = ['love', 'my', 'dalmation']
	thisDoc = array(setOfWords2Vec(myVocabList, testEntry))  #thisDoc - 在词汇库中标记测试文章中出现的词汇
	print(testEntry, 'classified as: ', classifyNB(thisDoc, p0V, p1V, pAb))

	testEntry = ['stupid', 'garbage']
	thisDoc = array(setOfWords2Vec(myVocabList, testEntry))
	print(testEntry, 'classified as: ', classifyNB(thisDoc, p0V, p1V, pAb))



















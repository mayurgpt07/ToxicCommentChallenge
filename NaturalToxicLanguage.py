import pandas as pd
import numpy as np
from wordcloud import WordCloud 
import matplotlib.pyplot as plt
from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.svm import SVC
from sklearn.feature_extraction.text import TfidfVectorizer
from pattern.en import suggest
import re
from gensim.models import Word2Vec
from numba import jit
from scipy.sparse import hstack
from imblearn.under_sampling import NearMiss
from imblearn.over_sampling import SMOTE

def reduce_lengthening(text):
    pattern = re.compile(r"(.)\1{2,}")
    return pattern.sub(r"\1\1", text)


def top_tfidf_feats(row, features, top_n=20):
    topn_ids = np.argsort(row)[::-1][:top_n]
    top_feats = [(features[i], row[i]) for i in topn_ids]
    df = pd.DataFrame(top_feats)
    df.columns = ['feature', 'tfidf']
    return df

def featureEngineer(toxic_data):
	toxic_data['Comment_Text'] = toxic_data['Comments'].apply(lambda x: re.sub('\ |\!|\/|\;|\:|\=|\"|\:|\]|\[|\<|\>|\{|\}|\'|\?|\.|\,|\|',' ', str(x)))
	toxic_data['new_Comment_Text'] = toxic_data['Comment_Text'].apply(lambda x: re.sub('\s+',' ',str(x).strip().lower()))
	toxic_data['CommentTokenize'] = toxic_data['new_Comment_Text'].apply(lambda x: word_tokenize(x))
	sentenceList = []
	sentenceListTest = []
	for i in toxic_data['CommentTokenize']:
		wordList = ''	
		for k in i:
			if lemmatizer.lemmatize(k) not in stopwordsList:
				wordList = wordList + reduce_lengthening(lemmatizer.lemmatize(k)) + ' '
		sentenceList.append(wordList.strip().lower())
	#wordList.clear()

	toxic_data['RemovedStopWords'] = pd.Series(sentenceList)
	toxic_data['RemovedStopWords'] = toxic_data['RemovedStopWords'].apply(lambda x: re.sub('\s+', ' ', str(x).strip()))
	return toxic_data

def createVisualization(train_data, outputCol):
	VisualizationDataFrame = pd.DataFrame()
	outputList = []
	uniqueOutputCol = train_data[outputCol].unique()
	for j in uniqueOutputCol:
		interValue = np.where((train_data[outputCol] == j))
		outputList.append(np.count_nonzero(interValue))
	VisualizationDataFrame['Y Label'] = pd.Series(outputList)
	plt.hist(VisualizationDataFrame['Y Label'], color = 'green')
	plt.xlabel(outputCol, fontsize=10)
	plt.ylabel('Count of '+outputCol, fontsize=10)
	plt.show()

def createFeatures(train_data):
	toxic_data = pd.DataFrame()
	toxic_data['Comments'] = train_data.loc[:,'comment_text']
	print(toxic_data['Comments'].head(10))
	toxic_data['ToxicClassResult'] = train_data.loc[:, 'toxic']
	toxic_data['SevereToxicClassResult'] = train_data.loc[:, 'severe_toxic']
	toxic_data['ObsceneToxicClassResult'] = train_data.loc[:, 'obscene']
	toxic_data['ThreatToxicClassResult'] = train_data.loc[:, 'threat']
	toxic_data['InsultToxicClassResult'] = train_data.loc[:, 'insult']
	toxic_data['IdentityHateToxicClassResult'] = train_data.loc[:, 'identity_hate']
	toxic_data['NumberOfSentences'] = toxic_data['Comments'].apply(lambda x: len(re.findall('\n',str(x.strip())))+1)
	toxic_data['MeanLengthOfSentences'] = toxic_data['Comments'].apply(lambda x: np.mean([len(w) for w in x.strip().split("\n")]))
	toxic_data['MeanLengthOfWords'] = toxic_data['Comments'].apply(lambda x: np.mean([len(w) for w in x.strip().split(" ")]))
	toxic_data['NumberOfUniqueWords'] = toxic_data['Comments'].apply(lambda x: len(set(x.split())))
	toxic_data['numberOfWords'] = toxic_data['Comments'].apply(lambda x: len(x.split()))
	return toxic_data

def createFeaturesForTest(test_data):
	toxic_data = pd.DataFrame()
	toxic_data['Comments'] = test_data.loc[:,'comment_text']
	print(toxic_data['Comments'].head(10))
	toxic_data['NumberOfSentences'] = toxic_data['Comments'].apply(lambda x: len(re.findall('\n',str(x.strip())))+1)
	toxic_data['MeanLengthOfSentences'] = toxic_data['Comments'].apply(lambda x: np.mean([len(w) for w in x.strip().split("\n")]))
	toxic_data['MeanLengthOfWords'] = toxic_data['Comments'].apply(lambda x: np.mean([len(w) for w in x.strip().split(" ")]))
	toxic_data['NumberOfUniqueWords'] = toxic_data['Comments'].apply(lambda x: len(set(x.split())))
	toxic_data['numberOfWords'] = toxic_data['Comments'].apply(lambda x: len(x.split()))
	return toxic_data


train_data = pd.read_csv('./jigsaw-toxic-comment-classification-challenge/train.csv', sep = ',', header = 0)
#train_data['comment_text'].fillna('unknown', inplace = True)
test_data = pd.read_csv('./jigsaw-toxic-comment-classification-challenge/test.csv', sep = ',', header = 0)

stopwordsList = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

toxic_data_read = createFeatures(train_data)
#toxic_data_read.to_csv('IntermediateDataFrame.csv', sep = ',', header = True)
test_data_read = createFeaturesForTest(test_data)
#test_data_read.to_csv('IntermediateDataFrameTest.csv', sep = ',', header = True)
#toxic_data_read = pd.read_csv('./IntermediateDataFrame.csv', sep = ',', header = 0)

toxic_data_features = featureEngineer(toxic_data_read)
test_data_features = featureEngineer(test_data_read)
#test_data.to_csv('CompletedFeaturesTest.csv', sep = ',', header = True)
#toxic_data = featureEngineer(toxic_data_read)
#toxic_data.to_csv('CompletedFeatures.csv', sep = ',', header = True)
#toxic_data_test_newFeatures = createFeatures(toxic_data_read_test)
#toxic_data_test_newFeatures.to_csv('CompletedFeaturesTest.csv', sep = ',', header = True)

#toxic_data_features = pd.read_csv('./CompletedFeatures.csv', sep = ',', header = 0)

toxic_data = toxic_data_features.dropna(subset = ['RemovedStopWords'])
test_data = test_data_features.dropna(subset = ['RemovedStopWords'])
print('Shape of Test Data', test_data.shape)
print('Shape of Train Data', toxic_data.shape)
# plt.hist(toxic_data['ToxicClassResult'], color = 'green')
# plt.show()
# plt.hist(toxic_data['SevereToxicClassResult'], color = 'green')
# plt.show()
# plt.hist(toxic_data['ObsceneToxicClassResult'], color = 'green')
# plt.show()
# plt.hist(toxic_data['ThreatToxicClassResult'], color = 'green')
# plt.show()
# plt.hist(toxic_data['InsultToxicClassResult'], color = 'green')
# plt.show()
# plt.hist(toxic_data['IdentityHateToxicClassResult'], color = 'green')
# plt.show()
# empty_string = ''
# for i in toxic_data['RemovedStopWords']:
# 	empty_string = empty_string.strip() + ' ' + i.strip()

# print('Length of Total Comments', len(empty_string))

# wordcloud = WordCloud(width = 900, height = 900,
#                 background_color ='white',
#                 min_font_size = 10).generate(empty_string)
concatenatedComments = pd.concat([toxic_data['RemovedStopWords'], test_data['RemovedStopWords']])
vectorizer = TfidfVectorizer(min_df = 30,strip_accents = 'unicode', analyzer = 'word',token_pattern=r'\w{1,}',ngram_range = (1,3), stop_words = 'english', sublinear_tf = True, max_features = 40000)
multipleWordAnalyzer = vectorizer.fit(concatenatedComments)
train_ngrams = multipleWordAnalyzer.transform(toxic_data['RemovedStopWords'])
print('Shape of Train', train_ngrams.shape)
test_ngrams = multipleWordAnalyzer.transform(test_data['RemovedStopWords'])
print('Shape of Test', test_ngrams.shape)


wordVectorizer = TfidfVectorizer(strip_accents = 'unicode', analyzer = 'char', ngram_range = (2,6) , sublinear_tf = True, max_features = 10000)
charAnalyzer = wordVectorizer.fit(concatenatedComments)
train_1grams = charAnalyzer.transform(toxic_data['RemovedStopWords'])
test_1grams = charAnalyzer.transform(test_data['RemovedStopWords'])

# print(type(train_ngrams))
# print(np.shape(train_ngrams))
# print(np.ndim(train_ngrams))

trainingColumns = ['NumberOfSentences', 'NumberOfUniqueWords', 'numberOfWords', 'MeanLengthOfSentences']
testingColumns = ['ToxicClassResult','SevereToxicClassResult','ObsceneToxicClassResult','ThreatToxicClassResult','InsultToxicClassResult','IdentityHateToxicClassResult']

# print(toxic_data[trainingColumns].head(5))
# print(type(train_ngrams))

# print(type(train_1grams))
trainingFeatures = hstack((train_ngrams, train_1grams)).tocsr()
testingFeatures = hstack((test_ngrams, test_1grams)).tocsr()
#trainingFeatureDataFrame = pd.DataFrame(trainingFeatures.toarray())
#testingFeatureDataFrame = pd.DataFrame(testingFeatures.toarray())

#print(trainingFeatureDataFrame.shape)

X, y = trainingFeatures, toxic_data[testingColumns]

X_train, X_test, Y_train, Y_test = model_selection.train_test_split(X, y, test_size = 0.33)
# X_train = X_train.dropna()
# Y_train_underSampled = pd.DataFrame()
# sm = SMOTE(random_state = 2)
# for j in testingColumns:

# 	X_train_underSampledList, Y_train_underSampled_List = sm.fit_sample(X_train, Y_train[j].ravel())
# 	Y_train_underSampled[j] = pd.Series(Y_train_underSampled_List)
# 	#X_train_underSampled = X_train_underSampled.dropna()
# 	print("After Undersampling, counts of label "+j+": {}".format(sum(Y_train_underSampled[j] == 1)))
# 	print("After Undersampling, counts of label "+j+": {}".format(sum(Y_train_underSampled[j] == 0)))
# 	SupportVectorModel = SVC(kernel = 'rbf', C = 0.1, cache_size = 10000.0, decision_function_shape = 'ovo')
# 	LogisticModel = LogisticRegression(penalty = 'elasticnet', C = 0.01, solver = 'saga', max_iter = 4000, l1_ratio = 0.5)
# 	FittedSVModel = SupportVectorModel.fit(X_train_underSampledList, Y_train_underSampled_List)
# 	crossValidationScoreforSV = cross_val_score(SupportVectorModel, X_train_underSampledList, Y_train_underSampled_List, cv = 5)
# 	crossValidationScoreforLogstic = cross_val_score(LogisticModel, X_train_underSampledList, Y_train_underSampled_List, cv = 5)
# 	print('Cross Validation Score Support Vector', crossValidationScoreforSV, np.mean(crossValidationScoreforSV))
# 	print('Cross Validation Score Logistic Model', crossValidationScoreforLogstic, np.mean(crossValidationScoreforLogstic))

#print(Y_train.shape)
# print("Before Undersampling, counts of label 'Toxic Class': {}".format(sum(Y_train['ToxicClassResult'] == 1)))
# print("Before Undersampling, counts of label 'Severe Toxic Class': {}".format(sum(Y_train['SevereToxicClassResult'] == 1)))
# print("Before Undersampling, counts of label 'Obscene Toxic Class': {}".format(sum(Y_train['ObsceneToxicClassResult'] == 1)))
# print("Before Undersampling, counts of label 'Threat Class': {}".format(sum(Y_train['ThreatToxicClassResult'] == 1)))
# print("Before Undersampling, counts of label 'Insult Class': {}".format(sum(Y_train['InsultToxicClassResult'] == 1)))
# print("Before Undersampling, counts of label 'Identity Hate Class': {}".format(sum(Y_train['IdentityHateToxicClassResult'] == 1)))


for testColumns in testingColumns:
	SupportVectorModel = LogisticRegression(C=0.1, solver='sag')#SVC(kernel = 'rbf', C = 0.1, cache_size = 10000.0, decision_function_shape = 'ovo', probability = True)
	FittedSVModel = SupportVectorModel.fit(X_train, Y_train[testColumns])
	crossValidationScoreforSV = cross_val_score(SupportVectorModel, X_test, Y_test[testColumns], cv = 3	, scoring = 'roc_auc')
	test_data[testColumns] = FittedSVModel.predict_proba(testingFeatures)[:, 1]
	print(FittedSVModel.predict_proba(testingFeatures))
	print('Cross Validation Score Support Vector', crossValidationScoreforSV, np.mean(crossValidationScoreforSV))

test_data.to_csv('PredictedValue.csv', sep = ',', header = True)
# plot the WordCloud image                        
# plt.figure(figsize = (8, 8), facecolor = None)
# plt.imshow(wordcloud)
# plt.axis("off") 
# plt.tight_layout(pad = 0)
# plt.show()

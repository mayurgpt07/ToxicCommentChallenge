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



train_data = pd.read_csv('./jigsaw-toxic-comment-classification-challenge/train.csv', sep = ',', header = 0)
#train_data['comment_text'].fillna('unknown', inplace = True)
#test_data = pd.read_csv('./jigsaw-toxic-comment-classification-challenge/test.csv', sep = ',', header = 0)

stopwordsList = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

#toxic_data_read = createFeatures(train_data)
#toxic_data_read.to_csv('IntermediateDataFrame.csv', sep = ',', header = True)
toxic_data_read = pd.read_csv('./IntermediateDataFrame.csv', sep = ',', header = 0)
#toxic_data_read_test = featureEngineer(test_data)
#toxic_data_read_test.to_csv('IntermediateDataFrameTest.csv', sep = ',', header = True)

#toxic_data = featureEngineer(toxic_data_read)
#toxic_data.to_csv('CompletedFeatures.csv', sep = ',', header = True)
#toxic_data_test_newFeatures = createFeatures(toxic_data_read_test)
#toxic_data_test_newFeatures.to_csv('CompletedFeaturesTest.csv', sep = ',', header = True)

toxic_data_features = pd.read_csv('./CompletedFeatures.csv', sep = ',', header = 0)

toxic_data = toxic_data_features.sample(n=50000).dropna(subset = ['RemovedStopWords'])
# empty_string = ''
# for i in toxic_data['RemovedStopWords']:
# 	empty_string = empty_string.strip() + ' ' + i.strip()

# print('Length of Total Comments', len(empty_string))

# wordcloud = WordCloud(width = 900, height = 900,
#                 background_color ='white',
#                 min_font_size = 10).generate(empty_string)

vectorizer = TfidfVectorizer(min_df = 30,strip_accents = 'unicode', analyzer = 'word',token_pattern=r'\w{1,}',ngram_range = (1,3), stop_words = 'english', sublinear_tf = True, max_features = 40000)
X = vectorizer.fit_transform(toxic_data['RemovedStopWords'])
print('Length of features', len(vectorizer.get_feature_names()))
train_ngrams = vectorizer.transform(toxic_data['RemovedStopWords'])

wordVectorizer = TfidfVectorizer(strip_accents = 'unicode', analyzer = 'char', ngram_range = (2,6) , stop_words = 'english', sublinear_tf = True, max_features = 10000)
Y = wordVectorizer.fit_transform(toxic_data['RemovedStopWords'])
print('Length of one word features', len(wordVectorizer.get_feature_names()))
train_1grams = wordVectorizer.transform(toxic_data['RemovedStopWords'])

# print(type(train_ngrams))
# print(np.shape(train_ngrams))
# print(np.ndim(train_ngrams))

trainingColumns = ['NumberOfSentences', 'NumberOfUniqueWords', 'numberOfWords', 'MeanLengthOfSentences']
testingColumns = ['ToxicClassResult','SevereToxicClassResult','ObsceneToxicClassResult','ThreatToxicClassResult','InsultToxicClassResult','IdentityHateToxicClassResult']

# print(toxic_data[trainingColumns].head(5))
# print(type(train_ngrams))

# print(type(train_1grams))
trainingFeatures = hstack((toxic_data[trainingColumns],train_ngrams, train_1grams)).tocsr()
trainingFeatureDataFrame = pd.DataFrame(trainingFeatures.toarray())

X, y = trainingFeatureDataFrame, toxic_data[testingColumns]

X_train, X_test, Y_train, Y_test = model_selection.train_test_split(X, y, test_size = 0.33)

# LogisticModel = LogisticRegression(C = 0.1, penalty = 'elasticnet', solver = 'saga', l1_ratio = 0.5)
# FitteddLogistic = LogisticModel.fit(X_train, Y_train)
# crossValidationScore = cross_val_score(LogisticModel, X_train, Y_train, cv = 5, scoring = 'roc_auc')

# print('crossValidationScore', crossValidationScore, np.mean(crossValidationScore))
for testColumns in testingColumns:
	SupportVectorModel = SVC(kernel = 'rbf', C = 0.1, cache_size = 10000.0, decision_function_shape = 'ovo')
	FittedSVModel = SupportVectorModel.fit(X_train, Y_train[testColumns])
	crossValidationScoreforSV = cross_val_score(SupportVectorModel, X_train, Y_train[testColumns], cv = 5)

	print('Cross Validation Score Support Vector', crossValidationScoreforSV, np.mean(crossValidationScoreforSV))

# plot the WordCloud image                        
# plt.figure(figsize = (8, 8), facecolor = None)
# plt.imshow(wordcloud)
# plt.axis("off") 
# plt.tight_layout(pad = 0)
# plt.show()

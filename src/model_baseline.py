from Data_loader_nlp import DataLoaderUtil
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
import nltk
nltk.download('stopwords')
nltk.download('punkt')
from nltk.tokenize import word_tokenize
import numpy as np
import pandas as pd
import seaborn as sn
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
dataloader_util = DataLoaderUtil()
Test_labels,Test_data=dataloader_util.get_data_test()
Training_labels,Training_data=dataloader_util.get_data_training()
Validation_labels,Validation_data=dataloader_util.get_data_validation()

text_clf = Pipeline([('vect', CountVectorizer(stop_words='english')), ('tfidf', TfidfTransformer()), ('clf', MultinomialNB())])

## TFID transformer
# tokenization, lowercasing 

stop_words=stopwords.words('english')
stop_words.remove('not')
training_without_stop = []
for sentence in Training_data:
    words = sentence.split(" ")
    filtered_words = [word.lower() for word in words if not word in stop_words]
    training_without_stop.append(" ".join(filtered_words))

#training_data_tokens = word_tokenize(Training_data)
#training_without_stop = [word for word in training_data_tokens if not word in stop_words]


count_vectorizer = CountVectorizer(lowercase=True,stop_words=None)
training_data_count = count_vectorizer.fit_transform(training_without_stop)

tfidtransformer = TfidfTransformer()
training_data_tfidf=tfidtransformer.fit_transform(training_data_count)
classifier = MultinomialNB()
training_data_fitted= classifier.fit(training_data_tfidf, Training_labels)

############### testing the performance
test_data_count = count_vectorizer.transform(Test_data)
test_data_tfidf=tfidtransformer.transform(test_data_count)
predicted_labels = classifier.predict(test_data_tfidf)

accuracy= np.mean(predicted_labels == Test_labels)
print(accuracy)

## confusion matrix
def vis(conf_mat, labels):
    """Prettify the confusion matrix."""
    df_cm = pd.DataFrame(conf_mat, index=[i for i in labels], columns=[i for i in labels])
    plt.figure(figsize=(10, 7))
    plt.title('Confusion matrix')
    sn.heatmap(df_cm, annot=True, fmt='g')
    plt.show()


conf_matrix = confusion_matrix(Test_labels, predicted_labels)
vis(conf_matrix, set(Test_labels))


'''
## stemming 

import nltk
nltk.download()
from nltk.stem.snowball import SnowballStemmer
stemmer = SnowballStemmer("english", ignore_stopwords=True)
class StemmedCountVectorizer(CountVectorizer):
    def build_analyzer(self):
        analyzer = super(StemmedCountVectorizer, self).build_analyzer()
        return lambda doc: ([stemmer.stem(w) for w in analyzer(doc)])
stemmed_count_vect = StemmedCountVectorizer(stop_words='english')
text_mnb_stemmed = Pipeline([('vect', stemmed_count_vect),('tfidf', TfidfTransformer()), ('mnb', MultinomialNB(fit_prior=False))])
text_mnb_stemmed = text_mnb_stemmed.fit(twenty_train.data, twenty_train.target)
predicted_mnb_stemmed = text_mnb_stemmed.predict(twenty_test.data)
np.mean(predicted_mnb_stemmed == twenty_test.target)


'''
'''
tfidf_vec = TfidfVectorizer(preprocessor=lambda x: x, tokenizer=lambda x: x)
tfIdfVectorizer=TfidfVectorizer(use_idf=True)
tfIdf = tfIdfVectorizer.fit_transform(dataset)
df = pd.DataFrame(tfIdf[0].T.todense(), index=tfIdfVectorizer.get_feature_names(), columns=["TF-IDF"])
tfidf_df = pd.DataFrame(tfidf_vector.toarray(), index=text_titles, columns=tfidf_vectorizer.get_feature_names())

df = df.sort_values('TF-IDF', ascending=False)
print (df.head(25))

features.append(('tfidf', tfidf_vec))
vec = FeatureUnion(features)
classifier = Pipeline([('tfidf', TfidfVectorizer()), ('cls', MultinomialNB())])
## or [('vect', CountVectorizer()),('tfidf', TfidfTransformer())

classifier.fit(training_data, training_data_target)
'''
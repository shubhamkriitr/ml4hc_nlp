from Data_loader_nlp import DataLoaderUtil
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
import nltk
import re
nltk.download('stopwords')
nltk.download('punkt')
from nltk.tokenize import word_tokenize
import numpy as np
import pandas as pd
import seaborn as sn
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
from nltk.corpus import stopwords

from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix
#text_clf = Pipeline([('vect', CountVectorizer(stop_words='english')), ('tfidf', TfidfTransformer()), ('clf', MultinomialNB())])

## TFID transformer
# tokenization, lowercasing 
class preprocess_data():
    def __init__(self):
        stop_words_=stopwords.words('english')
        stop_words_.remove('not')
        self.stop_words = stop_words_
    def removing_stop_word(self,data):
        data_without_stop = []
        for sentence in data: #Training_data
            words = sentence.split(" ")
            filtered_words = [word.lower() for word in words if not word in self.stop_words]
            data_without_stop.append(" ".join(filtered_words))
        return data_without_stop
    def remove_punctuation(self,data):
        remove_punctuation_besides_numbers = r'(?<!\d)[.,;:](?!\d)'
        data_without_punctuation=[]
        for line in range(len(data)):
            without_punctuation = re.sub(remove_punctuation_besides_numbers, "", data[line], 0)
            without_punctuation=without_punctuation.strip()
            data_without_punctuation.append(without_punctuation)
        return data_without_punctuation
class baseline_model():   
    #training_data_tokens = word_tokenize(Training_data)
    #training_without_stop = [word for word in training_data_tokens if not word in stop_words]
    def __init__(self, vect_max_df, vect_max_features,vect_ngram_range,tfidf_norm, tfidf_use_idf,clf_alpha):
        self.pipeline = Pipeline([('vect', CountVectorizer(lowercase=True,stop_words=None,max_df=vect_max_df,\
            max_features=vect_max_features,ngram_range=vect_ngram_range)),\
            ('tfidf', TfidfTransformer(norm=tfidf_norm,use_idf=tfidf_use_idf)), ('clf', MultinomialNB(alpha=clf_alpha))])
        self.model = self.pipeline
        print(self.pipeline.get_params()) 
    def get_pipeline(self):
        return self.pipeline
    def tuning_parameters(self,test_data,test_data_labels):
        parameters= {
            "vect__max_df": (0.5,0.75,1),
            'vect__max_features': (None, 5000, 10000, 50000),
            "vect__ngram_range": ((1, 1), (1, 2)),  # unigrams or bigrams
            'tfidf__use_idf': (True, False),
            'tfidf__norm': ('l1', 'l2'),
            "clf__alpha": (0.00001, 0.000001),
            #'clf__class_prior': [None, 'balanced'],
            #'clf__n_jobs': [-1]
            # "clf__fit_prior": [True, False]
        }
        grid_search = GridSearchCV(self.pipeline, parameters, n_jobs=-1, scoring='f1_weighted',verbose=1,cv=5)
        print("parameters:")
        print(parameters)
        grid_search.fit(test_data, test_data_labels)
        print("Best parameters set:")
        best_parameters = grid_search.best_estimator_.get_params()
        for param_name in sorted(parameters.keys()):
            print("\t%s: %r" % (param_name, best_parameters[param_name]))
        self.model=grid_search
        return grid_search, best_parameters
    
    def fit(self,data,data_labels):
        return self.model.fit(data,data_labels)

    def predict(self,data):
        return self.model.predict(data)

    def metrics_score(self,true_data_labels,predicted_data_labels):
        accuracy=accuracy_score(predicted_data_labels, true_data_labels)
        print(accuracy)

        f1=f1_score(predicted_data_labels, true_data_labels,average='weighted')
        print(f1)

        conf_matrix = confusion_matrix(true_data_labels, predicted_data_labels)
        self.plotting_confusion_matrix(conf_matrix, set(true_data_labels))
        print('Confusion Matrix : \n' + str(confusion_matrix(true_data_labels,predicted_data_labels)))
        return accuracy,f1,conf_matrix
    ##plotting confusion matrix
    def plotting_confusion_matrix(self,conf_mat, labels):
        df_cm = pd.DataFrame(conf_mat, index=[i for i in labels], columns=[y for y in labels])
        plt.figure(figsize=(10, 7))
        plt.title('Confusion matrix')
        sn.heatmap(df_cm, annot=True, fmt='g')
        plt.show()

if __name__ == "__main__":
    # loading the data 
    dataloader_util = DataLoaderUtil()
    Test_labels,Test_data_raw=dataloader_util.get_data_test()
    Training_labels,Training_data_raw=dataloader_util.get_data_training()
    Validation_labels,Validation_data_raw=dataloader_util.get_data_validation()
    # preprocessing the data
    Data_preprocessor = preprocess_data()
    Test_data = Data_preprocessor.removing_stop_word(Test_data_raw)
    Test_data = Data_preprocessor.remove_punctuation(Test_data)

    Training_data = Data_preprocessor.removing_stop_word(Training_data_raw)
    Training_data = Data_preprocessor.remove_punctuation(Training_data)

    Validation_data = Data_preprocessor.removing_stop_word(Validation_data_raw)
    Validation_data = Data_preprocessor.remove_punctuation(Validation_data)
    # finding the best hyperparameters
    
    # creating and tunning the model
    '''
    Model_baseline = baseline_model(vect_max_df=1,vect_max_features=None,vect_ngram_range=(1,1),tfidf_norm="l2",tfidf_use_idf = True,clf_alpha=1)
    best_model,best_params = Model_baseline.tuning_parameters(Training_data,Training_labels)
    validation_predictions=Model_baseline.predict(Validation_data)
    accuracy,f1,conf_matrix = Model_baseline.metrics_score(Validation_labels,validation_predictions)
    '''
    # LIST OF THE BEST HYPERPARAMETERS
    # tfidf_norm = l2
    # use_idf = False
    # vect_max_df = 0.5
    # vect_max_features= 50000
    #vect_ngram_range= (1,2)
    # clf_alpha = 1e-05  
    # clf_class_prior = None
    # accuracy 0.7941 

    # training the model 
    Best_pipeline= baseline_model(vect_max_df=0.5,vect_max_features= 50000,vect_ngram_range=(1,2),tfidf_norm="l2",tfidf_use_idf = False,clf_alpha=0.00001)
    Best_pipeline.fit(Training_data, Training_labels)

    # getting validation score
    validation_predictions=Best_pipeline.predict(Validation_data)
    accuracy_val,f1_val,conf_matrix_val = Best_pipeline.metrics_score(Validation_labels,validation_predictions)

    # getting scores on the testing dataset
    test_predictions=Best_pipeline.predict(Test_data)
    accuracy_,f1,conf_matrix_ = Best_pipeline.metrics_score(Test_labels,test_predictions)
    


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
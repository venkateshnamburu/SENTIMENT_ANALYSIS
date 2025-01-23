import sqlite3
import pandas as pd
import numpy as np
import nltk
import string
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfTransformer, TfidfVectorizer, CountVectorizer
import sklearn.metrics as metrics
from sklearn.metrics import confusion_matrix, roc_curve, auc, precision_score, recall_score, f1_score
from nltk.stem.porter import PorterStemmer
import string
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from gensim.models import Word2Vec, KeyedVectors
import pickle
import warnings
warnings.filterwarnings("ignore")
from sklearn import datasets, neighbors
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from collections import Counter
from matplotlib.colors import ListedColormap
#import scikitplot.metrics as sciplot
from sklearn.metrics import accuracy_score
import math
import nltk
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
#nltk.download('stopwords')
import re
from nltk.stem.snowball import SnowballStemmer


# ##### Use below link to download the dataset
# ##### https://www.kaggle.com/datasets/snap/amazon-fine-food-reviews

# #### The immediate code block below does the following things :
# 
# 1. Load the Amazon dataset.
# 2. Classify the reviews initially based on their score rating and give them a 'Positve' or a 'Negative' tag.
# 3. Remove duplicate/redundant datas.
# 4. Get an idea of how much percentage data were actually duplicates.
# 5. Plot a histogram which will display the distribution of the number of positive and negative reviews after de-duplication.
# 
# ###### NOTE : If we dont' clean the data and feed them to an ML system, it basically means we are throwing in a lot of garbage data to the ML system. If we give it garbage, it will give us garbage back. So it's utmost important to clean the data before proceeding.
'''
def data_collection(df):
    code write
    return df_collected

def data_cleaning(df_collected):

    return df_cleaned

def data_preprocessing(df_cleaned):

    return df_processed
def convert_text_to_numeric(df_pre_processed):


def train_test_split():

def model_training():


def predicting_on_real_time_data():
'''

def Process(mode,user_input=None):
    if mode=="train":
        print(" Training is started ... ")
        filtered_data=pd.read_csv("Reviews.csv")

        #Give reviews with Score > 3 a 'Positive' tag, and reviews with a score < 3 a 'Negative' tag.
        filtered_data['SentimentPolarity'] = filtered_data['Score'].apply(lambda x : 'Positive' if x > 3 else 'Negative')
        filtered_data['Class_Labels'] = filtered_data['SentimentPolarity'].apply(lambda x : 1 if x == 'Positive' else 0)

        print("The number of positive and negative reviews before the removal of duplicate data.")
        print(filtered_data["SentimentPolarity"].value_counts())


        #Removing duplicate entries based on past knowledge.
        filtered_duplicates=filtered_data.drop_duplicates(subset={"UserId","ProfileName","Time","Text"}, keep='first', inplace=True)


        print("The number of positive and negative reviews after the removal of duplicate data.")
        print(filtered_data["SentimentPolarity"].value_counts())


        #Removing the entries where HelpfulnessNumerator > HelpfulnessDenominator.
        final_data=filtered_data[filtered_data.HelpfulnessNumerator <= filtered_data.HelpfulnessDenominator]

        final_data["SentimentPolarity"].value_counts()


        # #### In this code block :
        # 
        # 1. I am creating a copy of the final_data dataset called 'sampled_dataset' by dropping the unwanted columns that we don't need for this problem.
        # 2. Sorting the data according to time, such that the oldest reviews are displayed at the top and the latest reviews are displayed at the bottom.
        # 3. Displaying information about the number of postive and negative reviews in the sampled dataset, using a Histogram.


        #Dropping unwanted columns for now.
        sampled_dataset=final_data.drop(labels=['Id','ProductId', 'UserId', 'Score', 'ProfileName','HelpfulnessNumerator', 'HelpfulnessDenominator','Summary'], axis=1)
        print("The shape of the sampled dataset after dropping unwanted columns : ", sampled_dataset.shape)
    

        #Sorting data according to Time in ascending order => Time Based Splitting Step 1.
        sampled_dataset=sampled_dataset.sort_values('Time', axis=0, ascending=False, inplace=False, kind='quicksort', na_position='last')
        sampled_dataset = sampled_dataset.reset_index()
        sampled_dataset=sampled_dataset.drop(labels=['index'], axis=1)


        #Display distribution of Postive and Negative reviews in a bar graph
        #sampled_dataset["SentimentPolarity"].value_counts().plot(kind='bar',color=['green','red'],title='Distribution Of Positive and Negative reviews after De-Duplication.',figsize=(5,5))


        # #### In this code block :
        # 
        # 1. We define two functions which will remove the HTML tags and punctuations from each review.
        # 2. At the end of this code block, each review will contain texts which will only contain alphabetical strings. 
        # 3. We will apply techniques such as stemming and stopwords removal.
        # 3. We will create two columns in the sampled dataset - 'CleanedText' and 'RemovedHTML'.
        # 4. 'CleanedText' column will basically contain the data corpus after stemming the each reviews and removing stopwords from each review. We will use this for our Bag of Word model.
        # 5. 'RemovedHTML' column will contain the data corpus from which only the HTML tags and punctuations are removed. We will use this column for our TF-IDF model, Average Word2Vec model and TF-IDF weighted average Word2Vec model.
        # 6. Store the final table in a dataset called 'sampled_dataset' for future use.

        '''Data Cleaning Stage. Clean each review from the sampled Amazon Dataset.'''
        #Data Cleaning Stage. Clean each review from the sampled Amazon Dataset

        #Function to clean html tags from a sentence
        def removeHtml(sentence): 
            pattern = re.compile('<.*?>')
            cleaned_text = re.sub(pattern,' ',sentence)
            return cleaned_text


        #Function to keep only words containing letters A-Z and a-z. This will remove all punctuations, special characters etc.
        def removePunctuations(sentence):
            cleaned_text  = re.sub('[^a-zA-Z]',' ',sentence)
            return cleaned_text


        #Stemming and stopwords removal
        sno = SnowballStemmer(language='english')

        #Removing the word 'not' from stopwords
        default_stopwords = set(stopwords.words('english'))
        remove_not = set(['not'])
        custom_stopwords = default_stopwords - remove_not


        #Building a data corpus by removing all stopwords except 'not'. Because 'not' can be an important estimator to differentiate between positive and negative reviews.    
        count=0                   #Iterator to iterate through the list of reviews and check if a given review belongs to the positive or negative class
        string=' '    
        data_corpus=[]
        all_positive_words=[] #Store all the relevant words from Positive reviews
        all_negative_words=[] #Store all the relevant words from Negative reviews
        stemed_word=''
        for review in sampled_dataset['Text'].values:
            filtered_sentence=[]
            sentence=removeHtml(review) #Remove HTMl tags
            for word in sentence.split():
                for cleaned_words in removePunctuations(word).split():
                    if((cleaned_words.isalpha()) & (len(cleaned_words)>2)): #Checking if a word consists of only alphabets + word length is greater than 2.    
                        if(cleaned_words.lower() not in custom_stopwords):
                            stemed_word=(sno.stem(cleaned_words.lower()))
                            filtered_sentence.append(stemed_word)
                            if (sampled_dataset['SentimentPolarity'].values)[count] == 'Positive': 
                                all_positive_words.append(stemed_word) #List of all the relevant words from Positive reviews
                            if(sampled_dataset['SentimentPolarity'].values)[count] == 'Negative':
                                all_negative_words.append(stemed_word) #List of all the relevant words from Negative reviews
                        else:
                            continue
                    else:
                        continue 
            string = " ".join(filtered_sentence) #Final string of cleaned words    
            data_corpus.append(string) #Data corpus contaning cleaned reviews from the whole dataset
            count+=1
        
        print("The length of the data corpus is : {}".format(len(data_corpus)))

        #Adding a column of CleanedText to the table final which stores the data_corpus after pre-processing the reviews 
        sampled_dataset['CleanedText']=data_corpus 

            
        # Finding most frequently occuring Positive and Negative words 
        freq_positive=nltk.FreqDist(all_positive_words)
        freq_negative=nltk.FreqDist(all_negative_words)
        print("Most Common Positive Words : ",freq_positive.most_common(20))
        print("Most Common Negative Words : ",freq_negative.most_common(20))

        sampled_dataset = sampled_dataset[['Time','CleanedText','Class_Labels']]
        print(sampled_dataset.shape)

        #sampled_dataset['Class_Labels'].value_counts()


        X = sampled_dataset['CleanedText']
        y = sampled_dataset['Class_Labels']

        split = math.floor(0.8*len(X))
        X_train = X[0:split,] ; y_train = y[0:split,]

        X_test = X[split:,] ; y_test = y[split:,]

        # print(X_train.shape)
        # print(X_test.shape)
        # print(y_train.shape)
        # print(y_test.shape)

        #Initializing the BOW constructor
        cv_object = CountVectorizer().fit(X_train)

        #Creating the BOW matrix from cleaned data corpus. Only 'not' is preserved from stopwords. This is done for both train and test Vectors.
        #print("\nCreating the BOW vectors using the cleaned corpus")
        X_train_vectors = cv_object.transform(X_train)
        X_test_vectors = cv_object.transform(X_test)

        model = MultinomialNB()

        # Train the model using the training sets
        model.fit(X_train_vectors,y_train)

        #Predict the response for test dataset
        y_pred = model.predict(X_test_vectors)

        # Save the CountVectorizer and the model to pickle files
        with open('count_vectorizer.pkl', 'wb') as file:
            pickle.dump(cv_object, file)
        with open('sentiment_model.pkl', 'wb') as file:
            pickle.dump(model, file)

        # Model Accuracy, how often is the classifier correct?
        print("Accuracy:",metrics.accuracy_score(y_test, y_pred))

    elif mode == 'predict' and user_input is not None:
            print(" Predicting the user input .....")
            # Load the CountVectorizer and the model from pickle files
            with open('count_vectorizer.pkl', 'rb') as file:
                cv_object = pickle.load(file)
            with open('sentiment_model.pkl', 'rb') as file:
                model = pickle.load(file)
            # Transform user input using the loaded CountVectorizer
            user_input_vector = cv_object.transform([user_input])

            # Predict sentiment
            prediction = model.predict(user_input_vector)[0]
            print("Prediction :Result will display in streamlit page")
            return prediction

    else:
        return None 
if __name__ == "__main__":
    mode = "train"
    Process(mode)



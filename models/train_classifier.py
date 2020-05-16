#Import Libraries

import sys
import pandas as pd
import numpy as np
import nltk
import re
import pickle

nltk.download(['punkt', 'wordnet'])

from sqlalchemy import create_engine
from nltk.tokenize import word_tokenize, RegexpTokenizer
from nltk.stem import WordNetLemmatizer
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.metrics import classification_report,precision_recall_fscore_support
from sklearn.tree import DecisionTreeClassifier


def load_data(database_filepath):
    
    '''
    Load data from database using database filepath
    
    input: take input as a string of file path
    
    output:
    
    X: message column
    y: categories column
    category name: names of the categories
    '''

    engine= create_engine('sqlite:///{}'.format(database_filepath))
    df= pd.read_sql_table('DisasterResponse', engine)
    
    X= df['message']
    y= df.iloc[:, 4:]
    category_names= y.columns
    
    return X, y, category_names

#tokenizing and normalising the message column
def tokenize(text):
    '''
    We will use tokenize function to process our data
    
    input: strings in sentence format
    
    output: list of tokenize words from message'''
    
    
    url_regex = r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    
    detected_urls= re.findall(url_regex, text)
    
    for url in detected_urls:
        text= text.replace(url, 'urlplaceholder')
        
    
    tokens= word_tokenize(text)
    lemmatizer= WordNetLemmatizer()
    
    
    clean_tokens=[]
    for tok in tokens:
        
        clean_tok= lemmatizer.lemmatize(tok).lower().strip()
        
        clean_tokens.append(clean_tok)
        
    return clean_tokens

# Buliding Machine Learning Pipeline
def build_model():
    
    '''
    A machinelearning pipeline takes message column as input and 
    outputclassification results on the other 36 categories in the dataset.
    
    
    Input: none
    output: a model that predict the message columns token on the basis of 36 other categories
    '''

    
    pipeline = Pipeline([
        ('features', FeatureUnion([
            ('text_pipeline', Pipeline([
                ('vect', CountVectorizer(tokenizer=tokenize)),
                ('tfidf', TfidfTransformer())
            ]))
        ])),
        
        ('clf', MultiOutputClassifier(RandomForestClassifier()))
    ])

    parameters = {
        'features__text_pipeline__vect__ngram_range': ((1, 1), (1, 2)),
        'clf__estimator__min_samples_split': [2, 3, 4]}
    
    
    
    cv = GridSearchCV(pipeline, param_grid=parameters)

    return cv


# Evaluating the model
def evaluate_model(model, X_test, Y_test, category_names):
    
    ''' 
    Evaluate the model performance by f1 score, precision and recall
    '''

    y_pred = model.predict(X_test)

    # print scores
    print(classification_report(Y_test.iloc[:,1:].values, np.array([x[1:] for x in y_pred]), target_names=category_names))


def save_model(model, model_filepath):
    
    '''Input:
    model: a ML model
    model_filepath: the file path that the model will be saved
    Returns:
    None'''
    
    
    pickle.dump(model, open(model_filepath, 'wb'))
    

def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
       
    
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()
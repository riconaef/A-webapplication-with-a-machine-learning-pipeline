import sys
import numpy as np
import pandas as pd
import re
import pickle
from sqlalchemy import create_engine

from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
#from sklearn.tree import DecisionTreeClassifier

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.model_selection import GridSearchCV


def load_data(database_filepath):
    """
    Input: A filepath to the sql database
    Output: An X and Y array for the machine learning pipeline and the names of the category columns
    """
    engine = create_engine('sqlite:///'+str(database_filepath))
    df = pd.read_sql_table(database_filepath, engine)
    X = df['message'].values #extract the X values
    Y = df.drop(columns=['id','message','original','genre']).values #extract the Y values
    
    category_names = list(df.drop(columns=['id','message','original','genre']).columns) #save the category names
    
    return X, Y, category_names


def tokenize(text):
    """
    Input: A text string.
    Output: A tokenized text string
    
    The processing includes following steps:
        1.  Remove all signs (points, commas, ...)
        2.  Split the string into the different words.
        3.  Lemmatize the words: Group the words into the different roots.
    
    """
    text = re.sub(r"[^a-zA-Z0-9]", " ",text)
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()
    
    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens


def build_model():
    """
    Input: None
    Output: A machine learning pipeline. This includes following features:
        1.  A vectorizer for counting the words
        2.  A Tfidf Transformer
        3.  A machine learning Classifier: Random Forest Classifier
        4.  A grid search to improve the input parameters of the machine learning algorithm
    """
    pipeline = Pipeline([
        ('features', FeatureUnion([
            ('text_pipeline', Pipeline([
                ('vect', CountVectorizer(tokenizer=tokenize)),
                ('tfidf', TfidfTransformer())
            ])),
        ])),
        ('clf', MultiOutputClassifier(RandomForestClassifier())),
    ])

    parameters = {
        'features__text_pipeline__vect__ngram_range': ((1, 1), (1, 2)),
        #'clf__estimator__n_estimators': [50, 100, 200],
        #'clf__estimator__min_samples_split': [2, 3, 4]
    }
    cv = GridSearchCV(pipeline, param_grid=parameters)
    
    return cv


def evaluate_model(model, X_test, Y_test_array, category_names):
    """
    Input:  A machine learning model, 
            the X-test array, 
            the Y-test array, 
            the category names of the columns
    Output: None
    
    This function prints parameters which show the performance of the machine learning pipeline. 
    """
    Y_pred_array = model.predict(X_test) #make a prediction of the model
    Y_pred = pd.DataFrame(data = Y_pred_array, columns = category_names)
    Y_test = pd.DataFrame(data = Y_test_array, columns = category_names)

    for col in category_names:
        print(classification_report(Y_test[col],Y_pred[col]))


def save_model(model, model_filepath):
    """
    Input:  A trained machine learning model,
            A filepath at which the model has to be saved
    Output: None
    
    This function saves the model in a machine learning pipeline.
    """
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
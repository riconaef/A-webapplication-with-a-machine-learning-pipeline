import json
import plotly
import numpy as np
import pandas as pd
import re

from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar
from plotly.graph_objs import Pie

from sklearn.externals import joblib
from sqlalchemy import create_engine


app = Flask(__name__)

def tokenize(text):
    text = re.sub(r"[^a-zA-Z0-9]", " ",text)
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens

# load data
engine = create_engine('sqlite:///../data/DisasterResponse.db')
df = pd.read_sql_table('data/DisasterResponse.db', engine)

# load model
model = joblib.load("../models/classifier.pkl")


# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
    # extract data needed for visuals
    
    #figure 1
    x_pie = df.drop(['id', 'message', 'original', 'genre'], axis = 1).sum()
    names_pie = list(x_pie.index)
    
    #figure 2
    counts = []
    cat = ['direct', 'social', 'news']
    for i in cat:
        counts.append(df.groupby('genre').mean().loc[i].drop('id').round(2))
    names_bar = list(counts[0].index)
      
    # create visuals
    graphs = [
        {#figure 1
            'data': [
                Pie(
                    values = x_pie,
                    labels = names_pie
                ),
            ],

            'layout': {
                'title': 'Relative size of each category',
                'height':1000, 
                'width':1000,
            }
        },

        {#figure 2
            'data': [
                Bar(
                    x = names_bar,
                    y = counts[0],
                    name = cat[0]
                ),
                Bar(
                    x = names_bar,
                    y = counts[1],
                    name = cat[1]
                ),
                Bar(
                    x = names_bar,
                    y = counts[2],
                    name = cat[2]
                )
            ],

            'layout': {
                'title': 'Distribution of genres in each category',
                'height':500, 
                'width':1300,
                'yaxis': {
                    'title': "Mean Count"
                },
                'xaxis': {
                    'title': "Category"
                },

            }
        }
    ]
   
    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)
    
    # render web page with plotly graphs
    return render_template('master.html', ids=ids, graphJSON=graphJSON)


# web page that handles user query and displays model results
@app.route('/go')
def go():
    # save user input in query
    query = request.args.get('query', '') 

    # use model to predict classification for query
    classification_labels = model.predict([query])[0]
    classification_results = dict(zip(df.columns[4:], classification_labels))

    # This will render the go.html Please see that file. 
    return render_template(
        'go.html',
        query=query,
        classification_result=classification_results
    )


def main():
    app.run(host='0.0.0.0', port=3001, debug=True)


if __name__ == '__main__':
    main()
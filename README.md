# Disaster Response Pipeline Project

### Table of Contents

1. [Installation](#installation)
2. [Project Motivation](#motivation)
3. [File Descriptions](#files)
4. [Results](#results)
5. [Acknowledgements](#licensing)

## Installation <a name="installation"></a>

The code should run in Python 3.

Following libraries are needed:
- import numpy as np
- import pandas as pd
- import re
- import pickle
- from sqlalchemy import create_engine

- from nltk.tokenize import word_tokenize
- from nltk.stem import WordNetLemmatizer

- from sklearn.multioutput import MultiOutputClassifier
- from sklearn.ensemble import RandomForestClassifier
- from sklearn.model_selection import train_test_split
- from sklearn.metrics import classification_report
- from sklearn.pipeline import Pipeline, FeatureUnion
- from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
- from sklearn.model_selection import GridSearchCV
- from sklearn.externals import joblib

- from flask import Flask
- from flask import render_template, request, jsonify

- from plotly.graph_objs import Bar
- from plotly.graph_objs import Pie

## Project Motivation<a name="motivation"></a>
The goal of this webapplication is to classifie messages writen by people or media into different categories. This is especially interesting if a disaster occurs. In such situations there are many messages in a short timewindow. Some messages might be urgent and need some attention. So it can be very helpful if there is an automatic classification to transfer important messages to the right people or organisations.

## File Descriptions <a name="files"></a>

#### The data folder
This folder contains three files:
- disaster_categories.csv
- disaster_messages.csv
- process_data.py

The process_data.py file cleans the data and stores it in a database. It can be run with the following command: 

'python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db'

The three additional arguments are the two locations of the csv files and the pathname for the sql database.

#### The models folder
This folder contains one file:
- train_classifier.py

This file sets up a machine learning pipeline, trains it and saves the model to a pickle file. For running, following command can be used:

'python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl'

The two additional arguments are the location of the database and a pathname for the pickle file

#### The app folder
This folder contains a 'templates' folder with two html files:
- go.html
- master.html

These files are being used for the frontend visualization.
The run.py file is for running the webapp and can be used as follows:

'python run.py'

The machine learning model is used to classifie entrys in entrybox into the different categories. 

## Results<a name="results"></a>
At the website there are plotted two graphs to give an overview over the dataset. The first is a pie plot to show the relative amount of the different categories. The second plot shows the amount of the three genres 'direct', 'social' and 'news' for each category. 

By typing in a sentence or a word into the empty entry box, some related categories are popping up green. It is not working for all sentences or words. This might be due to the lack of training data. 

## Licensing, Authors, Acknowledgements<a name="licensing"></a>
I thank figure8 for offering the data.

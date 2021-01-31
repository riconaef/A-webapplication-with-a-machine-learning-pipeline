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



## File Descriptions <a name="files"></a>

The used data consists of two dataframes (listings and calendar) and can be downloaded here. The calendar dataframe is a large file, 
thus it was separated into two files.

The steps of the analysis can be viewed in the Crisp_DM_history.ipynb file. There is also a single jupyter file for each question to generate the desired plots.

## Results<a name="results"></a>

The main findings of the results can also be found [here](https://naefrico.medium.com/what-drives-prices-at-airbnb-accommodations-c60e4589a099).

## Licensing, Authors, Acknowledgements<a name="licensing"></a>

I thank airbnb for offering the data. You can find the Licensing for the data and other descriptive information at the Kaggle link available [here](https://www.kaggle.com/airbnb/seattle/data). You can also use the code here and run the model by yourself. Maybe you even have an idea for adjustments or improvements. 

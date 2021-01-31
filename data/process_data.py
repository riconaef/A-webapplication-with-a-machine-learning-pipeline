import sys
import numpy as np
import pandas as pd
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    """
    Input: two csv files
    Output: A pandas dataframe with the merged data of both csv files
    """
    
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    df = messages.merge(categories, how = 'outer', on = 'id') #merge datasets

    return df

def clean_data(df):
    """
    Input: A pandas dataframe
    Output: A cleaned dataframe which includes following steps:
        1.  Drops all duplicates in the dataframe
        2.  Each category in the column "category" is split into a different column
        3.  All category columns contain 1 for True or 0 for False
    """
    
    df = df.drop_duplicates()
    categories = df['categories'].str.split(';', expand=True) #split categories column
    category_colnames = categories.iloc[0].apply(lambda x: x[:-2]) #get names of all new columns
    categories.columns = category_colnames #set the column names

    for column in categories:
        categories[column] = categories[column].str[-1] #set each value to the last character
        categories[column] = categories[column].astype(int)  #convert column from string to numeric

    df = df.drop('categories', axis=1) #drop old categories column
    df = pd.concat([df, categories], axis=1) #put messages dataframe together with the new categories

    return df


def save_data(df, database_filename):
    """
    Input:  A pandas dataframe
            A name for the sql database
    Output: None
    
    This function takes a dataframe and saves it in a sql-database 
    """
    
    engine = create_engine('sqlite:///' + str(database_filename))
    df.to_sql(database_filename, engine, index=False)  

def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)

        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)

        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()
## Disaster_response_Udacity_Project_3
### Table of Contents
1. Introduction
2. ETL And Machine Lerning Pipeline
3. File Description
4. Libraries used
5. Executing Program
6. Snapshots
7. Author

### 1. Introduction

This project is a part of Udacity nano degree project. The data set is provided by Figure Eight for this project which contains pre labelled tweet and messages from real life disater-events. The project aim is to build a Natural Language Processing (NLP) model to categorize messages on a real time basis.  

The project is divided into three sections:
1. Processing data, building an ETL pipeline to extract data from source, clean the data and load the data to SQL db.
2. Build a machine learning pipeline to train the which can classify text message in various categories.
3. Run a web app by using flask to show the dashboard and demonstrate real time classification of messages.

### 2. ETL And Machine Lerning Pipeline
We move further in this project lets first understand what is ETL and what is machine learning pipeline model.

ETL- In computing, extract, transform, load (ETL) is the general procedure of copying data from one or more sources into a destination system which represents the data differently from the source(s) or in a different context than the source(s). For more details you can go through to this [link](https://www.webopedia.com/TERM/E/ETL.html)

Machine Lerning Pipeline- A machine learning pipeline is used to help automate machine learning workflows. They operate by enabling a sequence of data to be transformed and correlated together in a model that can be tested and evaluated to achieve an outcome, whether positive or negative.For more details you can go through to this [link](https://medium.com/analytics-vidhya/what-is-a-pipeline-in-machine-learning-how-to-create-one-bda91d0ceaca)

### 3. File Description

* process_data.py: This python excutuble code takes as its input csv files containing message data and message categories (labels), and then creates a SQL database
* train_classifier.py: This code trains the ML model with the SQL data base
* ETL Pipeline Preparation.ipynb: process_data.py development procces
* ML Pipeline Preparation.ipynb: train_classifier.py. development procces
* data: This folder contains sample messages and categories datasets in csv format.
* app: cointains the run.py to iniate the web app.

### 4. Libraries used

1. numpy
2. pandas
3. nltk
4. re
5. sys
6. pickle
7. sqlalchemy
8. sklearn

### 5. Executing Program

1. You can run the following commands in the project's directory to set up the database, train model and save the model.

* To run ETL pipeline to clean data and store the processed data in the database python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/disaster_response_db.db
* To run the ML pipeline that loads data from DB, trains classifier and saves the classifier as a pickle file python models/train_classifier.py data/disaster_response_db.db models/classifier.pkl
2. Run the following command in the app's directory to run your web app. python run.py

3. Go to http://0.0.0.0:3001/

### 6. Snapshots

The project also include a web app with bootstap and Flask where an emergency worker can input a new message and get classification results in several categories.

![alt text](https://github.com/AdityarNarayan/Disaster_Response_Udacity_project_3/raw/master/Disaster_response_project_dashboard_screenshot.JPG)

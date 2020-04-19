# Udacity data science nano degree pipeline project #

The objective of this exercise is to develop a combined ETL + ML pipeline for
predicting the topic(s) of text messages received during a disaster. While 
essentially a proof of concept in practice, in principle a workflow like this
could be used to filter through a large number of communications to identify
those related to a specific need or issue.

The process has the following steps:
1. ETL pipeline for cleaning and organizing disaster-related text messages
2. ML pipeline for predicting the topics of messages, based on their content
3. A webapp for displaying types of messages and predicting user-supplied text 

## 1. ETL pipeline ##

This pipeline deduplicates messages, unpacks the categories used to tag 
the message, and exports the data to the table `scored_messages` in 
`data\DisasterResponse`data\DisasterResponse.db`. 

### Running ###

The process can be run from the base of the repo with the command 
`python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
The files `disaster_messages.csv` and `disaster_categories.csv` need to be in 
the folder `data/` for this to work.

## 2. ML pipeline ##

This pipeline creates a modeling dataset from `scored_messages`, then uses a
Random Forest classifier to evaluate whether the messages belong to the tagged
categories. 

Message texts are tokenized, stripped of English stop words and lemmatized for
nouns, verbs, and adjectives. They are transformed with TF-IDF.

### Running ###

The process can be run from the base of the repo with the command 
`python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`
The database `data/DisasterResponse.db` needs to contain the table 
`scored_messages` for this to run. 
**NOTE**: the script is set up to perform a fairly ambitious grid search
and will take a *long* time to run. Parameters are stored in the `params` 
variable, in L24-29 of `train_classifier.py`, and should probably be limited if
you want to rerun this.

### Model evaluation ###

Many categories are quite rare, present in <5% of the corpus. As a result, 
precision is generally much higher than recall, and accuracy is unrealistically
high. The effectivness of scoring topics is also highly variable. Certain 
disasters (earthquakes, flood, storm) and certain needs (food, shelter, water)
can be classified quite well (F1-score > 0.7). On the lower end of the scale, 
more ambiguous terms like "infrastructure related" and "other weather" have
scores around 0.5, indicating essentially random classifications.

## 3. Flask app ##

The app has two components:
* You can enter text message and have it classified using the classifier 
developed in Step #2
* There's some graphing of the types of messages in the corpus (based on the
database generated in Step #1); this could stand to be developed further.


### Running ###

The webapp needs to be run from the `app/` folder, with the command 
`python run.py`. This will be hosted at `http://0.0.0.0:3001` or, if running
locally, `localhost:3001`.

## Licensing & Acknowledgements ## 

The data referenced in this project were provided by Figure Eight (now owned by
[Appen](https://appen.com/). The Flask app and graphs are minimally edited from
what was provided through the Udacity Data Science Nanodegree program.
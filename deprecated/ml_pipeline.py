"""
Pipeline for scoring texts
(Consumes effects of etl_pipeline.py)
"""

# Libraries
from collections import namedtuple
import nltk
import pandas as pd
import pickle
import re
from sqlalchemy import create_engine

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import RandomForestClassifier
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.multioutput import MultiOutputClassifier
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.preprocessing import OneHotEncoder

import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('words')

#######################
# Grid search parameters
params = {
    'clss__estimator__min_samples_split': range(2, 3),  # range(2,5),
    'clss__estimator__n_estimators': [10],  # [10, 50, 100, 120],
    'feats__msgs__tfidf__smooth_idf': [False],  # [False, True],
    'feats__msgs__vect__ngram_range': [(1, 1)],  # [(1,1), (1,2)],
}

###########
# Functions


def tokenize(text):
    """Turn string into lemmatized tokens"""

    # Basic cleaning
    text = text.lower()
    text = re.sub(r"[^a-z0-9]", " ", text)

    # Get word tokens and remove stopwords
    swrds = nltk.corpus.stopwords.words('english')
    tokens = nltk.word_tokenize(text)
    tokens = [tk.strip() for tk in tokens if tk not in swrds]

    # Lemmatize thoroughly
    lem = nltk.stem.wordnet.WordNetLemmatizer()
    lem_tok = []
    for tok in tokens:
        tok = lem.lemmatize(tok)
        tok = lem.lemmatize(tok, pos='v')
        tok = lem.lemmatize(tok, pos='a')
        lem_tok.append(tok)

    return lem_tok


def text_gscv(params):
    """Generate a GridSearchCV object for a text classification pipeline"""

    msgs = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer())
    ])

    pipeline = Pipeline([
        ('feats', ColumnTransformer([
            ('msgs', msgs, 'message'),
            ('ohe', OneHotEncoder(), ['genre', 'translated'])
        ], remainder='drop')),
        ('clss', MultiOutputClassifier(
            RandomForestClassifier()))
    ])

    gscv = GridSearchCV(pipeline, params, cv=5)

    return gscv


##############
# Main section

# Data preparation
print("Ingesting data from database")
engine = create_engine('sqlite:///data/DisasterResponse.db')
qry = "SELECT * FROM scored_messages ORDER BY message"
df = pd.read_sql(qry, engine)
engine.dispose()

X = df.iloc[:, :3]
y = df.iloc[:, 3:]

# Split data and train model
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=2290)
model = text_gscv(params)
print("Scoring model")
model.fit(X_train, y_train)

# Evaluate model
print("Evaluating model fit")
y_pred = model.predict(X_test)

scor_lst = []
for col in range(y_test.shape[1]):
    rpt = classification_report(
        y_test.iloc[:, col], y_pred[:, col], output_dict=True)
    scor_lst.append([
        y_test.columns[col],
        rpt['accuracy'],
        rpt['macro avg']['precision'],
        rpt['macro avg']['recall'],
        rpt['macro avg']['f1-score']
    ])

scor_df = pd.DataFrame(
    scor_lst, columns=['class', 'accuracy', 'precision', 'recall', 'f1-score'])

# Package model with evaluation and output to pickle
print("Exporting results")
ModelOutput = namedtuple('ModelOutput', 'class_rept mod_obj')
mod_out = ModelOutput(scor_df, model)

with open('text_scorer.pkl', 'wb') as f:
    pickle.dump(mod_out, f)

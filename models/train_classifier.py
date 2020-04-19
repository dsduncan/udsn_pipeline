
import os
import pandas as pd
import re
import sys

from sklearn.ensemble import RandomForestClassifier
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.multioutput import MultiOutputClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('words')

#######################
# Grid search parameters
PARAMS = {
    'clss__estimator__min_samples_split': range(2, 3),  # range(2,5),
    'clss__estimator__n_estimators': [10],  # [10, 50, 100, 120],
    'feats__msgs__tfidf__smooth_idf': [False],  # [False, True],
    'feats__msgs__vect__ngram_range': [(1, 1)],  # [(1,1), (1,2)],
}


def load_data(database_filepath, tn='scored_messages'):
    """Load data from SQLite database
    Returns:
        Dataframes of predictors and outcomes
    """
    from sqlalchemy import create_engine
    engine = create_engine(f'sqlite:///{database_filepath}')
    qry = f"SELECT * FROM {tn} ORDER BY message;"
    df = pd.read_sql(qry, engine)
    engine.dispose()

    X = df.iloc[:, :3]
    y = df.iloc[:, 3:]

    return X, y


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


def build_model(params):
    """Pipeline for feeding into Random Forest classifier
    Arguments:
        params {dict}: Parameters to consider during GridSearch
    Returns:
        Pipeline's model object
    """
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


def evaluate_model(model, X_test, Y_test):
    """Compute fit statistics
    Effect:
        Prints out table of fit statistics for each category being
        predicted, sorted by F1 scores
    """

    y_pred = model.predict(X_test)

    scor_lst = []
    for idx, col in enumerate(Y_test.columns):
        rpt = classification_report(
            Y_test[col], y_pred[:, idx], output_dict=True)
        scor_lst.append([
            col,
            rpt['accuracy'],
            rpt['macro avg']['precision'],
            rpt['macro avg']['recall'],
            rpt['macro avg']['f1-score']
        ])

    scor_cols = ['class', 'accuracy', 'precision', 'recall', 'f1-score']
    scor_df = pd.DataFrame(scor_lst, columns=scor_cols)

    print(scor_df.sort_values('f1-score', ascending=False))


def save_model(model, model_filepath):
    """Write model object to pickle file"""
    import bz2
    import pickle

    with bz2.BZ2File(model_filepath+'.bz2', 'wb') as f:
        pickle.dump(model, f)


def main():
    print(os.getcwd())
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(
            X, Y, test_size=0.2, random_state=2290)
        
        print('Building model...')
        model = build_model(PARAMS)
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '
              'as the first argument and the filepath of the pickle file to '
              'save the model to as the second argument. \n\nExample: python '
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()
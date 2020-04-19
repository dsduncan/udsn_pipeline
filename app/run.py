import bz2
import json
import nltk
import pandas as pd
import pickle
import plotly


from flask import Flask
from flask import render_template, request
from plotly.graph_objs import Bar
from sqlalchemy import create_engine

nltk.download('stopwords')
nltk.download('wordnet')


app = Flask(__name__)


def tokenize(text):

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


# load data
engine = create_engine('sqlite:///../data/DisasterResponse.db')
df = pd.read_sql_table('scored_messages', engine)

# load model
with bz2.BZ2File('../models/classifier.pkl.bz2', 'rb') as f:
    model = pickle.load(f)



# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
    
    # extract data needed for visuals
    # TODO: Below is an example - modify to extract data for your own visuals
    genre_counts = df.groupby('genre').count()['message']
    genre_names = list(genre_counts.index)
    
    # create visuals
    # TODO: Below is an example - modify to create your own visuals
    graphs = [
        {
            'data': [
                Bar(
                    x=genre_names,
                    y=genre_counts
                )
            ],

            'layout': {
                'title': 'Distribution of Message Genres',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Genre"
                }
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
    pred_df = pd.DataFrame({
        'message': query,
        'genre': 'direct',
        'translated': 0
    }, index=[0])

    # use model to predict classification for query
    classification_labels = model.predict(pred_df)[0]
    classification_results = dict(zip(df.columns[3:], classification_labels))

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
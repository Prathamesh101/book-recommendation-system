from distutils.log import debug
from flask import Flask ,render_template, request, redirect, url_for, jsonify
import numpy as np
from chat import get_response
import pandas as pd

popular_df = pd.read_pickle('popular.pkl')
pt = pd.read_pickle('pt.pkl')
books = pd.read_pickle('books.pkl')
similarity_scores = pd.read_pickle('similarity_scores.pkl')
pop_sort = pd.read_pickle('pop_sort.pkl')

app = Flask(__name__)


@app.route('/')
def index():
    return render_template('index.html',image = list(popular_df['Image-URL-M'].values),
                            sp_image = list(pop_sort['Image-URL-M'].values))

@app.post("/predict")
def predict():
    text=request.get_json().get("message")
    #TODO:check if text is valid
    response=get_response(text)
    message={"answer":response}
    return jsonify(message)

@app.route('/Recommend',methods=["GET","POST"])
def recommend():
    if request.method == 'POST':
        user_input = request.values['user_input']
        data = []
        if(len(np.where(pt.index == user_input)[0])):
            index = np.where(pt.index == user_input)[0][0]
            similar_items = sorted(list(enumerate(similarity_scores[index])), key=lambda x: x[1], reverse=True)[1:5]
            
            for i in similar_items:
                item = []
                temp_df = books[books['Book-Title'] == pt.index[i[0]]]
                item.extend(list(temp_df.drop_duplicates('Book-Title')['Book-Title'].values))
                item.extend(list(temp_df.drop_duplicates('Book-Title')['Book-Author'].values))
                item.extend(list(temp_df.drop_duplicates('Book-Title')['Image-URL-M'].values))
                data.append(item)
            f = 1
        else:
            f = 0
        return render_template('recommend.html',data=data,f =f)
    else:
        return render_template('recommend.html')


@app.route('/Searchbooks')
def searchbooks():
    return render_template('spselection.html',book_name = list(pop_sort['Book-Title'].values),
                            author = list(pop_sort['Book-Author'].values),
                            image = list(pop_sort['Image-URL-M'].values),
                            votes = list(pop_sort['number_of_ratings'].values),
                            rating = list(pop_sort['average_ratings'].values))

@app.route("/Topfiftybooks")
def topfiftybooks():
    return render_template('topbooks.html',book_name = list(popular_df['Book-Title'].values),
                            author = list(popular_df['Book-Author'].values),
                            image = list(popular_df['Image-URL-M'].values),
                            votes = list(popular_df['number_of_ratings'].values),
                            rating = list(popular_df['average_ratings'].values))

if __name__ == "__main__":
    app.run(debug = True)
#external libraries importation
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sys
import os
from math import sqrt
from flask import request
from flask import Flask

#internal functions importations
from content_based import recommendMovie
from collaborative_based import getRecommendation 

app = Flask(__name__)

users = pd.read_csv("data/users.csv")
movies = pd.read_csv("data/movies.csv")
ratings = pd.read_csv("data/ratings.csv")

@app.route('/content', methods=['GET'])
def content():
    user_id = int(request.args.get('user_id'))
    limit = int(request.args.get('limit'))
    recommendation = recommendMovie(user_id, movies, ratings, limit=limit)
    return recommendation.to_json()





@app.route('/collaborative', methods=['GET'])
def collaborative():
    user_id = int(request.args.get('user_id'))
    noCategoriesMovies = movies[["movie_id", "title", "year"]]
    recommendation = getRecommendation(user_id, ratings, noCategoriesMovies)
    return recommendation.to_json()
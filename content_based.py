'''
This file contains the functions related to the content_based recommendation approach
'''
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sys
import os
from math import sqrt

def getSimilarity(topMovies, allMovies):
  '''
  description: This function return the similarity score between a given movie and the other movies in the dataset
  parameters:
    movie: category one-hot encoded dataFrame.
    allMovies: dataset's movies dataFrame.
  output:
    similarity: array(number_of_movies, 1) of the similarity score between the movie and the other ones.
  '''
  similarity =  np.matmul(topMovies.values, allMovies.values.T)
  return similarity


def getUserTopMovies(user_id, movies, ratings, limit=None):
  '''
  description: This function returns the top rated movies by a given user.
  parameters:
    user_id: the user's id.
    movies: all the movies in the dataset.
    ratings: all the ratings in the dataset.
  output:
    topMovies: dataFrame of the top rated movies.
  '''
  topMovies = ratings[ratings["user_id"] == user_id].sort_values(by="rating", ascending=False)
  if(limit):
    topMovies = topMovies.head(limit)
  topMovies = movies.merge(topMovies, on="movie_id")
  return topMovies


def recommendMovie(user_id, movies, ratings, limit=None):
  '''
  description: this function gives a movie recommendation using content-based approach, given a user_id, movies of the dataset and ratings.
  parameters:
    user_id: the user_id we want to recommend to.
    movies: movies of our dataset.
    ratings: ratings of our dataset.
    limit: limit of top liked movies by user we want to keep.
  output:
    moviesRecommendation: dataFrame of the recommended movies.
  '''
  notCategory = ["movie_id", "title", "year"]
  #getting the user's top liked movies:
  topMovies = getUserTopMovies(user_id, movies, ratings, limit)
  topMoviesCategories = topMovies.drop(notCategory + ["user_id", "rating"], axis=1)
  moviesCategories = movies.drop(notCategory, axis=1)
  similarity = getSimilarity(topMoviesCategories, moviesCategories)
  similarity = pd.DataFrame(similarity, columns=list(movies["movie_id"]))
  recommendedMovieIds = list(similarity.idxmax(axis=1).drop_duplicates())
  return movies.iloc[recommendedMovieIds]
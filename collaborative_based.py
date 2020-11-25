'''
This file contains the functions related to the collaborative_based recommendation approach
'''
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sys
import os
from math import sqrt

def getWatchedMovies(user_id, ratings, movies):
  '''
  description: this function returns the watched movies given a user id.
  parameters:
    user_id: the user's id
    ratings: the whole ratings dataFrame
    movies: the whole movies dataFrame
  output:
    watchedMovies: a dataFrame containing the informations about the movies watched
  '''
  watchedMovies = ratings[ratings["user_id"] == user_id].sort_values(by="rating", ascending=False)
  watchedMovies =  movies.merge(watchedMovies, on="movie_id")
  similarUsers = ratings[ratings['movie_id'].isin(watchedMovies['movie_id'].tolist())]
  return watchedMovies, similarUsers




def getSimilarityScore(watchedMovies, similarUsers):
  '''
  description: this function calculates the pearson's similarity between our user and the other users who watched the same movies.
  parameters:
    watchedMovies: the targeted user's watch history.
    similarUsers: users that have seen the same movies.

  output:
    similarity score for each watched movie.
  '''
  groupedUsers = similarUsers.groupby(['user_id'])
  groupedUsers = sorted(groupedUsers,  key=lambda x: len(x[1]), reverse=True)
  #we will keep only the 50 first rows
  groupedUsers = groupedUsers[0:50]
  pearsonScore = {}
  #For every user group in our subset
  for name, group in groupedUsers:
      #Let's start by sorting the input and current user group so the values aren't mixed up later on
      group = group.sort_values(by='movie_id')
      watchedMovies = watchedMovies.sort_values(by='movie_id')
      #Get the N for the formula
      n = len(group)
      #Get the review scores for the movies that they both have in common
      temp = watchedMovies[watchedMovies['movie_id'].isin(group['movie_id'].tolist())]
      #And then store them in a temporary buffer variable in a list format to facilitate future calculations
      tempRatingList = temp['rating'].tolist()
      #put the current user group reviews in a list format
      tempGroupList = group['rating'].tolist()
      #Now let's calculate the pearson correlation between two users, so called, x and y
      Sxx = sum([i**2 for i in tempRatingList]) - pow(sum(tempRatingList),2)/float(n)
      Syy = sum([i**2 for i in tempGroupList]) - pow(sum(tempGroupList),2)/float(n)
      Sxy = sum( i*j for i, j in zip(tempRatingList, tempGroupList)) - sum(tempRatingList)*sum(tempGroupList)/float(n)
      
      #If the denominator is different than zero, then divide, else, 0 correlation.
      if Sxx != 0 and Syy != 0:
          pearsonScore[name] = Sxy/sqrt(Sxx*Syy)
      else:
          pearsonScore[name] = 0
  similarityScore = pd.DataFrame.from_dict(pearsonScore, orient='index')
  similarityScore.columns = ['similarity_score']
  similarityScore['user_id'] = similarityScore.index
  similarityScore.index = range(len(similarityScore))
  similarityScore = similarityScore.sort_values(by='similarity_score', ascending=False)[0:50]
  return similarityScore


def getRecommendation(user_id, ratings, movies):
  '''
  description: this function gives a collaborative based movie recommendation given a user, ratings, and movies.
  parameters:
    user_id: the targeted user's id (int).
    ratings: a dataFrame of the ratings of the whole dataset.
    movies: a dataFrame the movies of the whole dataset.
  output:
    recommendedMovies: a dataFrame of the recommendedMovies.
  '''
  watchedMovies, similarUsers = getWatchedMovies(0, ratings, movies)
  similarityScore = getSimilarityScore(watchedMovies, similarUsers)
  #removing the targeted user's score which is equal to 1.
  similarityScore = similarityScore.iloc[1:]
  #merging with the ratings dataFrame to get the ratings of the similar users.
  topUsersRating = similarityScore.merge(ratings, left_on='user_id', right_on='user_id', how='inner')
  #calculating a weighed value of the rating based on the similarity score
  topUsersRating['weighted_rating'] = topUsersRating['similarity_score'] * topUsersRating['rating']
  #calculating a sum of the weighted ratings for each movie
  tempTopUsersRating = topUsersRating.groupby('movie_id').sum()[['similarity_score','weighted_rating']]
  tempTopUsersRating.columns = ['sum_similarity_index', 'sum_weighted_rating']
  recommendedMovies = pd.DataFrame()
  #Getting the average of weighted ratings
  recommendedMovies['weighted_average_recommendation_score'] = tempTopUsersRating['sum_weighted_rating']/tempTopUsersRating['sum_similarity_index']
  recommendedMovies['movie_id'] = tempTopUsersRating.index
  #sorting by the value of the weighted average recommendation score
  recommendedMovies = recommendedMovies.sort_values(by='weighted_average_recommendation_score', ascending=False)
  #returning the top 10 most similar movies
  return movies.loc[movies['movie_id'].isin(recommendedMovies.head(10)['movie_id'].tolist())]
import pandas as pd
import numpy as np
from textblob import TextBlob
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_extraction.text import TfidfVectorizer

# Drop song duplicates
def drop_duplicates(df):
    df['artists_song'] = df.apply(lambda row: row['artist_name']+row['track_name'],axis = 1)
    return df.drop_duplicates('artists_song')

#Perform sentiment analysis on track_name
def get_sentiment(df):
    df['sentiment'] = df.apply(lambda row: TextBlob(row['track_name']).sentiment.polarity, axis = 1)
    return df

def convert_sentiment_label(sentiment):
    if sentiment < 0:
        return 'negative'
    elif sentiment > 0:
        return 'positive'
    else:
        return 'neutral'
    
#One hot enocode the sentiment label
def one_hot_encode(df, label, value):
    one_hot = pd.get_dummies(df[label]) * value
    feature_names = one_hot.columns
    one_hot.columns = [label + '_' + str(col) for col in feature_names]
    df = df.drop(label, axis = 1)
    df = df.join(one_hot)
    df.reset_index(drop=True, inplace=True)
    return df

#normalize the audio features
def normalization(all_songs, float_cols):
    pop = all_songs[["artist_pop"]].reset_index(drop = True)
    scaler = MinMaxScaler()
    pop_scaled = pd.DataFrame(scaler.fit_transform(pop), columns = pop.columns)
    floats = all_songs[float_cols].reset_index(drop = True)
    floats_scaled = pd.DataFrame(scaler.fit_transform(floats), columns = floats.columns) * 0.2
    return pop_scaled, floats_scaled

# TF-IDF implementation
def tfidf_vectorization(df, column):
    tfidf = TfidfVectorizer()
    tfidf_matrix =  tfidf.fit_transform(df[column])
    tfidf_df = pd.DataFrame(tfidf_matrix.toarray())
    tfidf_df.columns = [column + "|" + i for i in tfidf.vocabulary_]
    tfidf_df.drop(columns=column + "|unknown")
    tfidf_df.reset_index(drop = True, inplace=True)
    return tfidf_df

#Run Feature Engineering
def feature_engineering():
    all_songs = pd.read_csv('recommendation_set.csv')
    all_songs = drop_duplicates(all_songs)
    # print("Are all songs unique: ",len(pd.unique(all_songs.artists_song))==len(all_songs))
    # print(all_songs.shape)
    all_songs = get_sentiment(all_songs)
    all_songs['sentiment_label'] = all_songs['sentiment'].apply(convert_sentiment_label)
    all_songs.drop(['sentiment'], axis=1, inplace=True)
    float_cols = all_songs.dtypes[all_songs.dtypes == 'float64'].index.values
    all_songs = one_hot_encode(all_songs, 'sentiment_label', 0.5)
    all_songs = one_hot_encode(all_songs, 'key', 0.5)
    all_songs = one_hot_encode(all_songs, 'mode', 0.5)
    genre_df = tfidf_vectorization(all_songs, 'genres_list')
    pop_scaled, floats_scaled = normalization(all_songs, float_cols)
    feature_df = pd.concat([genre_df, floats_scaled, pop_scaled], axis=1)
    feature_df['track_uri'] = all_songs['track_uri']
    feature_df.to_csv('features.csv', index=False)
    all_songs.to_csv('all_songs_data.csv', index=False)
    return feature_df, all_songs

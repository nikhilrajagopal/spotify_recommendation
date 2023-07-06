import pandas as pd
from spotify_api import get_playlist_info
from sklearn.metrics.pairwise import cosine_similarity
from feature_engineering import feature_engineering

def generate_playlist_feature(complete_feature_set, playlist_df):
    # Find song features in the playlist
    complete_feature_set_playlist = complete_feature_set[complete_feature_set['track_uri'].isin(playlist_df['track_uri'].values)]
    # Find all non-playlist song features
    complete_feature_set_nonplaylist = complete_feature_set[~complete_feature_set['track_uri'].isin(playlist_df['track_uri'].values)]
    complete_feature_set_playlist_final = complete_feature_set_playlist.drop(columns = "track_uri")
    return complete_feature_set_playlist_final.sum(axis = 0), complete_feature_set_nonplaylist

def generate_playlist_recos(df, features, nonplaylist_features, n):
    non_playlist_df = df[df['track_uri'].isin(nonplaylist_features['track_uri'].values)].copy()
    # print(non_playlist_df.shape)
    # print(features.values.reshape(1, -1).shape)
    non_playlist_df['sim'] = cosine_similarity(nonplaylist_features.drop('track_uri', axis=1).values, features.values.reshape(1, -1))[:, 0]
    non_playlist_df = non_playlist_df.groupby('artist_name').head(2)
    non_playlist_df_top_n = non_playlist_df.sort_values('sim', ascending=False).head(n)
    return non_playlist_df_top_n

# def generate_playlist_recos(df, features, nonplaylist_features, n):
#     non_playlist_df = df[df['track_uri'].isin(nonplaylist_features['track_uri'].values)].copy()
#     print(non_playlist_df.shape)
#     print(features.values.reshape(1, -1).shape)
#     non_playlist_df.reset_index(drop=True, inplace=True)  # Reset the index
#     non_playlist_df['sim'] = cosine_similarity(nonplaylist_features.drop('track_uri', axis=1).values, features.values.reshape(1, -1))[:, 0]
#     non_playlist_df_top_n = non_playlist_df.sort_values('sim', ascending=False).head(n)
#     return non_playlist_df_top_n


def recommendations(input_link, n):
    input_df = get_playlist_info(input_link)
    song_dataset = pd.read_csv('song_dataset.csv')
    recommendation_set = pd.concat([song_dataset, input_df]).to_csv('recommendation_set.csv', index=False)
    feature_df, all_songs = feature_engineering()
    complete_feature_set_playlist_vector, complete_feature_set_nonplaylist = generate_playlist_feature(feature_df, input_df)
    # print(complete_feature_set_playlist_vector.shape)
    # print(complete_feature_set_nonplaylist.shape)
    answer = generate_playlist_recos(all_songs, complete_feature_set_playlist_vector, complete_feature_set_nonplaylist, n)[['track_name', 'artist_name', 'album_name']]
    answer.rename(columns={'track_name': 'Song', 'artist_name':'Artist', 'album_name':'Album'}, inplace=True)
    answer.to_csv('answer.csv', index=False)
    return answer

input_link = input("Enter the link of your playlist: ")
n = int(input("How many recommendations would you like? "))
recommendations(input_link, n)
print(pd.read_csv('answer.csv'))


import pandas as pd
from spotify_api import get_playlist_info

# Generate dataframe from Pop playlist
pop_link = 'https://open.spotify.com/playlist/37i9dQZF1DXcBWIGoYBM5M'
pop_df = get_playlist_info(pop_link)

# Generate dataframe from Rap playlist
rap_link = 'https://open.spotify.com/playlist/37i9dQZF1DX48TTZL62Yht'
rap_df = get_playlist_info(rap_link)

#Generate dataframe from Nikhil's playlist
nikhil_link = 'https://open.spotify.com/playlist/1Hy9gbtq2SjksaXzaBfr1Y'
nikhil_df = get_playlist_info(nikhil_link)

#Generate data from Weyer's playlist
weyers_link = 'https://open.spotify.com/playlist/4O3JaVW5tSQlBXTlDSVE86'
weyers_df = get_playlist_info(weyers_link)

#Generate data from Girly Pop playlist
girlie_pop_link = 'https://open.spotify.com/playlist/5WsFFLhcXth9Vl3BYKJtct'
girlie_pop_df = get_playlist_info(girlie_pop_link)

#Generate data from online preprocessed pdf
songs_df = pd.read_csv('processed_data.csv')
songs_df.drop(columns=['uri', 'type', 'name', 'Unnamed: 0','Unnamed: 0.1', 'pos', 'artist_uri', 'album_uri', 'id', 'duration_ms_y', 'analysis_url', 'track_href', 'track_pop'], inplace = True)
songs_df = songs_df[['track_uri', 'track_name', 'artist_name', 'album_name', 'duration_ms_x', 'danceability', 'energy', 'valence', 'tempo', 'loudness', 'mode', 'key', 'acousticness', 'instrumentalness', 'liveness', 'speechiness', 'time_signature', 'genres', 'artist_pop']]
songs_df['genres_list'] = songs_df['genres'].str.split()
songs_df['genres_list'] = songs_df['genres_list'].apply(lambda x: [genre.replace('_', ' ') for genre in x])
songs_df.drop(columns=['genres'], inplace=True)
songs_df = songs_df[['track_uri', 'track_name', 'artist_name', 'album_name', 'duration_ms_x', 'danceability', 'energy', 'valence', 'tempo', 'loudness', 'mode', 'key', 'acousticness', 'instrumentalness', 'liveness', 'speechiness', 'time_signature', 'genres_list', 'artist_pop']]
songs_df.rename(columns={'duration_ms_x': 'duration_ms'}, inplace=True)
songs_df['genres_list'] = songs_df['genres_list'].apply(lambda x: str(x))

final = pd.concat([songs_df, pop_df, rap_df, nikhil_df, weyers_df, girlie_pop_df]).to_csv('song_dataset.csv', index=False) 


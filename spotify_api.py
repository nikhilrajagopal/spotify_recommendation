import spotipy
import pandas as pd
from spotipy.oauth2 import SpotifyClientCredentials
import retrying
from tqdm import tqdm

# Spotify API credentials
client_id = '2bb3eb0754074b63b4983cc49ca1d8c4'
client_secret = 'f83faf67653e40f28ebbab57642dcb3b'

# Initialize Spotipy client
client_credentials_manager = SpotifyClientCredentials(client_id=client_id, client_secret=client_secret)
sp = spotipy.Spotify(client_credentials_manager=client_credentials_manager)

# Function to extract playlist information
#@retrying.retry(wait_exponential_multiplier=1000, wait_exponential_max=10000)
def get_playlist_info(playlist_link):
    playlist_id = playlist_link.split('/')[-1]
    playlist = sp.playlist(playlist_id)

    tracks = playlist['tracks']
    total_tracks = min(tracks['total'], 600)  # Limiting to 600 tracks
    offset = 0
    limit = 50  # Set the number of tracks to retrieve per request
    playlist_info = []

    with tqdm(total=total_tracks, desc='Fetching playlist tracks') as pbar:
        while offset < total_tracks:
            results = sp.playlist_items(playlist_id, offset=offset, limit=limit)
            for item in results['items']:
                track = item['track']
                track_info = {
                    'track_uri': track['id'],
                    'track_name': track['name'],
                    'artist_name': track['artists'][0]['name'],
                    'album_name': track['album']['name'],
                    'duration_ms': track['duration_ms'],
                }
                
                # Get additional features
                features = sp.audio_features(track['uri'])[0]
                track_info['danceability'] = features['danceability']
                track_info['energy'] = features['energy']
                track_info['valence'] = features['valence']
                track_info['tempo'] = features['tempo']
                track_info['loudness'] = features['loudness']
                track_info['mode'] = features['mode']
                track_info['key'] = features['key']
                track_info['acousticness'] = features['acousticness']
                track_info['instrumentalness'] = features['instrumentalness']
                track_info['liveness'] = features['liveness']
                track_info['speechiness'] = features['speechiness']
                track_info['time_signature'] = features['time_signature']
                artist_id = track['artists'][0]['id']
                artist = sp.artist(artist_id)
                track_info['genres_list'] = artist['genres']
                track_info['artist_pop'] = artist['popularity']
                playlist_info.append(track_info)
            
            offset += limit
            if offset >= 600:
                break
            pbar.update(limit)
    
    return pd.DataFrame(playlist_info)


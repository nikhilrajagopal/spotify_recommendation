# Spotify Playlist Recommendation System

The code in this repository processes Spotify playlist data, performs feature engineering on the songs, and generates song recommendations based on playlist input.

## Files

1. `spotify_api.py`: Python script that utilizes the Spotify API to extract playlist information and retrieve audio features of songs.

2. `data_generation.py`: Python script for generating dataframes from different Spotify playlists and merging them into a single dataset.

3. `feature_engineering.py`: Python script that performs feature engineering on the dataset, including dropping duplicates, performing sentiment analysis, normalizing audio features, and implementing TF-IDF vectorization.

4. `predictions.py`: Python script that generates playlist recommendations based on user input and the processed dataset.

## File Descriptions

1. **spotify_api.py**
   - This script contains the `get_playlist_info()` function, which retrieves information about a Spotify playlist, including track details and audio features.
   - Modify the `client_id` and `client_secret` variables in the script with your own Spotify API credentials.

2. **data_generation.py**
   - This script generates dataframes from different Spotify playlists and merges them into a single dataset.
   - Update the playlist links in the script with the links to the desired playlists.
   - The script uses the `get_playlist_info()` function from `spotify_api.py` to extract playlist information.

3. **feature_engineering.py**
   - This script performs feature engineering on the dataset, including dropping duplicates, performing sentiment analysis on track names, normalizing audio features, and implementing TF-IDF vectorization for genre information.
   - The script contains several functions:
     - `drop_duplicates(df)`: Drops duplicate songs from the dataset based on the combination of artist name and track name.
     - `get_sentiment(df)`: Performs sentiment analysis on the track names and adds a sentiment column to the dataframe.
     - `convert_sentiment_label(sentiment)`: Converts the sentiment score to a categorical label (negative, neutral, or positive).
     - `one_hot_encode(df, label, value)`: Performs one-hot encoding on a specified column and assigns a value to each category.
     - `normalization(all_songs, float_cols)`: Normalizes the audio features and artist popularity using Min-Max scaling.
     - `tfidf_vectorization(df, column)`: Implements TF-IDF vectorization on the specified column (e.g., genres_list) to represent genre information.
   - The script also generates two output files: `features.csv` containing the processed features and `all_songs_data.csv` containing the complete processed dataset.

4. **predictions.py**
   - This script generates playlist recommendations based on user input and the processed dataset.
   - It utilizes the `get_playlist_info()` function from `spotify_api.py` to retrieve the input playlist information.
   - The script uses cosine similarity to find similar songs in the processed dataset and generates recommendations based on the top matches.
   - The recommended songs are saved in the `answer.csv` file.

## How to Run Code

1. Install the required dependencies by running `pip install -r requirements.txt`.

2. Obtain Spotify API credentials (client ID and client secret) and update the `spotify_api.py` script with your credentials.

3. Run `python predictions.py`
    - The script will prompt you to enter the link to a Spotify playlist and the number of recommendations you want to generate.
    - The script will generate a list of recommended songs based on the input playlist and save the results in the `answer.csv` file

## Dependencies

- spotipy
- pandas
- retrying
- tqdm
- numpy
- textblob
- scikit-learn

## Future Improvements

Here are some potential improvements that can be made to enhance the functionality and performance of the scripts:

1. **Optimization**: The current implementation processes the entire dataset each time recommendations are generated. To improve performance, consider implementing indexing or caching techniques to speed up the cosine similarity calculation and recommendation generation.

2. **User Interface**: Enhance the user experience by developing a graphical user interface (GUI) or a web application that allows users to interact with the scripts more intuitively. This can include features such as playlist selection, recommendation customization, and visualizations.

3. **Advanced Recommendation Algorithms**: Explore and implement matrix factorization and deep learning-based models to improve the quality and diversity of the generated recommendations.






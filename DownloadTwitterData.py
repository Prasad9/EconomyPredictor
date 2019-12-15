# Credits: https://gist.github.com/yanofsky/5436496

import tweepy  # https://github.com/tweepy/tweepy
import csv
import shutil
import os
from datetime import datetime

# Twitter API credentials
# TODO: Enter your personal Twitter credentials
CONSUMER_KEY = None
CONSUMER_SECRET = None
ACCESS_KEY = None
ACCESS_SECRET = None


class DownloadTwitterData:
    def __init__(self, params):
        self._append_data = params['APPEND_DATA']
        self._twitter_data = params['TWITTER_DATA']
        self._download_folder = os.path.abspath(params['DOWNLOAD_FOLDER'])
        self._meta_file = os.path.join(self._download_folder, params['META_FILE'])

        self._tweepy_api = self._authorize_twitter()

    def _authorize_twitter(self):
        # authorize twitter, initialize tweepy
        auth = tweepy.OAuthHandler(CONSUMER_KEY, CONSUMER_SECRET)
        auth.set_access_token(ACCESS_KEY, ACCESS_SECRET)
        tweepy_api = tweepy.API(auth)
        return tweepy_api

    def _create_folder(self):
        folder_exists = os.path.exists(self._download_folder)
        if not folder_exists or not self._append_data:
            shutil.rmtree(self._download_folder, ignore_errors=True)
            os.makedirs(self._download_folder)

            with open(self._meta_file, 'w') as fid:
                csv_fid = csv.writer(fid)
                csv_fid.writerow(['Speaker', 'Time', 'FileId'])

    def _download_user_tweets(self, user_screen_name):
        # Twitter only allows access to a users most recent 3240 tweets with this method

        # initialize a list to hold all the tweepy Tweets
        all_tweets = []

        # make initial request for most recent tweets (200 is the maximum allowed count)
        new_tweets = self._tweepy_api.user_timeline(screen_name=user_screen_name, count=200)

        # save most recent tweets
        all_tweets.extend(new_tweets)

        # save the id of the oldest tweet less one
        oldest = all_tweets[-1].id - 1

        # keep grabbing tweets until there are no tweets left to grab
        while len(new_tweets) > 0:
            print("getting tweets before ", oldest)

            # all subsequent requests use the max_id param to prevent duplicates
            new_tweets = self._tweepy_api.user_timeline(screen_name=user_screen_name, count=200, max_id=oldest)

            # save most recent tweets
            all_tweets.extend(new_tweets)

            # update the id of the oldest tweet less one
            oldest = all_tweets[-1].id - 1

            print("...{} tweets downloaded so far".format(len(all_tweets)))

        return all_tweets

    def download_data(self):
        self._create_folder()

        for user_dict in self._twitter_data:
            user_name = user_dict['USER_NAME']
            screen_name = user_dict['SCREEN_NAME']

            print('Downloading data of {}'.format(user_name))
            user_tweets = self._download_user_tweets(screen_name)
            # transform the tweepy tweets into a 2D array that will populate the csv
            out_tweets = [[int(datetime.timestamp(tweet.created_at)), tweet.text] for tweet in user_tweets]

            for index, tweet_data in enumerate(out_tweets, 1):
                base_file_name = '{}_Twitter_{:05d}'.format(screen_name, index)
                save_file_path = os.path.join(self._download_folder, '{}.csv'.format(base_file_name))
                # write the csv
                with open(save_file_path, 'w') as f:
                    writer = csv.writer(f)
                    writer.writerow(['time', 'text'])
                    writer.writerow(tweet_data)

                with open(self._meta_file, 'a') as fid:
                    csv_fid = csv.writer(fid)
                    csv_fid.writerow([user_name, tweet_data[0], base_file_name])


if __name__ == '__main__':
    params = {
        'APPEND_DATA': True,   # If you wish to append the data of newly added user.
                               # Set False if you wish to start fresh again
        'TWITTER_DATA': [      # List of dictionaries
            {
                'USER_NAME': 'Jamie Dimon',            # The name of the user
                'SCREEN_NAME': 'emo_jamie_dimon'       # Twitter handle name
            }
        ],
        'DOWNLOAD_FOLDER': './Data',             # The place to download the data
        'META_FILE': 'meta.csv',                  # The meta file name to be stored inside the download folder
    }
    d = DownloadTwitterData(params)
    d.download_data()

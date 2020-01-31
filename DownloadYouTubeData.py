import os
import shutil
from youtube_transcript_api import YouTubeTranscriptApi
import pandas as pd
from datetime import datetime
import csv

from GenerateSentiment import GenerateSentiment
from Constants import *

class DownloadYouTubeData:
    def __init__(self, params):
        self._append_data = params['APPEND_DATA']
        self._video_data = params['VIDEO_DATA']
        self._download_folder = os.path.abspath(params['DOWNLOAD_FOLDER'])
        self._join_interval = params['JOIN_INTERVAL']
        self._min_text_length = params['MIN_TEXT_LENGTH']
        self._meta_file = os.path.join(self._download_folder, params['META_FILE'])
        self._gs = GenerateSentiment()

    def _create_folder(self):
        folder_exists = os.path.exists(self._download_folder)
        if not folder_exists or not self._append_data:
            shutil.rmtree(self._download_folder, ignore_errors=True)
            os.makedirs(self._download_folder)

            with open(self._meta_file, 'w') as fid:
                csv_fid = csv.writer(fid)
                csv_fid.writerow(['Speaker', 'Time', 'FileId'])

    def _process_data(self, transcript):
        transformed_arr = []
        start_arr = []
        prev_start = 0.0
        curr_text = ''

        for script in transcript:
            curr_start = script['start']
            if curr_start - prev_start > self._join_interval:
                transformed_arr.append(curr_text.strip())
                curr_text = ''
                start_arr.append(prev_start)
                prev_start = curr_start

            curr_text += script['text'] + ' '

        if len(curr_text) > 0:
            transformed_arr.append(curr_text.strip())
            start_arr.append(prev_start)

        return transformed_arr, start_arr

    def download_data(self):
        total_transcripts = 0
        total_videos = 0
        self._create_folder()

        for speaker, video_arr in self._video_data.items():
            video_info_list = {}
            for video_data in video_arr:
                video_id = video_data['VIDEO_ID']
                video_date = video_data['DATE']

                try:
                    transcript = YouTubeTranscriptApi.get_transcript(video_id)
                except:
                    print('Failed for ', video_id)
                    continue

                transcript, start_times = self._process_data(transcript)
                sentiments = self._generate_sentiments(transcript)
                df = pd.DataFrame(data={'time': start_times, 'text': transcript, kDocSentimentScore: sentiments})
                file_path = os.path.join(self._download_folder, video_id + '.csv')
                df.to_csv(file_path, index=False)

                datetime_object = datetime.strptime(video_date, '%d-%m-%Y')
                timestamp = datetime.timestamp(datetime_object)
                video_info_list[str(int(timestamp))] = video_id

                total_transcripts += len(transcript)
                total_videos += 1

            keys = list(video_info_list.keys())
            keys.sort()
            with open(self._meta_file, 'a') as fid:
                csv_fid = csv.writer(fid)
                for key in keys:
                    csv_fid.writerow([speaker, key, video_info_list[key]])

            print('Total transcripts created: {}, processed videos: {}'.format(total_transcripts, total_videos))

    def _generate_sentiments(self, sentences):
        sentiments = []
        for sentence in sentences:
            sentiment_dict = self._gs.generate_sentiment(sentence)
            sentiments.append(sentiment_dict[kDocSentimentScore])
        return sentiments


if __name__ == '__main__':
    params = {
        'APPEND_DATA': True,   # If you wish to append the data of newly added user.
                               # Set False if you wish to start fresh again
        'VIDEO_DATA': {                        # Dictionaries of videos
            'Warren Buffet': [                 # List of videos belonging to Warren Buffet
                {'VIDEO_ID': 'Tr6MMsoWAog', 'DATE': '20-09-2019'},     # YouTube Video ID and date of launch
                # Keep adding the videos you wish to process
            ]
            # Keep adding all the users you wish to analyse
        },
        'DOWNLOAD_FOLDER': './Data',      # Location to download the folder
        'JOIN_INTERVAL': 5.0,             # Splitting the video in successive duration (in seconds)
        'META_FILE': 'meta.csv',          # Meta file name to be stored in download folder
        'MIN_TEXT_LENGTH': 200,           # Minimum text length of transcript to be considered
    }

    d = DownloadYouTubeData(params)
    d.download_data()
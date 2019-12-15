### Requirements
Install all the requirements needed to run this repository through:
```bash
pip install -r requirements.txt
```

### Download data
There are two ways to download the data.
#### a) Twitter
Provide the Twitter user handles you wish to download the tweets from. Look into the main function for more details [here](./DownloadTwitterData.py/#L100). Enter your personal Twitter credentials as well. After the configuration is done, run the code as:

```python
python DownloadTwitterData.py
```

#### b) YouTube
Provide the YouTube video ids you wish to download the transcripts from. Look into the main function for more details [here](./DownloadYouTubeData.py/#L88). After the configuration is done, run the code as:

```python
python DownloadYouTubeData.py
```

### Sentiment Analysis
The code makes use of Google Cloud's [Natural Language API](https://cloud.google.com/natural-language/docs/) Sentiment Analysis. You will have to create your account and setup the downloaded private key. Follow all the instructions mentioned [here](https://cloud.google.com/natural-language/docs/quickstart#quickstart-analyze-entities-gcloud).


### Create your own language model
To create your own language model specialized in economics and financial text, collect all the relevant PDF files in this domain. Set the required configuration in the [main function](./GenerateWordEmbeddings.py#L141). Ensure that the Google Cloud's requirements are satisfied mentioned in sentiment analysis section. To run the code:

```python
python GenerateWordEmbeddings.py
```


# Predict the future of economy using Machine Learning
The detailed explanation about the approach used in building up this repository is showcased in my [Medium article](https://medium.com/ymedialabs-innovation/predicting-the-future-of-the-economy-with-machine-learning-81a0efc7a41e).

## Requirements
Install all the requirements needed to run this repository through:
```bash
pip install -r requirements.txt
```

## Download data
There are two ways to download the data. Along with data getting downloaded, sentiment score will be generated. Ensure that you look into [Sentiment Analysis](#sentiment-analysis) section to make necessary setup.
### a) Twitter
Provide the Twitter user handles you wish to download the tweets from. Look into the main function for more details [here](./DownloadTwitterData.py/#L100). Enter your personal Twitter credentials as well. After the configuration is done, run the code as:

```python
python DownloadTwitterData.py
```

### b) YouTube
Provide the YouTube video ids you wish to download the transcripts from. Look into the main function for more details [here](./DownloadYouTubeData.py/#L88). After the configuration is done, run the code as:

```python
python DownloadYouTubeData.py
```

## Sentiment Analysis
The code makes use of Google Cloud's [Natural Language API](https://cloud.google.com/natural-language/docs/) Sentiment Analysis. You will have to create your account and setup the downloaded private key. Follow all the instructions mentioned [here](https://cloud.google.com/natural-language/docs/quickstart#quickstart-analyze-entities-gcloud).


## Create your own language model
To create your own language model specialized in economics and financial text, collect all the relevant PDF files in this domain. Set the required configuration in the [main function](./GenerateWordEmbeddings.py#L141). Ensure that the Google Cloud's requirements are satisfied mentioned in sentiment analysis section. To run the code:

```python
python GenerateWordEmbeddings.py
```

## Visualize the sentence embeddings
You can visualize the sentence embeddings generated either using the in-built TensorFlow Hub modules or using your custom language model. To generate TF Hub's embedding, make use of the functions of `plot_tf1_hub_embeddings` or `plot_tf2_hub_embeddings`. To convert the word embeddings generated through your language model into sentence embeddings, you will have to make use of `plot_learned_embeddings` function. Look into the [main function](./PlotEmbeddings.py) for more details. To run the code:

```python
python PlotEmbeddings.py
```

These functions will generate the vector files which can be visualized in TensorFlow's [embedding projector website](https://projector.tensorflow.org/).

## Filter embeddings
Based on the dataset you have generated, you may have to filter out certain samples which doesn't fit your financial analysis type of dataset. In other words, these may correspond to outliers when you are visualizing the sentence embeddings. To generate filtered embeddings, run the code of:

```python
python FilterEmbeddings.py
```

## Training models
You can train the model in three different ways. Each of them you may have configure their respective main functions to set the required values of parameter. You can run them using the code of:
### a) LSTM
```python
python RunLSTM.py
```
Here is the learning curve plot what I got during training.
![LSTM Learning](https://user-images.githubusercontent.com/13696749/73459535-0ae61a00-439d-11ea-97eb-ca17f5b2b096.png "LSTM Learning")  

### b) ARIMA
```python
python RunArima.py
```
Here is the learning curve plot what I got during training.
![ARIMA Learning](https://user-images.githubusercontent.com/13696749/73512679-859c4d00-4410-11ea-98c5-c4649a24a58a.png "ARIMA Learning")


### c) TensorFlow Probability
```python
python RunSTS.py
```
Here is the learning curve plot what I got during training.
![TensorFlow Probability Learning](https://user-images.githubusercontent.com/13696749/73512720-a9f82980-4410-11ea-9e4e-dfc20d0dafdc.png "TensorFlow Probability Learning")

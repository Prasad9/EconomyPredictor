from google.cloud import language_v1
from google.cloud.language_v1 import enums

from Constants import *

class GenerateSentiment:
    def __init__(self):
        self._client = language_v1.LanguageServiceClient()

        # text_content = 'I am so happy and joyful.'

        # Available types: PLAIN_TEXT, HTML
        self._text_type = enums.Document.Type.PLAIN_TEXT

        # Optional. If not specified, the language is automatically detected.
        # For list of supported languages:
        # https://cloud.google.com/natural-language/docs/languages
        self._language = "en"

        # Available values: NONE, UTF8, UTF16, UTF32
        self._encoding_type = enums.EncodingType.UTF8

    def generate_sentiment(self, text_content):
        document = {"content": text_content, "type": self._text_type, "language": self._language}

        response = self._client.analyze_sentiment(document, encoding_type=self._encoding_type)

        document_sentiment_score = response.document_sentiment.score
        document_sentiment_magnitude = response.document_sentiment.magnitude

        sentence_sentiment_score = []
        sentence_sentiment_magnitude = []
        # Get sentiment for all sentences in the document
        for sentence in response.sentences:
            # sentence_text = sentence.text.content
            sentence_sentiment_score.append(sentence.sentiment.score)
            sentence_sentiment_magnitude.append(sentence.sentiment.magnitude)

        return {
            kDocSentimentScore: document_sentiment_score,
            kDocSentimentMagnitude: document_sentiment_magnitude,
            kSenSentimentScore: sentence_sentiment_score,
            kSenSentimentMagnitude: sentence_sentiment_magnitude
        }
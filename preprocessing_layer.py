# preprocessing_layer.py
import spacy
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from typing import Dict

class PreprocessingLayer:
    def __init__(self, model="en_core_web_sm"):
        self.nlp = spacy.load(model)
        self.vader = SentimentIntensityAnalyzer()
        self.output = {}

    def process(self, message: str) -> Dict:
        doc = self.nlp(message)
        tokens = []
        pos_tags = []
        lemmas = []
        wh_tags = []
        sentiment_tags = []

        for token in doc:
            if token.is_alpha:
                tokens.append(token.text)
                pos_tags.append(f"POS={token.tag_}")
                lemmas.append(f"lemma={token.lemma_}")
                if token.tag_ in ["WDT", "WP", "WRB", "WP$", "MD"]:
                    wh_tags.append(f"WH={token.text.lower()}")

        # VADER Sentiment Analysis
        sentiment_scores = self.vader.polarity_scores(message)
        compound = sentiment_scores['compound']
        if compound >= 0.05:
            sentiment_tags.append("sentiment=positive")
        elif compound <= -0.05:
            sentiment_tags.append("sentiment=negative")
        else:
            sentiment_tags.append("sentiment=neutral")

        features = pos_tags + lemmas + wh_tags + sentiment_tags
        self.output = {
            "features": sorted(set(features))
        }
        return self.output["features"]

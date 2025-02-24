from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

analyzer = SentimentIntensityAnalyzer()

def analyze_sentiment(news):
    if not news or not isinstance(news, list):
        return "Neutral"

    scores = [
        analyzer.polarity_scores(article.get("content", ""))["compound"]
        for article in news if "content" in article
    ]

    avg_sentiment = sum(scores) / len(scores) if scores else 0
    return "Positive" if avg_sentiment > 0 else "Negative"

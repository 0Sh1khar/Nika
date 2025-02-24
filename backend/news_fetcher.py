import requests

API_KEY = "a24a81d1361d484aa6e6b1d42392c385"

def fetch_stock_news(symbol):
    url = f"https://newsapi.org/v2/everything?q={symbol}&apiKey={API_KEY}"

    try:
        response = requests.get(url)
        response.raise_for_status()  # Raise an error for HTTP errors (e.g., 404, 500)

        data = response.json()

        # Validate response structure
        if "articles" not in data or not isinstance(data["articles"], list):
            return {"error": "Invalid response from news API"}

        # Extract first 5 articles
        articles = data["articles"][:5]
        if not articles:
            return {"error": "No news found"}

        return [
            {
                "title": article.get("title", "No title available"),
                "content": article.get("description", "No description available"),
                "url": article.get("url", "#"),
                "image": article.get("urlToImage", "https://via.placeholder.com/150")
            }
            for article in articles
        ]

    except requests.exceptions.RequestException as e:
        return {"error": f"Failed to fetch news: {str(e)}"}

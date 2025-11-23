# -------------------------------
# AI Financial Sentiment Tracker v2 (multi-day)
# -------------------------------

import requests
import pandas as pd
from transformers import pipeline
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import time

import yfinance as yf

def fetch_stock_price(symbol="TSLA", days=7):
    end = datetime.now()
    start = end - timedelta(days=days)
    df = yf.download(symbol, start=start, end=end)
    df = df.reset_index()
    df["date"] = df["Date"].dt.strftime("%Y-%m-%d")
    return df[["date", "Close"]]

def compare_sentiment_vs_price(sent_summary, stock_df):
    merged = sent_summary.merge(stock_df, on="date", how="outer").sort_values("date")

    plt.figure(figsize=(9, 4))
    plt.plot(merged["date"], merged["avg_sentiment"], marker="o", label="Sentiment")
    plt.plot(merged["date"], merged["Close"], marker="s", label="Stock Price")

    plt.title("Sentiment vs Tesla Stock Price")
    plt.xlabel("Date")
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()

    print(merged)  # debug
    return merged


# -------------------------------
# 1️⃣ CONFIG
# -------------------------------
NEWS_API_KEY = "2c5223b815864bdba1b2a691f5950b1c"  # <-- replace this
COMPANY = "Tenneco Clean air India"  # company to track
DAYS = 7  # number of past days

# -------------------------------
# 2️⃣ FETCH NEWS (multi-day)
# -------------------------------
def fetch_news(company, days=5):
    all_articles = []
    to_date = datetime.now()

    for i in range(days):
        day_to = to_date - timedelta(days=i)
        day_from = day_to - timedelta(days=1)

        url = (
            f"https://newsapi.org/v2/everything?"
            f"q={company}&"
            f"from={day_from.date()}&"
            f"to={day_to.date()}&"
            f"language=en&sortBy=publishedAt&"
            f"pageSize=100&"
            f"apiKey={NEWS_API_KEY}"
        )

        print(f"Fetching news for {day_from.date()} → {day_to.date()}")
        response = requests.get(url)
        data = response.json().get("articles", [])
        time.sleep(1)  # small delay to avoid rate limit

        for a in data:
            all_articles.append({
                "date": a["publishedAt"][:10],
                "title": a["title"],
                "source": a["source"]["name"]
            })

    return pd.DataFrame(all_articles)

# -------------------------------
# 3️⃣ SENTIMENT ANALYSIS
# -------------------------------
def analyze_sentiment(df):
    finbert = pipeline("sentiment-analysis", model="ProsusAI/finbert")
    scores = []
    for text in df["title"]:
        result = finbert(text)[0]
        if result["label"] == "positive":
            score = result["score"]
        elif result["label"] == "negative":
            score = -result["score"]
        else:
            score = 0
        scores.append(score)
    df["sentiment_score"] = scores
    return df

# -------------------------------
# 4️⃣ SUMMARIZE & VISUALIZE
# -------------------------------
def summarize_sentiment(df):
    summary = df.groupby("date")["sentiment_score"].mean().reset_index()
    summary.columns = ["date", "avg_sentiment"]

    plt.figure(figsize=(8, 4))
    plt.plot(summary["date"], summary["avg_sentiment"], marker='o', linewidth=2)
    plt.title(f"{COMPANY} Sentiment Trend (Last {DAYS} Days)")
    plt.xlabel("Date")
    plt.ylabel("Average Sentiment (-1 to +1)")
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    return summary

# -------------------------------
# 5️⃣ MAIN
# -------------------------------
if __name__ == "__main__":
    print(f"🔍 Fetching multi-day news for {COMPANY}...")
    df = fetch_news(COMPANY, DAYS)
    print(f"✅ Got {len(df)} articles.")

    if not df.empty:
        print("🧠 Running FinBERT sentiment analysis...")
        analyzed = analyze_sentiment(df)

        print("📊 Creating daily summary...")
        summary = summarize_sentiment(analyzed)

        # Save results
        analyzed.to_csv(f"{COMPANY}_news_sentiment.csv", index=False)
        summary.to_csv(f"{COMPANY}_sentiment_summary.csv", index=False)
        print("💾 Results saved.")
        print("📉 Fetching Tesla stock prices for comparison...")
        stock_df = fetch_stock_price("TSLA", DAYS)
        stock_df.to_csv("Tesla_stock_price.csv", index=False)

        print("📊 Comparing sentiment with real price movement...")
        merged = compare_sentiment_vs_price(summary, stock_df)
        merged.to_csv("Tesla_sentiment_vs_price.csv", index=False)

        print("✅ Comparison complete. Files saved.")
    else:
        print("⚠️ No news articles found. Try different keywords or increase DAYS.")
        

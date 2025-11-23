# 📈 AI Financial Sentiment Tracker
### Multi-day Financial News Sentiment Analysis + Stock Price Comparison (FinBERT + yfinance)

This project analyzes **financial news sentiment** for a given company using **FinBERT** and compares it with real **stock price movement** over multiple days.

It:
- Fetches news headlines using **NewsAPI**
- Runs **FinBERT** sentiment analysis on each headline
- Aggregates **daily average sentiment**
- Fetches stock prices via **yfinance**
- Plots **Sentiment vs Stock Price**
- Exports all key results to CSV

---

## 🔍 What This Project Does

1. **Fetch News**  
   Uses the NewsAPI `/v2/everything` endpoint to retrieve headlines for a given company over the past `DAYS` days.

2. **Run FinBERT Sentiment Analysis**  
   Each headline is scored as:
   - Positive → `+score`
   - Negative → `-score`
   - Neutral → `0`

3. **Daily Sentiment Summary**  
   Sentiment scores are grouped by date to compute **average sentiment per day**.

4. **Fetch Stock Price Data**  
   Uses `yfinance` to download daily closing prices for a chosen stock symbol (currently **TSLA** by default).

5. **Compare Sentiment vs Stock Price**  
   Merges daily sentiment with stock prices and plots both time series together.

6. **Save Outputs**  
   Saves:
   - All articles with sentiment scores
   - Daily sentiment summary
   - Stock price data
   - Sentiment vs stock price merged data

---

## 🧱 Tech Stack

- **Python**
- **NewsAPI** (news data)
- **Hugging Face Transformers** (`ProsusAI/finbert`)
- **PyTorch** (model backend)
- **pandas** (data wrangling)
- **matplotlib** (visualization)
- **yfinance** (stock data)

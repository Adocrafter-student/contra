import pandas as pd
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import os

# Ensure VADER lexicon is downloaded
try:
    nltk.data.find('sentiment/vader_lexicon.zip')
except LookupError:
    print("Downloading VADER lexicon...")
    nltk.download('vader_lexicon')

def analyze_sentiment(input_file="news_data.csv", output_file="generated_events.csv"):
    """
    Reads news data, performs sentiment analysis, and generates an events CSV.
    """
    print(f"Reading data from {input_file}...")
    if not os.path.exists(input_file):
        print(f"Error: {input_file} not found.")
        return

    try:
        df = pd.read_csv(input_file)
    except Exception as e:
        print(f"Error reading CSV: {e}")
        return

    if "date" not in df.columns or "text" not in df.columns:
        print("Error: Input CSV must have 'date' and 'text' columns.")
        return

    print("Initializing VADER sentiment analyzer...")
    sia = SentimentIntensityAnalyzer()

    events = []

    print("Analyzing news items...")
    for _, row in df.iterrows():
        text = str(row["text"])
        date = row["date"]
        
        scores = sia.polarity_scores(text)
        compound = scores["compound"]
        
        event_type = "neutral"
        if compound > 0.05:
            event_type = "positive"
        elif compound < -0.05:
            event_type = "negative"
        
        # We only care about positive or negative events for the model
        if event_type != "neutral":
            events.append({
                "date": date,
                "event_type": event_type,
                "description": text  # Using the headline/text as description
            })

    if not events:
        print("No significant events found.")
        return

    events_df = pd.DataFrame(events)
    
    # Sort by date
    events_df["date"] = pd.to_datetime(events_df["date"])
    events_df = events_df.sort_values("date")
    
    print(f"Saving {len(events_df)} events to {output_file}...")
    events_df.to_csv(output_file, index=False)
    print("Done.")

if __name__ == "__main__":
    analyze_sentiment()

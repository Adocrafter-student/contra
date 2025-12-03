"""
Fetch Wikipedia pageviews and Google Trends data for controversy events
Outputs Excel file with sentiment data for Tesla, Amazon, and Meta events
"""

import pandas as pd
import requests
from urllib.parse import quote
from datetime import datetime, timedelta
from pytrends.request import TrendReq
import time

# Event data: Tesla (from events.csv) + Amazon + Meta
EVENTS = [
    # Tesla events
    {"date": "2018-08-07", "company": "Tesla", "ceo": "Elon Musk", "event": "Funding secured tweet", "type": "positive"},
    {"date": "2018-08-17", "company": "Tesla", "ceo": "Elon Musk", "event": "SEC investigation announced", "type": "negative"},
    {"date": "2018-09-07", "company": "Tesla", "ceo": "Elon Musk", "event": "Joe Rogan podcast", "type": "negative"},
    {"date": "2018-09-27", "company": "Tesla", "ceo": "Elon Musk", "event": "SEC lawsuit filed", "type": "negative"},
    {"date": "2018-10-04", "company": "Tesla", "ceo": "Elon Musk", "event": "SEC settlement", "type": "positive"},
    {"date": "2019-02-19", "company": "Tesla", "ceo": "Elon Musk", "event": "SEC contempt motion", "type": "negative"},
    {"date": "2019-04-11", "company": "Tesla", "ceo": "Elon Musk", "event": "Autonomy Day", "type": "positive"},
    {"date": "2019-10-23", "company": "Tesla", "ceo": "Elon Musk", "event": "Q3 profit surprise", "type": "positive"},
    {"date": "2020-01-29", "company": "Tesla", "ceo": "Elon Musk", "event": "Q4 earnings beat", "type": "positive"},
    {"date": "2020-02-03", "company": "Tesla", "ceo": "Elon Musk", "event": "Stock crosses $900", "type": "positive"},
    {"date": "2020-02-19", "company": "Tesla", "ceo": "Elon Musk", "event": "Stock drops 17%", "type": "negative"},
    {"date": "2020-03-18", "company": "Tesla", "ceo": "Elon Musk", "event": "COVID factory controversy", "type": "negative"},
    {"date": "2020-05-01", "company": "Tesla", "ceo": "Elon Musk", "event": "Stock price too high tweet", "type": "negative"},
    {"date": "2020-07-22", "company": "Tesla", "ceo": "Elon Musk", "event": "Q2 earnings beat", "type": "positive"},
    {"date": "2020-12-21", "company": "Tesla", "ceo": "Elon Musk", "event": "S&P 500 inclusion", "type": "positive"},
    
    # Amazon events
    {"date": "2018-11-13", "company": "Amazon", "ceo": "Jeff Bezos", "event": "HQ2 announcement (NYC/Arlington)", "type": "positive"},
    {"date": "2019-01-09", "company": "Amazon", "ceo": "Jeff Bezos", "event": "Divorce announcement", "type": "negative"},
    {"date": "2019-02-07", "company": "Amazon", "ceo": "Jeff Bezos", "event": "National Enquirer extortion claims", "type": "negative"},
    {"date": "2019-02-14", "company": "Amazon", "ceo": "Jeff Bezos", "event": "HQ2 NYC cancellation", "type": "negative"},
    {"date": "2019-07-15", "company": "Amazon", "ceo": "Jeff Bezos", "event": "Prime Day technical outage", "type": "negative"},
    {"date": "2019-07-16", "company": "Amazon", "ceo": "Jeff Bezos", "event": "Antitrust hearing announced", "type": "negative"},
    {"date": "2020-04-30", "company": "Amazon", "ceo": "Jeff Bezos", "event": "Q1 earnings beat (COVID boost)", "type": "positive"},
    {"date": "2020-07-29", "company": "Amazon", "ceo": "Jeff Bezos", "event": "Congress antitrust testimony", "type": "negative"},
    
    # Meta/Facebook events
    {"date": "2018-03-17", "company": "Meta", "ceo": "Mark Zuckerberg", "event": "Cambridge Analytica scandal", "type": "negative"},
    {"date": "2018-04-10", "company": "Meta", "ceo": "Mark Zuckerberg", "event": "Congress testimony Day 1", "type": "negative"},
    {"date": "2018-04-11", "company": "Meta", "ceo": "Mark Zuckerberg", "event": "Congress testimony Day 2", "type": "negative"},
    {"date": "2018-07-26", "company": "Meta", "ceo": "Mark Zuckerberg", "event": "Stock crashes 19% on earnings", "type": "negative"},
    {"date": "2018-11-14", "company": "Meta", "ceo": "Mark Zuckerberg", "event": "NY Times scandal report", "type": "negative"},
    {"date": "2019-03-13", "company": "Meta", "ceo": "Mark Zuckerberg", "event": "Major platform outage", "type": "negative"},
    {"date": "2019-10-23", "company": "Meta", "ceo": "Mark Zuckerberg", "event": "Congress testimony on Libra", "type": "negative"},
    {"date": "2020-07-29", "company": "Meta", "ceo": "Mark Zuckerberg", "event": "Antitrust hearing with Bezos", "type": "negative"},
]

# Wikipedia page mappings
WIKI_PAGES = {
    "Tesla": ["Tesla,_Inc.", "Elon_Musk"],
    "Amazon": ["Amazon_(company)", "Jeff_Bezos"],
    "Meta": ["Meta_Platforms", "Mark_Zuckerberg"]
}

# Google Trends keyword mappings  
TRENDS_KEYWORDS = {
    "Tesla": ["Tesla", "Elon Musk"],
    "Amazon": ["Amazon", "Jeff Bezos"],
    "Meta": ["Facebook", "Mark Zuckerberg"]  # Facebook brand name in 2018-2020
}

def fetch_wiki_pageviews(page_title, date_str):
    """Fetch Wikipedia pageviews for a specific date"""
    try:
        date_obj = datetime.strptime(date_str, "%Y-%m-%d")
        date_formatted = date_obj.strftime("%Y%m%d")
        
        url = f"https://wikimedia.org/api/rest_v1/metrics/pageviews/per-article/en.wikipedia/all-access/all-agents/{quote(page_title)}/daily/{date_formatted}/{date_formatted}"
        
        headers = {'User-Agent': 'CONTRA-Research/1.0 (Educational Project)'}
        response = requests.get(url, headers=headers, timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            if data.get("items"):
                return data["items"][0]["views"]
        return None
    except Exception as e:
        print(f"  ⚠ Wiki error for {page_title} on {date_str}: {e}")
        return None

def fetch_trends_for_week(keywords, center_date):
    """Fetch Google Trends for a week around the event date"""
    try:
        date_obj = datetime.strptime(center_date, "%Y-%m-%d")
        start = (date_obj - timedelta(days=3)).strftime("%Y-%m-%d")
        end = (date_obj + timedelta(days=3)).strftime("%Y-%m-%d")
        
        pytrends = TrendReq(hl="en-US", tz=0, retries=2, backoff_factor=0.2)
        pytrends.build_payload(keywords, timeframe=f"{start} {end}", geo="US")
        
        df = pytrends.interest_over_time()
        
        if not df.empty and "isPartial" in df.columns:
            df = df.drop(columns=["isPartial"])
        
        if not df.empty:
            # Get value for the specific date
            date_value = df.loc[date_obj.strftime("%Y-%m-%d")] if date_obj.strftime("%Y-%m-%d") in df.index else None
            if date_value is not None:
                return date_value.mean()  # Average across keywords
        
        return None
    except Exception as e:
        print(f"  ⚠ Trends error for {keywords} on {center_date}: {e}")
        return None

def main():
    print("=" * 80)
    print("FETCHING SENTIMENT DATA FOR EVENTS")
    print("=" * 80)
    print(f"Total events: {len(EVENTS)}")
    print(f"  Tesla: {sum(1 for e in EVENTS if e['company'] == 'Tesla')}")
    print(f"  Amazon: {sum(1 for e in EVENTS if e['company'] == 'Amazon')}")
    print(f"  Meta: {sum(1 for e in EVENTS if e['company'] == 'Meta')}")
    print()
    
    results = []
    
    for i, event in enumerate(EVENTS, 1):
        print(f"[{i}/{len(EVENTS)}] {event['date']} - {event['company']}: {event['event']}")
        
        company = event['company']
        wiki_pages = WIKI_PAGES[company]
        trends_keywords = TRENDS_KEYWORDS[company]
        
        # Fetch Wikipedia pageviews
        wiki_views = []
        for page in wiki_pages:
            views = fetch_wiki_pageviews(page, event['date'])
            if views is not None:
                wiki_views.append(views)
            time.sleep(0.2)  # Rate limiting
        
        avg_wiki_views = sum(wiki_views) / len(wiki_views) if wiki_views else None
        
        # Fetch Google Trends
        trends_value = fetch_trends_for_week(trends_keywords, event['date'])
        time.sleep(1)  # Rate limiting for Trends API
        
        results.append({
            'Date': event['date'],
            'Company': company,
            'CEO': event['ceo'],
            'Event': event['event'],
            'Type': event['type'],
            'Wiki Pageviews': int(avg_wiki_views) if avg_wiki_views else '',
            'Google Trends': round(float(trends_value), 1) if trends_value is not None else ''
        })
        
        print(f"  Wiki: {avg_wiki_views if avg_wiki_views else 'N/A'} | Trends: {trends_value if trends_value else 'N/A'}")
    
    # Create DataFrame
    df = pd.DataFrame(results)
    
    # Save to Excel
    excel_filename = 'sentiment_data_events.xlsx'
    df.to_excel(excel_filename, index=False, sheet_name='Event Sentiment Data')
    
    print()
    print("=" * 80)
    print("DATA SAVED!")
    print("=" * 80)
    print(f"✓ Excel file: {excel_filename}")
    print(f"✓ Total records: {len(df)}")
    print()
    print("Summary by company:")
    print(df.groupby('Company').size())
    print()
    print("Open the Excel file to view Wikipedia pageviews and Google Trends data!")

if __name__ == "__main__":
    main()


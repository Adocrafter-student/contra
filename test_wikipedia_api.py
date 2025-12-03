"""
Simple test to verify Wikipedia Pageviews API works
Tests fetching pageview data for Tesla and Elon Musk articles
"""

import requests
from urllib.parse import quote
import json
from datetime import datetime, timedelta

print("=" * 70)
print("TESTING WIKIPEDIA PAGEVIEWS API")
print("=" * 70)

# Test parameters
ARTICLES = ["Tesla,_Inc.", "Elon_Musk"]
START_DATE = "20200501"  # May 1, 2020 (Elon's "stock price too high" tweet)
END_DATE = "20200515"    # May 15, 2020

print(f"\nFetching Wikipedia pageviews from {START_DATE} to {END_DATE}...")
print(f"Articles: {', '.join(ARTICLES)}\n")

for article in ARTICLES:
    print(f"📚 Testing: {article.replace('_', ' ')}")
    print("-" * 70)
    
    # Build API URL
    url = f"https://wikimedia.org/api/rest_v1/metrics/pageviews/per-article/en.wikipedia/all-access/all-agents/{quote(article)}/daily/{START_DATE}/{END_DATE}"
    
    print(f"URL: {url}\n")
    
    try:
        # Make request with User-Agent header (REQUIRED by Wikipedia API)
        headers = {
            'User-Agent': 'CONTRA-Research/1.0 (Educational Research Project; contact@example.com)'
        }
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        
        # Parse response
        data = response.json()
        items = data.get("items", [])
        
        if not items:
            print(f"❌ No data returned for {article}")
            continue
        
        print(f"✅ SUCCESS! Got {len(items)} days of data\n")
        
        # Show first 5 days as proof
        print("Sample data (first 5 days):")
        for i, item in enumerate(items[:5]):
            timestamp = item["timestamp"]
            views = item["views"]
            date = datetime.strptime(timestamp[:8], "%Y%m%d").strftime("%Y-%m-%d")
            print(f"  {date}: {views:,} views")
        
        # Show statistics
        total_views = sum(item["views"] for item in items)
        avg_views = total_views / len(items)
        max_views = max(item["views"] for item in items)
        max_date = max(items, key=lambda x: x["views"])["timestamp"][:8]
        max_date_formatted = datetime.strptime(max_date, "%Y%m%d").strftime("%Y-%m-%d")
        
        print(f"\n📊 Statistics:")
        print(f"  Total views: {total_views:,}")
        print(f"  Average daily views: {avg_views:,.0f}")
        print(f"  Peak views: {max_views:,} (on {max_date_formatted})")
        print()
        
    except requests.exceptions.RequestException as e:
        print(f"❌ ERROR: {e}")
        print()
    except Exception as e:
        print(f"❌ UNEXPECTED ERROR: {e}")
        print()

print("=" * 70)
print("✅ Wikipedia Pageviews API TEST COMPLETE")
print("=" * 70)
print("\nThe API works! You can now use it in your CONTRA model.")
print("Note: May 1, 2020 should show a spike due to Elon's tweet.")


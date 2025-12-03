# Assignment 4: Dataset & Methodology

## 1. Dataset Description

To rigorously evaluate the impact of CEO-centric sentiment on stock performance, this study constructs a multi-source dataset spanning from 2018 to 2020. This period was selected to capture high-volatility events, including the "funding secured" tweet saga and the COVID-19 market crash. The dataset integrates traditional financial metrics with alternative data streams representing public attention and sentiment.

### Financial Data
The core financial data for this research is sourced from the Yahoo Finance API via the `yfinance` Python library. We focus on three major technology companies with high-profile CEOs: Tesla Inc. (TSLA), Amazon (AMZN), and Meta Platforms (META). For each company, we collect daily historical price data, specifically utilizing the "Adjusted Close" price to account for any stock splits or dividend adjustments that occurred during the observation window. To provide a baseline for market performance and to calculate systematic risk (beta), we also retrieve data for the S&P 500 index (^GSPC). This financial data serves as the ground truth for our predictive models, allowing us to measure the deviation between expected market returns and actual price movements following sentiment shocks.

**Table 1: Sample Financial Data (Tesla, Inc.)**
| Date | Open | High | Low | Close | Volume |
| --- | --- | --- | --- | --- | --- |
| 2020-01-02 | 28.3 | 28.71 | 28.11 | 28.68 | 142981500 |
| 2020-01-03 | 29.37 | 30.27 | 29.13 | 29.53 | 266677500 |
| 2020-01-06 | 29.36 | 30.1 | 29.33 | 30.1 | 151995000 |
| 2020-01-07 | 30.76 | 31.44 | 30.22 | 31.27 | 268231500 |

### Sentiment and Event Data
Complementing the financial records, we employ a dual-layered approach to capture the "human factor." First, we utilize Google Trends and Wikipedia Pageviews to measure organic public interest and attention spikes. We track daily search volumes for terms like "Tesla" and "Elon Musk," as well as "Amazon" with "Jeff Bezos" and "Meta" with "Mark Zuckerberg." These metrics serve as a proxy for the intensity of public attention, helping to identify moments when a company or its leader enters the collective consciousness.

Second, we introduce a Custom Event Kernel, a manually curated dataset of significant CEO-related events. This kernel includes 35 distinct events across the three companies, ranging from regulatory investigations and earnings surprises to viral social media moments. Each event is assigned a structured sentiment weight based on its nature (e.g., a privacy scandal is weighted negatively, while a successful product launch is positive). This structured approach allows us to quantify the severity and direction of "sentiment shocks" that purely automated text analysis might misinterpret.

**Table 2: Sample Event & Sentiment Data**
| Date | Company | Event Description | Sentiment Score | Google Trends Index |
| --- | --- | --- | --- | --- |
| 2018-08-07 | Tesla | Elon Musk tweets "funding secured" | -0.85 | 100 |
| 2018-09-07 | Tesla | Elon Musk smokes weed on podcast | -0.6 | 85 |
| 2019-11-21 | Tesla | Cybertruck window shatters | -0.45 | 92 |
| 2020-05-01 | Tesla | Musk tweets "stock price too high" | -0.9 | 95 |

# Configuration settings for the investor tracking system

# List of notable investors to track with their CIK numbers
# INVESTORS = {
#     "Warren Buffett (Berkshire Hathaway)": "0001067983",
#     "Ray Dalio (Bridgewater)": "0001350694",
#     "Bill Ackman (Pershing Square)": "0001336528",
#     "Renaissance Technologies": "0001037389",
#     "BlackRock": "0001364742"
# }
INVESTORS = {
    "Warren Buffett (Berkshire Hathaway)": "0001067983",
    "Michael Burry (Scion Asset Management)": "0001649339",
    # "Ray Dalio (Bridgewater)": "0001350694",
    "Bill Ackman (Pershing Square)": "0001336528",
    # "Jim Simons (Renaissance Technologies)": "0001037389",
    "Mohnish Pabrai (Pabrai Investment Funds)": "0001173334"
}
# List of companies to track
COMPANIES = ["AAPL", "MSFT", "AMZN", "GOOGL", "META", "TSLA", "BRK-B", "JPM", "V", "JNJ", "PLTR", "NVDA", "ASML", "QCOM", "INTC", "AMD", "MU", "TSM"]
# COMPANIES = ["AAPL"]

# List of commodities to track
COMMODITIES = ["GC=F", "SI=F", "CL=F", "NG=F", "BTC-USD", "ETH-USD"]
# COMMODITIES = ["GC=F"]

# News sources
NEWS_SOURCES = {
    "WSJ": "https://www.wsj.com/search?query=",
    "Financial Times": "https://www.ft.com/search?q=",
    "Yahoo Finance": "https://finance.yahoo.com/quote/",
    "MarketWatch": "https://www.marketwatch.com/search?q="
}

# SEC EDGAR API settings
SEC_API_HEADERS = {
    'User-Agent': 'InvestorTrackingTool abhi.dtu11@gmail.com'  # Replace with your actual information
}

# Number of previous quarters to compare for investment changes
QUARTERS_TO_TRACK = 4

# Maximum number of news articles to retrieve per company
MAX_ARTICLES_PER_COMPANY = 20

# Waiting time between requests (in seconds) to avoid rate limiting
REQUEST_DELAY = 2

DATA_DIR = 'data'
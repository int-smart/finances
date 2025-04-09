import time
import pandas as pd
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager
from src.config import COMPANIES, NEWS_SOURCES, MAX_ARTICLES_PER_COMPANY, REQUEST_DELAY
import requests
import os
from dotenv import load_dotenv
from datetime import datetime


load_dotenv()

class NewsTracker:
    def __init__(self, tickers=COMPANIES, headless=True):
        """Initialize the news scraper with a Selenium WebDriver"""
        chrome_options = Options()
        if headless:
            chrome_options.add_argument("--headless")
        chrome_options.add_argument("--disable-gpu")
        chrome_options.add_argument("--window-size=1920,1080")
        chrome_options.add_argument("--disable-extensions")
        chrome_options.add_argument("--no-sandbox")
        chrome_options.add_argument("--disable-dev-shm-usage")
        
        # Specify Chrome binary location if needed
        # chrome_options.binary_location = "/usr/bin/google-chrome"  # Uncomment and adjust path as needed
        
        try:
            self.driver = webdriver.Chrome(
                service=Service(ChromeDriverManager().install()),
                options=chrome_options
            )
        except Exception as e:
            print(f"Error initializing Chrome WebDriver: {e}")
            print("Trying alternative approach...")
            try:
                # Try with explicit driver path
                driver_path = ChromeDriverManager().install()
                print(f"Using ChromeDriver at: {driver_path}")
                self.driver = webdriver.Chrome(
                    service=Service(driver_path),
                    options=chrome_options
                )
            except Exception as e:
                print(f"Failed to initialize Chrome WebDriver: {e}")
                raise
        
        self.articles = {}
        self.tickers = tickers
        
    def scrape_wsj(self, company, max_articles=MAX_ARTICLES_PER_COMPANY):
        """Scrape Wall Street Journal articles about a company"""
        print(f"Scraping WSJ for news about {company}...")
        search_url = f"{NEWS_SOURCES['WSJ']}{company}"
        
        self.driver.get(search_url)
        time.sleep(3)  # Allow time for the page to load
        
        articles = []
        try:
            article_elements = WebDriverWait(self.driver, 10).until(
                EC.presence_of_all_elements_located((By.CSS_SELECTOR, "article.article"))
            )
            
            for article in article_elements[:max_articles]:
                try:
                    headline = article.find_element(By.CSS_SELECTOR, "h3").text
                    summary = article.find_element(By.CSS_SELECTOR, "p").text
                    date = article.find_element(By.CSS_SELECTOR, "p.WSJTheme--timestamp--1K7Z5rGT").text
                    link = article.find_element(By.CSS_SELECTOR, "h3 a").get_attribute("href")
                    
                    articles.append({
                        "headline": headline,
                        "summary": summary,
                        "date": date,
                        "link": link,
                        "source": "Wall Street Journal"
                    })
                except Exception as e:
                    print(f"Error extracting article details: {e}")
        except Exception as e:
            print(f"Error scraping WSJ for {company}: {e}")
        
        return articles
    
    def scrape_ft(self, company, max_articles=MAX_ARTICLES_PER_COMPANY):
        """Scrape Financial Times articles about a company"""
        print(f"Scraping Financial Times for news about {company}...")
        search_url = f"{NEWS_SOURCES['Financial Times']}{company}"
        
        self.driver.get(search_url)
        time.sleep(3)  # Allow time for the page to load
        
        articles = []
        try:
            article_elements = WebDriverWait(self.driver, 10).until(
                EC.presence_of_all_elements_located((By.CSS_SELECTOR, "li.o-teaser-collection__item"))
            )
            
            for article in article_elements[:max_articles]:
                try:
                    headline = article.find_element(By.CSS_SELECTOR, "div.o-teaser__heading").text
                    summary_element = article.find_elements(By.CSS_SELECTOR, "p.o-teaser__standfirst")
                    summary = summary_element[0].text if summary_element else "No summary available"
                    date_element = article.find_elements(By.CSS_SELECTOR, "div.o-teaser__timestamp")
                    date = date_element[0].text if date_element else "No date available"
                    link = article.find_element(By.CSS_SELECTOR, "a.js-teaser-heading-link").get_attribute("href")
                    
                    articles.append({
                        "headline": headline,
                        "summary": summary,
                        "date": date,
                        "link": link,
                        "source": "Financial Times"
                    })
                except Exception as e:
                    print(f"Error extracting article details: {e}")
        except Exception as e:
            print(f"Error scraping FT for {company}: {e}")
        
        return articles
    
    def scrape_yahoo_finance(self, company, max_articles=MAX_ARTICLES_PER_COMPANY):
        """Scrape Yahoo Finance articles about a company"""
        print(f"Scraping Yahoo Finance for news about {company}...")
        search_url = f"https://finance.yahoo.com/quote/{company}/news"
        
        self.driver.get(search_url)
        time.sleep(REQUEST_DELAY)  # Allow time for the page to load

        articles = []
        try:
            # Accept cookies if a consent dialog appears
            try:
                consent_button = WebDriverWait(self.driver, 5).until(
                    EC.element_to_be_clickable((By.CSS_SELECTOR, "button[name='agree']"))
                )
                consent_button.click()
                time.sleep(1)
            except:
                pass  # No consent dialog or different selector

            # Find the news articles
            article_elements = WebDriverWait(self.driver, 10).until(
                EC.presence_of_all_elements_located((By.CSS_SELECTOR, "li.story-item"))
            )
            print(f"Found {len(article_elements)} articles.")
            for article in article_elements[:max_articles]:
                try:
                    headline = article.find_element(By.CSS_SELECTOR, "h3").text
                    link_element = article.find_element(By.CSS_SELECTOR, "a")
                    link = link_element.get_attribute("href")

                    # For summary, we either need to click into the article or use the preview if available
                    summary_element = article.find_elements(By.CSS_SELECTOR, "p")
                    summary = summary_element[0].text if summary_element else "No summary available"

                    articles.append({
                        "headline": headline,
                        "summary": summary,
                        "link": link
                    })                
                except Exception as e:
                    print(f"Error extracting article details: {e}")

            # After processing all articles, navigate to each link to get the full content
            for article in articles:
                try:
                    self.driver.get(article["link"])
                    time.sleep(REQUEST_DELAY)  # Allow time for the page to load
                    # Extract full content from the article page
                    content_wrapper = self.driver.find_element(By.CSS_SELECTOR, "div.atoms-wrapper")

                    if content_wrapper:
                        # Find all paragraph elements within the wrapper
                        paragraphs = content_wrapper.find_elements(By.CSS_SELECTOR, "p.yf-1090901")
                        # Join the text from each paragraph with newlines between them
                        full_content = "\n\n".join([p.text for p in paragraphs])
                    else:
                        full_content = "No content available"

                    article["full_content"] = full_content
                except Exception as e:
                    if "retry" in str(e).lower():
                        print(f"Retry error encountered, aborting article processing: {e}")
                        break
                    raise e
                    
        except Exception as e:
            print(f"Error scraping Yahoo Finance for {company}: {e}")
        
        return articles
    
    def scrape_marketwatch(self, company, max_articles=MAX_ARTICLES_PER_COMPANY):
        """Scrape MarketWatch articles about a company"""
        print(f"Scraping MarketWatch for news about {company}...")
        search_url = f"https://www.marketwatch.com/search?q={company}&m=Keyword"
        
        self.driver.get(search_url)
        time.sleep(3)  # Allow time for the page to load
        
        articles = []
        try:
            # First, switch to the iframe where the content is located
            iframe = WebDriverWait(self.driver, 10).until(
                EC.presence_of_element_located((By.TAG_NAME, "iframe"))
            )
            self.driver.switch_to.frame(iframe)
            print("Successfully switched to iframe")
            # import pdb
            # pdb.set_trace()
            article_elements = WebDriverWait(self.driver, 10).until(
                EC.presence_of_all_elements_located((By.CSS_SELECTOR, "div.element.element--article"))
            )
            print(f"Found {len(article_elements)} article elements")
            for article in article_elements[:max_articles]:
                try:
                    headline_element = article.find_element(By.CSS_SELECTOR, "h3.article__headline")
                    headline = headline_element.text
                    link = headline_element.find_element(By.CSS_SELECTOR, "a").get_attribute("href")
                    
                    summary_element = article.find_elements(By.CSS_SELECTOR, "p.article__summary")
                    summary = summary_element[0].text if summary_element else "No summary available"
                    
                    date_element = article.find_elements(By.CSS_SELECTOR, "div.article__details")
                    date = date_element[0].text if date_element else "No date available"
                    
                    articles.append({
                        "headline": headline,
                        "summary": summary,
                        "date": date,
                        "link": link,
                        "source": "MarketWatch"
                    })
                except Exception as e:
                    print(f"Error extracting article details: {e}")
        except Exception as e:
            print(f"Error scraping MarketWatch for {company}: {e}")
        
        return articles
    
    def scrape_reuters(self, company, max_articles=MAX_ARTICLES_PER_COMPANY):
        """Scrape Reuters articles about a company"""
        print(f"Scraping Reuters for news about {company}...")
        search_url = f"https://www.reuters.com/site-search/?query={company}"
        
        self.driver.get(search_url)
        time.sleep(3)  # Allow time for the page to load
        # import pdb
        # pdb.set_trace()
        articles = []
        try:
            # Try to handle cookie consent if it appears
            try:
                cookie_button = WebDriverWait(self.driver, 5).until(
                    EC.element_to_be_clickable((By.XPATH, "//button[contains(text(), 'Accept All')]"))
                )
                cookie_button.click()
                time.sleep(1)
            except:
                print("No cookie consent dialog found or unable to click it")
        
            # Find all article elements
            article_elements = WebDriverWait(self.driver, 10).until(
                EC.presence_of_all_elements_located((By.CSS_SELECTOR, "li[class*='search_results']"))
            )
        
            print(f"Found {len(article_elements)} Reuters articles about {company}")
        
            for article in article_elements[:max_articles]:
                try:
                    # Extract headline
                    headline_element = article.find_element(By.CSS_SELECTOR, "div[class*='search-result']")
                    headline = headline_element.text.strip()
                
                    # Extract link
                    link_element = article.find_element(By.CSS_SELECTOR, "a[data-testid='TitleLink']")
                    link = link_element.get_attribute("href")
                    if link and not link.startswith("http"):
                        link = "https://www.reuters.com" + link

                    # Extract summary
                    summary_element = self.get_full_article_content(link)
                    summary = summary_element[0].text.strip() if summary_element else "No summary available"
                
                    # Extract date
                    date_element = article.find_elements(By.CSS_SELECTOR, "div[class*='date']")
                    date = date_element[0].text.strip() if date_element else "No date available"
                
                    articles.append({
                        "headline": headline,
                        "summary": summary,
                        "date": date,
                        "link": link,
                        "source": "Reuters"
                    })
                except Exception as e:
                    print(f"Error extracting Reuters article details: {e}")
        except Exception as e:
            print(f"Error scraping Reuters for {company}: {e}")
        
        return articles

    def get_full_article_content(self, article_url):
        """Visit an article URL and extract the full article content"""
        print(f"Retrieving full article content from: {article_url}")
        
        self.driver.get(article_url)
        time.sleep(3)  # Allow time for the page to load
        
        # Handle potential cookie/consent dialogs
        try:
            cookie_buttons = self.driver.find_elements(By.XPATH, "//button[contains(text(), 'Accept') or contains(text(), 'I agree') or contains(text(), 'Continue')]")
            if cookie_buttons:
                cookie_buttons[0].click()
                time.sleep(1)
        except Exception as e:
            print(f"Note: {e}")
        
        article_content = ""
        try:
            # For Reuters
            if "reuters.com" in article_url:
                paragraphs = self.driver.find_elements(By.CSS_SELECTOR, "div[class*='paragraph-']")
                article_content = "\n\n".join([p.text for p in paragraphs if p.text.strip()])
            
            # For WSJ
            elif "wsj.com" in article_url:
                paragraphs = self.driver.find_elements(By.CSS_SELECTOR, "p.article-content")
                article_content = "\n\n".join([p.text for p in paragraphs if p.text.strip()])
            
            # For Financial Times
            elif "ft.com" in article_url:
                paragraphs = self.driver.find_elements(By.CSS_SELECTOR, "p.article__content")
                article_content = "\n\n".join([p.text for p in paragraphs if p.text.strip()])
            
            # Generic approach as fallback
            else:
                # Try common article content selectors
                selectors = [
                    "article p", 
                    ".article-body p", 
                    ".story-content p",
                    ".article__body p",
                    "main p"
                ]
                
                for selector in selectors:
                    paragraphs = self.driver.find_elements(By.CSS_SELECTOR, selector)
                    if paragraphs:
                        article_content = "\n\n".join([p.text for p in paragraphs if p.text.strip()])
                        break
            
            # If still no content, try a very broad approach
            if not article_content:
                paragraphs = self.driver.find_elements(By.TAG_NAME, "p")
                article_content = "\n\n".join([p.text for p in paragraphs if len(p.text.strip()) > 50])  # Only paragraphs with substantial text
        
        except Exception as e:
            print(f"Error extracting article content: {e}")
        
        return article_content

    def get_news_for_company(self, company):
        """Get news articles about a company from all configured sources"""
        all_articles = []
        
        yahoo_articles = self.scrape_yahoo_finance(company)
        all_articles.extend(yahoo_articles)
        time.sleep(REQUEST_DELAY)
        
        # marketwatch_articles = self.scrape_marketwatch(company)
        # all_articles.extend(marketwatch_articles)

        # reuters_articles = self.scrape_reuters(company)
        # all_articles.extend(reuters_articles)
    
        self.articles[company] = all_articles
        return all_articles
    
    def get_news_for_tickers(self, tickers):
        """Get news for a list of company tickers"""
        for ticker in tickers:
            self.get_news_for_company(ticker)
            time.sleep(REQUEST_DELAY)
            
        return self.articles

    def track(self):
        print("\n" + "-" * 80)
        print("Scraping news articles...")
        
        try:
            news_data = self.get_news_for_tickers(self.tickers)
            
            print("\nRecent news:")
            for ticker, articles in news_data.items():
                print(f"\n{ticker}:")
                for article in articles[:3]:  # Show top 3 articles
                    print(f"  {article['headline']}")
                    print(f"  {article['summary'][:100]}...")
                    if 'full_content' in article:
                        print(f"  {article['full_content'][:100]}...")        
            return news_data
        finally:
            self.close()        
    
    def close(self):
        """Close the Selenium WebDriver"""
        if self.driver:
            self.driver.quit()

    def get_news_from_fmp(self, ticker, time_from, time_to):
        """Get news from Financial Modeling Prep API for a given ticker and time range."""
        api_key = os.getenv("FMP_API_KEY")
        base_url = "https://financialmodelingprep.com/stable/fmp-articles"
        # base_url = "https://financialmodelingprep.com/stable/news/stock-latest"
        
        # Convert datetime objects to strings if necessary
        if isinstance(time_from, datetime):
            time_from = time_from.strftime('%Y-%m-%d')
        if isinstance(time_to, datetime):
            time_to = time_to.strftime('%Y-%m-%d')

        params = {
            'symbols': ticker,
            'from': time_from,
            'to': time_to,
            'apikey': api_key
        }
        
        try:
            response = requests.get(base_url, params=params)
            response.raise_for_status()  # Raise an error for bad responses
            news_data = response.json()
            
            articles = []
            for article in news_data:
                articles.append({
                    'headline': article.get('title', ''),
                    'link': article.get('url', ''),
                    'full_content': article.get('text', ''),
                    'summary': article.get('text', '')[:200] if article.get('text') else ''
                })
            return articles
            
        except requests.exceptions.RequestException as e:
            print(f"Error fetching news from FMP API: {e}")
            return []
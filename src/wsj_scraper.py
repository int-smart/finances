"""
WSJ Scraper Module using Playwright
Handles browser login and data scraping from Wall Street Journal
"""

import asyncio
import json
import time
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
import re
from pathlib import Path

from playwright.async_api import async_playwright, Browser, BrowserContext, Page
from bs4 import BeautifulSoup
import pandas as pd


class WSJScraper:
    """
    Wall Street Journal scraper using Playwright for authenticated sessions
    """
    
    def __init__(self, headless: bool = True, slow_mo: int = 1000):
        """
        Initialize the WSJ scraper
        
        Args:
            headless: Whether to run browser in headless mode
            slow_mo: Milliseconds to slow down operations (helps avoid detection)
        """
        self.headless = headless
        self.slow_mo = slow_mo
        self.browser: Optional[Browser] = None
        self.context: Optional[BrowserContext] = None
        self.page: Optional[Page] = None
        self.is_logged_in = False
        
        # WSJ URLs
        self.base_url = "https://www.wsj.com"
        self.login_url = "https://accounts.wsj.com/login"
        
        # Session storage path
        self.session_path = Path("wsj_session.json")
    
    async def __aenter__(self):
        """Async context manager entry"""
        await self.start_browser()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        await self.close()
    
    async def start_browser(self):
        """Start the Playwright browser"""
        self.playwright = await async_playwright().start()
        
        # Launch browser with options to avoid detection
        self.browser = await self.playwright.chromium.launch(
            headless=self.headless,
            slow_mo=self.slow_mo,
            args=[
                '--no-sandbox',
                '--disable-blink-features=AutomationControlled',
                '--disable-dev-shm-usage',
                '--disable-gpu',
                '--no-first-run',
                '--no-default-browser-check',
                '--disable-background-timer-throttling',
                '--disable-backgrounding-occluded-windows',
                '--disable-renderer-backgrounding'
            ]
        )
        
        # Create context with realistic user agent and viewport
        self.context = await self.browser.new_context(
            viewport={'width': 1920, 'height': 1080},
            user_agent='Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
        )
        
        # Create page
        self.page = await self.context.new_page()
        
        # Add stealth measures
        await self.page.add_init_script("""
            Object.defineProperty(navigator, 'webdriver', {
                get: () => undefined,
            })
        """)
    
    async def login_manual(self, save_session: bool = True, timeout_minutes: int = 5, use_confirmation: bool = True) -> bool:
        """
        Manual login to WSJ - opens browser for user to login manually
        
        Args:
            save_session: Whether to save session for future use
            timeout_minutes: How long to wait for manual login (in minutes)
            use_confirmation: If True, wait for user to press Enter after login
            
        Returns:
            bool: True if login successful
        """
        try:
            # Try to load existing session first
            if self.session_path.exists() and save_session:
                await self.load_session()
                if await self.verify_login():
                    print("✅ Loaded existing session successfully")
                    self.is_logged_in = True
                    return True
            
            print("🌐 Opening WSJ login page for manual login...")
            print("📝 Please login manually in the browser window that opens")
            if use_confirmation:
                print("🔑 After you successfully login, come back here and press Enter to continue")
            else:
                print("🔑 Complete your login, then wait - the script will detect when you're logged in")
            print("💡 Tip: Stay on any WSJ page after login - don't close the browser")
            
            # Open login page with faster loading
            try:
                await self.page.goto(self.login_url, wait_until='domcontentloaded', timeout=15000)
            except Exception as e:
                print(f"⚠️  Initial page load timeout: {e}")
                print("📝 Continuing anyway - you can navigate to the login page manually")
            
            if use_confirmation:
                # Wait for user confirmation
                return await self._wait_for_manual_confirmation(save_session)
            else:
                # Use automatic detection (original method)
                return await self._wait_for_automatic_detection(timeout_minutes, save_session)
                
        except Exception as e:
            print(f"❌ Login error: {str(e)}")
            return False

    async def _wait_for_manual_confirmation(self, save_session: bool = True) -> bool:
        """Wait for user to manually confirm they've logged in"""
        import threading
        import queue
        
        def wait_for_input(q):
            input("\n👆 Press Enter after you've successfully logged in to WSJ...")
            q.put("ready")
        
        # Start input thread
        q = queue.Queue()
        input_thread = threading.Thread(target=wait_for_input, args=(q,))
        input_thread.daemon = True
        input_thread.start()
        
        # Wait for user input or timeout (30 minutes max)
        max_wait_time = 30 * 60  # 30 minutes
        start_time = time.time()
        
        while time.time() - start_time < max_wait_time:
            try:
                # Check if user pressed Enter
                q.get_nowait()
                print("✅ Login confirmed! Proceeding with scraping...")
                self.is_logged_in = True
                
                if save_session:
                    await self.save_session()
                
                return True
                        
            except queue.Empty:
                pass
            
            await asyncio.sleep(0.5)
        
        print("⏰ Timeout reached - login confirmation not received")
        return False

    async def _wait_for_automatic_detection(self, timeout_minutes: int, save_session: bool = True) -> bool:
        """Original automatic detection method"""
        print(f"⏱️  Waiting up to {timeout_minutes} minutes for automatic login detection...")
        
        timeout_seconds = timeout_minutes * 60
        start_time = time.time()
        last_check_time = 0
        
        while time.time() - start_time < timeout_seconds:
            current_time = time.time()
            
            # Only check login status every 5 seconds to avoid interrupting user
            if current_time - last_check_time >= 5:
                last_check_time = current_time
                
                # Check current URL to see if user has moved past login
                current_url = self.page.url
                
                # Always try to check login status quietly first
                if await self._check_login_status_quietly():
                    print("✅ Login successful!")
                    self.is_logged_in = True
                    
                    if save_session:
                        await self.save_session()
                    
                    return True
                
                # If not logged in and not on a login page, give user guidance
                if not any(term in current_url.lower() for term in ['login', 'signin', 'accounts']):
                    print("💡 You're not on the login page. Please navigate to the WSJ login page to continue.")
            
            # Show progress every 15 seconds
            elapsed = int(time.time() - start_time)
            if elapsed > 0 and elapsed % 15 == 0:
                remaining = timeout_seconds - elapsed
                print(f"⏳ Still waiting for login... {remaining}s remaining")
            
            # Short sleep to not overload the CPU
            await asyncio.sleep(1)
        
        print("⏰ Timeout reached - login not completed")
        return False

    async def _check_login_status_quietly(self) -> bool:
        """Check login status without navigating away from current page"""
        try:
            # Method 1: Try to access a protected endpoint without navigation
            try:
                response = await self.page.evaluate("""
                    async () => {
                        try {
                            const response = await fetch('/my/', {
                                method: 'GET',
                                credentials: 'include'
                            });
                            return {
                                status: response.status,
                                redirected: response.redirected,
                                url: response.url
                            };
                        } catch (error) {
                            return { error: error.message };
                        }
                    }
                """)
                
                # If we get a successful response and not redirected to login
                if response and not response.get('error'):
                    status = response.get('status', 0)
                    redirected_url = response.get('url', '')
                    
                    # Success if status is 200 and not redirected to login
                    if status == 200 and not any(term in redirected_url.lower() for term in ['login', 'signin', 'accounts']):
                        return True
            except Exception:
                pass
            
            # Method 2: Check for login-specific elements on current page
            try:
                # Look for user-specific elements that indicate login
                user_elements = await self.page.query_selector_all(
                    '[data-module="MyWSJ"], .user-nav, .account-menu, .user-info, [class*="user"], [class*="account"]'
                )
                if len(user_elements) > 0:
                    return True
                
                # Check if current URL indicates successful login
                current_url = self.page.url
                if any(indicator in current_url.lower() for indicator in ['/my/', '/account/', '/profile/']):
                    return True
                    
            except Exception:
                pass
            
            return False
            
        except Exception:
            # Final fallback - just check if we're not on a login page
            try:
                current_url = self.page.url
                return not any(term in current_url.lower() for term in ['login', 'signin', 'accounts'])
            except:
                return False

    async def login_automated(self, username: str, password: str, save_session: bool = True) -> bool:
        """
        Automated login to WSJ (original method)
        
        Args:
            username: WSJ username/email
            password: WSJ password
            save_session: Whether to save session for future use
            
        Returns:
            bool: True if login successful
        """
        try:
            # Try to load existing session first
            if self.session_path.exists() and save_session:
                await self.load_session()
                if await self.verify_login():
                    print("✅ Loaded existing session successfully")
                    self.is_logged_in = True
                    return True
            
            print("🤖 Attempting automated login to WSJ...")
            await self.page.goto(self.login_url, wait_until='domcontentloaded', timeout=15000)
            
            # Wait for and fill username
            await self.page.wait_for_selector('input[name="username"], input[type="email"]', timeout=10000)
            await self.page.fill('input[name="username"], input[type="email"]', username)
            
            # Fill password
            await self.page.wait_for_selector('input[name="password"], input[type="password"]')
            await self.page.fill('input[name="password"], input[type="password"]', password)
            
            # Click login button
            login_button = await self.page.wait_for_selector('button[type="submit"], input[type="submit"], .login-button')
            await login_button.click()
            
            # Wait for navigation or error
            await asyncio.sleep(3)
            
            # Check if login was successful
            if await self.verify_login():
                print("✅ Automated login successful!")
                self.is_logged_in = True
                
                if save_session:
                    await self.save_session()
                
                return True
            else:
                print("❌ Automated login failed - try manual login instead")
                return False
                
        except Exception as e:
            print(f"❌ Automated login error: {str(e)}")
            return False
    
    async def verify_login(self) -> bool:
        """Verify if user is logged in by navigating to a protected page"""
        try:
            # Go to a page that requires login
            await self.page.goto(f"{self.base_url}/my/", wait_until='domcontentloaded', timeout=15000)
            
            # Check for login indicators
            current_url = self.page.url
            
            # If redirected to login page, not logged in
            if 'login' in current_url or 'signin' in current_url:
                return False
            
            # Check for user-specific elements
            user_elements = await self.page.query_selector_all('[data-module="MyWSJ"], .user-nav, .account-menu')
            return len(user_elements) > 0
            
        except Exception:
            return False

    async def verify_login_quiet(self) -> bool:
        """Verify if user is logged in without navigation (for manual login)"""
        return await self._check_login_status_quietly()
    
    async def save_session(self):
        """Save browser session"""
        try:
            state = await self.context.storage_state()
            with open(self.session_path, 'w') as f:
                json.dump(state, f)
            print("Session saved")
        except Exception as e:
            print(f"Failed to save session: {e}")
    
    async def load_session(self):
        """Load browser session"""
        try:
            with open(self.session_path, 'r') as f:
                state = json.load(f)
            
            # Close current context and create new one with saved state
            await self.context.close()
            self.context = await self.browser.new_context(
                storage_state=state,
                viewport={'width': 1920, 'height': 1080},
                user_agent='Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
            )
            self.page = await self.context.new_page()
            print("Session loaded")
        except Exception as e:
            print(f"Failed to load session: {e}")
    
    async def scrape_article(self, url: str) -> Dict[str, Any]:
        """
        Scrape a single WSJ article
        
        Args:
            url: Article URL
            
        Returns:
            Dictionary with article data
        """
        try:
            # Navigate with faster wait condition
            try:
                await self.page.goto(url, wait_until='domcontentloaded', timeout=30000)
            except Exception as e:
                print(f"❌ Failed to load article: {e}")
                return {'url': url, 'error': str(e), 'scraped_at': datetime.now().isoformat()}
            
            await asyncio.sleep(2)
            
            # Get page content
            content = await self.page.content()
            soup = BeautifulSoup(content, 'html.parser')
            
            # Extract article data
            article_data = {
                'url': url,
                'title': '',
                'subtitle': '',
                'author': '',
                'publish_date': '',
                'content': '',
                'summary': '',
                'scraped_at': datetime.now().isoformat()
            }
            
            # Title - updated selectors for current WSJ structure
            title_selectors = [
                'h1[data-testid="headline"]',  # Most common current selector
                'h1.wsj-article-headline',     # Legacy selector
                'h1',                          # Fallback
                '.headline-text',              # Alternative
                '.article-headline'            # Another fallback
            ]
            
            for selector in title_selectors:
                title_elem = soup.select_one(selector)
                if title_elem:
                    article_data['title'] = title_elem.get_text(strip=True)
                    break
            
            # Subtitle
            subtitle_elem = soup.find('h2', class_='sub-head') or soup.find('.article-sub-head')
            if subtitle_elem:
                article_data['subtitle'] = subtitle_elem.get_text(strip=True)
            
            # Author - updated selectors
            author_selectors = [
                '[data-testid="byline"] a',
                '[rel="author"]',
                '.author-name',
                '.byline'
            ]
            
            for selector in author_selectors:
                author_elem = soup.select_one(selector)
                if author_elem:
                    article_data['author'] = author_elem.get_text(strip=True)
                    break
            
            # Publish date - updated selectors
            date_selectors = [
                '[data-testid="timestamp-text"]',
                'time',
                '[datetime]',
                '.timestamp',
                '.publish-date'
            ]
            
            for selector in date_selectors:
                date_elem = soup.select_one(selector)
                if date_elem:
                    article_data['publish_date'] = date_elem.get('datetime') or date_elem.get_text(strip=True)
                    break
            
            # Article content - updated selectors for current WSJ structure
            content_selectors = [
                '.paywall p[data-type="paragraph"]',  # Paywall content paragraphs
                'p[data-type="paragraph"]',  # All paragraph elements with data-type
                '[data-module="ArticleBody"] p',
                '.article-content p',
                '.wsj-snippet-body',
                '.story-body p',
                '.article-wrap p',
                '.article-body p'
            ]
            
            for selector in content_selectors:
                content_elems = soup.select(selector)
                if content_elems:
                    # Filter out empty paragraphs and unwanted content
                    paragraphs = []
                    for elem in content_elems:
                        text = elem.get_text(strip=True)
                        # Skip short paragraphs, ads, and navigation text
                        if (len(text) > 20 and 
                            not any(skip_word in text.lower() for skip_word in [
                                'advertisement', 'subscribe', 'sign up', 'share your thoughts',
                                'write to', 'photo:', 'source:', 'chart', 'related coverage'
                            ])):
                            paragraphs.append(text)
                    
                    if paragraphs:
                        article_data['content'] = '\n\n'.join(paragraphs)
                        print(f"📄 Extracted {len(paragraphs)} paragraphs using selector: {selector}")
                        break
            
            # Summary (if available)
            summary_elem = soup.find('.article-summary') or soup.find('.snippet-body')
            if summary_elem:
                article_data['summary'] = summary_elem.get_text(strip=True)
            
            return article_data
            
        except Exception as e:
            print(f"Error scraping article {url}: {e}")
            return {'url': url, 'error': str(e), 'scraped_at': datetime.now().isoformat()}
    
    async def search_articles(self, query: str, limit: int = 10) -> List[str]:
        """
        Search for articles on WSJ
        
        Args:
            query: Search query
            limit: Maximum number of URLs to return
            
        Returns:
            List of article URLs
        """
        try:
            search_url = f"{self.base_url}/search?query={query}"
            # Navigate with faster wait condition
            try:
                await self.page.goto(search_url, wait_until='domcontentloaded', timeout=30000)
            except Exception as e:
                print(f"❌ Failed to load search page: {e}")
                return []
            
            await asyncio.sleep(2)
            
            # Get search results
            content = await self.page.content()
            soup = BeautifulSoup(content, 'html.parser')
            
            # Extract article links
            article_links = []
            link_selectors = [
                'a[data-testid="flexcard-headline"]',  # Primary flexcard headlines
                'h3 a[href*="/articles/"]',  # H3 headline links
                '.headline a[href*="/articles/"]',  # Generic headline links
                'a[href*="/articles/"]'  # All article links (fallback)
            ]
            
            import pdb; pdb.set_trace()
            for selector in link_selectors:
                links = soup.select(selector)
                for link in links:
                    href = link.get('href')
                    if href and '/articles/' in href:
                        if href.startswith('/'):
                            href = self.base_url + href
                        article_links.append(href)
                        
                        if len(article_links) >= limit:
                            break
                
                if len(article_links) >= limit:
                    break
            
            return list(set(article_links))[:limit]
            
        except Exception as e:
            print(f"Error searching articles: {e}")
            return []
    
    async def scrape_market_data(self) -> Dict[str, Any]:
        """
        Scrape market data from WSJ markets page
        
        Returns:
            Dictionary with market data
        """
        try:
            markets_url = f"{self.base_url}/market-data"
            
            # Navigate with faster wait condition
            try:
                await self.page.goto(markets_url, wait_until='domcontentloaded', timeout=30000)
            except Exception as e:
                print(f"❌ Failed to load market data page: {e}")
                return {'error': str(e), 'scraped_at': datetime.now().isoformat()}
            
            await asyncio.sleep(2)
            
            content = await self.page.content()
            soup = BeautifulSoup(content, 'html.parser')
            
            market_data = {
                'major_indices': {},
                'currencies': {},
                'commodities': {},
                'scraped_at': datetime.now().isoformat()
            }
            
            # Extract major indices (DOW, S&P, NASDAQ)
            index_elements = soup.select('.market-data-module .data-point, .index-item')
            for elem in index_elements:
                name_elem = elem.find('.name') or elem.find('.symbol')
                value_elem = elem.find('.value') or elem.find('.price')
                change_elem = elem.find('.change') or elem.find('.change-percent')
                
                if name_elem and value_elem:
                    name = name_elem.get_text(strip=True)
                    value = value_elem.get_text(strip=True)
                    change = change_elem.get_text(strip=True) if change_elem else ''
                    
                    market_data['major_indices'][name] = {
                        'value': value,
                        'change': change
                    }
            
            return market_data
            
        except Exception as e:
            print(f"Error scraping market data: {e}")
            return {'error': str(e), 'scraped_at': datetime.now().isoformat()}
    
    async def get_business_articles(self) -> List[Dict[str, Any]]:
        """
        Get all articles from WSJ business page sections
        
        Returns:
            List of article dictionaries with full content
        """
        try:
            business_url = f"{self.base_url}/business/"
            
            # Navigate to business page
            try:
                await self.page.goto(business_url, wait_until='domcontentloaded', timeout=60000)
            except Exception as e:
                print(f"❌ Failed to load business page: {e}")
                return []
            
            # Wait longer for dynamic content to load
            await asyncio.sleep(20)  # Let all content load completely
            
            # Try to wait for article containers to appear
            try:
                await self.page.wait_for_selector('.e1sf124z9, [data-testid="flexcard-headline"]', timeout=10000)
                print("✅ Article containers found on page")
            except Exception as e:
                print(f"⚠️  Article containers not found, proceeding anyway: {e}")
            
            # Additional wait to ensure all dynamic content is loaded
            await asyncio.sleep(5)
            
            # Scroll to bottom to load any lazy-loaded content
            print("📜 Scrolling to bottom to load all content...")
            await self.page.evaluate("""
                async () => {
                    return new Promise((resolve) => {
                        let totalHeight = 0;
                        let distance = 100;
                        let timer = setInterval(() => {
                            let scrollHeight = document.body.scrollHeight;
                            window.scrollBy(0, distance);
                            totalHeight += distance;
                            
                            if(totalHeight >= scrollHeight){
                                clearInterval(timer);
                                resolve();
                            }
                        }, 100);
                    });
                }
            """)
            
            # Wait a bit more after scrolling
            await asyncio.sleep(3)
            print("✅ Finished scrolling, extracting content...")
            
            content = await self.page.content()
            soup = BeautifulSoup(content, 'html.parser')
            
            # Extract all article links from the business page
            article_links = []
            
            # Look for article links in the business sections
            # Focus on the most reliable selectors based on the HTML structure
            link_selectors = [
                '.e1sf124z9 a[data-testid="flexcard-headline"]',  # Article containers with flexcard headlines
                'a[data-testid="flexcard-headline"]',  # Primary flexcard headlines (most reliable)
                '.e1sf124z9 a[href*="wsj.com"]',  # All WSJ links in article containers
                'h3 a[href*="wsj.com"]'  # H3 headline links to WSJ
            ]
            e1sf124z9_count = len(soup.find_all(class_='e1sf124z9'))
            print(f"Number of times 'e1sf124z9' appears: {e1sf124z9_count}")

            for selector in link_selectors:
                links = soup.select(selector)
                print(f"🔍 Selector '{selector}' found {len(links)} links")
                
                for i, link in enumerate(links):
                    href = link.get('href')
                    if href:
                        # Debug: Show first few URLs for this selector
                        if i < 3:
                            print(f"  🔗 Sample URL {i+1}: {href}")
                        
                        # Convert relative URLs to absolute
                        if href.startswith('/'):
                            href = self.base_url + href
                        
                        # Check if it's a WSJ article URL
                        is_article = (
                            '/articles/' in href or 
                            ('/business/' in href and any(section in href for section in [
                                'airlines', 'energy-oil', 'earnings', 'deals', 'tech', 'media', 
                                'autos', 'real-estate', 'retail', 'finance', 'logistics'
                            ])) or
                            # Also include URLs that look like article URLs with unique IDs
                            (href.count('-') >= 2 and len(href.split('-')[-1]) >= 6)  # Likely article ID
                        )
                        
                        if is_article and href not in article_links:
                            article_links.append(href)
                            print(f"📰 Found article: {href}")
                
                # Break early if we found enough articles with the primary selector
                if len(article_links) >= 15:
                    print(f"✅ Found {len(article_links)} articles, stopping early")
                    break
            
            print(f"📰 Found {len(article_links)} article links on business page")
            
            # Now scrape each article
            articles = []
            
            # Limit articles to scrape
            max_articles = min(100, len(article_links))
            print(f"🎯 Will attempt to scrape {max_articles} articles")
            
            for i, url in enumerate(article_links[:max_articles], 1):
                print(f"📖 Scraping article {i}/{max_articles}: {url[:80]}...")
                
                try:
                    article_data = await self.scrape_article(url)
                    
                    # Check if we got meaningful content
                    if article_data.get('title') and not article_data.get('error'):
                        articles.append(article_data)
                        title = article_data.get('title', 'No title')
                        print(f"✅ Successfully scraped: {title[:60]}{'...' if len(title) > 60 else ''}")
                    else:
                        error_msg = article_data.get('error', 'Missing title or content')
                        print(f"⚠️  Skipped article: {error_msg}")
                    
                    # Be respectful with requests
                    await asyncio.sleep(2)
                    
                except Exception as e:
                    print(f"❌ Error scraping {url}: {e}")
                    continue
            
            print(f"🎉 Successfully scraped {len(articles)} articles from business page")
            return articles
            
        except Exception as e:
            print(f"❌ Error getting business articles: {e}")
            return []

    async def get_trending_topics(self) -> List[str]:
        """
        Get trending topics from WSJ homepage
        
        Returns:
            List of trending topics/headlines
        """
        try:
            # Navigate with faster wait condition
            try:
                await self.page.goto(self.base_url, wait_until='domcontentloaded', timeout=30000)
            except Exception as e:
                print(f"❌ Failed to load homepage: {e}")
                return []
            
            await asyncio.sleep(2)
            
            content = await self.page.content()
            soup = BeautifulSoup(content, 'html.parser')
            
            topics = []
            # Extract trending headlines - updated for current WSJ structure
            headline_selectors = [
                '[data-testid="flexcard-headline"]',
                '.headline',
                '.wsj-headline',
                'h1, h2, h3',
                '.story-headline'
            ]
            
            for selector in headline_selectors:
                headlines = soup.select(selector)
                for headline in headlines[:10]:  # Limit to top 10
                    text = headline.get_text(strip=True)
                    if text and len(text) > 10:  # Filter out short/empty text
                        topics.append(text)
            
            return list(set(topics))[:15]  # Return unique topics, max 15
            
        except Exception as e:
            print(f"Error getting trending topics: {e}")
            return []
    
    async def save_articles_to_json(self, articles: List[Dict[str, Any]], filename: str = None) -> str:
        """
        Save scraped articles to a JSON file
        
        Args:
            articles: List of article dictionaries
            filename: Optional filename, defaults to timestamped filename
            
        Returns:
            str: Path to the saved file
        """
        try:
            if filename is None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"wsj_business_articles_{timestamp}.json"
            
            # Create data structure
            data = {
                'scraped_at': datetime.now().isoformat(),
                'total_articles': len(articles),
                'source': 'WSJ Business Page',
                'articles': articles
            }
            
            # Save to file
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            
            print(f"💾 Saved {len(articles)} articles to {filename}")
            return filename
            
        except Exception as e:
            print(f"❌ Error saving articles to JSON: {e}")
            return ""

    async def close(self):
        """Close browser and cleanup"""
        if self.page:
            await self.page.close()
        if self.context:
            await self.context.close()
        if self.browser:
            await self.browser.close()
        if hasattr(self, 'playwright'):
            await self.playwright.stop()


# Convenience functions for easy usage
async def scrape_wsj_articles_manual(query: str, limit: int = 5, use_confirmation: bool = True) -> List[Dict[str, Any]]:
    """
    Convenience function to scrape WSJ articles with manual login
    
    Args:
        query: Search query
        limit: Number of articles to scrape
        use_confirmation: If True, wait for Enter press after login (recommended)
        
    Returns:
        List of article data dictionaries
    """
    async with WSJScraper(headless=False) as scraper:  # Show browser for manual login
        if await scraper.login_manual(use_confirmation=use_confirmation):
            urls = await scraper.search_articles(query, limit)
            articles = []
            
            for url in urls:
                article_data = await scraper.scrape_article(url)
                articles.append(article_data)
                await asyncio.sleep(1)  # Be respectful with requests
            
            return articles
        else:
            raise Exception("Failed to login to WSJ manually")


async def scrape_wsj_articles_auto(username: str, password: str, query: str, limit: int = 5) -> List[Dict[str, Any]]:
    """
    Convenience function to scrape WSJ articles with automated login
    
    Args:
        username: WSJ username
        password: WSJ password
        query: Search query
        limit: Number of articles to scrape
        
    Returns:
        List of article data dictionaries
    """
    async with WSJScraper(headless=True) as scraper:
        if await scraper.login_automated(username, password):
            urls = await scraper.search_articles(query, limit)
            articles = []
            
            for url in urls:
                article_data = await scraper.scrape_article(url)
                articles.append(article_data)
                await asyncio.sleep(1)  # Be respectful with requests
            
            return articles
        else:
            raise Exception("Failed to login to WSJ automatically")


async def get_wsj_market_data_manual(use_confirmation: bool = True) -> Dict[str, Any]:
    """
    Convenience function to get WSJ market data with manual login
    
    Args:
        use_confirmation: If True, wait for Enter press after login (recommended)
        
    Returns:
        Market data dictionary
    """
    async with WSJScraper(headless=False) as scraper:  # Show browser for manual login
        if await scraper.login_manual(use_confirmation=use_confirmation):
            return await scraper.scrape_market_data()
        else:
            raise Exception("Failed to login to WSJ manually")


async def get_wsj_market_data_auto(username: str, password: str) -> Dict[str, Any]:
    """
    Convenience function to get WSJ market data with automated login
    
    Args:
        username: WSJ username
        password: WSJ password
        
    Returns:
        Market data dictionary
    """
    async with WSJScraper(headless=True) as scraper:
        if await scraper.login_automated(username, password):
            return await scraper.scrape_market_data()
        else:
            raise Exception("Failed to login to WSJ automatically")


async def scrape_wsj_business_articles_manual(save_to_file: bool = True, filename: str = None) -> List[Dict[str, Any]]:
    """
    Convenience function to scrape all business articles with manual login
    
    Args:
        save_to_file: Whether to save articles to JSON file
        filename: Optional filename for JSON file
        
    Returns:
        List of article dictionaries
    """
    async with WSJScraper(headless=False) as scraper:  # Show browser for manual login
        if await scraper.login_manual(use_confirmation=True):
            articles = await scraper.get_business_articles()
            
            if save_to_file and articles:
                await scraper.save_articles_to_json(articles, filename)
            
            return articles
        else:
            raise Exception("Failed to login to WSJ manually")


async def scrape_wsj_business_articles_auto(username: str, password: str, save_to_file: bool = True, filename: str = None) -> List[Dict[str, Any]]:
    """
    Convenience function to scrape all business articles with automated login
    
    Args:
        username: WSJ username
        password: WSJ password
        save_to_file: Whether to save articles to JSON file
        filename: Optional filename for JSON file
        
    Returns:
        List of article dictionaries
    """
    async with WSJScraper(headless=True) as scraper:
        if await scraper.login_automated(username, password):
            articles = await scraper.get_business_articles()
            
            if save_to_file and articles:
                await scraper.save_articles_to_json(articles, filename)
            
            return articles
        else:
            raise Exception("Failed to login to WSJ automatically")


# Example usage
if __name__ == "__main__":
    import os
    from dotenv import load_dotenv
    
    load_dotenv()
    
    async def main():
        print("🚀 WSJ Scraper - Basic Demo")
        print("For full examples, run: python examples/wsj_scraper_example.py")
        print()
        
        print("Choose demo:")
        print("1. Manual login demo")
        print("2. Automated login demo")
        
        choice = input("Enter choice (1 or 2): ").strip()
        
        if choice == "1":
            # Manual login demo
            print("\n📰 Manual login demo...")
            try:
                articles = await scrape_wsj_articles_manual("GOOG", limit=2, use_confirmation=True)
                print(f"✅ Demo complete - scraped {len(articles)} articles")
                for article in articles:
                    print(f"- {article.get('title', 'No title')}")
                    print(article)
                
            except Exception as e:
                print(f"❌ Error: {e}")
        
        elif choice == "2":
            # Automated login demo
            username = os.getenv('WSJ_USERNAME')
            password = os.getenv('WSJ_PASSWORD')
            
            if not username or not password:
                print("❌ Please set WSJ_USERNAME and WSJ_PASSWORD environment variables")
                return
            
            print(f"\n🤖 Automated login demo...")
            try:
                articles = await scrape_wsj_articles_auto(username, password, "technology", limit=2)
                print(f"✅ Demo complete - scraped {len(articles)} articles")
                for article in articles:
                    print(f"- {article.get('title', 'No title')}")
                
            except Exception as e:
                print(f"❌ Error: {e}")
        
        else:
            print("❌ Invalid choice")
    
    asyncio.run(main()) 
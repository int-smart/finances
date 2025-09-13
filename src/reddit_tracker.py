import os
import sys
import pickle
import re
from datetime import datetime
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from dotenv import load_dotenv

# Add the reddit_scraper path to import it
sys.path.append('/home/abhishek/Desktop/Projects/tools/scrapers/reddit_scraper')

try:
    from reddit_scraper import RedditScraper, RedditPost, SubredditSummarizer
except ImportError as e:
    print(f"Warning: Could not import reddit_scraper: {e}")
    print("Please ensure the reddit_scraper module is properly installed")
    # Create dummy classes to prevent errors
    class RedditScraper:
        def __init__(self): pass
    class RedditPost:
        def __init__(self): pass
    class SubredditSummarizer:
        def __init__(self): pass

from src.config import COMPANIES, DATA_DIR

load_dotenv()

@dataclass
class RedditSummary:
    """Data class for Reddit stock discussion summary"""
    ticker: str
    total_posts: int
    positive_sentiment_posts: int
    negative_sentiment_posts: int
    summary: str
    positive_factors: List[Dict[str, str]]
    negative_factors: List[Dict[str, str]]
    earnings_sinks: List[str]
    earnings_boosters: List[str]
    sentiment: str
    price_impact: str
    top_posts: List[RedditPost]
    subreddit_breakdown: Dict[str, int]

class RedditTracker:
    """Track stock discussions on Reddit and provide summaries similar to news summaries"""
    
    def __init__(self):
        self.scraper = RedditScraper()
        # Use the reddit_scraper's built-in summarizer
        try:
            self.summarizer = SubredditSummarizer()
        except:
            self.summarizer = None
        
        # Stock-focused subreddits
        self.stock_subreddits = [
            'wallstreetbets',
            'stocks', 
            'investing',
            'SecurityAnalysis',
            'ValueInvesting',
            'StockMarket',
            'pennystocks',
            'financialindependence',
            'SecurityAnalysis',
            'investing_discussion'
        ]
        
        # General finance subreddits that might discuss stocks
        self.finance_subreddits = [
            'personalfinance',
            'financialplanning',
            'financialindependence',
            'fire',
            'economics'
        ]
        
        self.all_subreddits = self.stock_subreddits + self.finance_subreddits
        
    def search_ticker(self, ticker: str, limit_per_subreddit: int = 10) -> List[RedditPost]:
        """Search for mentions of a specific ticker across relevant subreddits"""
        print(f"Searching Reddit for ticker: {ticker}")
        
        # Search for the main ticker across all subreddits
        all_posts = self.scraper.get_posts_from_multiple_subreddits(
            subreddit_names=self.stock_subreddits,
            limit_per_subreddit=limit_per_subreddit,
            keyword=ticker,
            ordered_by='new',
            merge_comments=True
        )
        
        # Also search for common ticker variations
        ticker_variations = [
            # f"${ticker}",
            # f"${ticker.lower()}",
            # ticker.upper(),
            # ticker.lower()
        ]
        
        for variation in ticker_variations:
            if variation != ticker:  # Avoid duplicate searches
                variation_posts = self.scraper.get_posts_from_multiple_subreddits(
                    subreddit_names=self.stock_subreddits,
                    limit_per_subreddit=2,  # Fewer for variations
                    keyword=variation
                )
                all_posts.extend(variation_posts)
        
        # Remove duplicates based on post_id
        unique_posts = []
        seen_ids = set()
        for post in all_posts:
            if post.post_id not in seen_ids:
                unique_posts.append(post)
                seen_ids.add(post.post_id)
        
        # Sort by engagement (score + comments)
        unique_posts.sort(key=lambda x: x.score + x.comments_count, reverse=True)
        
        return unique_posts
    
    def get_top_posts_from_subreddit(self, subreddit: str, limit: int = 20) -> List[RedditPost]:
        """Get top posts from a specific subreddit"""
        try:
            return self.scraper.get_top_posts_from_subreddit(
                subreddit, 
                time_filter='week', 
                limit=limit
            )
        except Exception as e:
            print(f"Error getting top posts from {subreddit}: {e}")
            return []
    
    def analyze_subreddit_discussions(self, subreddit: str, limit: int = 20) -> Dict[str, Any]:
        """Analyze general discussions in a subreddit for stock opportunities"""
        posts = self.get_top_posts_from_subreddit(subreddit, limit)
        
        if not posts:
            return {"subreddit": subreddit, "posts": [], "summary": "No posts found"}
        
        # Extract potential stock tickers from post titles and content
        mentioned_tickers = self._extract_tickers_from_posts(posts)
        
        # Try to use the reddit_scraper's AI summarizer if available
        analysis = {"subreddit": subreddit}
        
        if self.summarizer:
            try:
                # Use the reddit_scraper's stock-focused analysis method
                analysis = self.summarizer.analyze_subreddit_for_stocks(subreddit, posts, limit=limit)
                # Add the basic ticker extraction as backup
                if not analysis.get('mentioned_tickers'):
                    analysis['mentioned_tickers'] = list(mentioned_tickers.keys())
            except Exception as e:
                print(f"Error using AI summarizer: {e}")
                # Fall back to basic analysis
                analysis.update({
                    "total_posts": len(posts),
                    "mentioned_tickers": mentioned_tickers,
                    "summary": "",
                    "top_posts": posts
                })
        else:
            # Basic analysis without AI
            analysis.update({
                "total_posts": len(posts),
                "mentioned_tickers": mentioned_tickers,
                "summary": "",
                "top_posts": posts
            })
        
        return analysis
    
    def _extract_tickers_from_posts(self, posts: List[RedditPost]) -> Dict[str, int]:
        """Extract potential stock tickers from Reddit posts"""
        ticker_pattern = re.compile(r'\$([A-Z]{1,5})\b|\b([A-Z]{2,5})\b')
        ticker_counts = {}
        
        # Known tickers to focus on
        known_tickers = set(COMPANIES + ['SPY', 'QQQ', 'IWM', 'DIA', 'VTI', 'VXUS'])
        
        for post in posts:
            text = f"{post.title} {post.content or ''}"
            
            # Find potential tickers
            matches = ticker_pattern.findall(text)
            for match in matches:
                # match is a tuple, get the non-empty part
                ticker = match[0] or match[1]
                if ticker and (ticker in known_tickers or len(ticker) >= 2):
                    ticker_counts[ticker] = ticker_counts.get(ticker, 0) + 1
        
        # Filter out common words that match the pattern
        common_words = {'THE', 'AND', 'FOR', 'ARE', 'BUT', 'NOT', 'YOU', 'ALL', 'CAN', 'HER', 'WAS', 'ONE', 'OUR', 'HAD', 'BY', 'UP', 'DO', 'NO', 'IF', 'SO', 'MY', 'ME', 'AM', 'GO', 'TO', 'IT', 'IS', 'IN', 'ON', 'AS', 'AT', 'OR', 'OF', 'AN', 'US', 'HE', 'BE', 'HIS', 'HIM', 'NOW', 'GET', 'HAS', 'WHO', 'ITS', 'NEW', 'OUT', 'ANY', 'BAD', 'SEE', 'TOP', 'HOW', 'OLD', 'WAY', 'BIG', 'TOO', 'LOT', 'WHY', 'USE', 'OWN', 'SAY', 'SHE'}
        
        # Filter out common words and return sorted by count
        filtered_tickers = {
            ticker: count for ticker, count in ticker_counts.items() 
            if ticker not in common_words and count > 1
        }
        
        return dict(sorted(filtered_tickers.items(), key=lambda x: x[1], reverse=True))
    
    def generate_ticker_summary(self, ticker: str, posts: List[RedditPost]) -> RedditSummary:
        """Generate a comprehensive summary for a specific ticker using AI analysis"""
        if not posts:
            return RedditSummary(
                ticker=ticker,
                total_posts=0,
                positive_sentiment_posts=0,
                negative_sentiment_posts=0,
                summary=f"No Reddit discussions found for {ticker}",
                positive_factors=[],
                negative_factors=[],
                earnings_sinks=[],
                earnings_boosters=[],
                sentiment="neutral",
                price_impact="No impact data available",
                top_posts=[],
                subreddit_breakdown={}
            )
        
        # Get subreddit breakdown
        subreddit_breakdown = self._get_subreddit_breakdown(posts)
        
        # Generate AI-powered detailed analysis
        detailed_analysis = self._generate_detailed_ticker_analysis(ticker, posts)
        
        # Count positive/negative posts based on AI sentiment if available
        positive_posts = 0
        negative_posts = 0
        if detailed_analysis.get('sentiment') == 'positive':
            positive_posts = len(posts) // 2 + 1  # Rough estimate
        elif detailed_analysis.get('sentiment') == 'negative':
            negative_posts = len(posts) // 2 + 1  # Rough estimate
        elif detailed_analysis.get('sentiment') == 'mixed':
            positive_posts = len(posts) // 3
            negative_posts = len(posts) // 3
        
        return RedditSummary(
            ticker=ticker,
            total_posts=len(posts),
            positive_sentiment_posts=positive_posts,
            negative_sentiment_posts=negative_posts,
            summary=detailed_analysis.get('summary', f"Reddit analysis for {ticker} based on {len(posts)} posts"),
            positive_factors=detailed_analysis.get('positive_factors', []),
            negative_factors=detailed_analysis.get('negative_factors', []),
            earnings_sinks=detailed_analysis.get('earnings_sinks', []),
            earnings_boosters=detailed_analysis.get('earnings_boosters', []),
            sentiment=detailed_analysis.get('sentiment', 'neutral'),
            price_impact=detailed_analysis.get('price_impact', 'Unknown'),
            top_posts=posts[:10],
            subreddit_breakdown=subreddit_breakdown
        )

    def _get_subreddit_breakdown(self, posts: List[RedditPost]) -> Dict[str, int]:
        """Get breakdown of posts by subreddit"""
        breakdown = {}
        for post in posts:
            breakdown[post.subreddit] = breakdown.get(post.subreddit, 0) + 1
        return breakdown
    
    def _generate_detailed_ticker_analysis(self, ticker: str, posts: List[RedditPost]) -> Dict[str, Any]:
        """Generate detailed analysis using the reddit_scraper's AI summarizer"""
        if not self.summarizer:
            # Fallback basic analysis if no AI summarizer available
            return {
                'summary': f"Basic Reddit analysis for {ticker} based on {len(posts)} posts",
                'positive_factors': [],
                'negative_factors': [],
                'earnings_boosters': [],
                'earnings_sinks': [],
                'sentiment': 'neutral',
                'price_impact': 'Unable to determine without AI analysis'
            }
        
        try:
            # Use the reddit_scraper's AI-powered analysis
            analysis = self._analyze_ticker_with_ai(ticker, posts)
            return analysis
        except Exception as e:
            print(f"Error using AI analysis for {ticker}: {e}")
            # Fallback to basic analysis
            return {
                'summary': f"Reddit analysis for {ticker} based on {len(posts)} posts (AI analysis failed)",
                'positive_factors': [],
                'negative_factors': [],
                'earnings_boosters': [],
                'earnings_sinks': [],
                'sentiment': 'neutral',
                'price_impact': 'Unable to determine due to analysis error'
            }
    
    def _analyze_ticker_with_ai(self, ticker: str, posts: List[RedditPost]) -> Dict[str, Any]:
        """Use the reddit_scraper's AI summarizer to analyze ticker sentiment"""
        try:
            # Use the new analyze_stock_sentiment method from the summarizer
            return self.summarizer.analyze_stock_sentiment(ticker, posts)
        except Exception as e:
            print(f"AI analysis failed: {e}")
            raise e
    
    def track_all_tickers(self, tickers: List[str] = None, posts_per_ticker: int = 20) -> Dict[str, Any]:
        """Track Reddit discussions for all company tickers"""
        if tickers is None:
            tickers = COMPANIES
        
        print(f"Tracking Reddit discussions for {len(tickers)} tickers...")
        
        reddit_data = {}
        current_date = datetime.now().strftime("%Y-%m-%d")
        
        for ticker in tickers:
            try:
                print(f"Processing {ticker}...")
                posts = self.search_ticker(ticker, limit_per_subreddit=posts_per_ticker // len(self.stock_subreddits))
                summary = self.generate_ticker_summary(ticker, posts)
                
                # Save the summary to individual files
                if posts:  # Only save if we have posts
                    self.save_reddit_summary(ticker, summary)
                
                reddit_data[ticker] = {
                    'summary': summary,
                    'posts': posts,
                    'last_updated': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                }
                
            except Exception as e:
                print(f"Error processing {ticker}: {e}")
                reddit_data[ticker] = {
                    'summary': None,
                    'posts': [],
                    'error': str(e),
                    'last_updated': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                }
        
        return {current_date: reddit_data}
    
    def analyze_popular_subreddits(self, subreddits: List[str] = None) -> Dict[str, Any]:
        """Analyze discussions in popular financial subreddits for opportunities"""
        if subreddits is None:
            subreddits = ['wallstreetbets', 'stocks']
        
        print(f"Analyzing discussions in {len(subreddits)} subreddits...")
        
        subreddit_analyses = {}
        current_date = datetime.now().strftime("%Y-%m-%d")
        
        for subreddit in subreddits:
            try:
                analysis = self.analyze_subreddit_discussions(subreddit)
                subreddit_analyses[subreddit] = analysis
            except Exception as e:
                print(f"Error analyzing r/{subreddit}: {e}")
                subreddit_analyses[subreddit] = {
                    'error': str(e),
                    'subreddit': subreddit
                }
        
        return {current_date: subreddit_analyses}
    
    def save_data(self, filepath: str, data: Dict[str, Any]):
        """Save Reddit tracking data to file"""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        # If file exists, load and merge with existing data
        if os.path.exists(filepath):
            try:
                with open(filepath, 'rb') as f:
                    existing_data = pickle.load(f)
                # Merge with existing data
                existing_data.update(data)
                data = existing_data
            except Exception as e:
                print(f"Warning: Could not load existing data: {e}")
        
        with open(filepath, 'wb') as f:
            pickle.dump(data, f)
        
        print(f"Reddit data saved to {filepath}")
    
    def save_reddit_summary(self, ticker: str, summary: RedditSummary):
        """Save Reddit summary to file using the same format as news summaries"""
        summaries_dir = os.path.join(DATA_DIR, 'reddit_summaries')
        os.makedirs(summaries_dir, exist_ok=True)
        
        filename = f"{summaries_dir}/{ticker}.pkl"
        current_date = datetime.now().strftime("%Y-%m-%d")
        
        summary_data = {
            current_date: {
                "summary": summary.summary,
                "sentiment": summary.sentiment,
                "price_impact": summary.price_impact,
                "positive_factors": summary.positive_factors,
                "negative_factors": summary.negative_factors,
                "earnings_boosters": summary.earnings_boosters,
                "earnings_sinks": summary.earnings_sinks,
                "source": "Reddit",
                "total_posts": summary.total_posts,
                "subreddits": list(summary.subreddit_breakdown.keys()),
                "avg_score": sum(p.score for p in summary.top_posts) / len(summary.top_posts) if summary.top_posts else 0,
                "last_updated": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
        }
        
        if os.path.exists(filename):
            try:
                with open(filename, 'rb') as f:
                    existing_data = pickle.load(f)
                    existing_data.update(summary_data)
            except:
                existing_data = summary_data
            with open(filename, 'wb') as f:
                pickle.dump(existing_data, f)
        else:
            with open(filename, 'wb') as f:
                pickle.dump(summary_data, f)
        
        print(f"Reddit summary saved to {filename}")
        return filename

if __name__ == "__main__":
    # Test the Reddit tracker
    tracker = RedditTracker()
    
    # Test with a single ticker
    test_ticker = "AAPL"
    posts = tracker.search_ticker(test_ticker, limit_per_subreddit=5)
    print(f"Found {len(posts)} posts for {test_ticker}")
    
    if posts:
        summary = tracker.generate_ticker_summary(test_ticker, posts)
        print(f"Summary: {summary.summary}")
        print(f"Sentiment: {summary.sentiment}")
        print(f"Positive factors: {len(summary.positive_factors)}")
        print(f"Negative factors: {len(summary.negative_factors)}")

"""
Example usage of the WSJ Scraper module
This script demonstrates how to login to WSJ and scrape various types of data
"""

import asyncio
import sys
import os
from pathlib import Path
import dotenv

dotenv.load_dotenv()

# Add the src directory to the path so we can import our modules
sys.path.append(str(Path(__file__).parent.parent / 'src'))

from wsj_scraper import WSJScraper, scrape_wsj_articles_manual, scrape_wsj_articles_auto, get_wsj_market_data_manual, get_wsj_market_data_auto, scrape_wsj_business_articles_manual, scrape_wsj_business_articles_auto


async def example_manual_login():
    """Manual login example with context manager"""
    print("=== Manual Login WSJ Scraper Usage ===")
    
    async with WSJScraper(headless=False) as scraper:  # headless=False to see the browser
        # Manual login to WSJ
        print("🌐 Please login manually in the browser window...")
        if await scraper.login_manual(timeout_minutes=3):
            print("✅ Successfully logged in to WSJ")
            
            # Search for articles
            print("\n📰 Searching for articles about 'stock market'...")
            article_urls = await scraper.search_articles("Apple", limit=3)
            print(f"Found {len(article_urls)} articles:")
            for i, url in enumerate(article_urls, 1):
                print(f"  {i}. {url}")
            
            # Scrape the first article
            if article_urls:
                print(f"\n📖 Scraping first article...")
                article_data = await scraper.scrape_article(article_urls[0])
                print(f"Title: {article_data.get('title', 'No title')}")
                print(f"Author: {article_data.get('author', 'No author')}")
                print(f"Date: {article_data.get('publish_date', 'No date')}")
                print(f"Content preview: {article_data.get('content', '')[:200]}...")
            
            # Get trending topics
            print(f"\n🔥 Getting trending topics...")
            topics = await scraper.get_trending_topics()
            print("Trending topics:")
            for i, topic in enumerate(topics[:5], 1):
                print(f"  {i}. {topic}")
            
            # Get market data
            print(f"\n📊 Getting market data...")
            market_data = await scraper.scrape_market_data()
            if market_data.get('major_indices'):
                print("Major indices:")
                for name, data in market_data['major_indices'].items():
                    print(f"  {name}: {data.get('value', 'N/A')} ({data.get('change', 'N/A')})")
            
        else:
            print("❌ Failed to login to WSJ")


async def example_automated_login():
    """Automated login example (fallback method)"""
    print("=== Automated Login WSJ Scraper Usage ===")
    
    # Replace with your WSJ credentials
    username = os.getenv('WSJ_USERNAME', 'your_wsj_username@email.com')
    password = os.getenv('WSJ_PASSWORD', 'your_wsj_password')
    
    if username == 'your_wsj_username@email.com':
        print("⚠️  Please set WSJ_USERNAME and WSJ_PASSWORD environment variables")
        return
    
    async with WSJScraper(headless=True) as scraper:  # headless for automated
        # Automated login to WSJ
        if await scraper.login_automated(username, password):
            print("✅ Successfully logged in to WSJ automatically")
            
            # Search for articles
            print("\n📰 Searching for articles about 'technology'...")
            article_urls = await scraper.search_articles("technology", limit=2)
            print(f"Found {len(article_urls)} articles:")
            for i, url in enumerate(article_urls, 1):
                print(f"  {i}. {url}")
            
        else:
            print("❌ Failed to login to WSJ automatically")


async def example_convenience_functions_manual():
    """Example using convenience functions with manual login"""
    print("\n=== Using Convenience Functions (Manual Login) ===")
    
    try:
        # Use convenience function to scrape articles with manual login
        print("📰 Scraping articles about 'artificial intelligence' (manual login)...")
        articles = await scrape_wsj_articles_manual("artificial intelligence", limit=2, timeout_minutes=3)
        
        print(f"Successfully scraped {len(articles)} articles:")
        for i, article in enumerate(articles, 1):
            print(f"\n  Article {i}:")
            print(f"    Title: {article.get('title', 'No title')}")
            print(f"    URL: {article.get('url', 'No URL')}")
            print(f"    Author: {article.get('author', 'No author')}")
            
        # Use convenience function to get market data
        print(f"\n📊 Getting market data (manual login)...")
        market_data = await get_wsj_market_data_manual(timeout_minutes=2)
        print(f"Market data scraped at: {market_data.get('scraped_at', 'Unknown')}")
        
    except Exception as e:
        print(f"❌ Error: {e}")


async def example_convenience_functions_auto():
    """Example using convenience functions with automated login"""
    print("\n=== Using Convenience Functions (Automated Login) ===")
    
    # Get credentials from environment variables
    username = os.getenv('WSJ_USERNAME')
    password = os.getenv('WSJ_PASSWORD')
    
    if not username or not password:
        print("⚠️  Please set WSJ_USERNAME and WSJ_PASSWORD environment variables")
        return
    
    try:
        # Use convenience function to scrape articles with automated login
        print("📰 Scraping articles about 'artificial intelligence' (automated login)...")
        articles = await scrape_wsj_articles_auto(username, password, "artificial intelligence", limit=2)
        
        print(f"Successfully scraped {len(articles)} articles:")
        for i, article in enumerate(articles, 1):
            print(f"\n  Article {i}:")
            print(f"    Title: {article.get('title', 'No title')}")
            print(f"    URL: {article.get('url', 'No URL')}")
            print(f"    Author: {article.get('author', 'No author')}")
            
        # Use convenience function to get market data
        print(f"\n📊 Getting market data (automated login)...")
        market_data = await get_wsj_market_data_auto(username, password)
        print(f"Market data scraped at: {market_data.get('scraped_at', 'Unknown')}")
        
    except Exception as e:
        print(f"❌ Error: {e}")


async def example_business_articles_manual():
    """Example scraping all business articles with manual login"""
    print("\n=== Business Articles Scraper (Manual Login) ===")
    
    try:
        print("📰 Scraping ALL articles from WSJ business page (manual login)...")
        print("📋 This will scrape articles from all business sections like:")
        print("   • Autos, Energy & Oil, Airlines, C-Suite")
        print("   • Hospitality, Media, Logistics, Telecom")
        print("   • Entrepreneurship, and more...")
        print("⏱️  This may take several minutes to complete")
        
        articles = await scrape_wsj_business_articles_manual(save_to_file=True)
        
        if articles:
            print(f"\n🎉 Successfully scraped {len(articles)} business articles!")
            print("\n📋 Sample of scraped articles:")
            for i, article in enumerate(articles[:5], 1):  # Show first 5
                print(f"  {i}. {article.get('title', 'No title')[:70]}...")
                print(f"     Author: {article.get('author', 'Unknown')}")
                print(f"     Content: {len(article.get('content', ''))} characters")
            
            if len(articles) > 5:
                print(f"     ... and {len(articles) - 5} more articles")
            
            print(f"\n💾 All articles saved to JSON file")
            
        else:
            print("❌ No articles were scraped")
            
    except Exception as e:
        print(f"❌ Error: {e}")


async def example_business_articles_auto():
    """Example scraping all business articles with automated login"""
    print("\n=== Business Articles Scraper (Automated Login) ===")
    
    # Get credentials from environment variables
    username = os.getenv('WSJ_USERNAME')
    password = os.getenv('WSJ_PASSWORD')
    
    if not username or not password:
        print("⚠️  Please set WSJ_USERNAME and WSJ_PASSWORD environment variables")
        return
    
    try:
        print("📰 Scraping ALL articles from WSJ business page (automated login)...")
        print("📋 This will scrape articles from all business sections")
        print("⏱️  This may take several minutes to complete")
        
        articles = await scrape_wsj_business_articles_auto(username, password, save_to_file=True)
        
        if articles:
            print(f"\n🎉 Successfully scraped {len(articles)} business articles!")
            print("\n📋 Sample of scraped articles:")
            for i, article in enumerate(articles[:5], 1):  # Show first 5
                print(f"  {i}. {article.get('title', 'No title')[:70]}...")
                print(f"     Author: {article.get('author', 'Unknown')}")
                print(f"     Content: {len(article.get('content', ''))} characters")
            
            if len(articles) > 5:
                print(f"     ... and {len(articles) - 5} more articles")
            
            print(f"\n💾 All articles saved to JSON file")
            
        else:
            print("❌ No articles were scraped")
            
    except Exception as e:
        print(f"❌ Error: {e}")


async def example_session_persistence():
    """Example showing session persistence with manual login"""
    print("\n=== Session Persistence Example (Manual Login) ===")
    
    # First run - will login manually and save session
    print("🔑 First run - manual login and saving session...")
    async with WSJScraper(headless=False) as scraper:
        if await scraper.login_manual(save_session=True, timeout_minutes=3):
            print("✅ Logged in manually and session saved")
            topics = await scraper.get_trending_topics()
            print(f"Got {len(topics)} trending topics")
    
    # Second run - will load existing session
    print("\n🔄 Second run - loading existing session...")
    async with WSJScraper(headless=True) as scraper:
        if await scraper.login_manual(save_session=True, timeout_minutes=1):  # Shorter timeout since session should exist
            print("✅ Session loaded successfully")
            # Do more scraping...
            market_data = await scraper.scrape_market_data()
            print(f"Got market data: {len(market_data)} items")


def setup_environment_example():
    """Show how to setup environment variables"""
    print("\n=== Environment Setup ===")
    print("To use this scraper, you need to set up your WSJ credentials.")
    print("You can either:")
    print("1. Set environment variables:")
    print("   export WSJ_USERNAME='your_username@email.com'")
    print("   export WSJ_PASSWORD='your_password'")
    print("\n2. Or create a .env file in your project root:")
    print("   WSJ_USERNAME=your_username@email.com")
    print("   WSJ_PASSWORD=your_password")
    print("\n3. Or pass credentials directly to the functions (not recommended for production)")


async def main():
    """Main function to run examples"""
    print("🚀 WSJ Scraper Examples")
    print("=" * 50)
    
    print("Choose which examples to run:")
    print("1. Manual login example (recommended)")
    print("2. Automated login example") 
    print("3. Convenience functions (manual)")
    print("4. Convenience functions (automated)")
    print("5. Business articles scraper (manual) - 🆕 NEW!")
    print("6. Business articles scraper (automated) - 🆕 NEW!")
    print("7. Session persistence example")
    print("8. All examples")
    
    choice = input("\nEnter your choice (1-8): ").strip()
    
    try:
        if choice == "1":
            await example_manual_login()
        elif choice == "2":
            await example_automated_login()
        elif choice == "3":
            await example_convenience_functions_manual()
        elif choice == "4":
            await example_convenience_functions_auto()
        elif choice == "5":
            await example_business_articles_manual()
        elif choice == "6":
            await example_business_articles_auto()
        elif choice == "7":
            await example_session_persistence()
        elif choice == "8":
            print("\n🔄 Running all examples...")
            await example_manual_login()
            await example_automated_login()
            await example_convenience_functions_manual()
            await example_convenience_functions_auto()
            await example_business_articles_manual()
            await example_business_articles_auto()
            await example_session_persistence()
        else:
            print("❌ Invalid choice")
            return
        
    except KeyboardInterrupt:
        print("\n👋 Examples interrupted by user")
    except Exception as e:
        print(f"\n❌ Error running examples: {e}")


if __name__ == "__main__":
    # Install required packages first
    print("📦 Make sure you have installed the required packages:")
    print("pip install playwright beautifulsoup4 python-dotenv")
    print("playwright install chromium")
    print()
    
    # Run the examples
    asyncio.run(main()) 
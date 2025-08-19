#!/bin/bash

echo "🚀 Setting up WSJ Scraper dependencies..."

# Update pip
echo "📦 Updating pip..."
pip install --upgrade pip

# Install required packages
echo "📦 Installing Python packages..."
pip install playwright beautifulsoup4 lxml python-dotenv pandas

# Install playwright browsers
echo "🌐 Installing Playwright browsers..."
playwright install chromium

# Create examples directory if it doesn't exist
mkdir -p examples

echo "✅ Setup complete!"
echo ""
echo "📝 Next steps:"
echo "1. Set up your WSJ credentials:"
echo "   export WSJ_USERNAME='your_username@email.com'"
echo "   export WSJ_PASSWORD='your_password'"
echo ""
echo "   Or create a .env file in the project root:"
echo "   WSJ_USERNAME=your_username@email.com"
echo "   WSJ_PASSWORD=your_password"
echo ""
echo "2. Test the scraper:"
echo "   cd examples"
echo "   python wsj_scraper_example.py"
echo ""
echo "3. Use the scraper in your own code:"
echo "   from src.wsj_scraper import WSJScraper, scrape_wsj_articles"
echo ""
echo "�� Happy scraping!" 
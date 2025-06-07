import os
import pickle
from datetime import datetime
import json
import re
from dotenv import load_dotenv
from together import Together

class NewsSummarizer:
    def __init__(self):
        load_dotenv()
        api_key = os.environ.get("TOGETHER_API_KEY")
        if not api_key:
            print("Warning: TOGETHER_API_KEY environment variable not set. News summarization may not work.")
        self.client = Together(api_key=api_key)
        self.model = "deepseek-ai/DeepSeek-R1-Distill-Llama-70B-free"
        self.summaries_dir = "data/news_summaries"
        self.summaries = {}
        os.makedirs(self.summaries_dir, exist_ok=True)
        
    # Format articles for the prompt
    def process_article_batch(self, ticker, articles_batch):
        formatted_articles = []
        for i, article in enumerate(articles_batch):
            title = article.get('headline', 'Untitled')
            date = article.get('date', 'Unknown date')
            source = article.get('source', 'Unknown source')
            summary = article.get('summary', '')
            content = article.get('full_content', '')
            formatted_articles.append(f"Article {i+1}: '{title}' - {date} ({source})\n{summary}\n{content}\n")
        
        articles_text = "\n".join(formatted_articles)
        
        prompt = f"""Analyze the following news articles about {ticker} and create a detailed quantitative assessment about the company associated with {ticker}.

        Here are the recent news articles:
        {articles_text}

        Please provide for the {ticker}:

        1. A detailed executive summary (5-7 sentences) that includes specific figures, percentages, or monetary values mentioned in the articles
        2. Key positive developments with quantifiable metrics where available (e.g., "Revenue increased by 15% to $3.2B" rather than just "Revenue increased")
        3. Specific negative factors or risks with concrete examples and data points from the articles
        4. Nuanced sentiment analysis that goes beyond simple positive/negative/neutral and discusses the degree and context of the sentiment
        5. Impact assessment on stock price

        Format your response as JSON with the following structure:
        {{
        "summary": "Comprehensive analysis with specific figures and details for the {ticker}",
        "positive_factors": [
            {{"factor": "Description of positive factor 1 for the {ticker}", "metrics": "Specific numbers/percentages"}},
            {{"factor": "Description of positive factor 2 for the {ticker}", "metrics": "Specific numbers/percentages"}}
        ],
        "negative_factors": [
            {{"factor": "Description of negative factor 1 for the {ticker}", "metrics": "Specific numbers/percentages"}},
            {{"factor": "Description of negative factor 2 for the {ticker}", "metrics": "Specific numbers/percentages"}}
        ],
        "sentiment": "positive/negative/neutral/mixed",
        "price_impact": "likely impact on price for the {ticker}"        
    }}"""

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
            )
            summary = response.choices[0].message.content
            # Remove <think>...</think> blocks
            text = re.sub(r'<think>.*?</think>', '', summary, flags=re.DOTALL)
            # Check if there's a JSON code block
            json_block_match = re.search(r'```(?:json)?([\s\S]*?)```', text, re.DOTALL)
            if json_block_match:
                json_str = json_block_match.group(1).strip()
            else:
                json_str = text.strip()
            return json.loads(json_str)
        except Exception as e:
            return {"summary": f"Error generating summary: {str(e)}", "error": str(e)}

    def merge_summaries(self, summaries_list):
        merged = {
            "summary": "",
            "positive_factors": [],
            "negative_factors": [],
            "sentiment": "",
            "price_impact": ""
        }
        for summary in summaries_list:
            if isinstance(summary, dict) and "error" not in summary:
                merged["summary"] += summary.get("summary", "") + " "
                merged["positive_factors"].extend(summary.get("positive_factors", []))
                merged["negative_factors"].extend(summary.get("negative_factors", []))
                
                # Combine sentiment analysis
                if "sentiment" in summary:
                    merged["sentiment"] += summary["sentiment"] + " "
                            
                # Combine price impact assessment
                if "price_impact" in summary:
                    merged["price_impact"] += str(summary["price_impact"]) + " "
        # Determine overall sentiment
        if merged["sentiment"]:
            sentiments = merged["sentiment"].lower().split()
            positive_count = sentiments.count("positive")
            negative_count = sentiments.count("negative")
            merged["sentiment"] = "positive" if positive_count > negative_count else "negative" if negative_count > positive_count else "neutral"

        return merged

    def summarize_news(self, ticker, articles):
        """Summarize news articles for a ticker using Together AI"""
        if not articles:
            return {"summary": "No recent news articles found for this ticker."}
        
        batch_size = 4
        summaries = []
        
        for i in range(0, len(articles), batch_size):
            batch = articles[i:i + batch_size]
            batch_summary = self.process_article_batch(ticker, batch)
            summaries.append(batch_summary)
        
        merged_summary = self.merge_summaries(summaries)
        
        if ticker not in self.summaries:
            self.summaries[ticker] = {}
        self.summaries[ticker] = merged_summary
        
        self.save_summary(ticker, self.summaries[ticker])
        return merged_summary
    
    def save_summary(self, ticker, summary):
        """Save news summary to a file"""
        filename = f"{self.summaries_dir}/{ticker}.pkl"
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        
        current_date = datetime.now().strftime("%Y-%m-%d")
        new_data = {
            current_date: {
                "summary": summary["summary"],
                "sentiment": summary["sentiment"],
                "price_impact": summary["price_impact"],
                "positive_factors": summary["positive_factors"],
                "negative_factors": summary["negative_factors"],
                "last_updated": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }        
        }
        
        if os.path.exists(filename):
            with open(filename, 'rb') as f:
                existing_data = pickle.load(f)
                existing_data.update(new_data)
            with open(filename, 'wb') as f:
                pickle.dump(existing_data, f)
        else:
            with open(filename, 'wb') as f:
                pickle.dump(new_data, f)
        
        print(f"Data saved to {filename}")   
        return filename
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
    
    def summarize_news(self, ticker, articles):
        """Summarize news articles for a ticker using Together AI"""
        if not articles:
            return {"summary": "No recent news articles found for this ticker."}
        
        # Format articles for the prompt
        formatted_articles = []
        for i, article in enumerate(articles[:10]):  # Limit to 10 articles to avoid token limits
            title = article.get('headline', 'Untitled')
            date = article.get('date', 'Unknown date')
            source = article.get('source', 'Unknown source')
            summary = article.get('summary', '')
            content = article.get('full_content', '')
            formatted_articles.append(f"Article {i+1}: '{title}' - {date} ({source})\n{summary}\n")
        
        articles_text = "\n".join(formatted_articles)
        
        prompt = f"""
        I need you to analyze the following news articles about {ticker} stock and create a comprehensive summary.
        
        Here are the recent news articles:
        {articles_text}
        
        Please provide:
        1. A concise summary of the key developments (3-5 sentences)
        2. Any notable positive news factors
        3. Any notable negative news factors or risks
        4. Overall news sentiment (positive, negative, or neutral)
        5. Potential impact on stock price based on these news items
        Do not output any thinking 
        Format your response as JSON with the following structure:
        {{
            "summary": "Overall summary here",
            "positive_factors": ["factor1", "factor2", ...],
            "negative_factors": ["factor1", "factor2", ...],
            "sentiment": "positive/negative/neutral",
            "price_impact": "likely impact on price"
        }}
        """
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
            )
            summary = response.choices[0].message.content
            
            # Remove <think>...</think> blocks
            text = re.sub(r'<think>.*?</think>', '', summary, flags=re.DOTALL)
            
            # Check if there's a JSON code block
            json_block_match = re.search(r'```(?:json)?\s*([\s\S]*?)```', text, re.DOTALL)
            
            if json_block_match:
                # Extract the content inside the code block
                json_str = json_block_match.group(1).strip()
            else:
                # If no code block, use the entire text after removing think tags
                json_str = text.strip()
            
            # Try to parse as JSON
            try:
                summary_data = json.loads(json_str)
                today = datetime.now().strftime("%Y-%m-%d")
                
                if ticker not in self.summaries:
                    self.summaries[ticker] = {}
                self.summaries[ticker][today] = summary_data
                
                self.save_summary(ticker, self.summaries[ticker])
                return summary_data
            except json.JSONDecodeError as e:
                print(f"Error parsing JSON: {e}")
                return {"text": text, "error": str(e)}
            
        except Exception as e:
            return {"summary": f"Error generating summary: {str(e)}", "error": str(e)}
    
    def save_summary(self, ticker, summary):
        """Save news summary to a file"""
        filename = f"{self.summaries_dir}/{ticker}.pkl"
        
        with open(filename, 'wb') as f:
            pickle.dump(summary, f)
        
        return filename
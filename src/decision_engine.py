import pandas as pd
import numpy as np
from scipy.signal import argrelextrema
import pickle
from datetime import datetime
from src.news_summarizer import NewsSummarizer

class DecisionEngine:
    def __init__(self, investor_data=None, news_data=None, stock_data=None, fundamentals_data=None):
        self.investor_data = investor_data or {}
        self.news_data = news_data or {}
        self.stock_data = stock_data or {}
        self.fundamentals_data = fundamentals_data or {}
        self.recommendations = {}
        self.news_summarizer = NewsSummarizer()
        self.news_summaries = {}
    
    def analyze_position_changes(self):
        """Analyze the significant position changes by investors"""
        insights = {}
        
        if not self.investor_data or 'changes' not in self.investor_data:
            return {'error': 'No investor position change data available'}
        
        changes = self.investor_data.get('changes', {})
        
        # Aggregate changes by ticker
        ticker_changes = {}
        for investor, positions in changes.items():
            for position in positions:
                ticker = position.get('ticker')
                if ticker not in ticker_changes:
                    ticker_changes[ticker] = []
                
                position_copy = position.copy()
                position_copy['investor'] = investor
                ticker_changes[ticker].append(position_copy)
        
        # Generate insights for each ticker
        for ticker, positions in ticker_changes.items():
            buy_count = sum(1 for p in positions if p.get('change_type') in ['NEW POSITION', 'INCREASE'])
            sell_count = sum(1 for p in positions if p.get('change_type') in ['EXITED POSITION', 'DECREASE'])
            
            # Simple consensus calculation
            consensus = None
            if buy_count > sell_count:
                consensus = 'BULLISH'
            elif sell_count > buy_count:
                consensus = 'BEARISH'
            else:
                consensus = 'NEUTRAL'
            
            insights[ticker] = {
                'positions': positions,
                'buy_count': buy_count,
                'sell_count': sell_count,
                'consensus': consensus
            }
        
        return insights
    
    def analyze_news_sentiment(self):
        """Analyze news sentiment using Together AI via NewsSummarizer"""
        if not self.news_data:
            return {'error': 'No news data available'}
        
        sentiments = {}
        
        # Initialize the news summarizer if not already done
        if not hasattr(self, 'news_summarizer'):
            self.news_summarizer = NewsSummarizer()
        
        for company, articles in self.news_data.items():
            if not articles:
                continue
            
            print(f"Analyzing news sentiment for {company} using Together AI...")
            
            # Use the news summarizer to get a comprehensive analysis
            summary_result = self.news_summarizer.summarize_news(company, articles)
            
            # Extract sentiment from the summary result
            if isinstance(summary_result, dict):
                sentiment_value = summary_result.get('sentiment', 'NEUTRAL')
                
                # Standardize sentiment values
                if sentiment_value.lower() in ['positive', 'bullish', 'optimistic']:
                    sentiment = 'POSITIVE'
                elif sentiment_value.lower() in ['negative', 'bearish', 'pessimistic']:
                    sentiment = 'NEGATIVE'
                else:
                    sentiment = 'NEUTRAL'
                
                # Calculate a sentiment score based on positive and negative factors
                positive_factors = summary_result.get('positive_factors', [])
                negative_factors = summary_result.get('negative_factors', [])
                
                total_factors = len(positive_factors) + len(negative_factors)
                if total_factors > 0:
                    sentiment_score = (len(positive_factors) - len(negative_factors)) / total_factors
                else:
                    sentiment_score = 0
                
                sentiments[company] = {
                    'sentiment': sentiment,
                    'score': sentiment_score,
                    'summary': summary_result.get('summary', 'No summary available'),
                    'positive_factors': positive_factors,
                    'negative_factors': negative_factors,
                    'price_impact': summary_result.get('price_impact', 'Unknown'),
                    'article_count': len(articles)
                }
            else:
                # Fallback to the original keyword-based method if the API call fails
                print(f"Warning: Together AI summarization failed for {company}. Falling back to keyword method.")
                sentiments[company] = self._analyze_news_sentiment_keywords(articles)
        
        return sentiments

    def _analyze_news_sentiment_keywords(self, articles):
        """Legacy keyword-based sentiment analysis as a fallback"""
        positive_keywords = ['surge', 'jump', 'gain', 'rise', 'growth', 'profit', 
                            'boost', 'positive', 'success', 'strong', 'beat', 'exceed']
        negative_keywords = ['drop', 'fall', 'decline', 'loss', 'negative', 'weak', 
                            'cut', 'miss', 'below', 'trouble', 'risk', 'concern']
        
        positive_count = 0
        negative_count = 0
        neutral_count = 0
        
        for article in articles:
            text = (article.get('headline', '') + ' ' + article.get('summary', '')).lower()
            
            pos_matches = sum(1 for word in positive_keywords if word in text)
            neg_matches = sum(1 for word in negative_keywords if word in text)
            
            if pos_matches > neg_matches:
                positive_count += 1
            elif neg_matches > pos_matches:
                negative_count += 1
            else:
                neutral_count += 1
        
        total = positive_count + negative_count + neutral_count
        if total > 0:
            sentiment_score = (positive_count - negative_count) / total
            
            if sentiment_score > 0.2:
                sentiment = 'POSITIVE'
            elif sentiment_score < -0.2:
                sentiment = 'NEGATIVE'
            else:
                sentiment = 'NEUTRAL'
        else:
            sentiment = 'NEUTRAL'
            sentiment_score = 0
        
        return {
            'sentiment': sentiment,
            'score': sentiment_score,
            'positive_articles': positive_count,
            'negative_articles': negative_count,
            'neutral_articles': neutral_count,
            'article_count': len(articles)
        }
    
    def analyze_fundamentals(self):
        """Analyze company fundamentals"""
        if not self.fundamentals_data:
            return {'error': 'No fundamentals data available'}
        
        analysis = {}
        
        for ticker, data in self.fundamentals_data.items():
            if 'ratios' not in data:
                continue
                
            ratios = data['ratios']
            
            # Basic fundamental analysis
            pe_ratio = ratios.get('P/E')
            debt_equity = ratios.get('Debt/Equity')
            roe = ratios.get('ROE')
            profit_margin = ratios.get('Profit Margin')
            
            # Evaluate fundamentals (this is a simplified approach)
            valuation = None
            if pe_ratio is not None:
                if pe_ratio < 15:
                    valuation = 'UNDERVALUED'
                elif pe_ratio > 25:
                    valuation = 'OVERVALUED'
                else:
                    valuation = 'FAIR VALUE'
            
            financial_health = None
            if debt_equity is not None:
                if debt_equity < 0.5:
                    financial_health = 'STRONG'
                elif debt_equity > 1.5:
                    financial_health = 'WEAK'
                else:
                    financial_health = 'MODERATE'
            
            profitability = None
            if roe is not None and profit_margin is not None:
                if roe > 0.15 and profit_margin > 0.1:
                    profitability = 'HIGH'
                elif roe < 0.05 or profit_margin < 0.03:
                    profitability = 'LOW'
                else:
                    profitability = 'MODERATE'
            
            analysis[ticker] = {
                'valuation': valuation,
                'financial_health': financial_health,
                'profitability': profitability,
                'ratios': ratios
            }
        
        return analysis
    
    def analyze_price_trends(self):
        """Analyze price trends from stock data using multiple financial heuristics"""
        if not self.stock_data or 'stocks' not in self.stock_data:
            return {'error': 'No stock price data available'}
            
        trends = {}
        for ticker, data in self.stock_data.get('stocks', {}).items():
            if 'history' not in data or data['history'].empty:
                continue
                
            hist = data['history']
            
            # Calculate moving averages
            hist['MA50'] = hist['Close'].rolling(window=50).mean()
            hist['MA200'] = hist['Close'].rolling(window=200).mean()
            
            # Get latest values
            latest_close = hist['Close'].iloc[-1]
            latest_ma50 = hist['MA50'].iloc[-1] if not pd.isna(hist['MA50'].iloc[-1]) else None
            latest_ma200 = hist['MA200'].iloc[-1] if not pd.isna(hist['MA200'].iloc[-1]) else None
            
            # Determine trend
            trend = None
            trend_strength = None
            if latest_ma50 is not None and latest_ma200 is not None:
                if latest_close > latest_ma50 and latest_ma50 > latest_ma200:
                    trend = 'UPTREND'
                    if (latest_close / latest_ma200 - 1) > 0.1:
                        trend_strength = 'STRONG'
                    else:
                        trend_strength = 'MODERATE'
                elif latest_close < latest_ma50 and latest_ma50 < latest_ma200:
                    trend = 'DOWNTREND'
                    if (1 - latest_close / latest_ma200) > 0.1:
                        trend_strength = 'STRONG'
                    else:
                        trend_strength = 'MODERATE'
                else:
                    trend = 'SIDEWAYS'
                    trend_strength = 'NEUTRAL'
            
            # Calculate volatility (using standard deviation of returns)
            returns = hist['Close'].pct_change().dropna()
            volatility = returns.std() * np.sqrt(252)  # Annualized volatility
            
            # NEW: Calculate RSI (Relative Strength Index)
            delta = hist['Close'].diff()
            gain = delta.where(delta > 0, 0)
            loss = -delta.where(delta < 0, 0)
            avg_gain = gain.rolling(window=14).mean()
            avg_loss = loss.rolling(window=14).mean()
            rs = avg_gain / avg_loss
            hist['RSI'] = 100 - (100 / (1 + rs))
            latest_rsi = hist['RSI'].iloc[-1] if not pd.isna(hist['RSI'].iloc[-1]) else None
            
            # NEW: Calculate MACD
            hist['EMA12'] = hist['Close'].ewm(span=12, adjust=False).mean()
            hist['EMA26'] = hist['Close'].ewm(span=26, adjust=False).mean()
            hist['MACD'] = hist['EMA12'] - hist['EMA26']
            hist['Signal'] = hist['MACD'].ewm(span=9, adjust=False).mean()
            hist['MACD_Histogram'] = hist['MACD'] - hist['Signal']
            latest_macd = hist['MACD'].iloc[-1] if not pd.isna(hist['MACD'].iloc[-1]) else None
            latest_signal = hist['Signal'].iloc[-1] if not pd.isna(hist['Signal'].iloc[-1]) else None
            
            # NEW: Calculate Bollinger Bands
            hist['BB_Middle'] = hist['Close'].rolling(window=20).mean()
            hist['BB_Std'] = hist['Close'].rolling(window=20).std()
            hist['BB_Upper'] = hist['BB_Middle'] + (hist['BB_Std'] * 2)
            hist['BB_Lower'] = hist['BB_Middle'] - (hist['BB_Std'] * 2)
            bb_width = (hist['BB_Upper'].iloc[-1] - hist['BB_Lower'].iloc[-1]) / hist['BB_Middle'].iloc[-1]
            
            # NEW: Volume analysis
            avg_volume = hist['Volume'].rolling(window=50).mean().iloc[-1]
            latest_volume = hist['Volume'].iloc[-1]
            volume_trend = 'HIGH' if latest_volume > avg_volume * 1.5 else 'NORMAL'
            
            # NEW: Beta calculation (market correlation)
            if 'market' in self.stock_data and 'history' in self.stock_data['market']:
                market_hist = self.stock_data['market']['history']
                if len(market_hist) > 30 and len(hist) > 30:
                    # Align dates
                    stock_returns = hist['Close'].pct_change().dropna()
                    market_returns = market_hist['Close'].pct_change().dropna()
                    
                    # Calculate beta only if we have overlapping dates
                    common_dates = stock_returns.index.intersection(market_returns.index)
                    if len(common_dates) > 30:
                        stock_returns_aligned = stock_returns.loc[common_dates]
                        market_returns_aligned = market_returns.loc[common_dates]
                        covariance = stock_returns_aligned.cov(market_returns_aligned)
                        market_variance = market_returns_aligned.var()
                        beta = covariance / market_variance if market_variance != 0 else 1
                    else:
                        beta = None
                else:
                    beta = None
            else:
                beta = None
            
            # NEW: Price patterns
            # Simple check for potential double bottom
            if len(hist) > 60:
                last_60 = hist['Close'].iloc[-60:]
                recent_lows = argrelextrema(last_60.values, np.less)[0]
                double_bottom = False
                if len(recent_lows) >= 2:
                    last_two_lows = recent_lows[-2:]
                    if abs(last_60.iloc[last_two_lows[0]] - last_60.iloc[last_two_lows[1]]) < last_60.iloc[last_two_lows[0]] * 0.05:
                        if last_two_lows[1] - last_two_lows[0] > 10:  # At least 10 days apart
                            double_bottom = True
            else:
                double_bottom = None
            
            # Gather all insights
            trends[ticker] = {
                'trend': trend,
                'trend_strength': trend_strength,
                'volatility': volatility,
                'latest_close': latest_close,
                'ma50': latest_ma50,
                'ma200': latest_ma200,
                'rsi': latest_rsi,
                'rsi_status': 'OVERSOLD' if latest_rsi and latest_rsi < 30 else 'OVERBOUGHT' if latest_rsi and latest_rsi > 70 else 'NEUTRAL',
                'macd': latest_macd,
                'macd_signal': latest_signal,
                'macd_crossover': 'BULLISH' if latest_macd and latest_signal and latest_macd > latest_signal else 'BEARISH' if latest_macd and latest_signal and latest_macd < latest_signal else 'NONE',
                'bollinger_width': bb_width,
                'position_in_bb': 'UPPER' if latest_close > hist['BB_Upper'].iloc[-1] else 'LOWER' if latest_close < hist['BB_Lower'].iloc[-1] else 'MIDDLE',
                'volume_trend': volume_trend,
                'beta': beta,
                'pattern_double_bottom': double_bottom
            }
            
            # NEW: Add a summary recommendation based on multiple indicators
            bullish_signals = 0
            bearish_signals = 0
            
            if trend == 'UPTREND': bullish_signals += 1
            if trend == 'DOWNTREND': bearish_signals += 1
            
            if latest_rsi and latest_rsi < 30: bullish_signals += 1  # Oversold can be bullish
            if latest_rsi and latest_rsi > 70: bearish_signals += 1  # Overbought can be bearish
            
            if latest_macd and latest_signal and latest_macd > latest_signal: bullish_signals += 1
            if latest_macd and latest_signal and latest_macd < latest_signal: bearish_signals += 1
            
            if double_bottom: bullish_signals += 1
            
            if latest_close < hist['BB_Lower'].iloc[-1]: bullish_signals += 1
            if latest_close > hist['BB_Upper'].iloc[-1]: bearish_signals += 1
            
            if bullish_signals > bearish_signals + 1:
                trends[ticker]['recommendation'] = 'BUY'
            elif bearish_signals > bullish_signals + 1:
                trends[ticker]['recommendation'] = 'SELL'
            else:
                trends[ticker]['recommendation'] = 'HOLD'
                
        return trends

    
    def generate_recommendations(self):
        """Generate investment recommendations based on all available data"""
        # Get analyses
        investor_insights = self.analyze_position_changes()
        news_sentiment = self.analyze_news_sentiment()
        fundamental_analysis = self.analyze_fundamentals()
        price_trends = self.analyze_price_trends()
        
        recommendations = {}
        
        # Combine all analyses
        all_tickers = set()
        all_tickers.update(investor_insights.keys() if isinstance(investor_insights, dict) else [])
        all_tickers.update(news_sentiment.keys() if isinstance(news_sentiment, dict) else [])
        all_tickers.update(fundamental_analysis.keys() if isinstance(fundamental_analysis, dict) else [])
        all_tickers.update(price_trends.keys() if isinstance(price_trends, dict) else [])
        
        for ticker in all_tickers:
            if ticker == 'error':
                continue
                
            recommendation = {
                'ticker': ticker,
                'investor_consensus': investor_insights.get(ticker, {}).get('consensus', 'NO DATA'),
                'news_sentiment': news_sentiment.get(ticker, {}).get('sentiment', 'NO DATA'),
                'valuation': fundamental_analysis.get(ticker, {}).get('valuation', 'NO DATA'),
                'financial_health': fundamental_analysis.get(ticker, {}).get('financial_health', 'NO DATA'),
                'price_trend': price_trends.get(ticker, {}).get('trend', 'NO DATA')
            }
            
            # Simple scoring system
            score = 0
            
            if recommendation['investor_consensus'] == 'BULLISH':
                score += 2
            elif recommendation['investor_consensus'] == 'BEARISH':
                score -= 2
                
            if recommendation['news_sentiment'] == 'POSITIVE':
                score += 1
            elif recommendation['news_sentiment'] == 'NEGATIVE':
                score -= 1
                
            if recommendation['valuation'] == 'UNDERVALUED':
                score += 1
            elif recommendation['valuation'] == 'OVERVALUED':
                score -= 1
                
            if recommendation['financial_health'] == 'STRONG':
                score += 1
            elif recommendation['financial_health'] == 'WEAK':
                score -= 1
                
            if recommendation['price_trend'] == 'UPTREND':
                score += 1
            elif recommendation['price_trend'] == 'DOWNTREND':
                score -= 1
            
            # Get news summary and incorporate it
            recommendation['detailed_news_summary'] = news_sentiment.get(ticker, {}).get('summary', 'No summary available')
            recommendation['news_positive_factors'] = news_sentiment.get(ticker, {}).get('positive_factors', [])
            recommendation['news_negative_factors'] = news_sentiment.get(ticker, {}).get('negative_factors', [])
            
            # Final recommendation
            if score >= 2:
                final_rec = 'STRONG BUY'
            elif score > 0:
                final_rec = 'BUY'
            elif score == 0:
                final_rec = 'HOLD'
            elif score > -2:
                final_rec = 'SELL'
            else:
                final_rec = 'STRONG SELL'
            
            recommendation['final_recommendation'] = final_rec
            recommendation['score'] = score
            
            recommendations[ticker] = recommendation
        
        self.recommendations = recommendations
        return recommendations
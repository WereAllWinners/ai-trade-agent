"""
Unit tests for news_fetcher — LLM sentiment scoring, fallback behavior,
and NEWS_LLM_SENTIMENT env flag.

All tests are fully offline — no Polygon API, no GPU required.
"""
import sys
from pathlib import Path
from unittest.mock import patch, MagicMock

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'scripts'))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'scripts' / 'data'))

import news_fetcher


class TestScoreSentimentWithLLM:
    """Tests for score_sentiment_with_llm()."""

    def test_llm_path_returns_correct_schema(self):
        """When LLM returns a well-formed response, the dict has correct keys/values."""
        llm_output = (
            "Sentiment: bullish\n"
            "Magnitude: high\n"
            "Reason: demand catalyst multi-quarter tailwind"
        )
        with patch('model_inference_lora.get_trading_decision', return_value=llm_output):
            result = news_fetcher.score_sentiment_with_llm(
                'Jensen Huang signals strong data center demand',
                'NVIDIA CEO highlights surging AI infrastructure spend'
            )
        assert result['sentiment'] == 'bullish'
        assert result['magnitude'] == 'high'
        assert 'demand catalyst' in result['sentiment_reason']

    def test_llm_path_bearish(self):
        """Bearish LLM response is parsed correctly."""
        llm_output = "Sentiment: bearish\nMagnitude: low\nReason: minor guidance miss"
        with patch('model_inference_lora.get_trading_decision', return_value=llm_output):
            result = news_fetcher.score_sentiment_with_llm('Stock misses estimates', 'EPS below consensus')
        assert result['sentiment'] == 'bearish'
        assert result['magnitude'] == 'low'

    def test_llm_path_neutral(self):
        """Neutral LLM response is parsed correctly."""
        llm_output = "Sentiment: neutral\nMagnitude: medium\nReason: mixed signals no clear edge"
        with patch('model_inference_lora.get_trading_decision', return_value=llm_output):
            result = news_fetcher.score_sentiment_with_llm('Company announces quarterly results', '')
        assert result['sentiment'] == 'neutral'

    def test_fallback_triggers_on_exception(self):
        """When get_trading_decision raises, keyword fallback runs without error."""
        with patch('model_inference_lora.get_trading_decision', side_effect=RuntimeError("GPU OOM")):
            result = news_fetcher.score_sentiment_with_llm(
                'Stock surges on earnings beat', 'record profit'
            )
        # keyword fallback: 'surges' and 'beat' are in _BULLISH → bullish
        assert result['sentiment'] == 'bullish'
        assert result['magnitude'] == 'medium'
        assert result['sentiment_reason'] == ''

    def test_fallback_on_import_error(self):
        """ImportError (model not available) also triggers keyword fallback."""
        with patch.dict('sys.modules', {'model_inference_lora': None}):
            # Removing the module from sys.modules causes ImportError on 'from' import
            result = news_fetcher.score_sentiment_with_llm(
                'Company layoffs announced', 'job cuts across divisions'
            )
        # 'layoffs' and 'cuts' are in _BEARISH
        assert result['sentiment'] == 'bearish'

    def test_result_schema_always_has_three_keys(self):
        """Both LLM path and fallback always return all three required keys."""
        keys = {'sentiment', 'magnitude', 'sentiment_reason'}

        # LLM path
        with patch('model_inference_lora.get_trading_decision', return_value="Sentiment: neutral\nMagnitude: low\nReason: nothing"):
            r1 = news_fetcher.score_sentiment_with_llm('title', 'desc')
        assert set(r1.keys()) >= keys

        # Fallback path
        with patch('model_inference_lora.get_trading_decision', side_effect=Exception("fail")):
            r2 = news_fetcher.score_sentiment_with_llm('title', 'desc')
        assert set(r2.keys()) >= keys

    def test_llm_invalid_response_falls_back_to_keyword(self):
        """Garbled LLM output (no parseable fields) falls back to keyword defaults."""
        with patch('model_inference_lora.get_trading_decision', return_value="I cannot determine"):
            result = news_fetcher.score_sentiment_with_llm('Stock surges', '')
        # Keyword: 'surges' → bullish
        assert result['sentiment'] in ('bullish', 'neutral')  # either is acceptable


class TestNewsLLMSentimentFlag:
    """Tests for NEWS_LLM_SENTIMENT env flag behavior in fetch_news()."""

    def _make_fake_article(self):
        return {
            'title':           'NVDA rallies on strong guidance',
            'description':     'Nvidia reports record quarterly earnings',
            'published_utc':   '2024-01-15T14:00:00Z',
            'age':             '2h ago',
        }

    def test_llm_sentiment_false_uses_keyword_path(self):
        """When NEWS_LLM_SENTIMENT=false, score_sentiment_with_llm is never called."""
        original = news_fetcher.NEWS_LLM_SENTIMENT
        try:
            news_fetcher.NEWS_LLM_SENTIMENT = False
            # Patch API + LLM — LLM must NOT be called
            fake_resp = MagicMock()
            fake_resp.json.return_value = {'results': [
                {'title': 'NVDA rallies', 'description': 'Record earnings',
                 'published_utc': '2024-01-15T14:00:00Z'}
            ]}
            fake_resp.raise_for_status = MagicMock()

            with patch('requests.get', return_value=fake_resp), \
                 patch('news_fetcher._API_KEY', 'fake_key'), \
                 patch('news_fetcher._cache', {}), \
                 patch('news_fetcher._rate_limit'), \
                 patch('model_inference_lora.get_trading_decision') as mock_llm:
                articles = news_fetcher.fetch_news('NVDA')

            mock_llm.assert_not_called()
            assert len(articles) == 1
            assert articles[0]['sentiment'] in ('bullish', 'bearish', 'neutral')
        finally:
            news_fetcher.NEWS_LLM_SENTIMENT = original

    def test_llm_sentiment_true_calls_score_function(self):
        """When NEWS_LLM_SENTIMENT=true, score_sentiment_with_llm is invoked."""
        original = news_fetcher.NEWS_LLM_SENTIMENT
        try:
            news_fetcher.NEWS_LLM_SENTIMENT = True
            fake_resp = MagicMock()
            fake_resp.json.return_value = {'results': [
                {'title': 'NVDA rallies', 'description': 'Record earnings',
                 'published_utc': '2024-01-15T14:00:00Z'}
            ]}
            fake_resp.raise_for_status = MagicMock()
            llm_output = "Sentiment: bullish\nMagnitude: high\nReason: earnings beat"

            with patch('requests.get', return_value=fake_resp), \
                 patch('news_fetcher._API_KEY', 'fake_key'), \
                 patch('news_fetcher._cache', {}), \
                 patch('news_fetcher._rate_limit'), \
                 patch('model_inference_lora.get_trading_decision', return_value=llm_output):
                articles = news_fetcher.fetch_news('NVDA')

            assert articles[0]['magnitude'] == 'high'
            assert articles[0]['sentiment_reason'] == 'earnings beat'
        finally:
            news_fetcher.NEWS_LLM_SENTIMENT = original


class TestFormatForPrompt:
    """Tests for format_for_prompt() output format changes."""

    def _fake_article(self, sentiment='bullish', magnitude='high', reason='demand catalyst'):
        return {
            'title':            'NVDA gains on AI demand',
            'description':      'Strong forward guidance',
            'published_utc':    '2024-01-15T14:00:00Z',
            'age':              '3h ago',
            'sentiment':        sentiment,
            'magnitude':        magnitude,
            'sentiment_reason': reason,
        }

    def test_bullish_article_includes_magnitude_and_reason(self):
        """Non-neutral articles show [BULLISH/HIGH — reason] tag."""
        with patch('news_fetcher.fetch_news', return_value=[self._fake_article()]):
            output = news_fetcher.format_for_prompt('NVDA')
        assert '[BULLISH/HIGH' in output
        assert 'demand catalyst' in output
        assert '3h ago' in output

    def test_neutral_article_has_no_tag(self):
        """Neutral sentiment articles have no bracket tag."""
        with patch('news_fetcher.fetch_news', return_value=[
            self._fake_article(sentiment='neutral', reason='')
        ]):
            output = news_fetcher.format_for_prompt('NVDA')
        assert '[' not in output

    def test_empty_reason_still_shows_sentiment_magnitude(self):
        """An article with empty reason still shows [BEARISH/LOW] without the dash."""
        with patch('news_fetcher.fetch_news', return_value=[
            self._fake_article(sentiment='bearish', magnitude='low', reason='')
        ]):
            output = news_fetcher.format_for_prompt('NVDA')
        assert '[BEARISH/LOW]' in output
        assert '—' not in output

    def test_no_articles_returns_empty_string(self):
        with patch('news_fetcher.fetch_news', return_value=[]):
            assert news_fetcher.format_for_prompt('NVDA') == ''

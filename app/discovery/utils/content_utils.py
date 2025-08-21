"""
Unified content processing utilities.
Consolidates duplicate text processing logic across multiple files.
"""

import re
import hashlib
from collections import Counter
from typing import List, Set, Dict, Optional


class ContentUtils:
    """Unified content processing utilities with performance optimizations."""
    
    # Pre-compiled regex patterns for better performance
    HTML_TAG_PATTERN = re.compile(r'<[^>]+>', re.IGNORECASE)
    WHITESPACE_PATTERN = re.compile(r'\s+')
    WORD_PATTERN = re.compile(r'\b[a-zA-Z]{3,}\b')
    SENTENCE_PATTERN = re.compile(r'[.!?]+')
    URL_PATTERN = re.compile(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')
    EMAIL_PATTERN = re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b')
    QUOTE_PATTERN = re.compile(r'[""]')
    APOSTROPHE_PATTERN = re.compile(r'['']')
    PUNCTUATION_CLEANUP = re.compile(r'([!?]){2,}')
    
    # Comprehensive stop words set for better keyword extraction
    STOP_WORDS = frozenset([
        'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with',
        'by', 'from', 'up', 'about', 'into', 'through', 'during', 'before',
        'after', 'above', 'below', 'out', 'off', 'over', 'under', 'again',
        'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why',
        'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other',
        'some', 'such', 'only', 'own', 'same', 'than', 'too', 'very', 'can',
        'will', 'just', 'should', 'now', 'said', 'also', 'get', 'make', 'go',
        'see', 'know', 'take', 'use', 'find', 'give', 'tell', 'ask', 'work',
        'seem', 'feel', 'try', 'leave', 'call', 'well', 'may', 'come', 'could',
        'would', 'should', 'might', 'must', 'shall', 'will', 'can', 'one', 'two',
        'first', 'last', 'next', 'new', 'old', 'good', 'bad', 'big', 'small'
    ])
    
    # Common spam indicators for content quality assessment
    SPAM_INDICATORS = frozenset([
        'click here', 'buy now', 'limited time', 'act now', 'free trial',
        'amazing deal', 'shocking', 'you won\'t believe', 'this one trick',
        'doctors hate', 'make money fast', 'work from home', 'get rich quick'
    ])
    
    @classmethod
    def clean_html(cls, html_content: str) -> str:
        """Remove HTML tags and normalize content with comprehensive cleaning."""
        if not html_content:
            return ""
        
        # Remove HTML tags
        clean = cls.HTML_TAG_PATTERN.sub('', html_content)
        
        # Remove URLs and email addresses
        clean = cls.URL_PATTERN.sub('', clean)
        clean = cls.EMAIL_PATTERN.sub('', clean)
        
        # Normalize quotes and apostrophes
        clean = cls.QUOTE_PATTERN.sub('"', clean)
        clean = cls.APOSTROPHE_PATTERN.sub("'", clean)
        
        # Clean up excessive punctuation
        clean = cls.PUNCTUATION_CLEANUP.sub(r'\1', clean)
        
        # Normalize whitespace
        clean = cls.WHITESPACE_PATTERN.sub(' ', clean)
        
        # Remove common advertising/spam text
        clean_lower = clean.lower()
        for spam_term in cls.SPAM_INDICATORS:
            clean_lower = clean_lower.replace(spam_term, '')
        
        return clean.strip()
    
    @classmethod
    def calculate_text_similarity(cls, text1: str, text2: str) -> float:
        """Calculate Jaccard similarity between two texts with optimizations."""
        if not text1 or not text2:
            return 0.0
        
        # Quick length-based pre-filter
        len_ratio = min(len(text1), len(text2)) / max(len(text1), len(text2))
        if len_ratio < 0.1:  # Very different lengths, likely not similar
            return 0.0
        
        # Extract words and filter stop words
        words1 = set(cls.WORD_PATTERN.findall(text1.lower())) - cls.STOP_WORDS
        words2 = set(cls.WORD_PATTERN.findall(text2.lower())) - cls.STOP_WORDS
        
        if not words1 or not words2:
            return 0.0
        
        # Calculate Jaccard similarity
        intersection = len(words1.intersection(words2))
        union = len(words1.union(words2))
        
        return intersection / union if union > 0 else 0.0
    
    @classmethod
    def extract_keywords(cls, text: str, max_keywords: int = 10) -> List[str]:
        """Extract keywords with improved frequency analysis and filtering."""
        if not text:
            return []
        
        # Extract words and filter
        words = cls.WORD_PATTERN.findall(text.lower())
        filtered_words = [
            word for word in words 
            if word not in cls.STOP_WORDS and len(word) > 3
        ]
        
        if not filtered_words:
            return []
        
        # Calculate word frequencies
        word_freq = Counter(filtered_words)
        
        # Filter out words that appear only once if we have many words
        if len(word_freq) > max_keywords * 2:
            word_freq = {word: count for word, count in word_freq.items() if count > 1}
        
        # Return most common keywords
        return [word for word, count in word_freq.most_common(max_keywords)]
    
    @classmethod
    def generate_content_hash(cls, content: str, url: str = "", title: str = "") -> str:
        """Generate consistent hash for content deduplication."""
        # Normalize content for hashing
        normalized = cls.clean_html(content)
        hash_input = f"{url}|{title}|{normalized[:1000]}"  # Limit to first 1000 chars
        return hashlib.md5(hash_input.encode('utf-8')).hexdigest()
    
    @classmethod
    def assess_content_quality(cls, content: str, title: str = "", 
                             min_length: int = 200, max_length: int = 50000) -> float:
        """Assess content quality with comprehensive scoring."""
        if not content:
            return 0.0
        
        score = 0.0
        content_lower = content.lower()
        
        # Length scoring
        content_length = len(content)
        if content_length < min_length:
            return 0.1  # Too short
        elif content_length > max_length:
            score += 0.2  # Long content, but cap the benefit
        else:
            # Optimal length range
            length_score = min(content_length / 1000, 0.3)  # Up to 0.3 for good length
            score += length_score
        
        # Title quality
        if title and len(title) > 10:
            score += 0.2
            # Bonus for descriptive titles (not clickbait)
            if not any(spam in title.lower() for spam in cls.SPAM_INDICATORS):
                score += 0.1
        
        # Content structure indicators
        sentences = len(cls.SENTENCE_PATTERN.findall(content))
        if sentences > 3:
            score += 0.2
        
        # Check for spam indicators
        spam_count = sum(1 for spam in cls.SPAM_INDICATORS if spam in content_lower)
        spam_penalty = min(spam_count * 0.1, 0.3)  # Max 0.3 penalty
        score -= spam_penalty
        
        # Word variety (vocabulary richness)
        words = cls.WORD_PATTERN.findall(content_lower)
        if words:
            unique_words = set(words) - cls.STOP_WORDS
            if len(words) > 0:
                vocabulary_richness = len(unique_words) / len(words)
                score += min(vocabulary_richness * 0.2, 0.2)
        
        return max(0.0, min(score, 1.0))  # Ensure score is between 0 and 1
    
    @classmethod
    def extract_sentences(cls, text: str, max_sentences: int = 5) -> List[str]:
        """Extract meaningful sentences from text for summarization."""
        if not text:
            return []
        
        sentences = cls.SENTENCE_PATTERN.split(text)
        
        # Filter and clean sentences
        meaningful_sentences = []
        for sentence in sentences:
            sentence = sentence.strip()
            if len(sentence) > 20 and not any(spam in sentence.lower() for spam in cls.SPAM_INDICATORS):
                meaningful_sentences.append(sentence)
        
        return meaningful_sentences[:max_sentences]
    
    @classmethod
    def normalize_url(cls, url: str) -> str:
        """Normalize URL for better deduplication."""
        if not url:
            return ""
        
        url = url.lower().strip()
        
        # Remove common tracking parameters
        tracking_params = [
            'utm_source', 'utm_medium', 'utm_campaign', 'utm_term', 'utm_content',
            'fbclid', 'gclid', 'msclkid', 'ref', 'source', 'medium', 'campaign'
        ]
        
        # Simple parameter removal (more robust than regex for this case)
        if '?' in url:
            base_url, params = url.split('?', 1)
            if params:
                param_pairs = params.split('&')
                clean_params = []
                for param in param_pairs:
                    if '=' in param:
                        key = param.split('=')[0]
                        if key not in tracking_params:
                            clean_params.append(param)
                
                if clean_params:
                    url = base_url + '?' + '&'.join(clean_params)
                else:
                    url = base_url
        
        # Remove trailing slash and fragments
        url = re.sub(r'[/#]$', '', url)
        url = re.sub(r'#.*$', '', url)
        
        return url
    
    @classmethod
    def calculate_readability_score(cls, text: str) -> float:
        """Calculate readability score (simplified Flesch reading ease)."""
        if not text:
            return 0.0
        
        # Count sentences, words, and estimate syllables
        sentences = len(cls.SENTENCE_PATTERN.findall(text))
        if sentences == 0:
            sentences = 1
        
        words = cls.WORD_PATTERN.findall(text)
        word_count = len(words)
        
        if word_count == 0:
            return 0.0
        
        # Estimate syllables (approximation using vowel clusters)
        syllable_count = 0
        for word in words:
            syllables = max(1, len(re.findall(r'[aeiouAEIOU]+', word)))
            syllable_count += syllables
        
        # Simplified Flesch formula
        avg_sentence_length = word_count / sentences
        avg_syllables_per_word = syllable_count / word_count
        
        score = 206.835 - (1.015 * avg_sentence_length) - (84.6 * avg_syllables_per_word)
        
        # Normalize to 0-1 scale (Flesch scores range from 0-100+)
        return max(0.0, min(1.0, score / 100))


# Pre-compiled patterns for performance (module-level for singleton behavior)
_content_utils = ContentUtils()
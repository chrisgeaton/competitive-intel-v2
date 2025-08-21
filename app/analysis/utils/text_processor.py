"""
Text processing utilities for Analysis Service.

Centralized text processing, cleaning, and extraction functions.
"""

import re
from typing import Dict, List, Optional, Tuple
from collections import Counter


class ContentExtractor:
    """Extract and analyze content from various text sources."""
    
    # Pre-compiled regex patterns for performance
    PATTERNS = {
        "email": re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', re.IGNORECASE),
        "url": re.compile(r'https?://(?:[-\w.])+(?:[:\d]+)?(?:/(?:[\w/_.])*(?:\?(?:[\w&=%.])*)?(?:#(?:\w*))?)?', re.IGNORECASE),
        "phone": re.compile(r'\b(?:\+?1[-.\s]?)?\(?([0-9]{3})\)?[-.\s]?([0-9]{3})[-.\s]?([0-9]{4})\b'),
        "currency": re.compile(r'\$[\d,]+(?:\.\d{2})?|\b\d+(?:\.\d{2})?\s*(?:USD|dollars?)\b', re.IGNORECASE),
        "percentage": re.compile(r'\b\d+(?:\.\d+)?%\b'),
        "date": re.compile(r'\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{1,2},?\s+\d{4}\b', re.IGNORECASE),
        "whitespace": re.compile(r'\s+'),
        "word_boundary": re.compile(r'\b'),
        "sentence_end": re.compile(r'[.!?]+\s+')
    }
    
    @classmethod
    def extract_entities(cls, text: str) -> Dict[str, List[str]]:
        """Extract entities from text using regex patterns."""
        if not text:
            return {}
            
        entities = {}
        for entity_type, pattern in cls.PATTERNS.items():
            if entity_type in ["whitespace", "word_boundary", "sentence_end"]:
                continue  # Skip utility patterns
                
            matches = pattern.findall(text)
            if matches:
                entities[entity_type] = list(set(matches))  # Remove duplicates
                
        return entities
    
    @classmethod
    def extract_keywords(cls, text: str, min_length: int = 3, max_keywords: int = 20) -> List[str]:
        """Extract keywords from text using frequency analysis."""
        if not text:
            return []
            
        # Clean and tokenize text
        cleaned_text = re.sub(r'[^\w\s]', ' ', text.lower())
        words = cleaned_text.split()
        
        # Filter words by length and remove common stop words
        stop_words = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
            'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'have',
            'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should',
            'this', 'that', 'these', 'those', 'there', 'here', 'where', 'when',
            'how', 'what', 'who', 'why', 'which', 'can', 'may', 'might', 'must'
        }
        
        filtered_words = [
            word for word in words 
            if len(word) >= min_length and word not in stop_words
        ]
        
        # Count word frequencies and return top keywords
        word_counts = Counter(filtered_words)
        return [word for word, count in word_counts.most_common(max_keywords)]
    
    @classmethod
    def extract_sentences(cls, text: str, max_length: int = 200) -> List[str]:
        """Extract sentences from text with length filtering."""
        if not text:
            return []
            
        # Split by sentence patterns
        sentences = cls.PATTERNS["sentence_end"].split(text.strip())
        
        # Filter and clean sentences
        cleaned_sentences = []
        for sentence in sentences:
            sentence = sentence.strip()
            if sentence and len(sentence) <= max_length and len(sentence) > 10:
                cleaned_sentences.append(sentence)
                
        return cleaned_sentences


class TextProcessor:
    """Main text processing class with optimization features."""
    
    def __init__(self):
        self.extractor = ContentExtractor()
        self._cache = {}  # Simple content cache
        self._cache_size_limit = 1000
        
    def preprocess_text(self, text: str, normalize: bool = True) -> str:
        """Preprocess text for analysis with caching."""
        if not text:
            return ""
            
        # Check cache first
        cache_key = hash(text)
        if cache_key in self._cache:
            return self._cache[cache_key]
            
        # Process text
        if text is None:
            processed = ""
        else:
            processed = str(text).strip()
            
            if normalize:
                processed = processed.lower()
                # Normalize whitespace
                processed = ContentExtractor.PATTERNS["whitespace"].sub(' ', processed)
                
        # Cache result (with size limit)
        if len(self._cache) < self._cache_size_limit:
            self._cache[cache_key] = processed
            
        return processed
    
    def extract_content_features(self, content: Dict[str, any]) -> Dict[str, any]:
        """Extract comprehensive features from content."""
        title = self.preprocess_text(content.get("title", ""))
        text = self.preprocess_text(content.get("content_text", ""))
        full_text = f"{title} {text}"
        
        features = {
            "title": title,
            "text": text,
            "full_text": full_text,
            "title_length": len(title),
            "text_length": len(text),
            "total_length": len(full_text),
            "word_count": len(full_text.split()) if full_text else 0
        }
        
        # Extract entities and keywords
        features["entities"] = self.extractor.extract_entities(full_text)
        features["keywords"] = self.extractor.extract_keywords(full_text)
        features["sentences"] = self.extractor.extract_sentences(text)
        
        return features
    
    def find_keyword_matches(
        self,
        text: str,
        keywords: List[str],
        match_type: str = "partial"
    ) -> List[Tuple[str, str, int]]:
        """Find keyword matches in text with position information."""
        matches = []
        if not text or not keywords:
            return matches
            
        processed_text = self.preprocess_text(text, normalize=True)
        
        for keyword in keywords:
            if not keyword:
                continue
                
            processed_keyword = self.preprocess_text(keyword, normalize=True)
            
            if match_type == "exact":
                # Word boundary matching
                pattern = r'\b' + re.escape(processed_keyword) + r'\b'
                for match in re.finditer(pattern, processed_text):
                    matches.append((keyword, match.group(), match.start()))
                    
            elif match_type == "partial":
                # Simple substring matching
                pos = processed_text.find(processed_keyword)
                if pos >= 0:
                    matches.append((keyword, processed_keyword, pos))
                    
            elif match_type == "fuzzy":
                # Simple fuzzy matching (can be enhanced)
                if self._fuzzy_match(processed_keyword, processed_text, threshold=0.8):
                    matches.append((keyword, processed_keyword, 0))
                    
        return matches
    
    def _fuzzy_match(self, keyword: str, text: str, threshold: float = 0.8) -> bool:
        """Simple fuzzy matching implementation."""
        if not keyword or not text:
            return False
            
        # Simple character-based similarity
        keyword_chars = set(keyword.lower())
        text_chars = set(text.lower())
        
        if len(keyword_chars) == 0:
            return False
            
        overlap = len(keyword_chars & text_chars)
        similarity = overlap / len(keyword_chars)
        
        return similarity >= threshold
    
    def extract_context_around_match(
        self,
        text: str,
        match_position: int,
        match_length: int,
        context_size: int = 50
    ) -> str:
        """Extract context around a match position."""
        if not text or match_position < 0:
            return ""
            
        start = max(0, match_position - context_size)
        end = min(len(text), match_position + match_length + context_size)
        
        return text[start:end].strip()
    
    def calculate_text_similarity(self, text1: str, text2: str) -> float:
        """Calculate similarity between two texts."""
        if not text1 or not text2:
            return 0.0
            
        # Simple word-based Jaccard similarity
        words1 = set(self.preprocess_text(text1).split())
        words2 = set(self.preprocess_text(text2).split())
        
        if not words1 and not words2:
            return 1.0
        if not words1 or not words2:
            return 0.0
            
        intersection = len(words1 & words2)
        union = len(words1 | words2)
        
        return intersection / union if union > 0 else 0.0
    
    def clear_cache(self):
        """Clear the text processing cache."""
        self._cache.clear()
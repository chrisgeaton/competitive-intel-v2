"""
Advanced content extraction, normalization, and processing for discovered content.
Handles text cleaning, entity extraction, sentiment analysis, and content enrichment.
"""

import re
import asyncio
import logging
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple, Set
from dataclasses import dataclass, field
from collections import Counter
import hashlib

from .engines.base_engine import DiscoveredItem, ContentExtractor


@dataclass
class ProcessedContent:
    """Processed and enriched content."""
    original_item: DiscoveredItem
    cleaned_text: str
    extracted_entities: List[Dict[str, Any]]
    key_phrases: List[str]
    sentiment_score: float
    sentiment_label: str
    readability_score: float
    language: str
    content_category: str
    topics: List[str]
    summary: str
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ContentAnalytics:
    """Analytics for content processing."""
    word_count: int
    sentence_count: int
    paragraph_count: int
    avg_sentence_length: float
    reading_time_minutes: int
    complexity_score: float
    keyword_density: Dict[str, float]
    content_structure: Dict[str, Any]


class EntityExtractor:
    """Extract entities from content using pattern matching and heuristics."""
    
    def __init__(self):
        # Pattern definitions for different entity types
        self.patterns = {
            'companies': [
                r'\b([A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+)*)\s+(?:Inc|Corp|Corporation|Ltd|LLC|Company|Co)\b',
                r'\b([A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+)*)\s+(?:Inc\.|Corp\.|Ltd\.|LLC\.)\b',
            ],
            'people': [
                r'\b(?:CEO|CTO|CFO|President|Director|Manager)\s+([A-Z][a-z]+\s+[A-Z][a-z]+)\b',
                r'\b([A-Z][a-z]+\s+[A-Z][a-z]+),?\s+(?:said|announced|stated|reported|told)\b',
                r'\baccording\s+to\s+([A-Z][a-z]+\s+[A-Z][a-z]+)\b',
            ],
            'locations': [
                r'\bin\s+([A-Z][a-zA-Z\s]+?)(?:,|\.|$|\s+said|\s+announced)',
                r'\b([A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+)?),?\s+(?:headquarters|office|facility|plant)\b',
            ],
            'technologies': [
                r'\b(artificial\s+intelligence|machine\s+learning|blockchain|cryptocurrency|IoT|5G|cloud\s+computing)\b',
                r'\b(Python|JavaScript|Java|React|Node\.js|AWS|Azure|Google\s+Cloud)\b',
            ],
            'financial': [
                r'\$([0-9,]+(?:\.[0-9]{2})?)\s*(?:million|billion|trillion)?\b',
                r'\b([0-9,]+(?:\.[0-9]{2})?)\s*(?:million|billion|trillion)\s+(?:dollars?|USD)\b',
                r'\bstock\s+price|share\s+price|market\s+cap|revenue|profit|loss\b',
            ]
        }
        
        # Common stop words for entity filtering
        self.stop_words = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
            'of', 'with', 'by', 'from', 'up', 'about', 'into', 'through', 'during',
            'before', 'after', 'above', 'below', 'out', 'off', 'over', 'under'
        }
    
    def extract_entities(self, text: str) -> List[Dict[str, Any]]:
        """Extract entities from text using pattern matching."""
        entities = []
        
        for entity_type, patterns in self.patterns.items():
            type_entities = set()
            
            for pattern in patterns:
                matches = re.findall(pattern, text, re.IGNORECASE)
                for match in matches:
                    if isinstance(match, tuple):
                        entity_text = ' '.join(match)
                    else:
                        entity_text = match
                    
                    # Clean and validate entity
                    cleaned_entity = self._clean_entity(entity_text)
                    if self._is_valid_entity(cleaned_entity, entity_type):
                        type_entities.add(cleaned_entity)
            
            # Add entities with confidence scores
            for entity in type_entities:
                entities.append({
                    'text': entity,
                    'type': entity_type,
                    'confidence': self._calculate_entity_confidence(entity, entity_type, text)
                })
        
        return entities
    
    def _clean_entity(self, entity_text: str) -> str:
        """Clean extracted entity text."""
        # Remove extra whitespace
        cleaned = re.sub(r'\s+', ' ', entity_text.strip())
        
        # Remove common prefixes/suffixes that shouldn't be part of entities
        cleaned = re.sub(r'^(?:the|a|an)\s+', '', cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r'\s+(?:the|a|an)$', '', cleaned, flags=re.IGNORECASE)
        
        return cleaned
    
    def _is_valid_entity(self, entity_text: str, entity_type: str) -> bool:
        """Validate extracted entity."""
        if len(entity_text) < 2:
            return False
        
        # Filter out common stop words
        words = entity_text.lower().split()
        if all(word in self.stop_words for word in words):
            return False
        
        # Type-specific validation
        if entity_type == 'people':
            # Should have at least first and last name
            if len(words) < 2:
                return False
            # Should not contain common non-name words
            non_name_words = {'said', 'announced', 'reported', 'told', 'according'}
            if any(word in non_name_words for word in words):
                return False
        
        elif entity_type == 'companies':
            # Should not be too generic
            generic_terms = {'company', 'corporation', 'business', 'firm'}
            if entity_text.lower() in generic_terms:
                return False
        
        return True
    
    def _calculate_entity_confidence(self, entity: str, entity_type: str, 
                                   full_text: str) -> float:
        """Calculate confidence score for extracted entity."""
        confidence = 0.5  # Base confidence
        
        # Frequency boost
        frequency = full_text.lower().count(entity.lower())
        confidence += min(frequency * 0.1, 0.3)
        
        # Context boost
        if entity_type == 'companies':
            context_indicators = ['announced', 'reported', 'said', 'CEO', 'headquarters']
            for indicator in context_indicators:
                if indicator.lower() in full_text.lower():
                    confidence += 0.1
        
        elif entity_type == 'people':
            context_indicators = ['CEO', 'president', 'director', 'said', 'announced']
            for indicator in context_indicators:
                if indicator.lower() in full_text.lower():
                    confidence += 0.1
        
        return min(confidence, 1.0)


class SentimentAnalyzer:
    """Simple sentiment analysis using lexicon-based approach."""
    
    def __init__(self):
        # Simple sentiment lexicons
        self.positive_words = {
            'good', 'great', 'excellent', 'amazing', 'outstanding', 'positive',
            'success', 'win', 'growth', 'improve', 'increase', 'profit', 'gain',
            'up', 'rise', 'boost', 'advance', 'progress', 'achievement', 'breakthrough'
        }
        
        self.negative_words = {
            'bad', 'terrible', 'awful', 'poor', 'negative', 'failure', 'lose',
            'decline', 'decrease', 'loss', 'drop', 'fall', 'down', 'crash',
            'crisis', 'problem', 'issue', 'concern', 'worry', 'risk', 'threat'
        }
    
    def analyze_sentiment(self, text: str) -> Tuple[float, str]:
        """Analyze sentiment of text and return score and label."""
        words = re.findall(r'\b[a-zA-Z]+\b', text.lower())
        
        positive_count = sum(1 for word in words if word in self.positive_words)
        negative_count = sum(1 for word in words if word in self.negative_words)
        
        total_sentiment_words = positive_count + negative_count
        
        if total_sentiment_words == 0:
            return 0.0, 'neutral'
        
        score = (positive_count - negative_count) / len(words)
        
        if score > 0.02:
            label = 'positive'
        elif score < -0.02:
            label = 'negative'
        else:
            label = 'neutral'
        
        # Normalize score to -1 to 1 range
        normalized_score = max(-1.0, min(1.0, score * 10))
        
        return normalized_score, label


class TopicExtractor:
    """Extract topics from content using keyword clustering."""
    
    def __init__(self):
        self.topic_keywords = {
            'technology': [
                'software', 'hardware', 'computer', 'digital', 'internet',
                'AI', 'machine learning', 'blockchain', 'cloud', 'data'
            ],
            'finance': [
                'money', 'investment', 'stock', 'market', 'economy',
                'banking', 'financial', 'revenue', 'profit', 'budget'
            ],
            'business': [
                'company', 'business', 'corporate', 'enterprise', 'startup',
                'strategy', 'management', 'leadership', 'growth', 'expansion'
            ],
            'healthcare': [
                'health', 'medical', 'hospital', 'doctor', 'patient',
                'medicine', 'treatment', 'disease', 'therapy', 'clinical'
            ],
            'energy': [
                'energy', 'oil', 'gas', 'renewable', 'solar', 'wind',
                'electric', 'battery', 'power', 'electricity'
            ],
            'research': [
                'research', 'study', 'analysis', 'experiment', 'data',
                'findings', 'results', 'methodology', 'conclusion', 'hypothesis'
            ]
        }
    
    def extract_topics(self, text: str, limit: int = 5) -> List[str]:
        """Extract topics from text based on keyword matching."""
        text_lower = text.lower()
        topic_scores = {}
        
        for topic, keywords in self.topic_keywords.items():
            score = 0
            for keyword in keywords:
                count = text_lower.count(keyword)
                score += count
            
            if score > 0:
                topic_scores[topic] = score
        
        # Sort by score and return top topics
        sorted_topics = sorted(topic_scores.items(), key=lambda x: x[1], reverse=True)
        return [topic for topic, score in sorted_topics[:limit]]


class ContentProcessor:
    """Advanced content processor with multiple analysis capabilities."""
    
    def __init__(self):
        self.entity_extractor = EntityExtractor()
        self.sentiment_analyzer = SentimentAnalyzer()
        self.topic_extractor = TopicExtractor()
        self.logger = logging.getLogger("discovery.content_processor")
        
        # Processing cache to avoid reprocessing identical content
        self.processing_cache: Dict[str, ProcessedContent] = {}
        self.max_cache_size = 1000
    
    async def process_content(self, item: DiscoveredItem) -> ProcessedContent:
        """Process and enrich discovered content."""
        # Check cache first
        content_hash = self._generate_content_hash(item)
        if content_hash in self.processing_cache:
            cached_result = self.processing_cache[content_hash]
            cached_result.original_item = item  # Update with current item
            return cached_result
        
        try:
            # Clean text
            cleaned_text = await self._clean_content(item.content)
            
            # Extract entities
            entities = self.entity_extractor.extract_entities(cleaned_text)
            
            # Extract key phrases
            key_phrases = await self._extract_key_phrases(cleaned_text)
            
            # Analyze sentiment
            sentiment_score, sentiment_label = self.sentiment_analyzer.analyze_sentiment(cleaned_text)
            
            # Calculate readability
            readability_score = self._calculate_readability(cleaned_text)
            
            # Detect language (simple heuristic)
            language = self._detect_language(cleaned_text)
            
            # Categorize content
            content_category = await self._categorize_content(item, cleaned_text)
            
            # Extract topics
            topics = self.topic_extractor.extract_topics(cleaned_text)
            
            # Generate summary
            summary = await self._generate_summary(cleaned_text, item.title)
            
            # Create analytics
            analytics = self._generate_analytics(cleaned_text)
            
            processed = ProcessedContent(
                original_item=item,
                cleaned_text=cleaned_text,
                extracted_entities=entities,
                key_phrases=key_phrases,
                sentiment_score=sentiment_score,
                sentiment_label=sentiment_label,
                readability_score=readability_score,
                language=language,
                content_category=content_category,
                topics=topics,
                summary=summary,
                metadata={
                    'processing_time': datetime.now().isoformat(),
                    'content_hash': content_hash,
                    'analytics': analytics.__dict__
                }
            )
            
            # Cache result
            self._cache_processed_content(content_hash, processed)
            
            return processed
            
        except Exception as e:
            self.logger.error(f"Content processing failed for {item.url}: {e}")
            
            # Return minimal processed content on error
            return ProcessedContent(
                original_item=item,
                cleaned_text=item.content,
                extracted_entities=[],
                key_phrases=[],
                sentiment_score=0.0,
                sentiment_label='neutral',
                readability_score=0.5,
                language='en',
                content_category='unknown',
                topics=[],
                summary=item.description or item.title,
                metadata={'processing_error': str(e)}
            )
    
    async def _clean_content(self, content: str) -> str:
        """Clean and normalize content text."""
        if not content:
            return ""
        
        # Remove HTML tags if any remain
        cleaned = ContentExtractor.clean_html(content)
        
        # Normalize whitespace
        cleaned = re.sub(r'\s+', ' ', cleaned)
        
        # Remove excessive punctuation
        cleaned = re.sub(r'[!]{2,}', '!', cleaned)
        cleaned = re.sub(r'[?]{2,}', '?', cleaned)
        
        # Remove URLs
        cleaned = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', cleaned)
        
        # Remove email addresses
        cleaned = re.sub(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', '', cleaned)
        
        # Normalize quotes
        cleaned = re.sub(r'[""]', '"', cleaned)
        cleaned = re.sub(r'['']', "'", cleaned)
        
        return cleaned.strip()
    
    async def _extract_key_phrases(self, text: str, limit: int = 10) -> List[str]:
        """Extract key phrases from text."""
        # Simple n-gram extraction
        words = re.findall(r'\b[a-zA-Z]{3,}\b', text.lower())
        
        # Remove stop words
        stop_words = {
            'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with',
            'by', 'from', 'up', 'about', 'into', 'through', 'during', 'before',
            'after', 'above', 'below', 'out', 'off', 'over', 'under', 'again',
            'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why',
            'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other',
            'some', 'such', 'only', 'own', 'same', 'than', 'too', 'very', 'can',
            'will', 'just', 'should', 'now', 'said', 'also', 'get', 'make', 'go',
            'see', 'know', 'take', 'use', 'find', 'give', 'tell', 'ask', 'work',
            'seem', 'feel', 'try', 'leave', 'call'
        }
        
        filtered_words = [word for word in words if word not in stop_words and len(word) > 3]
        
        # Generate bigrams and trigrams
        phrases = []
        
        # Bigrams
        for i in range(len(filtered_words) - 1):
            bigram = f"{filtered_words[i]} {filtered_words[i + 1]}"
            phrases.append(bigram)
        
        # Trigrams
        for i in range(len(filtered_words) - 2):
            trigram = f"{filtered_words[i]} {filtered_words[i + 1]} {filtered_words[i + 2]}"
            phrases.append(trigram)
        
        # Count frequencies
        phrase_counts = Counter(phrases)
        
        # Return most common phrases
        return [phrase for phrase, count in phrase_counts.most_common(limit)]
    
    def _calculate_readability(self, text: str) -> float:
        """Calculate readability score (simplified Flesch reading ease)."""
        if not text:
            return 0.0
        
        # Count sentences
        sentence_count = len(re.findall(r'[.!?]+', text))
        if sentence_count == 0:
            sentence_count = 1
        
        # Count words
        words = re.findall(r'\b\w+\b', text)
        word_count = len(words)
        
        if word_count == 0:
            return 0.0
        
        # Count syllables (approximation)
        syllable_count = 0
        for word in words:
            syllables = max(1, len(re.findall(r'[aeiouAEIOU]', word)))
            syllable_count += syllables
        
        # Simplified Flesch formula
        avg_sentence_length = word_count / sentence_count
        avg_syllables_per_word = syllable_count / word_count
        
        score = 206.835 - (1.015 * avg_sentence_length) - (84.6 * avg_syllables_per_word)
        
        # Normalize to 0-1 scale
        return max(0.0, min(1.0, score / 100))
    
    def _detect_language(self, text: str) -> str:
        """Detect language (simple heuristic)."""
        # Very simple language detection based on common words
        english_indicators = ['the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for']
        
        text_lower = text.lower()
        english_count = sum(1 for indicator in english_indicators if indicator in text_lower)
        
        if english_count >= 3:
            return 'en'
        else:
            return 'unknown'
    
    async def _categorize_content(self, item: DiscoveredItem, text: str) -> str:
        """Categorize content based on source and content analysis."""
        # Start with source type
        if item.content_type:
            base_category = item.content_type.value
        else:
            base_category = 'unknown'
        
        # Refine based on content analysis
        text_lower = text.lower()
        url_lower = item.url.lower() if item.url else ""
        
        # News indicators
        if any(indicator in text_lower for indicator in ['breaking', 'reported', 'announced', 'according to']):
            return 'news'
        
        # Research indicators
        if any(indicator in text_lower for indicator in ['study', 'research', 'analysis', 'methodology', 'findings']):
            return 'research'
        
        # Blog indicators
        if any(indicator in url_lower for indicator in ['blog', 'post']) or \
           any(indicator in text_lower for indicator in ['i think', 'in my opinion', 'personally']):
            return 'blog'
        
        # Press release indicators
        if any(indicator in text_lower for indicator in ['press release', 'announces', 'company news']):
            return 'press_release'
        
        return base_category
    
    async def _generate_summary(self, text: str, title: str) -> str:
        """Generate a summary of the content."""
        if not text:
            return title or "No content available"
        
        # Simple extractive summarization
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip() and len(s.strip()) > 20]
        
        if not sentences:
            return title or "No content available"
        
        # Take first sentence and a representative middle sentence
        if len(sentences) == 1:
            return sentences[0]
        elif len(sentences) == 2:
            return f"{sentences[0]}. {sentences[1]}"
        else:
            # First sentence + middle sentence
            middle_idx = len(sentences) // 2
            return f"{sentences[0]}. {sentences[middle_idx]}"
    
    def _generate_analytics(self, text: str) -> ContentAnalytics:
        """Generate content analytics."""
        if not text:
            return ContentAnalytics(0, 0, 0, 0.0, 0, 0.0, {}, {})
        
        # Basic counts
        words = re.findall(r'\b\w+\b', text)
        word_count = len(words)
        
        sentences = re.split(r'[.!?]+', text)
        sentence_count = len([s for s in sentences if s.strip()])
        
        paragraphs = text.split('\n\n')
        paragraph_count = len([p for p in paragraphs if p.strip()])
        
        # Average sentence length
        avg_sentence_length = word_count / max(sentence_count, 1)
        
        # Reading time (average 200 words per minute)
        reading_time_minutes = max(1, word_count // 200)
        
        # Complexity score (based on average word length and sentence length)
        avg_word_length = sum(len(word) for word in words) / max(word_count, 1)
        complexity_score = min(1.0, (avg_word_length * avg_sentence_length) / 100)
        
        # Keyword density
        word_freq = Counter(word.lower() for word in words if len(word) > 3)
        total_words = sum(word_freq.values())
        keyword_density = {
            word: count / total_words
            for word, count in word_freq.most_common(10)
        }
        
        # Content structure
        content_structure = {
            'has_headings': bool(re.search(r'^#{1,6}\s', text, re.MULTILINE)),
            'has_lists': bool(re.search(r'^\s*[-*+]\s', text, re.MULTILINE)),
            'has_quotes': bool(re.search(r'^\s*>', text, re.MULTILINE)),
            'has_code': bool(re.search(r'```|`[^`]+`', text)),
        }
        
        return ContentAnalytics(
            word_count=word_count,
            sentence_count=sentence_count,
            paragraph_count=paragraph_count,
            avg_sentence_length=avg_sentence_length,
            reading_time_minutes=reading_time_minutes,
            complexity_score=complexity_score,
            keyword_density=keyword_density,
            content_structure=content_structure
        )
    
    def _generate_content_hash(self, item: DiscoveredItem) -> str:
        """Generate hash for content caching."""
        content_to_hash = f"{item.url}{item.title}{item.content[:500]}"
        return hashlib.md5(content_to_hash.encode()).hexdigest()
    
    def _cache_processed_content(self, content_hash: str, processed: ProcessedContent):
        """Cache processed content with size limit."""
        if len(self.processing_cache) >= self.max_cache_size:
            # Remove oldest entries (simple FIFO)
            keys_to_remove = list(self.processing_cache.keys())[:100]
            for key in keys_to_remove:
                del self.processing_cache[key]
        
        self.processing_cache[content_hash] = processed
    
    async def batch_process_content(self, items: List[DiscoveredItem]) -> List[ProcessedContent]:
        """Process multiple content items efficiently."""
        processed_items = []
        
        # Process in batches to avoid memory issues
        batch_size = 10
        for i in range(0, len(items), batch_size):
            batch = items[i:i + batch_size]
            
            # Process batch concurrently
            tasks = [self.process_content(item) for item in batch]
            batch_results = await asyncio.gather(*tasks, return_exceptions=True)
            
            for result in batch_results:
                if isinstance(result, ProcessedContent):
                    processed_items.append(result)
                else:
                    self.logger.error(f"Batch processing error: {result}")
        
        return processed_items
    
    def get_processing_stats(self) -> Dict[str, Any]:
        """Get processing statistics."""
        return {
            'cache_size': len(self.processing_cache),
            'max_cache_size': self.max_cache_size,
            'cache_hit_rate': self._estimate_cache_hit_rate()
        }
    
    def _estimate_cache_hit_rate(self) -> float:
        """Estimate cache hit rate (placeholder)."""
        # This would need proper tracking in a real implementation
        return 0.3  # Placeholder value
    
    def clear_cache(self):
        """Clear processing cache."""
        self.processing_cache.clear()
        self.logger.info("Cleared content processing cache")
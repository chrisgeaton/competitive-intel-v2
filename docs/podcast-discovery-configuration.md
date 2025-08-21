# Podcast Discovery Engine Configuration

The Podcast Discovery Engine integrates with PodcastIndex.org API to discover podcasts and episodes based on user focus areas and tracked entities. This document provides configuration examples and integration guidance.

## PodcastIndex.org API Setup

### 1. Get API Credentials
Visit [PodcastIndex.org](https://podcastindex.org) to get your free API credentials:
- **API Key**: Your unique API key
- **API Secret**: Your API secret for authentication

### 2. Basic Configuration

```python
# Example configuration in your discovery service config
discovery_config = {
    "podcast_index": {
        "enabled": True,
        "api_key": "your_podcastindex_api_key",
        "api_secret": "your_podcastindex_api_secret",
        
        # Optional configuration parameters
        "max_episodes_per_show": 5,         # Max episodes per podcast show
        "max_shows_per_query": 10,          # Max shows per search query
        "recency_threshold_days": 30,       # Only include recent episodes
        "min_episode_duration": 300,        # 5 minutes minimum
        "max_episode_duration": 7200,       # 2 hours maximum
        "language_filter": ["en"],          # Language preferences
        "exclude_explicit": False           # Filter explicit content
    }
}
```

### 3. Environment Variables (Recommended)

```bash
# Set in your environment
export PODCASTINDEX_API_KEY="your_api_key_here"
export PODCASTINDEX_API_SECRET="your_api_secret_here"
```

```python
import os

discovery_config = {
    "podcast_index": {
        "enabled": True,
        "api_key": os.getenv("PODCASTINDEX_API_KEY"),
        "api_secret": os.getenv("PODCASTINDEX_API_SECRET"),
        "recency_threshold_days": 30,
        "language_filter": ["en"]
    }
}
```

## Integration Examples

### 1. Direct Engine Usage

```python
from app.discovery.engines import create_podcast_engine

# Create podcast engine instance
podcast_engine = create_podcast_engine(
    api_key="your_api_key",
    api_secret="your_api_secret",
    max_episodes_per_show=5,
    recency_threshold_days=30
)

# Discover content
user_context = {
    'focus_areas': ['artificial intelligence', 'machine learning'],
    'tracked_entities': ['OpenAI', 'Google', 'Microsoft'],
    'language': 'en'
}

keywords = ['AI', 'deep learning', 'neural networks']
discovered_items = await podcast_engine.discover_content(
    keywords=keywords,
    limit=20,
    user_context=user_context
)

# Process results
for item in discovered_items:
    print(f"Podcast: {item.metadata['podcast_title']}")
    print(f"Episode: {item.title}")
    print(f"Duration: {item.metadata['duration_formatted']}")
    print(f"Relevance Score: {item.relevance_score:.2f}")
    print("---")
```

### 2. Source Manager Integration

```python
from app.discovery.engines import SourceManager

# Full discovery service configuration
config = {
    "max_concurrent_engines": 3,
    "default_timeout": 30,
    
    # News APIs
    "news_apis": {
        "newsapi_key": "your_newsapi_key"
    },
    
    # RSS Monitor
    "rss_monitor": {
        "enabled": True,
        "max_feeds": 100
    },
    
    # Podcast Discovery
    "podcast_index": {
        "enabled": True,
        "api_key": "your_podcastindex_api_key",
        "api_secret": "your_podcastindex_api_secret",
        "recency_threshold_days": 30,
        "max_episodes_per_show": 5
    },
    
    # Web Scraper (optional)
    "web_scraper": {
        "enabled": False
    }
}

# Initialize source manager with all engines
source_manager = SourceManager(config)

# Discover content from all sources including podcasts
results = await source_manager.discover_content(
    keywords=['AI', 'machine learning'],
    focus_areas=['artificial intelligence'],
    entities=['OpenAI', 'Google'],
    limit=50
)
```

### 3. Pipeline Integration

```python
from app.discovery.pipeline import DailyDiscoveryPipeline

# Configure daily pipeline with podcast discovery
pipeline_config = {
    'batch_size': 50,
    'max_concurrent_users': 20,
    'content_limit_per_user': 100,
    'quality_threshold': 0.5,
    
    # Source manager configuration including podcasts
    'source_manager': {
        "podcast_index": {
            "enabled": True,
            "api_key": os.getenv("PODCASTINDEX_API_KEY"),
            "api_secret": os.getenv("PODCASTINDEX_API_SECRET"),
            "recency_threshold_days": 14,  # More recent for daily pipeline
            "max_episodes_per_show": 3
        }
    }
}

# Run daily discovery including podcast content
pipeline = DailyDiscoveryPipeline(pipeline_config)
metrics = await pipeline.run_daily_discovery()
```

## Configuration Parameters

### Required Parameters
- `api_key`: PodcastIndex.org API key
- `api_secret`: PodcastIndex.org API secret

### Optional Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `enabled` | `True` | Enable/disable podcast discovery |
| `max_episodes_per_show` | `5` | Maximum episodes per podcast show |
| `max_shows_per_query` | `10` | Maximum shows per search query |
| `recency_threshold_days` | `30` | Only include episodes from last N days |
| `min_episode_duration` | `300` | Minimum episode duration (seconds) |
| `max_episode_duration` | `7200` | Maximum episode duration (seconds) |
| `language_filter` | `["en"]` | Preferred languages |
| `exclude_explicit` | `False` | Filter out explicit content |

## Discovered Content Structure

### Podcast Episode Metadata

Each discovered podcast episode includes rich metadata:

```python
{
    'podcast_id': 123456,
    'podcast_title': 'AI Insights Podcast',
    'podcast_author': 'John Smith',
    'podcast_owner': 'Tech Media Inc',
    'episode_id': 789012,
    'episode_number': 42,
    'season_number': 2,
    'episode_type': 'full',
    'duration_seconds': 1800,
    'duration_formatted': '30m',
    'enclosure_url': 'https://example.com/episode.mp3',
    'enclosure_type': 'audio/mpeg',
    'enclosure_length': 15728640,
    'podcast_image': 'https://example.com/podcast.jpg',
    'episode_image': 'https://example.com/episode.jpg',
    'podcast_categories': {'Technology': 'Tech', 'Business': 'Business'},
    'language': 'en',
    'explicit': false,
    'transcript_url': 'https://example.com/transcript.txt',
    'chapters_url': 'https://example.com/chapters.json',
    'search_term': 'machine learning',
    'discovery_source': 'podcastindex',
    'ready_for_transcription': true
}
```

### Content Types and Source Types

- **Content Type**: `ContentType.PODCAST`
- **Source Type**: `SourceType.PODCAST`
- **Database Source Type**: `'podcast'`

## Database Integration

### DiscoveredSource Table

Podcast sources are stored with:
```python
{
    'source_type': 'podcast',
    'source_url': 'https://feeds.example.com/podcast.xml',
    'source_name': 'AI Insights Podcast',
    'source_description': 'Weekly insights on artificial intelligence'
}
```

### DiscoveredContent Table

Podcast episodes are stored as:
```python
{
    'content_type': 'podcast',
    'title': 'Episode 42: The Future of Machine Learning',
    'content_url': 'https://example.com/episode.mp3',
    'content_text': 'Formatted episode content with metadata',
    'author': 'John Smith',
    'published_at': datetime(2023, 12, 1),
    # ML scores calculated by content processor
    'relevance_score': 0.85,
    'credibility_score': 0.92,
    'overall_score': 0.88
}
```

## ML Scoring and Quality Assessment

### Relevance Scoring (40% of total score)
- Keyword matching in title and description
- User focus area alignment
- Tracked entity mentions

### Context Scoring (30% of total score)
- User strategic profile matching
- Company/technology mentions
- Industry relevance

### Recency Bonus (15% of total score)
- Recent episodes (last 7 days): +15%
- Recent episodes (last 30 days): +10%
- Older episodes (last 90 days): +5%

### Duration Bonus (15% of total score)
- Ideal length (10-60 minutes): +15%
- Good length (5-10 min or 1-1.5 hours): +10%
- Any valid duration: +5%

### Quality Assessment
- Title quality (length and informativeness)
- Description quality and completeness
- Metadata completeness (author, categories, images)
- Transcription readiness

## Future Transcription Workflow

The engine prepares for selective transcription:

```python
# Check if episode is ready for transcription
if item.metadata.get('ready_for_transcription'):
    transcript_url = item.metadata.get('transcript_url')
    audio_url = item.metadata.get('enclosure_url')
    
    if transcript_url:
        # Direct transcript available
        transcript = await fetch_transcript(transcript_url)
    elif audio_url:
        # Audio file available for transcription
        transcript = await transcribe_audio(audio_url)
```

## Performance and Rate Limiting

### API Limits
- PodcastIndex.org: Generous rate limits (60 requests/minute)
- Automatic rate limiting with 1-second minimum between requests
- Intelligent backoff for rate limit exceeded scenarios

### Caching
- Podcast metadata cached for 1 hour
- Episode data cached for 30 minutes
- Efficient deduplication to avoid redundant API calls

### Error Handling
- Comprehensive exception handling for API failures
- Automatic retry with exponential backoff
- Graceful degradation when API is unavailable

## Monitoring and Analytics

### Engine Metrics
```python
status = await podcast_engine.get_engine_status()
# Returns:
{
    "engine_name": "PodcastIndex",
    "source_type": "podcast",
    "is_available": true,
    "requests_made": 1250,
    "success_rate": 0.98,
    "avg_response_time": 0.45,
    "configuration": {...},
    "cache_stats": {...}
}
```

### Performance Tracking
- Request success rates
- Average response times
- Cache hit rates
- Content discovery rates
- User engagement metrics

## Security Considerations

### API Key Management
- Store API credentials in environment variables
- Never commit API keys to version control
- Use different keys for development/staging/production
- Regularly rotate API credentials

### Content Filtering
- Optional explicit content filtering
- Language-based content filtering
- Duration-based quality filtering
- Relevance threshold enforcement

## Troubleshooting

### Common Issues

**Authentication Errors**
```
Error: PodcastIndex search failed: 401
```
- Verify API key and secret are correct
- Check API key hasn't expired
- Ensure proper hash generation in authentication headers

**No Results Found**
- Check if search terms are too specific
- Verify language filter settings
- Adjust recency threshold
- Check if explicit content filtering is too restrictive

**Rate Limiting**
```
Error: PodcastIndex search failed: 429
```
- Automatic backoff will handle this
- Consider reducing concurrent requests
- Check rate limiting configuration

### Debug Mode

```python
config = {
    "podcast_index": {
        "enabled": True,
        "api_key": "your_key",
        "api_secret": "your_secret",
        "debug_mode": True  # Enable detailed logging
    }
}
```

## Example Use Cases

### 1. Technology Company Intelligence
```python
user_context = {
    'focus_areas': ['artificial intelligence', 'cloud computing', 'cybersecurity'],
    'tracked_entities': ['Amazon', 'Microsoft', 'Google', 'IBM'],
    'language': 'en'
}

keywords = ['AWS', 'Azure', 'machine learning', 'data security']
```

### 2. Investment Research
```python
user_context = {
    'focus_areas': ['fintech', 'blockchain', 'startup funding'],
    'tracked_entities': ['Stripe', 'Coinbase', 'Robinhood', 'Square'],
    'language': 'en'
}

keywords = ['venture capital', 'IPO', 'cryptocurrency', 'payments']
```

### 3. Industry Analysis
```python
user_context = {
    'focus_areas': ['renewable energy', 'electric vehicles', 'sustainability'],
    'tracked_entities': ['Tesla', 'BYD', 'CATL', 'Rivian'],
    'language': 'en'
}

keywords = ['battery technology', 'solar power', 'EV charging', 'carbon neutral']
```

This configuration enables comprehensive podcast discovery that integrates seamlessly with the existing Discovery Service pipeline while providing rich metadata for future transcription and analysis workflows.
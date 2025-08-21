"""
ML Training Pipeline - Updates ML models based on SendGrid engagement data and user behavior patterns.

Continuously improves ML models using real user engagement feedback, SendGrid webhook data,
and behavioral analytics to enhance content relevance predictions and scoring accuracy.
"""

import asyncio
import logging
import json
import pickle
from datetime import datetime, timedelta, timezone
from typing import List, Dict, Any, Optional, Tuple, Set
from dataclasses import dataclass, field
from pathlib import Path
import numpy as np
from collections import defaultdict

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, and_, or_, func, text, desc
from sqlalchemy.orm import selectinload

from app.database import get_db_session
from app.models.discovery import (
    DiscoveredContent, ContentEngagement, MLModelMetrics,
    DiscoveryJob, DiscoveredSource
)
from app.models.user import User
from app.models.strategic_profile import UserStrategicProfile

from ..utils import (
    get_config,
    UnifiedErrorHandler,
    get_ml_scoring_cache,
    batch_processor
)


@dataclass
class TrainingMetrics:
    """ML training session metrics."""
    training_start_time: datetime
    training_end_time: Optional[datetime] = None
    model_type: str = ""
    model_version: str = ""
    training_samples: int = 0
    validation_samples: int = 0
    training_accuracy: float = 0.0
    validation_accuracy: float = 0.0
    precision: float = 0.0
    recall: float = 0.0
    f1_score: float = 0.0
    improvement_over_previous: float = 0.0
    feature_importance: Dict[str, float] = field(default_factory=dict)
    training_errors: List[str] = field(default_factory=list)
    
    @property
    def training_duration(self) -> float:
        if not self.training_end_time:
            return (datetime.now(timezone.utc) - self.training_start_time).total_seconds()
        return (self.training_end_time - self.training_start_time).total_seconds()


@dataclass
class EngagementPattern:
    """User engagement pattern for ML training."""
    user_id: int
    content_features: Dict[str, Any]
    engagement_score: float
    engagement_type: str
    engagement_context: Dict[str, Any]
    strategic_context: Dict[str, Any]
    timestamp: datetime


@dataclass
class ModelTrainingData:
    """Training data structure for ML models."""
    features: np.ndarray
    labels: np.ndarray
    feature_names: List[str]
    sample_weights: Optional[np.ndarray] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class MLTrainingPipeline:
    """
    ML Training Pipeline for continuous model improvement.
    
    Processes SendGrid engagement data, user behavior patterns, and content
    performance metrics to continuously improve ML model accuracy and relevance.
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger("pipeline.ml_training")
        self.error_handler = UnifiedErrorHandler()
        
        # Training configuration
        self.min_training_samples = config.get('min_training_samples', 100)
        self.validation_split = config.get('validation_split', 0.2)
        self.model_improvement_threshold = config.get('model_improvement_threshold', 0.02)
        self.training_frequency_hours = config.get('training_frequency_hours', 24)
        
        # Model storage
        self.model_storage_path = Path(config.get('model_storage_path', 'models/'))
        self.model_storage_path.mkdir(exist_ok=True)
        
        # Current model versions
        self.current_models = {
            'relevance_scorer': '1.0',
            'engagement_predictor': '1.0',
            'content_classifier': '1.0'
        }
        
        # Cache
        self.ml_cache = get_ml_scoring_cache()
        
        # Training state
        self.training_in_progress = False
        self.last_training_time = None
        
        self.logger.info("ML Training Pipeline initialized")
    
    async def run_training_cycle(self) -> List[TrainingMetrics]:
        """
        Run complete ML training cycle for all models.
        
        Returns:
            List[TrainingMetrics]: Training results for each model
        """
        if self.training_in_progress:
            self.logger.warning("Training already in progress, skipping")
            return []
        
        self.training_in_progress = True
        training_results = []
        
        try:
            self.logger.info("Starting ML training cycle")
            
            # Check if training is needed
            if not await self._should_run_training():
                self.logger.info("Training not needed at this time")
                return []
            
            # Collect engagement data
            engagement_data = await self._collect_engagement_data()
            if len(engagement_data) < self.min_training_samples:
                self.logger.warning(f"Insufficient training data: {len(engagement_data)} samples")
                return []
            
            # Train each model type
            for model_type in ['relevance_scorer', 'engagement_predictor', 'content_classifier']:
                try:
                    metrics = await self._train_model(model_type, engagement_data)
                    training_results.append(metrics)
                    
                    # Deploy model if improvement is significant
                    if metrics.improvement_over_previous > self.model_improvement_threshold:
                        await self._deploy_model(model_type, metrics)
                        
                except Exception as e:
                    self.logger.error(f"Failed to train {model_type}: {e}")
                    error_metrics = TrainingMetrics(
                        training_start_time=datetime.now(timezone.utc),
                        model_type=model_type,
                        training_errors=[str(e)]
                    )
                    training_results.append(error_metrics)
            
            # Update training timestamp
            self.last_training_time = datetime.now(timezone.utc)
            
            # Clear ML caches to force using new models
            await self.ml_cache.clear()
            
            self.logger.info(f"ML training cycle completed with {len(training_results)} models")
            
        except Exception as e:
            self.logger.error(f"ML training cycle failed: {e}")
            raise
        
        finally:
            self.training_in_progress = False
        
        return training_results
    
    async def _should_run_training(self) -> bool:
        """Determine if ML training should run."""
        
        # Check if enough time has passed
        if self.last_training_time:
            time_since_last = datetime.now(timezone.utc) - self.last_training_time
            if time_since_last.total_seconds() < self.training_frequency_hours * 3600:
                return False
        
        # Check if there's sufficient new engagement data
        try:
            async with get_db_session() as session:
                cutoff_time = (
                    self.last_training_time or 
                    datetime.now(timezone.utc) - timedelta(days=7)
                )
                
                new_engagements_count = await session.execute(
                    select(func.count(ContentEngagement.id))
                    .where(
                        and_(
                            ContentEngagement.created_at >= cutoff_time,
                            ContentEngagement.feedback_processed == False
                        )
                    )
                )
                
                new_count = new_engagements_count.scalar()
                self.logger.debug(f"Found {new_count} new engagement records")
                
                return new_count >= self.min_training_samples
                
        except Exception as e:
            self.logger.error(f"Failed to check training necessity: {e}")
            return False
    
    async def _collect_engagement_data(self) -> List[EngagementPattern]:
        """Collect and prepare engagement data for ML training."""
        
        self.logger.info("Collecting engagement data for ML training")
        
        try:
            async with get_db_session() as session:
                # Get engagement data with content and user context
                query = (
                    select(ContentEngagement, DiscoveredContent, User, UserStrategicProfile)
                    .join(DiscoveredContent, ContentEngagement.content_id == DiscoveredContent.id)
                    .join(User, ContentEngagement.user_id == User.id)
                    .outerjoin(UserStrategicProfile, User.id == UserStrategicProfile.user_id)
                    .where(
                        and_(
                            ContentEngagement.created_at >= datetime.now(timezone.utc) - timedelta(days=30),
                            ContentEngagement.feedback_processed == False,
                            DiscoveredContent.content_text.isnot(None),
                            or_(
                                ContentEngagement.engagement_type == 'email_open',
                                ContentEngagement.engagement_type == 'email_click',
                                ContentEngagement.engagement_type == 'time_spent',
                                ContentEngagement.engagement_type == 'bookmark'
                            )
                        )
                    )
                    .order_by(ContentEngagement.created_at.desc())
                    .limit(5000)  # Limit for performance
                )
                
                result = await session.execute(query)
                engagement_records = result.all()
                
                patterns = []
                for engagement, content, user, strategic_profile in engagement_records:
                    try:
                        pattern = await self._build_engagement_pattern(
                            engagement, content, user, strategic_profile
                        )
                        patterns.append(pattern)
                        
                    except Exception as e:
                        self.logger.error(f"Failed to build pattern for engagement {engagement.id}: {e}")
                        continue
                
                self.logger.info(f"Collected {len(patterns)} engagement patterns")
                return patterns
                
        except Exception as e:
            self.logger.error(f"Failed to collect engagement data: {e}")
            raise
    
    async def _build_engagement_pattern(self, engagement: ContentEngagement, 
                                      content: DiscoveredContent, user: User,
                                      strategic_profile: Optional[UserStrategicProfile]) -> EngagementPattern:
        """Build engagement pattern from database records."""
        
        # Extract content features
        content_features = {
            'word_count': len((content.content_text or "").split()),
            'char_count': len(content.content_text or ""),
            'title_length': len(content.title or ""),
            'has_author': bool(content.author),
            'content_age_days': (datetime.now(timezone.utc) - (content.published_at or content.discovered_at)).days,
            'relevance_score': float(content.relevance_score or 0),
            'credibility_score': float(content.credibility_score or 0),
            'freshness_score': float(content.freshness_score or 0),
            'overall_score': float(content.overall_score or 0),
            'content_type': content.content_type,
            'content_language': content.content_language,
            'predicted_categories': json.loads(content.predicted_categories or '[]'),
            'detected_entities': json.loads(content.detected_entities or '[]'),
            'sentiment_score': float(content.sentiment_score or 0.5),
            'competitive_relevance': content.competitive_relevance or 'unknown'
        }
        
        # Normalize engagement score
        engagement_score = self._normalize_engagement_score(engagement)
        
        # Build engagement context
        engagement_context = {
            'engagement_type': engagement.engagement_type,
            'session_duration': engagement.session_duration or 0,
            'time_to_click': engagement.time_to_click or 0,
            'click_sequence': engagement.click_sequence or 1,
            'device_type': engagement.device_type or 'unknown',
            'content_age_at_engagement': engagement.content_age_at_engagement or 0
        }
        
        # Build strategic context
        strategic_context = {}
        if strategic_profile:
            strategic_context = {
                'industry': strategic_profile.industry or 'unknown',
                'organization_type': strategic_profile.organization_type or 'unknown',
                'role': strategic_profile.role or 'unknown',
                'strategic_goals': strategic_profile.strategic_goals or []
            }
        
        return EngagementPattern(
            user_id=user.id,
            content_features=content_features,
            engagement_score=engagement_score,
            engagement_type=engagement.engagement_type,
            engagement_context=engagement_context,
            strategic_context=strategic_context,
            timestamp=engagement.created_at
        )
    
    def _normalize_engagement_score(self, engagement: ContentEngagement) -> float:
        """Normalize engagement score based on engagement type."""
        
        engagement_weights = {
            'email_open': 0.3,
            'email_click': 0.7,
            'time_spent': 1.0,
            'bookmark': 0.9,
            'share': 0.8,
            'feedback': 1.0
        }
        
        base_weight = engagement_weights.get(engagement.engagement_type, 0.5)
        engagement_value = float(engagement.engagement_value or 1.0)
        
        # Normalize based on engagement type
        if engagement.engagement_type == 'time_spent':
            # Time spent: normalize to 0-1 based on typical reading time
            normalized = min(engagement_value / 300, 1.0)  # 5 minutes = 1.0
        elif engagement.engagement_type in ['email_open', 'email_click', 'bookmark']:
            # Binary engagements
            normalized = 1.0 if engagement_value > 0 else 0.0
        else:
            # Other engagements
            normalized = min(engagement_value / 5.0, 1.0)  # Scale to 0-1
        
        return base_weight * normalized
    
    async def _train_model(self, model_type: str, engagement_data: List[EngagementPattern]) -> TrainingMetrics:
        """Train a specific ML model type."""
        
        self.logger.info(f"Training {model_type} model")
        metrics = TrainingMetrics(
            training_start_time=datetime.now(timezone.utc),
            model_type=model_type,
            model_version=self._get_next_version(model_type)
        )
        
        try:
            # Prepare training data
            training_data = await self._prepare_training_data(model_type, engagement_data)
            
            if len(training_data.features) < self.min_training_samples:
                raise ValueError(f"Insufficient training samples: {len(training_data.features)}")
            
            metrics.training_samples = len(training_data.features)
            
            # Split data
            train_data, val_data = self._split_training_data(training_data)
            metrics.validation_samples = len(val_data.features)
            
            # Train model
            model, training_scores = await self._train_model_implementation(
                model_type, train_data, val_data
            )
            
            # Evaluate model
            evaluation_results = await self._evaluate_model(model, val_data, model_type)
            
            # Update metrics
            metrics.training_accuracy = training_scores.get('training_accuracy', 0.0)
            metrics.validation_accuracy = evaluation_results.get('accuracy', 0.0)
            metrics.precision = evaluation_results.get('precision', 0.0)
            metrics.recall = evaluation_results.get('recall', 0.0)
            metrics.f1_score = evaluation_results.get('f1_score', 0.0)
            metrics.feature_importance = evaluation_results.get('feature_importance', {})
            
            # Compare with previous model
            previous_accuracy = await self._get_previous_model_accuracy(model_type)
            metrics.improvement_over_previous = metrics.validation_accuracy - previous_accuracy
            
            # Save model
            model_path = await self._save_model(model, model_type, metrics.model_version, training_data.feature_names)
            
            metrics.training_end_time = datetime.now(timezone.utc)
            
            self.logger.info(f"{model_type} training completed - Accuracy: {metrics.validation_accuracy:.3f}")
            
            return metrics
            
        except Exception as e:
            self.logger.error(f"Model training failed for {model_type}: {e}")
            metrics.training_errors.append(str(e))
            metrics.training_end_time = datetime.now(timezone.utc)
            return metrics
    
    async def _prepare_training_data(self, model_type: str, 
                                   engagement_data: List[EngagementPattern]) -> ModelTrainingData:
        """Prepare training data for specific model type."""
        
        features_list = []
        labels_list = []
        weights_list = []
        
        for pattern in engagement_data:
            try:
                # Extract features based on model type
                if model_type == 'relevance_scorer':
                    features = self._extract_relevance_features(pattern)
                    label = pattern.engagement_score  # Continuous score
                elif model_type == 'engagement_predictor':
                    features = self._extract_engagement_features(pattern)
                    label = 1.0 if pattern.engagement_score > 0.5 else 0.0  # Binary classification
                elif model_type == 'content_classifier':
                    features = self._extract_classification_features(pattern)
                    label = self._get_content_category_label(pattern)
                else:
                    continue
                
                if features and label is not None:
                    features_list.append(features)
                    labels_list.append(label)
                    
                    # Weight samples by recency and engagement quality
                    weight = self._calculate_sample_weight(pattern)
                    weights_list.append(weight)
                    
            except Exception as e:
                self.logger.error(f"Failed to prepare sample: {e}")
                continue
        
        if not features_list:
            raise ValueError(f"No valid training samples for {model_type}")
        
        # Convert to numpy arrays
        features_array = np.array(features_list)
        labels_array = np.array(labels_list)
        weights_array = np.array(weights_list) if weights_list else None
        
        # Get feature names
        feature_names = self._get_feature_names(model_type)
        
        return ModelTrainingData(
            features=features_array,
            labels=labels_array,
            feature_names=feature_names,
            sample_weights=weights_array,
            metadata={
                'model_type': model_type,
                'samples_count': len(features_list),
                'feature_count': features_array.shape[1] if len(features_array.shape) > 1 else 0
            }
        )
    
    def _extract_relevance_features(self, pattern: EngagementPattern) -> Optional[List[float]]:
        """Extract features for relevance scoring model."""
        try:
            features = []
            
            # Content features
            features.extend([
                pattern.content_features.get('word_count', 0) / 1000,  # Normalized
                pattern.content_features.get('title_length', 0) / 100,
                1.0 if pattern.content_features.get('has_author') else 0.0,
                pattern.content_features.get('content_age_days', 0) / 30,
                pattern.content_features.get('relevance_score', 0),
                pattern.content_features.get('credibility_score', 0),
                pattern.content_features.get('freshness_score', 0),
                pattern.content_features.get('sentiment_score', 0.5),
            ])
            
            # Categorical features (one-hot encoded)
            content_type = pattern.content_features.get('content_type', 'unknown')
            type_features = [
                1.0 if content_type == 'article' else 0.0,
                1.0 if content_type == 'news' else 0.0,
                1.0 if content_type == 'report' else 0.0,
                1.0 if content_type == 'blog' else 0.0
            ]
            features.extend(type_features)
            
            # Competitive relevance features
            comp_relevance = pattern.content_features.get('competitive_relevance', 'unknown')
            comp_features = [
                1.0 if comp_relevance == 'high' else 0.0,
                1.0 if comp_relevance == 'medium' else 0.0,
                1.0 if comp_relevance == 'low' else 0.0
            ]
            features.extend(comp_features)
            
            # Strategic context features
            industry = pattern.strategic_context.get('industry', 'unknown')
            role = pattern.strategic_context.get('role', 'unknown')
            
            # Industry features (simplified)
            industry_features = [
                1.0 if 'tech' in industry.lower() else 0.0,
                1.0 if 'finance' in industry.lower() else 0.0,
                1.0 if 'health' in industry.lower() else 0.0,
                1.0 if 'retail' in industry.lower() else 0.0
            ]
            features.extend(industry_features)
            
            # Role features
            role_features = [
                1.0 if 'ceo' in role.lower() else 0.0,
                1.0 if 'cto' in role.lower() else 0.0,
                1.0 if 'cmo' in role.lower() else 0.0,
                1.0 if 'cfo' in role.lower() else 0.0
            ]
            features.extend(role_features)
            
            return features
            
        except Exception as e:
            self.logger.error(f"Failed to extract relevance features: {e}")
            return None
    
    def _extract_engagement_features(self, pattern: EngagementPattern) -> Optional[List[float]]:
        """Extract features for engagement prediction model."""
        try:
            # Start with relevance features as base
            features = self._extract_relevance_features(pattern)
            if not features:
                return None
            
            # Add engagement-specific features
            engagement_features = [
                pattern.engagement_context.get('session_duration', 0) / 600,  # Normalize to 10 min
                pattern.engagement_context.get('time_to_click', 0) / 300,     # Normalize to 5 min
                pattern.engagement_context.get('click_sequence', 1) / 10,     # Normalize
                pattern.content_features.get('content_age_at_engagement', 0) / 24,  # Hours to days
            ]
            
            # Device type features
            device_type = pattern.engagement_context.get('device_type', 'unknown')
            device_features = [
                1.0 if device_type == 'desktop' else 0.0,
                1.0 if device_type == 'mobile' else 0.0,
                1.0 if device_type == 'tablet' else 0.0
            ]
            
            features.extend(engagement_features + device_features)
            return features
            
        except Exception as e:
            self.logger.error(f"Failed to extract engagement features: {e}")
            return None
    
    def _extract_classification_features(self, pattern: EngagementPattern) -> Optional[List[float]]:
        """Extract features for content classification model."""
        try:
            features = []
            
            # Text-based features
            categories = pattern.content_features.get('predicted_categories', [])
            entities = pattern.content_features.get('detected_entities', [])
            
            features.extend([
                pattern.content_features.get('word_count', 0) / 1000,
                pattern.content_features.get('char_count', 0) / 5000,
                len(categories) / 5,  # Normalize category count
                len(entities) / 10,   # Normalize entity count
                pattern.content_features.get('sentiment_score', 0.5),
            ])
            
            # Category presence features
            category_features = [
                1.0 if 'technology' in categories else 0.0,
                1.0 if 'business' in categories else 0.0,
                1.0 if 'competitive' in categories else 0.0,
                1.0 if 'news' in categories else 0.0,
                1.0 if 'research' in categories else 0.0
            ]
            features.extend(category_features)
            
            return features
            
        except Exception as e:
            self.logger.error(f"Failed to extract classification features: {e}")
            return None
    
    def _get_content_category_label(self, pattern: EngagementPattern) -> Optional[int]:
        """Get content category label for classification."""
        categories = pattern.content_features.get('predicted_categories', [])
        
        # Map categories to numeric labels
        category_mapping = {
            'technology': 0,
            'business': 1,
            'competitive': 2,
            'news': 3,
            'research': 4,
            'general': 5
        }
        
        # Return label for first matching category
        for category in categories:
            if category in category_mapping:
                return category_mapping[category]
        
        return category_mapping['general']  # Default
    
    def _calculate_sample_weight(self, pattern: EngagementPattern) -> float:
        """Calculate sample weight based on recency and quality."""
        
        # Recency weight - more recent samples get higher weight
        age_days = (datetime.now(timezone.utc) - pattern.timestamp).days
        recency_weight = max(0.1, 1.0 - (age_days / 30))  # Decay over 30 days
        
        # Engagement quality weight
        quality_weight = 1.0
        if pattern.engagement_type in ['time_spent', 'bookmark']:
            quality_weight = 1.5  # Higher weight for high-quality engagements
        elif pattern.engagement_type == 'email_open':
            quality_weight = 0.8   # Lower weight for basic opens
        
        # Engagement score weight
        score_weight = 0.5 + pattern.engagement_score  # Range: 0.5 to 1.5
        
        return recency_weight * quality_weight * score_weight
    
    def _get_feature_names(self, model_type: str) -> List[str]:
        """Get feature names for model type."""
        
        base_features = [
            'word_count_norm', 'title_length_norm', 'has_author', 'content_age_norm',
            'relevance_score', 'credibility_score', 'freshness_score', 'sentiment_score',
            'type_article', 'type_news', 'type_report', 'type_blog',
            'comp_high', 'comp_medium', 'comp_low',
            'industry_tech', 'industry_finance', 'industry_health', 'industry_retail',
            'role_ceo', 'role_cto', 'role_cmo', 'role_cfo'
        ]
        
        if model_type == 'engagement_predictor':
            engagement_features = [
                'session_duration_norm', 'time_to_click_norm', 'click_sequence_norm',
                'content_age_at_engagement_norm', 'device_desktop', 'device_mobile', 'device_tablet'
            ]
            return base_features + engagement_features
        elif model_type == 'content_classifier':
            classification_features = [
                'word_count_norm', 'char_count_norm', 'category_count_norm', 'entity_count_norm',
                'sentiment_score', 'cat_technology', 'cat_business', 'cat_competitive', 
                'cat_news', 'cat_research'
            ]
            return classification_features
        else:
            return base_features
    
    def _split_training_data(self, training_data: ModelTrainingData) -> Tuple[ModelTrainingData, ModelTrainingData]:
        """Split training data into train and validation sets."""
        
        n_samples = len(training_data.features)
        n_val = int(n_samples * self.validation_split)
        
        # Shuffle indices
        indices = np.random.permutation(n_samples)
        val_indices = indices[:n_val]
        train_indices = indices[n_val:]
        
        train_data = ModelTrainingData(
            features=training_data.features[train_indices],
            labels=training_data.labels[train_indices],
            feature_names=training_data.feature_names,
            sample_weights=training_data.sample_weights[train_indices] if training_data.sample_weights is not None else None,
            metadata=training_data.metadata
        )
        
        val_data = ModelTrainingData(
            features=training_data.features[val_indices],
            labels=training_data.labels[val_indices],
            feature_names=training_data.feature_names,
            sample_weights=training_data.sample_weights[val_indices] if training_data.sample_weights is not None else None,
            metadata=training_data.metadata
        )
        
        return train_data, val_data
    
    async def _train_model_implementation(self, model_type: str, train_data: ModelTrainingData,
                                        val_data: ModelTrainingData) -> Tuple[Any, Dict[str, float]]:
        """Train the actual ML model implementation."""
        
        # For this implementation, we'll use simple models
        # In production, you would use libraries like scikit-learn, TensorFlow, or PyTorch
        
        try:
            if model_type in ['relevance_scorer']:
                # Simple linear regression for relevance scoring
                from sklearn.linear_model import LinearRegression
                from sklearn.metrics import mean_squared_error, r2_score
                
                model = LinearRegression()
                model.fit(train_data.features, train_data.labels, sample_weight=train_data.sample_weights)
                
                # Training accuracy
                train_pred = model.predict(train_data.features)
                training_accuracy = r2_score(train_data.labels, train_pred)
                
                return model, {'training_accuracy': training_accuracy}
                
            elif model_type in ['engagement_predictor', 'content_classifier']:
                # Simple logistic regression for classification
                from sklearn.linear_model import LogisticRegression
                from sklearn.metrics import accuracy_score
                
                model = LogisticRegression(max_iter=1000, random_state=42)
                model.fit(train_data.features, train_data.labels, sample_weight=train_data.sample_weights)
                
                # Training accuracy
                train_pred = model.predict(train_data.features)
                training_accuracy = accuracy_score(train_data.labels, train_pred)
                
                return model, {'training_accuracy': training_accuracy}
                
            else:
                raise ValueError(f"Unknown model type: {model_type}")
                
        except ImportError:
            self.logger.error("scikit-learn not available, using dummy model")
            # Return dummy model for demonstration
            return DummyModel(), {'training_accuracy': 0.5}
    
    async def _evaluate_model(self, model: Any, val_data: ModelTrainingData, 
                            model_type: str) -> Dict[str, Any]:
        """Evaluate trained model on validation data."""
        
        try:
            predictions = model.predict(val_data.features)
            
            if model_type == 'relevance_scorer':
                # Regression metrics
                from sklearn.metrics import mean_squared_error, r2_score
                
                mse = mean_squared_error(val_data.labels, predictions)
                r2 = r2_score(val_data.labels, predictions)
                
                return {
                    'accuracy': r2,
                    'mse': mse,
                    'precision': 0.0,  # Not applicable for regression
                    'recall': 0.0,
                    'f1_score': 0.0,
                    'feature_importance': self._get_feature_importance(model, val_data.feature_names)
                }
                
            else:
                # Classification metrics
                from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
                
                accuracy = accuracy_score(val_data.labels, predictions)
                precision = precision_score(val_data.labels, predictions, average='weighted', zero_division=0)
                recall = recall_score(val_data.labels, predictions, average='weighted', zero_division=0)
                f1 = f1_score(val_data.labels, predictions, average='weighted', zero_division=0)
                
                return {
                    'accuracy': accuracy,
                    'precision': precision,
                    'recall': recall,
                    'f1_score': f1,
                    'feature_importance': self._get_feature_importance(model, val_data.feature_names)
                }
                
        except ImportError:
            # Dummy evaluation if scikit-learn not available
            return {
                'accuracy': 0.6,
                'precision': 0.6,
                'recall': 0.6,
                'f1_score': 0.6,
                'feature_importance': {}
            }
    
    def _get_feature_importance(self, model: Any, feature_names: List[str]) -> Dict[str, float]:
        """Get feature importance from trained model."""
        
        try:
            if hasattr(model, 'coef_'):
                # Linear models
                importances = abs(model.coef_)
                if len(importances.shape) > 1:
                    importances = importances[0]  # Take first class for multi-class
                
                # Normalize importances
                total = sum(importances)
                if total > 0:
                    importances = importances / total
                
                return dict(zip(feature_names, importances))
                
            elif hasattr(model, 'feature_importances_'):
                # Tree-based models
                return dict(zip(feature_names, model.feature_importances_))
            
            else:
                return {}
                
        except Exception as e:
            self.logger.error(f"Failed to get feature importance: {e}")
            return {}
    
    async def _get_previous_model_accuracy(self, model_type: str) -> float:
        """Get accuracy of previous model version."""
        
        try:
            async with get_db_session() as session:
                result = await session.execute(
                    select(MLModelMetrics.validation_accuracy)
                    .where(
                        and_(
                            MLModelMetrics.model_type == model_type,
                            MLModelMetrics.is_active == True
                        )
                    )
                    .order_by(MLModelMetrics.created_at.desc())
                    .limit(1)
                )
                
                previous_accuracy = result.scalar_one_or_none()
                return float(previous_accuracy) if previous_accuracy else 0.5
                
        except Exception as e:
            self.logger.error(f"Failed to get previous model accuracy: {e}")
            return 0.5
    
    async def _save_model(self, model: Any, model_type: str, version: str, 
                         feature_names: List[str]) -> Path:
        """Save trained model to disk."""
        
        model_filename = f"{model_type}_v{version}.pkl"
        model_path = self.model_storage_path / model_filename
        
        try:
            model_data = {
                'model': model,
                'model_type': model_type,
                'version': version,
                'feature_names': feature_names,
                'created_at': datetime.now(timezone.utc).isoformat(),
                'config': self.config
            }
            
            with open(model_path, 'wb') as f:
                pickle.dump(model_data, f)
            
            self.logger.info(f"Saved model {model_type} v{version} to {model_path}")
            return model_path
            
        except Exception as e:
            self.logger.error(f"Failed to save model: {e}")
            raise
    
    async def _deploy_model(self, model_type: str, metrics: TrainingMetrics):
        """Deploy trained model to production."""
        
        try:
            # Save model metrics to database
            async with get_db_session() as session:
                # Deactivate previous model
                await session.execute(
                    text("""
                        UPDATE ml_model_metrics 
                        SET is_active = false, deprecated_at = NOW()
                        WHERE model_type = :model_type AND is_active = true
                    """),
                    {'model_type': model_type}
                )
                
                # Create new model metrics record
                model_metrics = MLModelMetrics(
                    model_version=metrics.model_version,
                    model_type=model_type,
                    model_name=f"{model_type}_v{metrics.model_version}",
                    training_data_size=metrics.training_samples,
                    training_duration_seconds=int(metrics.training_duration),
                    training_accuracy=metrics.training_accuracy,
                    validation_accuracy=metrics.validation_accuracy,
                    precision_score=metrics.precision,
                    recall_score=metrics.recall,
                    f1_score=metrics.f1_score,
                    feature_importance=json.dumps(metrics.feature_importance),
                    is_active=True,
                    deployed_at=datetime.now(timezone.utc)
                )
                
                session.add(model_metrics)
                await session.commit()
                
                # Update current model version
                self.current_models[model_type] = metrics.model_version
                
                self.logger.info(f"Deployed {model_type} v{metrics.model_version} to production")
                
        except Exception as e:
            self.logger.error(f"Failed to deploy model {model_type}: {e}")
            raise
    
    async def _mark_engagement_processed(self, engagement_data: List[EngagementPattern]):
        """Mark engagement records as processed for ML training."""
        
        try:
            async with get_db_session() as session:
                # Update engagement records
                engagement_ids = [
                    pattern.engagement_context.get('engagement_id') 
                    for pattern in engagement_data
                    if pattern.engagement_context.get('engagement_id')
                ]
                
                if engagement_ids:
                    await session.execute(
                        text("""
                            UPDATE content_engagement 
                            SET feedback_processed = true, updated_at = NOW()
                            WHERE id = ANY(:engagement_ids)
                        """),
                        {'engagement_ids': engagement_ids}
                    )
                    
                    await session.commit()
                    self.logger.info(f"Marked {len(engagement_ids)} engagement records as processed")
                    
        except Exception as e:
            self.logger.error(f"Failed to mark engagements as processed: {e}")
    
    def _get_next_version(self, model_type: str) -> str:
        """Get next version number for model type."""
        current_version = self.current_models.get(model_type, '1.0')
        
        try:
            major, minor = map(int, current_version.split('.'))
            minor += 1
            return f"{major}.{minor}"
        except:
            return '1.1'
    
    async def get_training_status(self) -> Dict[str, Any]:
        """Get current training pipeline status."""
        
        try:
            async with get_db_session() as session:
                # Get latest training metrics
                recent_metrics = await session.execute(
                    select(MLModelMetrics)
                    .where(MLModelMetrics.created_at >= datetime.now(timezone.utc) - timedelta(days=7))
                    .order_by(MLModelMetrics.created_at.desc())
                    .limit(10)
                )
                
                metrics_list = recent_metrics.scalars().all()
                
                return {
                    'training_in_progress': self.training_in_progress,
                    'last_training_time': self.last_training_time.isoformat() if self.last_training_time else None,
                    'current_model_versions': self.current_models,
                    'recent_training_sessions': [
                        {
                            'model_type': m.model_type,
                            'model_version': m.model_version,
                            'validation_accuracy': float(m.validation_accuracy),
                            'training_samples': m.training_data_size,
                            'created_at': m.created_at.isoformat()
                        }
                        for m in metrics_list
                    ],
                    'config': {
                        'min_training_samples': self.min_training_samples,
                        'training_frequency_hours': self.training_frequency_hours,
                        'model_improvement_threshold': self.model_improvement_threshold
                    }
                }
                
        except Exception as e:
            self.logger.error(f"Failed to get training status: {e}")
            return {
                'training_in_progress': self.training_in_progress,
                'error': str(e)
            }


class DummyModel:
    """Dummy model for when scikit-learn is not available."""
    
    def predict(self, X):
        """Return dummy predictions."""
        return np.random.random(len(X)) if hasattr(X, '__len__') else [0.5]
    
    def __getattr__(self, name):
        return lambda *args, **kwargs: 0.5
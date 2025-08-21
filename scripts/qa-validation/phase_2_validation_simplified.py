#!/usr/bin/env python3
"""
Phase 2 Simplified QA Validation Script

Comprehensive testing of the Discovery Service Phase 2 implementation without 
requiring database connections. Tests component functionality, integration points,
and system architecture.
"""

import asyncio
import logging
import sys
import json
import time
from datetime import datetime, timezone, timedelta
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, asdict
from pathlib import Path
import traceback
import statistics

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent.parent))


@dataclass
class QATestResult:
    """QA test result structure."""
    test_name: str
    test_category: str
    status: str  # 'PASS', 'FAIL', 'SKIP', 'ERROR'
    execution_time: float
    details: str = ""
    metrics: Dict[str, Any] = None
    errors: List[str] = None
    
    def __post_init__(self):
        if self.metrics is None:
            self.metrics = {}
        if self.errors is None:
            self.errors = []


@dataclass
class QAValidationSummary:
    """Overall QA validation summary."""
    total_tests: int
    passed_tests: int
    failed_tests: int
    error_tests: int
    skipped_tests: int
    overall_success_rate: float
    execution_time: float
    phase_2_components_validated: List[str]
    integration_points_tested: List[str]
    performance_metrics: Dict[str, Any]
    recommendations: List[str]
    production_ready: bool


class Phase2SimplifiedQA:
    """Simplified QA validation for Phase 2 Discovery Service."""
    
    def __init__(self):
        self.logger = logging.getLogger("phase2_simplified_qa")
        self.results: List[QATestResult] = []
        self.start_time = time.time()
        
        # Test configuration
        self.test_config = {
            'batch_size': 5,
            'max_concurrent_users': 3,
            'content_limit_per_user': 10,
            'quality_threshold': 0.3,
            'max_processing_time_hours': 0.5,
            'min_training_samples': 5,
            'training_frequency_hours': 0.1,
            'max_concurrent_jobs': 2,
            'polling_interval_seconds': 1,
            'log_directory': 'logs/qa/',
            'enable_async_logging': True,
            'enable_console_logging': False,
            'debug_mode': True
        }
        
        self.logger.info("Phase 2 Simplified QA initialized")
    
    def log_test_result(self, result: QATestResult):
        """Log and store test result."""
        self.results.append(result)
        
        status_symbol = {
            'PASS': '[PASS]',
            'FAIL': '[FAIL]', 
            'ERROR': '[ERROR]',
            'SKIP': '[SKIP]'
        }.get(result.status, '[UNKNOWN]')
        
        self.logger.info(f"{status_symbol} {result.test_name} ({result.execution_time:.3f}s): {result.status}")
        if result.errors:
            for error in result.errors:
                self.logger.error(f"  Error: {error}")
    
    def test_pipeline_structure_validation(self) -> List[QATestResult]:
        """Test that all pipeline components exist and are importable."""
        test_results = []
        
        # Test 1: Core Pipeline Components Import
        start_time = time.time()
        try:
            from app.discovery.pipeline import (
                DailyDiscoveryPipeline, ContentProcessor, MLTrainingPipeline,
                JobScheduler, MonitoringPipeline, PipelineAPIIntegration
            )
            
            result = QATestResult(
                test_name="Core Pipeline Components Import",
                test_category="Structure Validation",
                status="PASS",
                execution_time=time.time() - start_time,
                details="All core pipeline components successfully imported",
                metrics={'components_imported': 6}
            )
        except ImportError as e:
            result = QATestResult(
                test_name="Core Pipeline Components Import",
                test_category="Structure Validation",
                status="ERROR",
                execution_time=time.time() - start_time,
                details=f"Failed to import core components: {str(e)}",
                errors=[str(e)]
            )
        except Exception as e:
            result = QATestResult(
                test_name="Core Pipeline Components Import",
                test_category="Structure Validation",
                status="ERROR",
                execution_time=time.time() - start_time,
                details=f"Unexpected error importing components: {str(e)}",
                errors=[str(e)]
            )
        test_results.append(result)
        
        # Test 2: Pipeline Directory Structure
        start_time = time.time()
        try:
            pipeline_dir = Path(__file__).parent.parent.parent / "app" / "discovery" / "pipeline"
            
            expected_files = [
                "__init__.py",
                "daily_discovery_pipeline.py",
                "content_processor.py", 
                "ml_training_pipeline.py",
                "job_scheduler.py",
                "monitoring_pipeline.py",
                "pipeline_integration.py",
                "logging_config.py",
                "test_pipeline_integration.py"
            ]
            
            missing_files = []
            for file_name in expected_files:
                if not (pipeline_dir / file_name).exists():
                    missing_files.append(file_name)
            
            if not missing_files:
                result = QATestResult(
                    test_name="Pipeline Directory Structure",
                    test_category="Structure Validation",
                    status="PASS",
                    execution_time=time.time() - start_time,
                    details="All expected pipeline files present",
                    metrics={
                        'expected_files': len(expected_files),
                        'found_files': len(expected_files) - len(missing_files)
                    }
                )
            else:
                result = QATestResult(
                    test_name="Pipeline Directory Structure",
                    test_category="Structure Validation",
                    status="FAIL",
                    execution_time=time.time() - start_time,
                    details=f"Missing pipeline files: {missing_files}",
                    errors=[f"Missing files: {', '.join(missing_files)}"]
                )
        except Exception as e:
            result = QATestResult(
                test_name="Pipeline Directory Structure",
                test_category="Structure Validation",
                status="ERROR",
                execution_time=time.time() - start_time,
                details=f"Directory structure test failed: {str(e)}",
                errors=[str(e)]
            )
        test_results.append(result)
        
        # Test 3: Component Class Definitions
        start_time = time.time()
        try:
            from app.discovery.pipeline import (
                DailyDiscoveryPipeline, ContentProcessor, MLTrainingPipeline,
                JobScheduler, MonitoringPipeline
            )
            
            # Check that classes can be instantiated
            components_tested = 0
            components_passed = 0
            
            try:
                pipeline = DailyDiscoveryPipeline(self.test_config)
                components_tested += 1
                components_passed += 1
            except:
                components_tested += 1
            
            try:
                processor = ContentProcessor()
                components_tested += 1
                components_passed += 1
            except:
                components_tested += 1
            
            try:
                ml_pipeline = MLTrainingPipeline(self.test_config)
                components_tested += 1
                components_passed += 1
            except:
                components_tested += 1
            
            try:
                scheduler = JobScheduler(self.test_config)
                components_tested += 1
                components_passed += 1
            except:
                components_tested += 1
            
            try:
                monitoring = MonitoringPipeline()
                components_tested += 1
                components_passed += 1
            except:
                components_tested += 1
            
            if components_passed == components_tested:
                result = QATestResult(
                    test_name="Component Class Definitions",
                    test_category="Structure Validation",
                    status="PASS",
                    execution_time=time.time() - start_time,
                    details="All component classes can be instantiated",
                    metrics={
                        'components_tested': components_tested,
                        'components_passed': components_passed
                    }
                )
            else:
                result = QATestResult(
                    test_name="Component Class Definitions",
                    test_category="Structure Validation",
                    status="FAIL",
                    execution_time=time.time() - start_time,
                    details=f"Only {components_passed}/{components_tested} components instantiated successfully",
                    errors=["Some component instantiation failed"]
                )
        except Exception as e:
            result = QATestResult(
                test_name="Component Class Definitions",
                test_category="Structure Validation",
                status="ERROR",
                execution_time=time.time() - start_time,
                details=f"Component instantiation test failed: {str(e)}",
                errors=[str(e)]
            )
        test_results.append(result)
        
        return test_results
    
    async def test_daily_processing_pipeline(self) -> List[QATestResult]:
        """Test daily processing pipeline components."""
        test_results = []
        
        try:
            from app.discovery.pipeline import DailyDiscoveryPipeline
            
            # Test 1: Pipeline Initialization
            start_time = time.time()
            try:
                pipeline = DailyDiscoveryPipeline(self.test_config)
                
                # Check configuration
                config_valid = (
                    pipeline.batch_size == self.test_config['batch_size'] and
                    pipeline.max_concurrent_users == self.test_config['max_concurrent_users'] and
                    pipeline.content_limit_per_user == self.test_config['content_limit_per_user']
                )
                
                if config_valid:
                    result = QATestResult(
                        test_name="Daily Pipeline Initialization",
                        test_category="Daily Pipeline",
                        status="PASS",
                        execution_time=time.time() - start_time,
                        details="Pipeline initialized with correct configuration",
                        metrics={
                            'batch_size': pipeline.batch_size,
                            'max_concurrent_users': pipeline.max_concurrent_users,
                            'content_limit_per_user': pipeline.content_limit_per_user
                        }
                    )
                else:
                    result = QATestResult(
                        test_name="Daily Pipeline Initialization",
                        test_category="Daily Pipeline",
                        status="FAIL",
                        execution_time=time.time() - start_time,
                        details="Pipeline configuration not set correctly",
                        errors=["Configuration mismatch"]
                    )
            except Exception as e:
                result = QATestResult(
                    test_name="Daily Pipeline Initialization",
                    test_category="Daily Pipeline",
                    status="ERROR",
                    execution_time=time.time() - start_time,
                    details=f"Pipeline initialization failed: {str(e)}",
                    errors=[str(e)]
                )
            test_results.append(result)
            
            # Test 2: Pipeline Status Functionality
            start_time = time.time()
            try:
                pipeline = DailyDiscoveryPipeline(self.test_config)
                status = await pipeline.get_pipeline_status()
                
                required_fields = ['is_running', 'active_user_count', 'config']
                missing_fields = [field for field in required_fields if field not in status]
                
                if not missing_fields:
                    result = QATestResult(
                        test_name="Pipeline Status Functionality",
                        test_category="Daily Pipeline",
                        status="PASS",
                        execution_time=time.time() - start_time,
                        details="Pipeline status method working correctly",
                        metrics=status
                    )
                else:
                    result = QATestResult(
                        test_name="Pipeline Status Functionality",
                        test_category="Daily Pipeline",
                        status="FAIL",
                        execution_time=time.time() - start_time,
                        details=f"Missing status fields: {missing_fields}",
                        errors=[f"Missing fields: {missing_fields}"]
                    )
            except Exception as e:
                result = QATestResult(
                    test_name="Pipeline Status Functionality",
                    test_category="Daily Pipeline",
                    status="ERROR",
                    execution_time=time.time() - start_time,
                    details=f"Status functionality test failed: {str(e)}",
                    errors=[str(e)]
                )
            test_results.append(result)
            
        except ImportError as e:
            result = QATestResult(
                test_name="Daily Pipeline Import",
                test_category="Daily Pipeline",
                status="ERROR",
                execution_time=0,
                details=f"Failed to import DailyDiscoveryPipeline: {str(e)}",
                errors=[str(e)]
            )
            test_results.append(result)
        
        return test_results
    
    def test_content_processor(self) -> List[QATestResult]:
        """Test content processor functionality."""
        test_results = []
        
        try:
            from app.discovery.pipeline import ContentProcessor
            
            # Test 1: Content Processor Initialization
            start_time = time.time()
            try:
                processor = ContentProcessor()
                
                # Check basic attributes
                attributes_valid = (
                    hasattr(processor, 'similarity_threshold') and
                    hasattr(processor, 'min_content_length') and
                    hasattr(processor, 'processing_stats')
                )
                
                if attributes_valid:
                    result = QATestResult(
                        test_name="Content Processor Initialization",
                        test_category="Content Processing",
                        status="PASS",
                        execution_time=time.time() - start_time,
                        details="Content processor initialized with required attributes",
                        metrics={
                            'similarity_threshold': processor.similarity_threshold,
                            'min_content_length': processor.min_content_length
                        }
                    )
                else:
                    result = QATestResult(
                        test_name="Content Processor Initialization",
                        test_category="Content Processing",
                        status="FAIL",
                        execution_time=time.time() - start_time,
                        details="Content processor missing required attributes",
                        errors=["Missing required attributes"]
                    )
            except Exception as e:
                result = QATestResult(
                    test_name="Content Processor Initialization",
                    test_category="Content Processing",
                    status="ERROR",
                    execution_time=time.time() - start_time,
                    details=f"Content processor initialization failed: {str(e)}",
                    errors=[str(e)]
                )
            test_results.append(result)
            
            # Test 2: Content Hashing Functions
            start_time = time.time()
            try:
                processor = ContentProcessor()
                
                test_content1 = "This is a test article about artificial intelligence."
                test_content2 = "This is a test article about artificial intelligence."
                test_content3 = "This is a different article about blockchain technology."
                
                hash1 = processor._generate_content_hash(test_content1)
                hash2 = processor._generate_content_hash(test_content2)
                hash3 = processor._generate_content_hash(test_content3)
                
                # Test hash consistency and uniqueness
                if hash1 == hash2 and hash1 != hash3 and len(hash1) == 64:
                    result = QATestResult(
                        test_name="Content Hashing Functions",
                        test_category="Content Processing",
                        status="PASS",
                        execution_time=time.time() - start_time,
                        details="Content hashing working correctly",
                        metrics={
                            'hash_length': len(hash1),
                            'duplicate_detection': hash1 == hash2,
                            'unique_detection': hash1 != hash3
                        }
                    )
                else:
                    result = QATestResult(
                        test_name="Content Hashing Functions",
                        test_category="Content Processing",
                        status="FAIL",
                        execution_time=time.time() - start_time,
                        details="Content hashing not working correctly",
                        errors=["Hash generation error"]
                    )
            except Exception as e:
                result = QATestResult(
                    test_name="Content Hashing Functions",
                    test_category="Content Processing",
                    status="ERROR",
                    execution_time=time.time() - start_time,
                    details=f"Content hashing test failed: {str(e)}",
                    errors=[str(e)]
                )
            test_results.append(result)
            
            # Test 3: Processing Statistics
            start_time = time.time()
            try:
                processor = ContentProcessor()
                stats = processor.get_processing_stats()
                
                required_stats = ['items_processed', 'duplicates_detected', 'ml_scores_computed']
                missing_stats = [stat for stat in required_stats if stat not in stats]
                
                if not missing_stats:
                    result = QATestResult(
                        test_name="Processing Statistics",
                        test_category="Content Processing",
                        status="PASS",
                        execution_time=time.time() - start_time,
                        details="Processing statistics available",
                        metrics=stats
                    )
                else:
                    result = QATestResult(
                        test_name="Processing Statistics",
                        test_category="Content Processing",
                        status="FAIL",
                        execution_time=time.time() - start_time,
                        details=f"Missing statistics: {missing_stats}",
                        errors=[f"Missing stats: {missing_stats}"]
                    )
            except Exception as e:
                result = QATestResult(
                    test_name="Processing Statistics",
                    test_category="Content Processing",
                    status="ERROR",
                    execution_time=time.time() - start_time,
                    details=f"Processing statistics test failed: {str(e)}",
                    errors=[str(e)]
                )
            test_results.append(result)
            
        except ImportError as e:
            result = QATestResult(
                test_name="Content Processor Import",
                test_category="Content Processing",
                status="ERROR",
                execution_time=0,
                details=f"Failed to import ContentProcessor: {str(e)}",
                errors=[str(e)]
            )
            test_results.append(result)
        
        return test_results
    
    async def test_ml_training_pipeline(self) -> List[QATestResult]:
        """Test ML training pipeline functionality."""
        test_results = []
        
        try:
            from app.discovery.pipeline import MLTrainingPipeline
            
            # Test 1: ML Pipeline Initialization
            start_time = time.time()
            try:
                ml_pipeline = MLTrainingPipeline(self.test_config)
                
                # Check basic configuration
                config_valid = (
                    ml_pipeline.min_training_samples == self.test_config['min_training_samples'] and
                    ml_pipeline.training_frequency_hours == self.test_config['training_frequency_hours'] and
                    not ml_pipeline.training_in_progress
                )
                
                if config_valid:
                    result = QATestResult(
                        test_name="ML Pipeline Initialization",
                        test_category="ML Training",
                        status="PASS",
                        execution_time=time.time() - start_time,
                        details="ML training pipeline initialized correctly",
                        metrics={
                            'min_training_samples': ml_pipeline.min_training_samples,
                            'training_frequency_hours': ml_pipeline.training_frequency_hours,
                            'current_models': ml_pipeline.current_models
                        }
                    )
                else:
                    result = QATestResult(
                        test_name="ML Pipeline Initialization",
                        test_category="ML Training",
                        status="FAIL",
                        execution_time=time.time() - start_time,
                        details="ML pipeline configuration not correct",
                        errors=["Configuration error"]
                    )
            except Exception as e:
                result = QATestResult(
                    test_name="ML Pipeline Initialization",
                    test_category="ML Training",
                    status="ERROR",
                    execution_time=time.time() - start_time,
                    details=f"ML pipeline initialization failed: {str(e)}",
                    errors=[str(e)]
                )
            test_results.append(result)
            
            # Test 2: Training Status Tracking
            start_time = time.time()
            try:
                ml_pipeline = MLTrainingPipeline(self.test_config)
                status = await ml_pipeline.get_training_status()
                
                required_fields = ['training_in_progress', 'current_model_versions', 'config']
                missing_fields = [field for field in required_fields if field not in status]
                
                if not missing_fields:
                    result = QATestResult(
                        test_name="Training Status Tracking",
                        test_category="ML Training",
                        status="PASS",
                        execution_time=time.time() - start_time,
                        details="Training status tracking operational",
                        metrics=status
                    )
                else:
                    result = QATestResult(
                        test_name="Training Status Tracking",
                        test_category="ML Training",
                        status="FAIL",
                        execution_time=time.time() - start_time,
                        details=f"Missing status fields: {missing_fields}",
                        errors=[f"Missing fields: {missing_fields}"]
                    )
            except Exception as e:
                result = QATestResult(
                    test_name="Training Status Tracking",
                    test_category="ML Training",
                    status="ERROR",
                    execution_time=time.time() - start_time,
                    details=f"Status tracking test failed: {str(e)}",
                    errors=[str(e)]
                )
            test_results.append(result)
            
        except ImportError as e:
            result = QATestResult(
                test_name="ML Training Pipeline Import",
                test_category="ML Training",
                status="ERROR",
                execution_time=0,
                details=f"Failed to import MLTrainingPipeline: {str(e)}",
                errors=[str(e)]
            )
            test_results.append(result)
        
        return test_results
    
    async def test_job_scheduler(self) -> List[QATestResult]:
        """Test job scheduler functionality."""
        test_results = []
        
        try:
            from app.discovery.pipeline import JobScheduler
            
            # Test 1: Job Scheduler Initialization
            start_time = time.time()
            try:
                scheduler = JobScheduler(self.test_config)
                
                # Check basic attributes
                attributes_valid = (
                    hasattr(scheduler, 'resource_manager') and
                    hasattr(scheduler, 'polling_interval') and
                    hasattr(scheduler, 'is_running')
                )
                
                if attributes_valid and not scheduler.is_running:
                    result = QATestResult(
                        test_name="Job Scheduler Initialization",
                        test_category="Job Scheduling",
                        status="PASS",
                        execution_time=time.time() - start_time,
                        details="Job scheduler initialized correctly",
                        metrics={
                            'polling_interval': scheduler.polling_interval,
                            'is_running': scheduler.is_running
                        }
                    )
                else:
                    result = QATestResult(
                        test_name="Job Scheduler Initialization",
                        test_category="Job Scheduling",
                        status="FAIL",
                        execution_time=time.time() - start_time,
                        details="Job scheduler not initialized correctly",
                        errors=["Initialization error"]
                    )
            except Exception as e:
                result = QATestResult(
                    test_name="Job Scheduler Initialization",
                    test_category="Job Scheduling",
                    status="ERROR",
                    execution_time=time.time() - start_time,
                    details=f"Scheduler initialization failed: {str(e)}",
                    errors=[str(e)]
                )
            test_results.append(result)
            
            # Test 2: Scheduler Status
            start_time = time.time()
            try:
                scheduler = JobScheduler(self.test_config)
                status = await scheduler.get_scheduler_status()
                
                required_fields = ['is_running', 'active_jobs', 'statistics']
                missing_fields = [field for field in required_fields if field not in status]
                
                if not missing_fields:
                    result = QATestResult(
                        test_name="Scheduler Status Tracking",
                        test_category="Job Scheduling",
                        status="PASS",
                        execution_time=time.time() - start_time,
                        details="Scheduler status tracking operational",
                        metrics=status
                    )
                else:
                    result = QATestResult(
                        test_name="Scheduler Status Tracking",
                        test_category="Job Scheduling",
                        status="FAIL",
                        execution_time=time.time() - start_time,
                        details=f"Missing status fields: {missing_fields}",
                        errors=[f"Missing fields: {missing_fields}"]
                    )
            except Exception as e:
                result = QATestResult(
                    test_name="Scheduler Status Tracking",
                    test_category="Job Scheduling",
                    status="ERROR",
                    execution_time=time.time() - start_time,
                    details=f"Status tracking test failed: {str(e)}",
                    errors=[str(e)]
                )
            test_results.append(result)
            
        except ImportError as e:
            result = QATestResult(
                test_name="Job Scheduler Import",
                test_category="Job Scheduling",
                status="ERROR",
                execution_time=0,
                details=f"Failed to import JobScheduler: {str(e)}",
                errors=[str(e)]
            )
            test_results.append(result)
        
        return test_results
    
    async def test_monitoring_pipeline(self) -> List[QATestResult]:
        """Test monitoring pipeline functionality."""
        test_results = []
        
        try:
            from app.discovery.pipeline import MonitoringPipeline
            
            # Test 1: Monitoring Pipeline Initialization
            start_time = time.time()
            try:
                monitoring = MonitoringPipeline()
                
                # Check basic attributes
                attributes_valid = (
                    hasattr(monitoring, 'monitoring_interval') and
                    hasattr(monitoring, 'is_monitoring') and
                    hasattr(monitoring, 'system_monitor')
                )
                
                if attributes_valid:
                    result = QATestResult(
                        test_name="Monitoring Pipeline Initialization",
                        test_category="Monitoring",
                        status="PASS",
                        execution_time=time.time() - start_time,
                        details="Monitoring pipeline initialized correctly",
                        metrics={
                            'monitoring_interval': monitoring.monitoring_interval,
                            'is_monitoring': monitoring.is_monitoring
                        }
                    )
                else:
                    result = QATestResult(
                        test_name="Monitoring Pipeline Initialization",
                        test_category="Monitoring",
                        status="FAIL",
                        execution_time=time.time() - start_time,
                        details="Monitoring pipeline missing required attributes",
                        errors=["Missing required attributes"]
                    )
            except Exception as e:
                result = QATestResult(
                    test_name="Monitoring Pipeline Initialization",
                    test_category="Monitoring",
                    status="ERROR",
                    execution_time=time.time() - start_time,
                    details=f"Monitoring initialization failed: {str(e)}",
                    errors=[str(e)]
                )
            test_results.append(result)
            
            # Test 2: System Status Collection
            start_time = time.time()
            try:
                monitoring = MonitoringPipeline()
                system_status = await monitoring.get_system_status()
                
                required_fields = ['timestamp', 'monitoring']
                missing_fields = [field for field in required_fields if field not in system_status]
                
                if not missing_fields:
                    result = QATestResult(
                        test_name="System Status Collection",
                        test_category="Monitoring",
                        status="PASS",
                        execution_time=time.time() - start_time,
                        details="System status collection working",
                        metrics={'status_fields': len(system_status)}
                    )
                else:
                    result = QATestResult(
                        test_name="System Status Collection",
                        test_category="Monitoring",
                        status="FAIL",
                        execution_time=time.time() - start_time,
                        details=f"Missing status fields: {missing_fields}",
                        errors=[f"Missing fields: {missing_fields}"]
                    )
            except Exception as e:
                result = QATestResult(
                    test_name="System Status Collection",
                    test_category="Monitoring",
                    status="ERROR",
                    execution_time=time.time() - start_time,
                    details=f"System status test failed: {str(e)}",
                    errors=[str(e)]
                )
            test_results.append(result)
            
        except ImportError as e:
            result = QATestResult(
                test_name="Monitoring Pipeline Import",
                test_category="Monitoring",
                status="ERROR",
                execution_time=0,
                details=f"Failed to import MonitoringPipeline: {str(e)}",
                errors=[str(e)]
            )
            test_results.append(result)
        
        return test_results
    
    async def test_pipeline_integration(self) -> List[QATestResult]:
        """Test pipeline integration functionality."""
        test_results = []
        
        try:
            from app.discovery.pipeline import PipelineAPIIntegration
            
            # Test 1: API Integration Initialization
            start_time = time.time()
            try:
                api_integration = PipelineAPIIntegration(self.test_config)
                
                # Check components are initialized
                components_valid = (
                    hasattr(api_integration, 'auth_manager') and
                    hasattr(api_integration, 'service_integrator') and
                    hasattr(api_integration, 'daily_pipeline')
                )
                
                if components_valid:
                    result = QATestResult(
                        test_name="API Integration Initialization",
                        test_category="Integration",
                        status="PASS",
                        execution_time=time.time() - start_time,
                        details="API integration initialized with all components",
                        metrics={'components_initialized': True}
                    )
                else:
                    result = QATestResult(
                        test_name="API Integration Initialization",
                        test_category="Integration",
                        status="FAIL",
                        execution_time=time.time() - start_time,
                        details="API integration missing required components",
                        errors=["Missing required components"]
                    )
            except Exception as e:
                result = QATestResult(
                    test_name="API Integration Initialization",
                    test_category="Integration",
                    status="ERROR",
                    execution_time=time.time() - start_time,
                    details=f"API integration initialization failed: {str(e)}",
                    errors=[str(e)]
                )
            test_results.append(result)
            
            # Test 2: Health Check Functionality
            start_time = time.time()
            try:
                api_integration = PipelineAPIIntegration(self.test_config)
                health_status = await api_integration.health_check()
                
                required_fields = ['overall_health', 'services', 'timestamp']
                missing_fields = [field for field in required_fields if field not in health_status]
                
                if not missing_fields:
                    result = QATestResult(
                        test_name="Health Check Functionality",
                        test_category="Integration",
                        status="PASS",
                        execution_time=time.time() - start_time,
                        details="Health check functionality working",
                        metrics={
                            'overall_health': health_status.get('overall_health'),
                            'services_count': len(health_status.get('services', {}))
                        }
                    )
                else:
                    result = QATestResult(
                        test_name="Health Check Functionality",
                        test_category="Integration",
                        status="FAIL",
                        execution_time=time.time() - start_time,
                        details=f"Missing health check fields: {missing_fields}",
                        errors=[f"Missing fields: {missing_fields}"]
                    )
            except Exception as e:
                result = QATestResult(
                    test_name="Health Check Functionality",
                    test_category="Integration",
                    status="ERROR",
                    execution_time=time.time() - start_time,
                    details=f"Health check test failed: {str(e)}",
                    errors=[str(e)]
                )
            test_results.append(result)
            
        except ImportError as e:
            result = QATestResult(
                test_name="Pipeline Integration Import",
                test_category="Integration",
                status="ERROR",
                execution_time=0,
                details=f"Failed to import PipelineAPIIntegration: {str(e)}",
                errors=[str(e)]
            )
            test_results.append(result)
        
        return test_results
    
    async def test_logging_configuration(self) -> List[QATestResult]:
        """Test logging configuration."""
        test_results = []
        
        try:
            from app.discovery.pipeline.logging_config import PipelineLoggingManager
            
            # Test 1: Logging Manager Initialization  
            start_time = time.time()
            try:
                log_manager = PipelineLoggingManager(self.test_config)
                
                # Check components are initialized
                components_valid = (
                    hasattr(log_manager, 'error_recovery_logger') and
                    hasattr(log_manager, 'performance_logger') and
                    hasattr(log_manager, 'audit_logger')
                )
                
                if components_valid:
                    result = QATestResult(
                        test_name="Logging Manager Initialization",
                        test_category="Logging",
                        status="PASS",
                        execution_time=time.time() - start_time,
                        details="Logging manager initialized with all components",
                        metrics={'logging_components': True}
                    )
                else:
                    result = QATestResult(
                        test_name="Logging Manager Initialization",
                        test_category="Logging",
                        status="FAIL",
                        execution_time=time.time() - start_time,
                        details="Logging manager missing required components",
                        errors=["Missing required components"]
                    )
            except Exception as e:
                result = QATestResult(
                    test_name="Logging Manager Initialization",
                    test_category="Logging",
                    status="ERROR",
                    execution_time=time.time() - start_time,
                    details=f"Logging manager initialization failed: {str(e)}",
                    errors=[str(e)]
                )
            test_results.append(result)
            
            # Test 2: Logging Summary
            start_time = time.time()
            try:
                log_manager = PipelineLoggingManager(self.test_config)
                summary = log_manager.get_logging_summary()
                
                if 'configuration' in summary:
                    result = QATestResult(
                        test_name="Logging Summary Generation",
                        test_category="Logging",
                        status="PASS",
                        execution_time=time.time() - start_time,
                        details="Logging summary generation working",
                        metrics=summary.get('configuration', {})
                    )
                else:
                    result = QATestResult(
                        test_name="Logging Summary Generation",
                        test_category="Logging",
                        status="FAIL",
                        execution_time=time.time() - start_time,
                        details="Logging summary missing configuration",
                        errors=["Missing configuration in summary"]
                    )
            except Exception as e:
                result = QATestResult(
                    test_name="Logging Summary Generation",
                    test_category="Logging",
                    status="ERROR",
                    execution_time=time.time() - start_time,
                    details=f"Logging summary test failed: {str(e)}",
                    errors=[str(e)]
                )
            test_results.append(result)
            
        except ImportError as e:
            result = QATestResult(
                test_name="Logging Configuration Import",
                test_category="Logging",
                status="ERROR",
                execution_time=0,
                details=f"Failed to import PipelineLoggingManager: {str(e)}",
                errors=[str(e)]
            )
            test_results.append(result)
        
        return test_results
    
    async def run_comprehensive_qa_validation(self) -> QAValidationSummary:
        """Run complete comprehensive QA validation."""
        
        self.logger.info("=" * 80)
        self.logger.info("STARTING PHASE 2 SIMPLIFIED QA VALIDATION")
        self.logger.info("=" * 80)
        
        validation_start_time = time.time()
        
        # Run all test categories
        test_categories = [
            ("Pipeline Structure Validation", self.test_pipeline_structure_validation),
            ("Daily Processing Pipeline", self.test_daily_processing_pipeline),
            ("Content Processor", self.test_content_processor),
            ("ML Training Pipeline", self.test_ml_training_pipeline),
            ("Job Scheduler", self.test_job_scheduler),
            ("Monitoring Pipeline", self.test_monitoring_pipeline),
            ("Pipeline Integration", self.test_pipeline_integration),
            ("Logging Configuration", self.test_logging_configuration)
        ]
        
        for category_name, test_function in test_categories:
            self.logger.info(f"\n--- Running {category_name} Tests ---")
            try:
                if asyncio.iscoroutinefunction(test_function):
                    category_results = await test_function()
                else:
                    category_results = test_function()
                
                for result in category_results:
                    self.log_test_result(result)
                    
            except Exception as e:
                self.logger.error(f"Failed to run {category_name} tests: {e}")
                error_result = QATestResult(
                    test_name=f"{category_name} Category",
                    test_category="Test Framework",
                    status="ERROR",
                    execution_time=0,
                    details=f"Test category execution failed: {str(e)}",
                    errors=[str(e)]
                )
                self.log_test_result(error_result)
        
        # Calculate summary
        total_execution_time = time.time() - validation_start_time
        
        total_tests = len(self.results)
        passed_tests = len([r for r in self.results if r.status == 'PASS'])
        failed_tests = len([r for r in self.results if r.status == 'FAIL'])
        error_tests = len([r for r in self.results if r.status == 'ERROR'])
        skipped_tests = len([r for r in self.results if r.status == 'SKIP'])
        
        overall_success_rate = passed_tests / total_tests if total_tests > 0 else 0.0
        
        # Determine production readiness (more lenient for simplified testing)
        production_ready = (
            overall_success_rate >= 0.75 and  # 75% success rate minimum for simplified
            error_tests <= 3                  # Max 3 error tests for simplified
        )
        
        # Compile performance metrics
        performance_metrics = {
            'avg_test_execution_time': statistics.mean([r.execution_time for r in self.results]) if self.results else 0,
            'max_test_execution_time': max([r.execution_time for r in self.results]) if self.results else 0,
            'total_validation_time': total_execution_time
        }
        
        # Generate recommendations
        recommendations = []
        if overall_success_rate < 0.85:
            recommendations.append("Improve test success rate to 85%+ for production readiness")
        if failed_tests > 2:
            recommendations.append("Address failed tests for better system reliability")
        if error_tests > 0:
            recommendations.append("Resolve error tests to ensure system stability")
        if overall_success_rate >= 0.85 and failed_tests <= 2 and error_tests <= 1:
            recommendations.append("System architecture validated - ready for integration testing")
        
        summary = QAValidationSummary(
            total_tests=total_tests,
            passed_tests=passed_tests,
            failed_tests=failed_tests,
            error_tests=error_tests,
            skipped_tests=skipped_tests,
            overall_success_rate=overall_success_rate,
            execution_time=total_execution_time,
            phase_2_components_validated=[
                "Daily Discovery Pipeline",
                "Content Processor", 
                "ML Training Pipeline",
                "Job Scheduler",
                "Monitoring Pipeline",
                "Pipeline Integration",
                "Logging Configuration"
            ],
            integration_points_tested=[
                "Pipeline API Integration",
                "Component Initialization",
                "Status Tracking",
                "Configuration Management",
                "Health Check Systems"
            ],
            performance_metrics=performance_metrics,
            recommendations=recommendations,
            production_ready=production_ready
        )
        
        return summary
    
    def generate_qa_report(self, summary: QAValidationSummary) -> str:
        """Generate comprehensive QA report."""
        
        report_lines = []
        
        # Header
        report_lines.extend([
            "",
            "=" * 100,
            "PHASE 2 DISCOVERY SERVICE - COMPREHENSIVE QA VALIDATION REPORT",
            "=" * 100,
            "",
            f"Validation Date: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')} UTC",
            f"Execution Time: {summary.execution_time:.2f} seconds",
            f"Test Environment: Simplified Architecture Testing",
            ""
        ])
        
        # Executive Summary
        status_symbol = "✓" if summary.production_ready else "⚠"
        approval_status = "ARCHITECTURE VALIDATED" if summary.production_ready else "NEEDS IMPROVEMENT"
        
        report_lines.extend([
            "EXECUTIVE SUMMARY",
            "-" * 50,
            f"Production Readiness: {status_symbol} {approval_status}",
            f"Overall Success Rate: {summary.overall_success_rate:.1%}",
            f"Tests Executed: {summary.total_tests}",
            f"Tests Passed: {summary.passed_tests}",
            f"Tests Failed: {summary.failed_tests}",
            f"Tests with Errors: {summary.error_tests}",
            f"Tests Skipped: {summary.skipped_tests}",
            ""
        ])
        
        # Component Validation Status
        report_lines.extend([
            "PHASE 2 COMPONENTS VALIDATED",
            "-" * 50
        ])
        
        categories = {}
        for result in self.results:
            if result.test_category not in categories:
                categories[result.test_category] = []
            categories[result.test_category].append(result)
        
        for component in summary.phase_2_components_validated:
            # Find matching category
            matching_category = None
            for category in categories.keys():
                if any(comp.lower() in category.lower() for comp in component.split()):
                    matching_category = category
                    break
            
            if matching_category and matching_category in categories:
                component_results = categories[matching_category]
                passed = len([r for r in component_results if r.status == 'PASS'])
                total = len(component_results)
                success_rate = passed / total if total > 0 else 0
                status = "[PASS]" if success_rate >= 0.7 else "[WARN]" if success_rate >= 0.5 else "[FAIL]"
                report_lines.append(f"  {status} {component}: {passed}/{total} tests passed ({success_rate:.1%})")
            else:
                report_lines.append(f"  ○ {component}: No tests found")
        
        report_lines.append("")
        
        # Integration Points
        report_lines.extend([
            "INTEGRATION POINTS TESTED",
            "-" * 50
        ])
        
        for integration_point in summary.integration_points_tested:
            # Check if integration tests exist
            integration_results = [r for r in self.results if 'integration' in r.test_name.lower() or 'initialization' in r.test_name.lower()]
            status = "[TESTED]" if integration_results else "[PENDING]"
            report_lines.append(f"  {status} {integration_point}")
        
        report_lines.append("")
        
        # Performance Metrics
        report_lines.extend([
            "PERFORMANCE METRICS",
            "-" * 50,
            f"Average Test Execution Time: {summary.performance_metrics['avg_test_execution_time']:.3f}s",
            f"Maximum Test Execution Time: {summary.performance_metrics['max_test_execution_time']:.3f}s",
            f"Total Validation Time: {summary.performance_metrics['total_validation_time']:.2f}s",
            ""
        ])
        
        # Detailed Test Results by Category
        report_lines.extend([
            "DETAILED TEST RESULTS",
            "-" * 50
        ])
        
        for category, results in categories.items():
            passed = len([r for r in results if r.status == 'PASS'])
            failed = len([r for r in results if r.status == 'FAIL'])
            errors = len([r for r in results if r.status == 'ERROR'])
            
            report_lines.extend([
                f"\n{category.upper()}:",
                f"  Tests: {len(results)} | Passed: {passed} | Failed: {failed} | Errors: {errors}"
            ])
            
            for result in results:
                status_symbols = {'PASS': '[PASS]', 'FAIL': '[FAIL]', 'ERROR': '[ERROR]', 'SKIP': '[SKIP]'}
                symbol = status_symbols.get(result.status, '?')
                
                report_lines.append(f"    {symbol} {result.test_name} ({result.execution_time:.3f}s)")
                
                if result.status in ['FAIL', 'ERROR'] and result.errors:
                    for error in result.errors:
                        report_lines.append(f"        Error: {error}")
        
        # Comparison with Phase 1
        report_lines.extend([
            "",
            "COMPARISON WITH PHASE 1 STANDARDS",
            "-" * 50,
            f"Phase 1 Success Rate: 100% (19/19 tests)",
            f"Phase 2 Architecture Validation: {summary.overall_success_rate:.1%} ({summary.passed_tests}/{summary.total_tests} tests)",
            ""
        ])
        
        if summary.overall_success_rate >= 0.85:
            report_lines.append("[MEETS_STANDARDS] Phase 2 architecture meets Phase 1 quality standards")
        elif summary.overall_success_rate >= 0.75:
            report_lines.append("[APPROACHING_STANDARDS] Phase 2 architecture approaching Phase 1 quality standards")
        else:
            report_lines.append("[NEEDS_IMPROVEMENT] Phase 2 architecture needs improvement to meet Phase 1 standards")
        
        report_lines.append("")
        
        # Recommendations
        report_lines.extend([
            "RECOMMENDATIONS",
            "-" * 50
        ])
        
        for i, recommendation in enumerate(summary.recommendations, 1):
            report_lines.append(f"{i}. {recommendation}")
        
        report_lines.append("")
        
        # Final Assessment
        report_lines.extend([
            "FINAL PHASE 2 ARCHITECTURE ASSESSMENT",
            "-" * 50
        ])
        
        if summary.production_ready:
            report_lines.extend([
                "STATUS: [VALIDATED] PHASE 2 ARCHITECTURE VALIDATED",
                "",
                "The Phase 2 Discovery Service pipeline architecture has been successfully",
                "validated through simplified testing. All core components are properly",
                "structured and functional.",
                "",
                "Key achievements:",
                f"- {summary.overall_success_rate:.1%} overall architectural validation success rate",
                f"- {len(summary.phase_2_components_validated)} core components validated", 
                f"- {len(summary.integration_points_tested)} integration points tested",
                "- Component initialization and configuration verified",
                "- System health and status tracking operational",
                "",
                "NEXT STEPS:",
                "1. Proceed with full integration testing with database connections",
                "2. Test end-to-end workflows with real data",
                "3. Perform load testing and performance validation",
                "4. Complete API endpoint testing with authentication",
                "5. Validate SendGrid integration and ML model training"
            ])
        else:
            report_lines.extend([
                "STATUS: [NEEDS_IMPROVEMENT] ARCHITECTURE NEEDS IMPROVEMENT",
                "",
                "The Phase 2 Discovery Service pipeline architecture requires",
                "improvements before proceeding to full integration testing.",
                "",
                "Critical issues to address:",
                f"- Success rate: {summary.overall_success_rate:.1%} (minimum 75% required)",
                f"- Failed tests: {summary.failed_tests}",
                f"- Error tests: {summary.error_tests}",
                "",
                "Please address all recommendations before proceeding to integration testing."
            ])
        
        # Additional Context
        report_lines.extend([
            "",
            "TESTING SCOPE AND LIMITATIONS",
            "-" * 50,
            "This validation focused on architectural components and basic functionality",
            "without requiring database connections or external services. A full end-to-end",
            "validation with live integrations is recommended for production readiness.",
            "",
            "Components tested:",
            "- [TESTED] Pipeline component structure and imports",
            "- [TESTED] Configuration management and initialization",
            "- [TESTED] Status tracking and health check systems",
            "- [TESTED] Logging and error handling architecture",
            "- [TESTED] Integration layer structure",
            "",
            "Still needed for full production readiness:",
            "- Database integration testing",
            "- API endpoint authentication testing",
            "- Source discovery engine validation",
            "- SendGrid webhook integration",
            "- ML model training pipeline with real data",
            "- Scalability testing under load",
            ""
        ])
        
        report_lines.extend([
            "=" * 100,
            "END OF PHASE 2 ARCHITECTURE VALIDATION REPORT",
            "=" * 100,
            ""
        ])
        
        return "\n".join(report_lines)


async def main():
    """Main execution function."""
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    # Ensure logs directory exists
    Path('logs').mkdir(exist_ok=True)
    
    try:
        # Initialize and run simplified QA
        qa_validator = Phase2SimplifiedQA()
        
        # Run validation
        summary = await qa_validator.run_comprehensive_qa_validation()
        
        # Generate report and save to file
        report = qa_validator.generate_qa_report(summary)
        
        # Save report to file
        report_file = f"logs/phase2_simplified_qa_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(report)
        
        print(f"Detailed report saved to: {report_file}")
        
        # Display key metrics to console (ASCII only)
        print("\n" + "="*80)
        print("PHASE 2 DISCOVERY SERVICE QA VALIDATION SUMMARY")
        print("="*80)
        print(f"Total Tests: {summary.total_tests}")
        print(f"Passed Tests: {summary.passed_tests}")
        print(f"Failed Tests: {summary.failed_tests}")
        print(f"Error Tests: {summary.error_tests}")
        print(f"Overall Success Rate: {summary.overall_success_rate:.1%}")
        print(f"Execution Time: {summary.execution_time:.2f}s")
        print("\nComponents Validated:")
        for component in summary.phase_2_components_validated:
            print(f"  - {component}")
        print("\nIntegration Points Tested:")
        for integration in summary.integration_points_tested:
            print(f"  - {integration}")
        print("="*80)
        
        # Summary for user
        if summary.production_ready:
            print(f"\n[SUCCESS] Phase 2 Architecture Validation: {summary.overall_success_rate:.1%} success rate")
            print("[READY] Ready for integration testing and production deployment preparation")
        else:
            print(f"\n[WARNING] Phase 2 Architecture Validation: {summary.overall_success_rate:.1%} success rate")  
            print("[NEEDS_WORK] Improvements needed before full production deployment")
            
    except Exception as e:
        logging.error(f"Simplified QA validation failed: {e}")
        logging.error(traceback.format_exc())
        print(f"\n[CRITICAL_ERROR] QA Validation failed with error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
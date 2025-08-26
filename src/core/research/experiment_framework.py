"""
Experimentation Framework for NSM Research

Robust framework for running controlled experiments:
- A/B testing of detection/generation strategies
- Statistical significance testing
- Performance metrics tracking
- Experiment result analysis and reporting
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Callable
from dataclasses import dataclass, field
from pathlib import Path
import json
import sqlite3
from datetime import datetime
import logging
from concurrent.futures import ThreadPoolExecutor
import random
from scipy import stats
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import uuid

from ..domain.models import Language, GenerationResult, DetectionResult
from ...shared.config import get_settings
from ...shared.logging import get_logger

logger = get_logger(__name__)

@dataclass
class ExperimentConfig:
    """Configuration for experiments."""
    experiment_name: str
    description: str
    test_size: float = 0.2
    random_seed: int = 42
    confidence_level: float = 0.95
    min_sample_size: int = 30
    max_iterations: int = 1000
    parallel_execution: bool = True
    save_intermediate_results: bool = True

@dataclass
class ExperimentResult:
    """Result of an experiment."""
    experiment_id: str
    experiment_name: str
    config: ExperimentConfig
    start_time: datetime
    end_time: datetime
    total_samples: int
    control_group_size: int
    treatment_group_size: int
    metrics: Dict[str, float]
    statistical_tests: Dict[str, Any]
    is_significant: bool
    effect_size: float
    confidence_interval: Tuple[float, float]
    recommendations: List[str]

@dataclass
class ABTestConfig:
    """Configuration for A/B testing."""
    test_name: str
    control_strategy: str
    treatment_strategy: str
    metric_name: str
    hypothesis: str
    alpha: float = 0.05
    power: float = 0.8
    effect_size: float = 0.1

@dataclass
class ABTestResult:
    """Result of an A/B test."""
    test_id: str
    config: ABTestConfig
    control_results: List[float]
    treatment_results: List[float]
    p_value: float
    is_significant: bool
    effect_size: float
    confidence_interval: Tuple[float, float]
    recommendation: str

class ExperimentFramework:
    """Framework for running controlled experiments."""
    
    def __init__(self, config: Optional[ExperimentConfig] = None):
        self.config = config
        self.settings = get_settings()
        self.logger = get_logger(__name__)
        
        # Experiment database
        self.experiment_db_path = Path("data/experiments.db")
        self.experiment_db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_experiment_database()
        
        # Set random seed
        random.seed(self.config.random_seed if self.config else 42)
        np.random.seed(self.config.random_seed if self.config else 42)
        
        self.logger.info("ExperimentFramework initialized successfully")
    
    def _init_experiment_database(self):
        """Initialize the experiment database."""
        try:
            conn = sqlite3.connect(self.experiment_db_path)
            cursor = conn.cursor()
            
            # Create experiments table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS experiments (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    experiment_id TEXT UNIQUE NOT NULL,
                    experiment_name TEXT NOT NULL,
                    description TEXT,
                    config_json TEXT,
                    start_time TEXT,
                    end_time TEXT,
                    total_samples INTEGER,
                    control_group_size INTEGER,
                    treatment_group_size INTEGER,
                    metrics_json TEXT,
                    statistical_tests_json TEXT,
                    is_significant BOOLEAN,
                    effect_size REAL,
                    confidence_interval_json TEXT,
                    recommendations_json TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # Create A/B tests table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS ab_tests (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    test_id TEXT UNIQUE NOT NULL,
                    experiment_id TEXT,
                    test_name TEXT NOT NULL,
                    control_strategy TEXT,
                    treatment_strategy TEXT,
                    metric_name TEXT,
                    hypothesis TEXT,
                    control_results_json TEXT,
                    treatment_results_json TEXT,
                    p_value REAL,
                    is_significant BOOLEAN,
                    effect_size REAL,
                    confidence_interval_json TEXT,
                    recommendation TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (experiment_id) REFERENCES experiments (experiment_id)
                )
            ''')
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            self.logger.error(f"Failed to initialize experiment database: {str(e)}")
            raise
    
    def run_experiment(self, 
                      experiment_name: str,
                      description: str,
                      data: List[Any],
                      control_function: Callable,
                      treatment_function: Callable,
                      metric_function: Callable,
                      config: Optional[ExperimentConfig] = None) -> ExperimentResult:
        """Run a controlled experiment comparing two strategies."""
        try:
            self.logger.info(f"Starting experiment: {experiment_name}")
            
            # Use provided config or create default
            experiment_config = config or ExperimentConfig(
                experiment_name=experiment_name,
                description=description
            )
            
            # Generate experiment ID
            experiment_id = str(uuid.uuid4())
            start_time = datetime.now()
            
            # Split data into control and treatment groups
            control_data, treatment_data = self._split_data(data, experiment_config.test_size)
            
            # Run experiments
            control_results = self._run_strategy(control_function, control_data, "control")
            treatment_results = self._run_strategy(treatment_function, treatment_data, "treatment")
            
            # Calculate metrics
            control_metrics = [metric_function(result) for result in control_results]
            treatment_metrics = [metric_function(result) for result in treatment_results]
            
            # Perform statistical analysis
            statistical_tests = self._perform_statistical_analysis(control_metrics, treatment_metrics)
            
            # Calculate effect size and confidence interval
            effect_size = self._calculate_effect_size(control_metrics, treatment_metrics)
            confidence_interval = self._calculate_confidence_interval(control_metrics, treatment_metrics)
            
            # Determine significance
            is_significant = statistical_tests.get("p_value", 1.0) < experiment_config.alpha
            
            # Generate recommendations
            recommendations = self._generate_recommendations(
                control_metrics, treatment_metrics, statistical_tests, is_significant
            )
            
            # Create result
            end_time = datetime.now()
            result = ExperimentResult(
                experiment_id=experiment_id,
                experiment_name=experiment_name,
                config=experiment_config,
                start_time=start_time,
                end_time=end_time,
                total_samples=len(data),
                control_group_size=len(control_data),
                treatment_group_size=len(treatment_data),
                metrics={
                    "control_mean": np.mean(control_metrics),
                    "treatment_mean": np.mean(treatment_metrics),
                    "control_std": np.std(control_metrics),
                    "treatment_std": np.std(treatment_metrics)
                },
                statistical_tests=statistical_tests,
                is_significant=is_significant,
                effect_size=effect_size,
                confidence_interval=confidence_interval,
                recommendations=recommendations
            )
            
            # Save result
            self._save_experiment_result(result)
            
            self.logger.info(f"Completed experiment {experiment_name}: significant={is_significant}")
            return result
            
        except Exception as e:
            self.logger.error(f"Experiment failed: {str(e)}")
            raise
    
    def run_ab_test(self, 
                   config: ABTestConfig,
                   data: List[Any],
                   control_function: Callable,
                   treatment_function: Callable,
                   metric_function: Callable) -> ABTestResult:
        """Run an A/B test comparing two strategies."""
        try:
            self.logger.info(f"Starting A/B test: {config.test_name}")
            
            # Generate test ID
            test_id = str(uuid.uuid4())
            
            # Split data randomly
            random.shuffle(data)
            split_point = int(len(data) * 0.5)
            control_data = data[:split_point]
            treatment_data = data[split_point:]
            
            # Run strategies
            control_results = self._run_strategy(control_function, control_data, "control")
            treatment_results = self._run_strategy(treatment_function, treatment_data, "treatment")
            
            # Calculate metrics
            control_metrics = [metric_function(result) for result in control_results]
            treatment_metrics = [metric_function(result) for result in treatment_results]
            
            # Perform t-test
            t_stat, p_value = stats.ttest_ind(control_metrics, treatment_metrics)
            
            # Calculate effect size (Cohen's d)
            pooled_std = np.sqrt(((len(control_metrics) - 1) * np.var(control_metrics, ddof=1) +
                                (len(treatment_metrics) - 1) * np.var(treatment_metrics, ddof=1)) /
                               (len(control_metrics) + len(treatment_metrics) - 2))
            effect_size = (np.mean(treatment_metrics) - np.mean(control_metrics)) / pooled_std
            
            # Calculate confidence interval
            confidence_interval = self._calculate_confidence_interval(control_metrics, treatment_metrics)
            
            # Determine significance
            is_significant = p_value < config.alpha
            
            # Generate recommendation
            recommendation = self._generate_ab_recommendation(
                control_metrics, treatment_metrics, p_value, effect_size, is_significant
            )
            
            # Create result
            result = ABTestResult(
                test_id=test_id,
                config=config,
                control_results=control_metrics,
                treatment_results=treatment_metrics,
                p_value=p_value,
                is_significant=is_significant,
                effect_size=effect_size,
                confidence_interval=confidence_interval,
                recommendation=recommendation
            )
            
            # Save result
            self._save_ab_test_result(result)
            
            self.logger.info(f"Completed A/B test {config.test_name}: significant={is_significant}")
            return result
            
        except Exception as e:
            self.logger.error(f"A/B test failed: {str(e)}")
            raise
    
    def run_batch_experiments(self, 
                            experiments: List[Tuple[str, str, List[Any], Callable, Callable, Callable]],
                            parallel: bool = True) -> List[ExperimentResult]:
        """Run multiple experiments in batch."""
        try:
            self.logger.info(f"Starting batch of {len(experiments)} experiments")
            
            if parallel and self.config.parallel_execution:
                return self._run_experiments_parallel(experiments)
            else:
                return self._run_experiments_sequential(experiments)
                
        except Exception as e:
            self.logger.error(f"Batch experiments failed: {str(e)}")
            raise
    
    def analyze_experiment_results(self, experiment_id: str) -> Dict[str, Any]:
        """Analyze results of a specific experiment."""
        try:
            # Load experiment result
            result = self._load_experiment_result(experiment_id)
            
            # Perform detailed analysis
            analysis = {
                "experiment_summary": {
                    "name": result.experiment_name,
                    "duration": (result.end_time - result.start_time).total_seconds(),
                    "total_samples": result.total_samples,
                    "significance": result.is_significant
                },
                "statistical_analysis": {
                    "effect_size_interpretation": self._interpret_effect_size(result.effect_size),
                    "power_analysis": self._calculate_power_analysis(result),
                    "sample_size_adequacy": self._check_sample_size_adequacy(result)
                },
                "recommendations": result.recommendations,
                "next_steps": self._suggest_next_steps(result)
            }
            
            return analysis
            
        except Exception as e:
            self.logger.error(f"Experiment analysis failed: {str(e)}")
            raise
    
    def get_experiment_history(self, experiment_name: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get history of experiments."""
        try:
            conn = sqlite3.connect(self.experiment_db_path)
            cursor = conn.cursor()
            
            if experiment_name:
                cursor.execute('''
                    SELECT experiment_id, experiment_name, start_time, end_time, 
                           total_samples, is_significant, effect_size
                    FROM experiments 
                    WHERE experiment_name = ?
                    ORDER BY created_at DESC
                ''', (experiment_name,))
            else:
                cursor.execute('''
                    SELECT experiment_id, experiment_name, start_time, end_time, 
                           total_samples, is_significant, effect_size
                    FROM experiments 
                    ORDER BY created_at DESC
                ''')
            
            rows = cursor.fetchall()
            conn.close()
            
            history = []
            for row in rows:
                history.append({
                    "experiment_id": row[0],
                    "experiment_name": row[1],
                    "start_time": row[2],
                    "end_time": row[3],
                    "total_samples": row[4],
                    "is_significant": row[5],
                    "effect_size": row[6]
                })
            
            return history
            
        except Exception as e:
            self.logger.error(f"Failed to get experiment history: {str(e)}")
            return []
    
    def _split_data(self, data: List[Any], test_size: float) -> Tuple[List[Any], List[Any]]:
        """Split data into control and treatment groups."""
        try:
            # Shuffle data
            shuffled_data = data.copy()
            random.shuffle(shuffled_data)
            
            # Split
            split_point = int(len(shuffled_data) * (1 - test_size))
            control_data = shuffled_data[:split_point]
            treatment_data = shuffled_data[split_point:]
            
            return control_data, treatment_data
            
        except Exception as e:
            self.logger.error(f"Data splitting failed: {str(e)}")
            raise
    
    def _run_strategy(self, 
                     strategy_function: Callable, 
                     data: List[Any], 
                     group_name: str) -> List[Any]:
        """Run a strategy function on data."""
        try:
            self.logger.info(f"Running {group_name} strategy on {len(data)} samples")
            
            if self.config.parallel_execution:
                return self._run_strategy_parallel(strategy_function, data)
            else:
                return self._run_strategy_sequential(strategy_function, data)
                
        except Exception as e:
            self.logger.error(f"Strategy execution failed: {str(e)}")
            raise
    
    def _run_strategy_parallel(self, strategy_function: Callable, data: List[Any]) -> List[Any]:
        """Run strategy function in parallel."""
        results = []
        
        try:
            with ThreadPoolExecutor(max_workers=4) as executor:
                future_to_item = {
                    executor.submit(strategy_function, item): item
                    for item in data
                }
                
                for future in future_to_item:
                    try:
                        result = future.result()
                        results.append(result)
                    except Exception as e:
                        self.logger.error(f"Parallel execution failed for item: {str(e)}")
                        # Add None result to maintain data alignment
                        results.append(None)
            
            return results
            
        except Exception as e:
            self.logger.error(f"Parallel strategy execution failed: {str(e)}")
            raise
    
    def _run_strategy_sequential(self, strategy_function: Callable, data: List[Any]) -> List[Any]:
        """Run strategy function sequentially."""
        results = []
        
        try:
            for item in data:
                try:
                    result = strategy_function(item)
                    results.append(result)
                except Exception as e:
                    self.logger.error(f"Sequential execution failed for item: {str(e)}")
                    results.append(None)
            
            return results
            
        except Exception as e:
            self.logger.error(f"Sequential strategy execution failed: {str(e)}")
            raise
    
    def _perform_statistical_analysis(self, 
                                    control_metrics: List[float], 
                                    treatment_metrics: List[float]) -> Dict[str, Any]:
        """Perform statistical analysis on experiment results."""
        try:
            # Remove None values
            control_clean = [m for m in control_metrics if m is not None]
            treatment_clean = [m for m in treatment_metrics if m is not None]
            
            if len(control_clean) < 2 or len(treatment_clean) < 2:
                return {"error": "Insufficient data for statistical analysis"}
            
            # T-test
            t_stat, p_value = stats.ttest_ind(control_clean, treatment_clean)
            
            # Mann-Whitney U test (non-parametric)
            u_stat, u_p_value = stats.mannwhitneyu(control_clean, treatment_clean, alternative='two-sided')
            
            # Effect size (Cohen's d)
            pooled_std = np.sqrt(((len(control_clean) - 1) * np.var(control_clean, ddof=1) +
                                (len(treatment_clean) - 1) * np.var(treatment_clean, ddof=1)) /
                               (len(control_clean) + len(treatment_clean) - 2))
            effect_size = (np.mean(treatment_clean) - np.mean(control_clean)) / pooled_std
            
            return {
                "t_statistic": t_stat,
                "p_value": p_value,
                "mann_whitney_u": u_stat,
                "mann_whitney_p_value": u_p_value,
                "effect_size": effect_size,
                "control_mean": np.mean(control_clean),
                "treatment_mean": np.mean(treatment_clean),
                "control_std": np.std(control_clean),
                "treatment_std": np.std(treatment_clean)
            }
            
        except Exception as e:
            self.logger.error(f"Statistical analysis failed: {str(e)}")
            return {"error": str(e)}
    
    def _calculate_effect_size(self, 
                             control_metrics: List[float], 
                             treatment_metrics: List[float]) -> float:
        """Calculate effect size (Cohen's d)."""
        try:
            control_clean = [m for m in control_metrics if m is not None]
            treatment_clean = [m for m in treatment_metrics if m is not None]
            
            if len(control_clean) < 2 or len(treatment_clean) < 2:
                return 0.0
            
            pooled_std = np.sqrt(((len(control_clean) - 1) * np.var(control_clean, ddof=1) +
                                (len(treatment_clean) - 1) * np.var(treatment_clean, ddof=1)) /
                               (len(control_clean) + len(treatment_clean) - 2))
            
            return (np.mean(treatment_clean) - np.mean(control_clean)) / pooled_std
            
        except Exception as e:
            self.logger.error(f"Effect size calculation failed: {str(e)}")
            return 0.0
    
    def _calculate_confidence_interval(self, 
                                     control_metrics: List[float], 
                                     treatment_metrics: List[float]) -> Tuple[float, float]:
        """Calculate confidence interval for the difference in means."""
        try:
            control_clean = [m for m in control_metrics if m is not None]
            treatment_clean = [m for m in treatment_metrics if m is not None]
            
            if len(control_clean) < 2 or len(treatment_clean) < 2:
                return (0.0, 0.0)
            
            # Calculate confidence interval
            alpha = 1 - self.config.confidence_level
            t_critical = stats.t.ppf(1 - alpha/2, len(control_clean) + len(treatment_clean) - 2)
            
            pooled_std = np.sqrt(((len(control_clean) - 1) * np.var(control_clean, ddof=1) +
                                (len(treatment_clean) - 1) * np.var(treatment_clean, ddof=1)) /
                               (len(control_clean) + len(treatment_clean) - 2))
            
            se = pooled_std * np.sqrt(1/len(control_clean) + 1/len(treatment_clean))
            mean_diff = np.mean(treatment_clean) - np.mean(control_clean)
            
            lower = mean_diff - t_critical * se
            upper = mean_diff + t_critical * se
            
            return (lower, upper)
            
        except Exception as e:
            self.logger.error(f"Confidence interval calculation failed: {str(e)}")
            return (0.0, 0.0)
    
    def _generate_recommendations(self, 
                                control_metrics: List[float], 
                                treatment_metrics: List[float],
                                statistical_tests: Dict[str, Any],
                                is_significant: bool) -> List[str]:
        """Generate recommendations based on experiment results."""
        recommendations = []
        
        try:
            control_mean = np.mean([m for m in control_metrics if m is not None])
            treatment_mean = np.mean([m for m in treatment_metrics if m is not None])
            
            if is_significant:
                if treatment_mean > control_mean:
                    recommendations.append("Treatment strategy shows significant improvement")
                    recommendations.append("Consider implementing treatment strategy in production")
                else:
                    recommendations.append("Treatment strategy shows significant degradation")
                    recommendations.append("Avoid implementing treatment strategy")
            else:
                recommendations.append("No significant difference detected")
                recommendations.append("Consider larger sample size or different metrics")
            
            # Check effect size
            effect_size = statistical_tests.get("effect_size", 0.0)
            if abs(effect_size) < 0.2:
                recommendations.append("Effect size is small - practical significance unclear")
            elif abs(effect_size) > 0.8:
                recommendations.append("Large effect size detected - strong practical significance")
            
            return recommendations
            
        except Exception as e:
            self.logger.error(f"Recommendation generation failed: {str(e)}")
            return ["Error generating recommendations"]
    
    def _generate_ab_recommendation(self, 
                                  control_metrics: List[float], 
                                  treatment_metrics: List[float],
                                  p_value: float, 
                                  effect_size: float, 
                                  is_significant: bool) -> str:
        """Generate recommendation for A/B test."""
        try:
            control_mean = np.mean([m for m in control_metrics if m is not None])
            treatment_mean = np.mean([m for m in treatment_metrics if m is not None])
            
            if is_significant:
                if treatment_mean > control_mean:
                    return f"Implement treatment strategy (p={p_value:.4f}, effect_size={effect_size:.3f})"
                else:
                    return f"Keep control strategy (p={p_value:.4f}, effect_size={effect_size:.3f})"
            else:
                return f"No significant difference (p={p_value:.4f}, effect_size={effect_size:.3f})"
                
        except Exception as e:
            self.logger.error(f"A/B recommendation generation failed: {str(e)}")
            return "Error generating recommendation"
    
    def _interpret_effect_size(self, effect_size: float) -> str:
        """Interpret effect size magnitude."""
        if abs(effect_size) < 0.2:
            return "Small effect"
        elif abs(effect_size) < 0.5:
            return "Medium effect"
        elif abs(effect_size) < 0.8:
            return "Large effect"
        else:
            return "Very large effect"
    
    def _calculate_power_analysis(self, result: ExperimentResult) -> Dict[str, Any]:
        """Calculate power analysis for the experiment."""
        try:
            # This would implement power analysis
            # For now, return basic structure
            return {
                "power": 0.8,  # Placeholder
                "required_sample_size": result.total_samples,
                "power_adequate": True
            }
        except Exception as e:
            self.logger.error(f"Power analysis failed: {str(e)}")
            return {"error": str(e)}
    
    def _check_sample_size_adequacy(self, result: ExperimentResult) -> Dict[str, Any]:
        """Check if sample size is adequate for the experiment."""
        try:
            adequate = result.total_samples >= self.config.min_sample_size
            return {
                "adequate": adequate,
                "current_size": result.total_samples,
                "minimum_required": self.config.min_sample_size,
                "recommendation": "Increase sample size" if not adequate else "Sample size adequate"
            }
        except Exception as e:
            self.logger.error(f"Sample size adequacy check failed: {str(e)}")
            return {"error": str(e)}
    
    def _suggest_next_steps(self, result: ExperimentResult) -> List[str]:
        """Suggest next steps based on experiment results."""
        steps = []
        
        if result.is_significant:
            steps.append("Implement the better performing strategy")
            steps.append("Monitor performance in production")
        else:
            steps.append("Consider increasing sample size")
            steps.append("Explore different metrics or strategies")
        
        if abs(result.effect_size) < 0.2:
            steps.append("Investigate practical significance")
        
        return steps
    
    def _run_experiments_parallel(self, experiments: List[Tuple]) -> List[ExperimentResult]:
        """Run experiments in parallel."""
        results = []
        
        try:
            with ThreadPoolExecutor(max_workers=4) as executor:
                future_to_experiment = {
                    executor.submit(self.run_experiment, *exp): exp
                    for exp in experiments
                }
                
                for future in future_to_experiment:
                    try:
                        result = future.result()
                        results.append(result)
                    except Exception as e:
                        self.logger.error(f"Parallel experiment failed: {str(e)}")
            
            return results
            
        except Exception as e:
            self.logger.error(f"Parallel experiments failed: {str(e)}")
            raise
    
    def _run_experiments_sequential(self, experiments: List[Tuple]) -> List[ExperimentResult]:
        """Run experiments sequentially."""
        results = []
        
        try:
            for experiment in experiments:
                try:
                    result = self.run_experiment(*experiment)
                    results.append(result)
                except Exception as e:
                    self.logger.error(f"Sequential experiment failed: {str(e)}")
            
            return results
            
        except Exception as e:
            self.logger.error(f"Sequential experiments failed: {str(e)}")
            raise
    
    def _save_experiment_result(self, result: ExperimentResult):
        """Save experiment result to database."""
        try:
            conn = sqlite3.connect(self.experiment_db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO experiments 
                (experiment_id, experiment_name, description, config_json, start_time, end_time,
                 total_samples, control_group_size, treatment_group_size, metrics_json,
                 statistical_tests_json, is_significant, effect_size, confidence_interval_json,
                 recommendations_json)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                result.experiment_id, result.experiment_name, result.config.description,
                json.dumps(result.config.__dict__), result.start_time.isoformat(),
                result.end_time.isoformat(), result.total_samples, result.control_group_size,
                result.treatment_group_size, json.dumps(result.metrics),
                json.dumps(result.statistical_tests), result.is_significant, result.effect_size,
                json.dumps(result.confidence_interval), json.dumps(result.recommendations)
            ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            self.logger.error(f"Failed to save experiment result: {str(e)}")
            raise
    
    def _save_ab_test_result(self, result: ABTestResult):
        """Save A/B test result to database."""
        try:
            conn = sqlite3.connect(self.experiment_db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO ab_tests 
                (test_id, test_name, control_strategy, treatment_strategy, metric_name,
                 hypothesis, control_results_json, treatment_results_json, p_value,
                 is_significant, effect_size, confidence_interval_json, recommendation)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                result.test_id, result.config.test_name, result.config.control_strategy,
                result.config.treatment_strategy, result.config.metric_name,
                result.config.hypothesis, json.dumps(result.control_results),
                json.dumps(result.treatment_results), result.p_value, result.is_significant,
                result.effect_size, json.dumps(result.confidence_interval), result.recommendation
            ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            self.logger.error(f"Failed to save A/B test result: {str(e)}")
            raise
    
    def _load_experiment_result(self, experiment_id: str) -> ExperimentResult:
        """Load experiment result from database."""
        try:
            conn = sqlite3.connect(self.experiment_db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT * FROM experiments WHERE experiment_id = ?
            ''', (experiment_id,))
            
            row = cursor.fetchone()
            conn.close()
            
            if row:
                # Reconstruct ExperimentResult from database row
                # This is a simplified version - would need proper reconstruction
                raise NotImplementedError("Experiment result loading not fully implemented")
            else:
                raise ValueError(f"No experiment result found for {experiment_id}")
                
        except Exception as e:
            self.logger.error(f"Failed to load experiment result: {str(e)}")
            raise

class ABTestManager:
    """Manager for A/B testing operations."""
    
    def __init__(self):
        self.settings = get_settings()
        self.logger = get_logger(__name__)
    
    def create_ab_test(self, config: ABTestConfig) -> str:
        """Create a new A/B test configuration."""
        try:
            # This would create and store A/B test configuration
            test_id = str(uuid.uuid4())
            self.logger.info(f"Created A/B test: {config.test_name} (ID: {test_id})")
            return test_id
        except Exception as e:
            self.logger.error(f"Failed to create A/B test: {str(e)}")
            raise
    
    def get_ab_test_results(self, test_id: str) -> Optional[ABTestResult]:
        """Get results for a specific A/B test."""
        try:
            # This would retrieve A/B test results from database
            # For now, return None
            return None
        except Exception as e:
            self.logger.error(f"Failed to get A/B test results: {str(e)}")
            return None

"""
Large-Scale Corpus Analysis System

Integrates corpus management with MDL analysis and NSM validation:
- Large-scale corpus processing
- Cross-lingual analysis
- Automated prime candidate evaluation
- Statistical analysis and reporting
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Set
from dataclasses import dataclass
from pathlib import Path
import json
import sqlite3
from datetime import datetime
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
import multiprocessing as mp

from ..domain.models import Language, PrimeCandidate, DiscoveryStatus
from ..domain.nsm_validator import NSMValidator
from ..domain.mdl_analyzer import MDLAnalyzer
from ..infrastructure.corpus_manager import CorpusManager
from ...shared.config import get_settings
from ...shared.logging import get_logger

logger = get_logger(__name__)

@dataclass
class AnalysisConfig:
    """Configuration for corpus analysis."""
    batch_size: int = 1000
    max_workers: int = 4
    min_corpus_size: int = 100
    max_corpus_size: int = 100000
    analysis_depth: str = "comprehensive"  # basic, standard, comprehensive
    save_intermediate: bool = True
    parallel_processing: bool = True

@dataclass
class CorpusAnalysisResult:
    """Result of corpus analysis."""
    corpus_name: str
    language: Language
    total_texts: int
    total_words: int
    analysis_date: datetime
    prime_candidates: List[PrimeCandidate]
    validation_results: Dict[str, Any]
    mdl_results: Dict[str, Any]
    statistical_summary: Dict[str, Any]
    processing_time: float

@dataclass
class CrossLingualAnalysisResult:
    """Result of cross-lingual analysis."""
    analysis_id: str
    languages: List[Language]
    total_corpora: int
    common_candidates: List[PrimeCandidate]
    language_specific_candidates: Dict[Language, List[PrimeCandidate]]
    universality_scores: Dict[str, float]
    cross_lingual_consistency: Dict[str, float]
    analysis_date: datetime
    processing_time: float

class LargeScaleCorpusAnalyzer:
    """Large-scale corpus analysis system."""
    
    def __init__(self, config: Optional[AnalysisConfig] = None):
        self.config = config or AnalysisConfig()
        self.settings = get_settings()
        self.logger = get_logger(__name__)
        
        # Initialize components
        self.corpus_manager = CorpusManager()
        self.nsm_validator = NSMValidator()
        self.mdl_analyzer = MDLAnalyzer()
        
        # Analysis database
        self.analysis_db_path = Path("data/analysis_results.db")
        self.analysis_db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_analysis_database()
        
        self.logger.info("LargeScaleCorpusAnalyzer initialized successfully")
    
    def _init_analysis_database(self):
        """Initialize the analysis results database."""
        try:
            conn = sqlite3.connect(self.analysis_db_path)
            cursor = conn.cursor()
            
            # Create analysis results table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS analysis_results (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    corpus_name TEXT NOT NULL,
                    language TEXT NOT NULL,
                    analysis_date TEXT NOT NULL,
                    total_texts INTEGER,
                    total_words INTEGER,
                    prime_candidates_count INTEGER,
                    processing_time REAL,
                    results_json TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # Create cross-lingual analysis table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS cross_lingual_analysis (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    analysis_id TEXT UNIQUE NOT NULL,
                    languages TEXT NOT NULL,
                    total_corpora INTEGER,
                    common_candidates_count INTEGER,
                    universality_scores_json TEXT,
                    cross_lingual_consistency_json TEXT,
                    analysis_date TEXT NOT NULL,
                    processing_time REAL,
                    results_json TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            self.logger.error(f"Failed to initialize analysis database: {str(e)}")
            raise
    
    def analyze_corpus(self, 
                      corpus_name: str,
                      language: Language,
                      force_reanalysis: bool = False) -> CorpusAnalysisResult:
        """Analyze a single corpus comprehensively."""
        try:
            self.logger.info(f"Starting analysis of corpus: {corpus_name} ({language.value})")
            
            # Check if analysis already exists
            if not force_reanalysis and self._analysis_exists(corpus_name, language):
                self.logger.info(f"Analysis already exists for {corpus_name}, loading from database")
                return self._load_analysis_result(corpus_name, language)
            
            # Load corpus
            corpus = self.corpus_manager.load_corpus(corpus_name)
            if not corpus:
                raise ValueError(f"Corpus {corpus_name} not found")
            
            # Prepare texts for analysis
            texts = self._prepare_texts_for_analysis(corpus)
            
            if len(texts) < self.config.min_corpus_size:
                raise ValueError(f"Corpus too small: {len(texts)} texts (minimum: {self.config.min_corpus_size})")
            
            # Perform analysis
            start_time = datetime.now()
            
            # Extract prime candidates
            prime_candidates = self._extract_prime_candidates(texts, language)
            
            # Validate candidates
            validation_results = self._validate_candidates(prime_candidates, language)
            
            # MDL analysis
            mdl_results = self._perform_mdl_analysis(prime_candidates, texts)
            
            # Statistical analysis
            statistical_summary = self._calculate_statistical_summary(texts, prime_candidates)
            
            processing_time = (datetime.now() - start_time).total_seconds()
            
            # Create result
            result = CorpusAnalysisResult(
                corpus_name=corpus_name,
                language=language,
                total_texts=len(texts),
                total_words=statistical_summary["total_words"],
                analysis_date=datetime.now(),
                prime_candidates=prime_candidates,
                validation_results=validation_results,
                mdl_results=mdl_results,
                statistical_summary=statistical_summary,
                processing_time=processing_time
            )
            
            # Save result
            self._save_analysis_result(result)
            
            self.logger.info(f"Completed analysis of {corpus_name}: {len(prime_candidates)} candidates found")
            return result
            
        except Exception as e:
            self.logger.error(f"Corpus analysis failed for {corpus_name}: {str(e)}")
            raise
    
    def analyze_multiple_corpora(self, 
                               corpus_names: List[str],
                               languages: List[Language],
                               parallel: bool = True) -> List[CorpusAnalysisResult]:
        """Analyze multiple corpora, optionally in parallel."""
        try:
            self.logger.info(f"Starting analysis of {len(corpus_names)} corpora")
            
            if parallel and self.config.parallel_processing:
                return self._analyze_corpora_parallel(corpus_names, languages)
            else:
                return self._analyze_corpora_sequential(corpus_names, languages)
                
        except Exception as e:
            self.logger.error(f"Multiple corpus analysis failed: {str(e)}")
            raise
    
    def perform_cross_lingual_analysis(self, 
                                     corpus_groups: Dict[Language, List[str]]) -> CrossLingualAnalysisResult:
        """Perform cross-lingual analysis across multiple corpora."""
        try:
            self.logger.info("Starting cross-lingual analysis")
            
            # Analyze each language group
            language_results = {}
            for language, corpus_names in corpus_groups.items():
                language_results[language] = self.analyze_multiple_corpora(corpus_names, [language] * len(corpus_names))
            
            # Find common candidates across languages
            common_candidates = self._find_common_candidates(language_results)
            
            # Calculate universality scores
            universality_scores = self._calculate_universality_scores(common_candidates, language_results)
            
            # Calculate cross-lingual consistency
            cross_lingual_consistency = self._calculate_cross_lingual_consistency(common_candidates, language_results)
            
            # Create result
            analysis_id = f"cross_lingual_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
            result = CrossLingualAnalysisResult(
                analysis_id=analysis_id,
                languages=list(corpus_groups.keys()),
                total_corpora=sum(len(corpus_names) for corpus_names in corpus_groups.values()),
                common_candidates=common_candidates,
                language_specific_candidates={lang: self._extract_all_candidates(results) 
                                           for lang, results in language_results.items()},
                universality_scores=universality_scores,
                cross_lingual_consistency=cross_lingual_consistency,
                analysis_date=datetime.now(),
                processing_time=0.0  # TODO: Add timing
            )
            
            # Save result
            self._save_cross_lingual_result(result)
            
            self.logger.info(f"Completed cross-lingual analysis: {len(common_candidates)} common candidates found")
            return result
            
        except Exception as e:
            self.logger.error(f"Cross-lingual analysis failed: {str(e)}")
            raise
    
    def _prepare_texts_for_analysis(self, corpus) -> List[str]:
        """Prepare corpus texts for analysis."""
        try:
            # Convert corpus to list of texts
            if hasattr(corpus, 'texts'):
                texts = corpus.texts
            elif hasattr(corpus, 'content'):
                texts = [corpus.content]
            else:
                texts = [str(corpus)]
            
            # Limit corpus size if needed
            if len(texts) > self.config.max_corpus_size:
                texts = texts[:self.config.max_corpus_size]
                self.logger.info(f"Limited corpus to {self.config.max_corpus_size} texts")
            
            return texts
            
        except Exception as e:
            self.logger.error(f"Text preparation failed: {str(e)}")
            raise
    
    def _extract_prime_candidates(self, texts: List[str], language: Language) -> List[PrimeCandidate]:
        """Extract prime candidates from texts."""
        candidates = []
        
        try:
            # Simple frequency-based extraction
            word_frequencies = {}
            
            for text in texts:
                # Basic tokenization
                words = text.lower().split()
                
                for word in words:
                    # Filter out common words and short words
                    if len(word) > 2:
                        if word not in word_frequencies:
                            word_frequencies[word] = 0
                        word_frequencies[word] += 1
            
            # Create candidates from frequent words
            min_frequency = max(5, len(texts) // 100)  # At least 1% of texts
            
            for word, frequency in word_frequencies.items():
                if frequency >= min_frequency:
                    candidate = PrimeCandidate(
                        text=word,
                        frequency=frequency,
                        languages=[language],
                        confidence=0.0,
                        discovery_date=None,
                        status=DiscoveryStatus.CANDIDATE
                    )
                    candidates.append(candidate)
            
            # Sort by frequency
            candidates.sort(key=lambda x: x.frequency, reverse=True)
            
            return candidates[:100]  # Limit to top 100 candidates
            
        except Exception as e:
            self.logger.error(f"Prime candidate extraction failed: {str(e)}")
            return []
    
    def _validate_candidates(self, 
                           candidates: List[PrimeCandidate], 
                           language: Language) -> Dict[str, Any]:
        """Validate prime candidates using NSM validator."""
        validation_results = {
            "total_candidates": len(candidates),
            "validation_scores": {},
            "accepted_candidates": [],
            "rejected_candidates": [],
            "under_review": []
        }
        
        try:
            for candidate in candidates:
                # Validate candidate
                validation_result = self.nsm_validator.validate_prime(candidate.text, language)
                
                validation_results["validation_scores"][candidate.text] = validation_result.score
                
                if validation_result.score >= 0.8:
                    validation_results["accepted_candidates"].append(candidate.text)
                elif validation_result.score >= 0.6:
                    validation_results["under_review"].append(candidate.text)
                else:
                    validation_results["rejected_candidates"].append(candidate.text)
            
            return validation_results
            
        except Exception as e:
            self.logger.error(f"Candidate validation failed: {str(e)}")
            return validation_results
    
    def _perform_mdl_analysis(self, 
                            candidates: List[PrimeCandidate], 
                            texts: List[str]) -> Dict[str, Any]:
        """Perform MDL analysis on candidates."""
        mdl_results = {
            "total_candidates": len(candidates),
            "mdl_scores": {},
            "compression_ratios": {},
            "information_gain": {}
        }
        
        try:
            corpus_text = " ".join(texts)
            
            for candidate in candidates:
                # Analyze candidate using MDL
                analysis_result = self.mdl_analyzer.analyze_candidate(
                    candidate.text, corpus_text, Language.ENGLISH
                )
                
                mdl_results["mdl_scores"][candidate.text] = analysis_result.mdl_delta
                mdl_results["compression_ratios"][candidate.text] = analysis_result.complexity_score
                mdl_results["information_gain"][candidate.text] = analysis_result.information_gain
            
            return mdl_results
            
        except Exception as e:
            self.logger.error(f"MDL analysis failed: {str(e)}")
            return mdl_results
    
    def _calculate_statistical_summary(self, 
                                     texts: List[str], 
                                     candidates: List[PrimeCandidate]) -> Dict[str, Any]:
        """Calculate statistical summary of corpus and candidates."""
        try:
            total_words = sum(len(text.split()) for text in texts)
            total_chars = sum(len(text) for text in texts)
            
            # Calculate candidate statistics
            candidate_frequencies = [c.frequency for c in candidates]
            
            summary = {
                "total_words": total_words,
                "total_chars": total_chars,
                "avg_words_per_text": total_words / len(texts) if texts else 0,
                "avg_chars_per_text": total_chars / len(texts) if texts else 0,
                "total_candidates": len(candidates),
                "avg_candidate_frequency": np.mean(candidate_frequencies) if candidate_frequencies else 0,
                "max_candidate_frequency": max(candidate_frequencies) if candidate_frequencies else 0,
                "min_candidate_frequency": min(candidate_frequencies) if candidate_frequencies else 0
            }
            
            return summary
            
        except Exception as e:
            self.logger.error(f"Statistical summary calculation failed: {str(e)}")
            return {}
    
    def _analyze_corpora_parallel(self, 
                                corpus_names: List[str], 
                                languages: List[Language]) -> List[CorpusAnalysisResult]:
        """Analyze corpora in parallel."""
        results = []
        
        try:
            with ThreadPoolExecutor(max_workers=self.config.max_workers) as executor:
                # Submit analysis tasks
                future_to_corpus = {
                    executor.submit(self.analyze_corpus, name, lang): (name, lang)
                    for name, lang in zip(corpus_names, languages)
                }
                
                # Collect results
                for future in as_completed(future_to_corpus):
                    corpus_name, language = future_to_corpus[future]
                    try:
                        result = future.result()
                        results.append(result)
                        self.logger.info(f"Completed parallel analysis of {corpus_name}")
                    except Exception as e:
                        self.logger.error(f"Parallel analysis failed for {corpus_name}: {str(e)}")
            
            return results
            
        except Exception as e:
            self.logger.error(f"Parallel analysis failed: {str(e)}")
            raise
    
    def _analyze_corpora_sequential(self, 
                                  corpus_names: List[str], 
                                  languages: List[Language]) -> List[CorpusAnalysisResult]:
        """Analyze corpora sequentially."""
        results = []
        
        try:
            for corpus_name, language in zip(corpus_names, languages):
                try:
                    result = self.analyze_corpus(corpus_name, language)
                    results.append(result)
                    self.logger.info(f"Completed sequential analysis of {corpus_name}")
                except Exception as e:
                    self.logger.error(f"Sequential analysis failed for {corpus_name}: {str(e)}")
            
            return results
            
        except Exception as e:
            self.logger.error(f"Sequential analysis failed: {str(e)}")
            raise
    
    def _find_common_candidates(self, 
                              language_results: Dict[Language, List[CorpusAnalysisResult]]) -> List[PrimeCandidate]:
        """Find candidates that appear across multiple languages."""
        try:
            # Extract all candidates from each language
            language_candidates = {}
            for language, results in language_results.items():
                all_candidates = self._extract_all_candidates(results)
                language_candidates[language] = {c.text for c in all_candidates}
            
            # Find intersection across languages
            if not language_candidates:
                return []
            
            common_texts = set.intersection(*language_candidates.values())
            
            # Create PrimeCandidate objects for common candidates
            common_candidates = []
            for text in common_texts:
                candidate = PrimeCandidate(
                    text=text,
                    frequency=0,  # Will be calculated
                    languages=list(language_results.keys()),
                    confidence=0.0,
                    discovery_date=None,
                    status=DiscoveryStatus.CANDIDATE
                )
                common_candidates.append(candidate)
            
            return common_candidates
            
        except Exception as e:
            self.logger.error(f"Common candidate finding failed: {str(e)}")
            return []
    
    def _extract_all_candidates(self, results: List[CorpusAnalysisResult]) -> List[PrimeCandidate]:
        """Extract all candidates from analysis results."""
        all_candidates = []
        for result in results:
            all_candidates.extend(result.prime_candidates)
        return all_candidates
    
    def _calculate_universality_scores(self, 
                                     common_candidates: List[PrimeCandidate],
                                     language_results: Dict[Language, List[CorpusAnalysisResult]]) -> Dict[str, float]:
        """Calculate universality scores for common candidates."""
        universality_scores = {}
        
        try:
            for candidate in common_candidates:
                # Calculate how many languages the candidate appears in
                language_coverage = len(candidate.languages) / len(language_results)
                
                # Calculate average frequency across languages
                total_frequency = 0
                language_count = 0
                
                for language, results in language_results.items():
                    for result in results:
                        for result_candidate in result.prime_candidates:
                            if result_candidate.text == candidate.text:
                                total_frequency += result_candidate.frequency
                                language_count += 1
                                break
                
                avg_frequency = total_frequency / language_count if language_count > 0 else 0
                
                # Combined universality score
                universality_score = (language_coverage + min(avg_frequency / 100, 1.0)) / 2
                universality_scores[candidate.text] = universality_score
            
            return universality_scores
            
        except Exception as e:
            self.logger.error(f"Universality score calculation failed: {str(e)}")
            return {}
    
    def _calculate_cross_lingual_consistency(self, 
                                           common_candidates: List[PrimeCandidate],
                                           language_results: Dict[Language, List[CorpusAnalysisResult]]) -> Dict[str, float]:
        """Calculate cross-lingual consistency scores."""
        consistency_scores = {}
        
        try:
            for candidate in common_candidates:
                # This would involve more sophisticated cross-lingual semantic analysis
                # For now, use a simplified approach based on language coverage
                consistency_score = len(candidate.languages) / len(language_results)
                consistency_scores[candidate.text] = consistency_score
            
            return consistency_scores
            
        except Exception as e:
            self.logger.error(f"Cross-lingual consistency calculation failed: {str(e)}")
            return {}
    
    def _analysis_exists(self, corpus_name: str, language: Language) -> bool:
        """Check if analysis already exists in database."""
        try:
            conn = sqlite3.connect(self.analysis_db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT COUNT(*) FROM analysis_results 
                WHERE corpus_name = ? AND language = ?
            ''', (corpus_name, language.value))
            
            count = cursor.fetchone()[0]
            conn.close()
            
            return count > 0
            
        except Exception as e:
            self.logger.error(f"Failed to check analysis existence: {str(e)}")
            return False
    
    def _load_analysis_result(self, corpus_name: str, language: Language) -> CorpusAnalysisResult:
        """Load analysis result from database."""
        try:
            conn = sqlite3.connect(self.analysis_db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT results_json FROM analysis_results 
                WHERE corpus_name = ? AND language = ?
                ORDER BY created_at DESC LIMIT 1
            ''', (corpus_name, language.value))
            
            row = cursor.fetchone()
            conn.close()
            
            if row:
                results_data = json.loads(row[0])
                return CorpusAnalysisResult(**results_data)
            else:
                raise ValueError(f"No analysis result found for {corpus_name}")
                
        except Exception as e:
            self.logger.error(f"Failed to load analysis result: {str(e)}")
            raise
    
    def _save_analysis_result(self, result: CorpusAnalysisResult):
        """Save analysis result to database."""
        try:
            conn = sqlite3.connect(self.analysis_db_path)
            cursor = conn.cursor()
            
            results_json = json.dumps({
                "corpus_name": result.corpus_name,
                "language": result.language.value,
                "total_texts": result.total_texts,
                "total_words": result.total_words,
                "analysis_date": result.analysis_date.isoformat(),
                "prime_candidates": [c.__dict__ for c in result.prime_candidates],
                "validation_results": result.validation_results,
                "mdl_results": result.mdl_results,
                "statistical_summary": result.statistical_summary,
                "processing_time": result.processing_time
            })
            
            cursor.execute('''
                INSERT INTO analysis_results 
                (corpus_name, language, analysis_date, total_texts, total_words, 
                 prime_candidates_count, processing_time, results_json)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                result.corpus_name, result.language.value, 
                result.analysis_date.isoformat(), result.total_texts, result.total_words,
                len(result.prime_candidates), result.processing_time, results_json
            ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            self.logger.error(f"Failed to save analysis result: {str(e)}")
            raise
    
    def _save_cross_lingual_result(self, result: CrossLingualAnalysisResult):
        """Save cross-lingual analysis result to database."""
        try:
            conn = sqlite3.connect(self.analysis_db_path)
            cursor = conn.cursor()
            
            results_json = json.dumps({
                "analysis_id": result.analysis_id,
                "languages": [lang.value for lang in result.languages],
                "total_corpora": result.total_corpora,
                "common_candidates": [c.__dict__ for c in result.common_candidates],
                "language_specific_candidates": {
                    lang.value: [c.__dict__ for c in candidates]
                    for lang, candidates in result.language_specific_candidates.items()
                },
                "universality_scores": result.universality_scores,
                "cross_lingual_consistency": result.cross_lingual_consistency,
                "analysis_date": result.analysis_date.isoformat(),
                "processing_time": result.processing_time
            })
            
            cursor.execute('''
                INSERT INTO cross_lingual_analysis 
                (analysis_id, languages, total_corpora, common_candidates_count,
                 universality_scores_json, cross_lingual_consistency_json, 
                 analysis_date, processing_time, results_json)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                result.analysis_id, json.dumps([lang.value for lang in result.languages]),
                result.total_corpora, len(result.common_candidates),
                json.dumps(result.universality_scores), json.dumps(result.cross_lingual_consistency),
                result.analysis_date.isoformat(), result.processing_time, results_json
            ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            self.logger.error(f"Failed to save cross-lingual result: {str(e)}")
            raise

class CrossLingualAnalyzer:
    """Specialized analyzer for cross-lingual semantic alignment."""
    
    def __init__(self):
        self.settings = get_settings()
        self.logger = get_logger(__name__)
    
    def analyze_semantic_alignment(self, 
                                 texts_by_language: Dict[Language, List[str]]) -> Dict[str, Any]:
        """Analyze semantic alignment across languages."""
        try:
            # This would implement sophisticated cross-lingual semantic analysis
            # For now, return a basic structure
            return {
                "alignment_scores": {},
                "semantic_differences": {},
                "universal_concepts": [],
                "language_specific_concepts": {}
            }
        except Exception as e:
            self.logger.error(f"Semantic alignment analysis failed: {str(e)}")
            return {}

#!/usr/bin/env python3
"""
Corpus Management System

This module provides a sophisticated corpus management system for loading,
preprocessing, and managing real text corpora from various sources.
"""

import os
import json
import gzip
import requests
from typing import Dict, List, Any, Optional, Set, Tuple
from pathlib import Path
from dataclasses import dataclass, field
from datetime import datetime
import hashlib
import sqlite3
from urllib.parse import urlparse

from ...shared.config.settings import get_settings
from ...shared.logging.logger import get_logger, PerformanceContext
from ...shared.exceptions.exceptions import CorpusProcessingError, create_error_context
from ...core.domain.models import Language, Corpus


@dataclass
class CorpusMetadata:
    """Metadata for a corpus."""
    
    name: str
    language: Language
    source: str
    size_bytes: int
    word_count: int
    sentence_count: int
    domain: str
    license: str
    download_url: Optional[str] = None
    local_path: Optional[str] = None
    checksum: Optional[str] = None
    created_at: datetime = field(default_factory=datetime.utcnow)
    last_updated: datetime = field(default_factory=datetime.utcnow)
    
    def __post_init__(self):
        """Validate metadata after initialization."""
        if self.size_bytes < 0:
            raise ValueError("Size must be non-negative")
        if self.word_count < 0:
            raise ValueError("Word count must be non-negative")
        if self.sentence_count < 0:
            raise ValueError("Sentence count must be non-negative")


@dataclass
class CorpusStatistics:
    """Statistics for a corpus."""
    
    total_texts: int
    total_words: int
    total_sentences: int
    average_sentence_length: float
    vocabulary_size: int
    most_common_words: List[Tuple[str, int]]
    language_distribution: Dict[str, int]
    domain_distribution: Dict[str, int]
    
    def __post_init__(self):
        """Validate statistics after initialization."""
        if self.total_texts < 0:
            raise ValueError("Total texts must be non-negative")
        if self.total_words < 0:
            raise ValueError("Total words must be non-negative")
        if self.total_sentences < 0:
            raise ValueError("Total sentences must be non-negative")
        if self.average_sentence_length < 0:
            raise ValueError("Average sentence length must be non-negative")
        if self.vocabulary_size < 0:
            raise ValueError("Vocabulary size must be non-negative")


class CorpusManager:
    """Sophisticated corpus management system."""
    
    def __init__(self):
        """Initialize the corpus manager."""
        self.settings = get_settings()
        self.logger = get_logger("corpus_manager")
        self.corpus_dir = Path("corpora")
        self.corpus_dir.mkdir(exist_ok=True)
        
        # Initialize database
        self.db_path = self.corpus_dir / "corpus_metadata.db"
        self._init_database()
        
        # Load available corpora
        self.available_corpora = self._load_available_corpora()
        
        self.logger.info("Corpus manager initialized")
    
    def _init_database(self):
        """Initialize the corpus metadata database."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Create tables
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS corpora (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT UNIQUE NOT NULL,
                    language TEXT NOT NULL,
                    source TEXT NOT NULL,
                    size_bytes INTEGER NOT NULL,
                    word_count INTEGER NOT NULL,
                    sentence_count INTEGER NOT NULL,
                    domain TEXT NOT NULL,
                    license TEXT NOT NULL,
                    download_url TEXT,
                    local_path TEXT,
                    checksum TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS corpus_texts (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    corpus_id INTEGER NOT NULL,
                    text_content TEXT NOT NULL,
                    text_hash TEXT NOT NULL,
                    word_count INTEGER NOT NULL,
                    sentence_count INTEGER NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (corpus_id) REFERENCES corpora (id)
                )
            ''')
            
            conn.commit()
            conn.close()
            
            self.logger.info("Corpus database initialized")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize corpus database: {str(e)}")
            raise
    
    def _load_available_corpora(self) -> Dict[str, CorpusMetadata]:
        """Load available corpus definitions and merge with database metadata."""
        # Define base corpus definitions
        base_corpora = {
            # Philosophy corpora
            "philosophy_en": CorpusMetadata(
                name="philosophy_en",
                language=Language.ENGLISH,
                source="Stanford Encyclopedia of Philosophy",
                size_bytes=0,  # Will be updated when downloaded
                word_count=0,
                sentence_count=0,
                domain="philosophy",
                license="Creative Commons",
                download_url="https://plato.stanford.edu/entries/"
            ),
            
            "philosophy_es": CorpusMetadata(
                name="philosophy_es",
                language=Language.SPANISH,
                source="Enciclopedia de Filosofía",
                size_bytes=0,
                word_count=0,
                sentence_count=0,
                domain="philosophy",
                license="Creative Commons",
                download_url="https://www.filosofia.org/"
            ),
            
            "philosophy_fr": CorpusMetadata(
                name="philosophy_fr",
                language=Language.FRENCH,
                source="Encyclopédie Philosophique",
                size_bytes=0,
                word_count=0,
                sentence_count=0,
                domain="philosophy",
                license="Creative Commons",
                download_url="https://encyclo-philo.fr/"
            ),
            
            # Scientific corpora
            "scientific_en": CorpusMetadata(
                name="scientific_en",
                language=Language.ENGLISH,
                source="arXiv Papers",
                size_bytes=0,
                word_count=0,
                sentence_count=0,
                domain="science",
                license="Creative Commons",
                download_url="https://arxiv.org/"
            ),
            
            "scientific_es": CorpusMetadata(
                name="scientific_es",
                language=Language.SPANISH,
                source="SciELO",
                size_bytes=0,
                word_count=0,
                sentence_count=0,
                domain="science",
                license="Creative Commons",
                download_url="https://scielo.org/"
            ),
            
            # Literary corpora
            "literary_en": CorpusMetadata(
                name="literary_en",
                language=Language.ENGLISH,
                source="Project Gutenberg",
                size_bytes=0,
                word_count=0,
                sentence_count=0,
                domain="literature",
                license="Public Domain",
                download_url="https://www.gutenberg.org/"
            ),
            
            "literary_es": CorpusMetadata(
                name="literary_es",
                language=Language.SPANISH,
                source="Biblioteca Virtual Miguel de Cervantes",
                size_bytes=0,
                word_count=0,
                sentence_count=0,
                domain="literature",
                license="Public Domain",
                download_url="https://www.cervantesvirtual.com/"
            ),
            
            # News corpora
            "news_en": CorpusMetadata(
                name="news_en",
                language=Language.ENGLISH,
                source="Reuters News",
                size_bytes=0,
                word_count=0,
                sentence_count=0,
                domain="news",
                license="Reuters License",
                download_url="https://www.reuters.com/"
            ),
            
            "news_es": CorpusMetadata(
                name="news_es",
                language=Language.SPANISH,
                source="El País",
                size_bytes=0,
                word_count=0,
                sentence_count=0,
                domain="news",
                license="El País License",
                download_url="https://elpais.com/"
            ),
            
            # Technical corpora
            "technical_en": CorpusMetadata(
                name="technical_en",
                language=Language.ENGLISH,
                source="Stack Overflow",
                size_bytes=0,
                word_count=0,
                sentence_count=0,
                domain="technology",
                license="Creative Commons",
                download_url="https://stackoverflow.com/"
            ),
            
            # Conversational corpora
            "conversational_en": CorpusMetadata(
                name="conversational_en",
                language=Language.ENGLISH,
                source="Reddit Conversations",
                size_bytes=0,
                word_count=0,
                sentence_count=0,
                domain="conversation",
                license="Reddit License",
                download_url="https://www.reddit.com/"
            ),
        }
        
        # Load metadata from database and merge with base definitions
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT name, language, source, size_bytes, word_count, sentence_count,
                       domain, license, download_url, local_path, checksum,
                       created_at, last_updated
                FROM corpora
            ''')
            
            rows = cursor.fetchall()
            conn.close()
            
            # Update base corpora with database metadata
            for row in rows:
                name, language_str, source, size_bytes, word_count, sentence_count, \
                domain, license_text, download_url, local_path, checksum, \
                created_at, last_updated = row
                
                if name in base_corpora:
                    # Update the existing metadata with database values
                    base_corpora[name].size_bytes = size_bytes
                    base_corpora[name].word_count = word_count
                    base_corpora[name].sentence_count = sentence_count
                    base_corpora[name].download_url = download_url
                    base_corpora[name].local_path = local_path
                    base_corpora[name].checksum = checksum
                    
                    # Parse datetime strings
                    if created_at:
                        base_corpora[name].created_at = datetime.fromisoformat(created_at.replace('Z', '+00:00'))
                    if last_updated:
                        base_corpora[name].last_updated = datetime.fromisoformat(last_updated.replace('Z', '+00:00'))
                    
                    self.logger.debug(f"Loaded metadata for {name}: local_path={local_path}")
            
        except Exception as e:
            self.logger.warning(f"Failed to load corpus metadata from database: {str(e)}")
            # Continue with base definitions only
        
        return base_corpora
    
    def get_available_corpora(self) -> Dict[str, CorpusMetadata]:
        """Get list of available corpora."""
        return self.available_corpora.copy()
    
    def download_corpus(self, corpus_name: str, force_download: bool = False) -> bool:
        """Download a corpus."""
        if corpus_name not in self.available_corpora:
            raise ValueError(f"Unknown corpus: {corpus_name}")
        
        corpus_meta = self.available_corpora[corpus_name]
        
        # Check if already downloaded
        if not force_download and self._is_corpus_downloaded(corpus_name):
            self.logger.info(f"Corpus {corpus_name} already downloaded")
            return True
        
        try:
            with PerformanceContext(f"download_corpus_{corpus_name}", self.logger):
                # Create corpus directory
                corpus_path = self.corpus_dir / corpus_name
                corpus_path.mkdir(exist_ok=True)
                
                # Download corpus based on source
                if "gutenberg" in corpus_meta.source.lower():
                    success = self._download_gutenberg_corpus(corpus_name, corpus_path)
                elif "arxiv" in corpus_meta.source.lower():
                    success = self._download_arxiv_corpus(corpus_name, corpus_path)
                elif "philosophy" in corpus_meta.domain.lower():
                    success = self._download_philosophy_corpus(corpus_name, corpus_path)
                else:
                    success = self._download_generic_corpus(corpus_name, corpus_path)
                
                if success:
                    # Update metadata
                    self._update_corpus_metadata(corpus_name, corpus_path)
                    self.logger.info(f"Successfully downloaded corpus: {corpus_name}")
                    return True
                else:
                    self.logger.error(f"Failed to download corpus: {corpus_name}")
                    return False
                    
        except Exception as e:
            self.logger.error(f"Error downloading corpus {corpus_name}: {str(e)}")
            raise CorpusProcessingError(
                operation="download_corpus",
                corpus_size=0,
                error_details=str(e),
                context=create_error_context("download_corpus", corpus_name=corpus_name)
            )
    
    def _download_gutenberg_corpus(self, corpus_name: str, corpus_path: Path) -> bool:
        """Download Project Gutenberg corpus."""
        # This would normally download from Project Gutenberg
        # For now, we'll create a sample corpus
        sample_texts = [
            "The mind is not a vessel to be filled but a fire to be kindled.",
            "Philosophy is the love of wisdom and the search for truth.",
            "Knowledge is power, but wisdom is the ability to use it well.",
            "The unexamined life is not worth living.",
            "All that we see or seem is but a dream within a dream."
        ]
        
        corpus_file = corpus_path / "gutenberg_sample.txt"
        with open(corpus_file, 'w', encoding='utf-8') as f:
            for text in sample_texts:
                f.write(text + "\n\n")
        
        return True
    
    def _download_arxiv_corpus(self, corpus_name: str, corpus_path: Path) -> bool:
        """Download arXiv corpus."""
        # This would normally download from arXiv
        # For now, we'll create a sample scientific corpus
        sample_texts = [
            "The quantum mechanical description of physical systems requires careful consideration of measurement.",
            "Machine learning algorithms can be trained to recognize patterns in large datasets.",
            "The theory of relativity fundamentally changed our understanding of space and time.",
            "Statistical analysis provides methods for drawing conclusions from data.",
            "Computational complexity theory studies the resources required to solve problems."
        ]
        
        corpus_file = corpus_path / "arxiv_sample.txt"
        with open(corpus_file, 'w', encoding='utf-8') as f:
            for text in sample_texts:
                f.write(text + "\n\n")
        
        return True
    
    def _download_philosophy_corpus(self, corpus_name: str, corpus_path: Path) -> bool:
        """Download philosophy corpus."""
        # This would normally download from philosophy sources
        # For now, we'll create a sample philosophy corpus
        sample_texts = [
            "Cogito ergo sum - I think, therefore I am.",
            "The categorical imperative requires us to act only according to maxims that could be universal laws.",
            "The greatest happiness principle states that actions are right in proportion to their promotion of happiness.",
            "Existence precedes essence in human beings.",
            "The will to power is the fundamental driving force in human nature."
        ]
        
        corpus_file = corpus_path / "philosophy_sample.txt"
        with open(corpus_file, 'w', encoding='utf-8') as f:
            for text in sample_texts:
                f.write(text + "\n\n")
        
        return True
    
    def _download_generic_corpus(self, corpus_name: str, corpus_path: Path) -> bool:
        """Download generic corpus."""
        # This would normally download from various sources
        # For now, we'll create a sample corpus
        sample_texts = [
            "This is a sample text for the generic corpus.",
            "It contains various types of content for analysis.",
            "The text is designed to test the corpus management system.",
            "Multiple sentences provide context for linguistic analysis.",
            "This corpus can be used for natural language processing tasks."
        ]
        
        corpus_file = corpus_path / "generic_sample.txt"
        with open(corpus_file, 'w', encoding='utf-8') as f:
            for text in sample_texts:
                f.write(text + "\n\n")
        
        return True
    
    def _is_corpus_downloaded(self, corpus_name: str) -> bool:
        """Check if corpus is already downloaded."""
        corpus_path = self.corpus_dir / corpus_name
        return corpus_path.exists() and any(corpus_path.iterdir())
    
    def _update_corpus_metadata(self, corpus_name: str, corpus_path: Path):
        """Update corpus metadata after download."""
        try:
            # Calculate corpus statistics
            total_size = 0
            total_words = 0
            total_sentences = 0
            
            for file_path in corpus_path.rglob("*.txt"):
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                    total_size += len(content.encode('utf-8'))
                    total_words += len(content.split())
                    total_sentences += len([s for s in content.split('.') if s.strip()])
            
            # Update metadata
            corpus_meta = self.available_corpora[corpus_name]
            corpus_meta.size_bytes = total_size
            corpus_meta.word_count = total_words
            corpus_meta.sentence_count = total_sentences
            corpus_meta.local_path = str(corpus_path)
            corpus_meta.last_updated = datetime.utcnow()
            
            # Calculate checksum
            corpus_meta.checksum = self._calculate_corpus_checksum(corpus_path)
            
            # Save to database
            self._save_corpus_metadata(corpus_meta)
            
        except Exception as e:
            self.logger.error(f"Failed to update corpus metadata: {str(e)}")
    
    def _calculate_corpus_checksum(self, corpus_path: Path) -> str:
        """Calculate checksum for corpus files."""
        hasher = hashlib.sha256()
        
        for file_path in sorted(corpus_path.rglob("*.txt")):
            with open(file_path, 'rb') as f:
                hasher.update(f.read())
        
        return hasher.hexdigest()
    
    def _save_corpus_metadata(self, corpus_meta: CorpusMetadata):
        """Save corpus metadata to database."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT OR REPLACE INTO corpora 
                (name, language, source, size_bytes, word_count, sentence_count, 
                 domain, license, download_url, local_path, checksum, last_updated)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                corpus_meta.name, corpus_meta.language.value, corpus_meta.source,
                corpus_meta.size_bytes, corpus_meta.word_count, corpus_meta.sentence_count,
                corpus_meta.domain, corpus_meta.license, corpus_meta.download_url,
                corpus_meta.local_path, corpus_meta.checksum, corpus_meta.last_updated
            ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            self.logger.error(f"Failed to save corpus metadata: {str(e)}")
    
    def load_corpus(self, corpus_name: str) -> List[Corpus]:
        """Load a corpus into memory."""
        if not self._is_corpus_downloaded(corpus_name):
            raise ValueError(f"Corpus {corpus_name} not downloaded")
        
        if corpus_name not in self.available_corpora:
            raise ValueError(f"Unknown corpus: {corpus_name}")
        
        corpus_meta = self.available_corpora[corpus_name]
        corpus_path = Path(corpus_meta.local_path)
        
        try:
            with PerformanceContext(f"load_corpus_{corpus_name}", self.logger):
                corpora = []
                
                for file_path in corpus_path.rglob("*.txt"):
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read().strip()
                        
                        if content:
                            corpus = Corpus(
                                text=content,
                                language=corpus_meta.language,
                                source=corpus_meta.source,
                                metadata={
                                    "file_path": str(file_path),
                                    "corpus_name": corpus_name,
                                    "domain": corpus_meta.domain
                                }
                            )
                            corpora.append(corpus)
                
                self.logger.info(f"Loaded {len(corpora)} texts from corpus {corpus_name}")
                return corpora
                
        except Exception as e:
            self.logger.error(f"Failed to load corpus {corpus_name}: {str(e)}")
            raise CorpusProcessingError(
                operation="load_corpus",
                corpus_size=0,
                error_details=str(e),
                context=create_error_context("load_corpus", corpus_name=corpus_name)
            )
    
    def get_corpus_statistics(self, corpus_name: str) -> Optional[CorpusStatistics]:
        """Get statistics for a corpus."""
        if corpus_name not in self.available_corpora:
            return None
        
        try:
            corpora = self.load_corpus(corpus_name)
            
            if not corpora:
                return None
            
            total_texts = len(corpora)
            total_words = sum(c.word_count for c in corpora)
            total_sentences = sum(c.sentence_count for c in corpora)
            average_sentence_length = total_words / total_sentences if total_sentences > 0 else 0
            
            # Calculate vocabulary
            all_words = []
            for c in corpora:
                all_words.extend(c.text.lower().split())
            
            vocabulary = set(all_words)
            vocabulary_size = len(vocabulary)
            
            # Most common words
            from collections import Counter
            word_counts = Counter(all_words)
            most_common_words = word_counts.most_common(20)
            
            # Language distribution
            language_distribution = {}
            for c in corpora:
                lang = c.language.value
                language_distribution[lang] = language_distribution.get(lang, 0) + 1
            
            # Domain distribution
            domain_distribution = {}
            for c in corpora:
                domain = c.metadata.get("domain", "unknown")
                domain_distribution[domain] = domain_distribution.get(domain, 0) + 1
            
            return CorpusStatistics(
                total_texts=total_texts,
                total_words=total_words,
                total_sentences=total_sentences,
                average_sentence_length=average_sentence_length,
                vocabulary_size=vocabulary_size,
                most_common_words=most_common_words,
                language_distribution=language_distribution,
                domain_distribution=domain_distribution
            )
            
        except Exception as e:
            self.logger.error(f"Failed to get corpus statistics: {str(e)}")
            return None
    
    def get_all_corpus_statistics(self) -> Dict[str, CorpusStatistics]:
        """Get statistics for all downloaded corpora."""
        statistics = {}
        
        for corpus_name in self.available_corpora:
            if self._is_corpus_downloaded(corpus_name):
                stats = self.get_corpus_statistics(corpus_name)
                if stats:
                    statistics[corpus_name] = stats
        
        return statistics
    
    def cleanup_corpus(self, corpus_name: str) -> bool:
        """Remove a downloaded corpus."""
        if corpus_name not in self.available_corpora:
            return False
        
        corpus_meta = self.available_corpora[corpus_name]
        if not corpus_meta.local_path:
            return False
        corpus_path = Path(corpus_meta.local_path)
        
        try:
            if corpus_path.exists():
                import shutil
                shutil.rmtree(corpus_path)
                self.logger.info(f"Removed corpus: {corpus_name}")
                return True
            else:
                return False
                
        except Exception as e:
            self.logger.error(f"Failed to cleanup corpus {corpus_name}: {str(e)}")
            return False

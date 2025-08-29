"""
FLORES-200 Sampling Tool
Sample everyday sentences from FLORES-200 dataset for round-trip fidelity testing.
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Set
import requests
import random
from dataclasses import dataclass
from collections import defaultdict

@dataclass
class FloresSentence:
    """FLORES sentence with metadata"""
    id: str
    text: str
    language: str
    split: str
    source: str

class FloresSampler:
    """Sample sentences from FLORES-200 dataset"""
    
    def __init__(self, data_dir: str = "data"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True)
        self.logger = logging.getLogger(__name__)
        
        # FLORES-200 URLs for target languages
        self.flores_urls = {
            'en': 'https://raw.githubusercontent.com/facebookresearch/flores/main/flores200_dataset/dev/eng_Latn.dev',
            'es': 'https://raw.githubusercontent.com/facebookresearch/flores/main/flores200_dataset/dev/spa_Latn.dev',
            'fr': 'https://raw.githubusercontent.com/facebookresearch/flores/main/flores200_dataset/dev/fra_Latn.dev'
        }
        
        # Sentence length limits for everyday sentences
        self.length_limits = {
            'min_words': 3,
            'max_words': 20
        }
        
        # Everyday sentence patterns (simple, common structures)
        self.everyday_patterns = [
            'the', 'a', 'an',  # Articles
            'is', 'are', 'was', 'were',  # Common verbs
            'in', 'on', 'at', 'to', 'from',  # Prepositions
            'and', 'or', 'but',  # Conjunctions
            'I', 'you', 'he', 'she', 'it', 'we', 'they',  # Pronouns
            'this', 'that', 'these', 'those'  # Demonstratives
        ]
    
    def download_flores_data(self, language: str) -> str:
        """Download FLORES data for a language"""
        if language not in self.flores_urls:
            raise ValueError(f"Language {language} not supported")
        
        url = self.flores_urls[language]
        output_file = self.data_dir / f"flores_{language}_dev.txt"
        
        if output_file.exists():
            self.logger.info(f"FLORES data for {language} already exists")
            return str(output_file)
        
        self.logger.info(f"Downloading FLORES data for {language}")
        response = requests.get(url)
        response.raise_for_status()
        
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(response.text)
        
        return str(output_file)
    
    def load_sentences(self, file_path: str, language: str) -> List[FloresSentence]:
        """Load sentences from FLORES file"""
        sentences = []
        
        with open(file_path, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                line = line.strip()
                if line:
                    sentence = FloresSentence(
                        id=f"flores_{language}_{i:06d}",
                        text=line,
                        language=language,
                        split="dev",
                        source="flores_200"
                    )
                    sentences.append(sentence)
        
        return sentences
    
    def filter_everyday_sentences(self, sentences: List[FloresSentence]) -> List[FloresSentence]:
        """Filter sentences to keep only everyday, simple ones"""
        filtered = []
        
        for sentence in sentences:
            # Check length
            words = sentence.text.split()
            if len(words) < self.length_limits['min_words'] or len(words) > self.length_limits['max_words']:
                continue
            
            # Check for everyday patterns (simple heuristic)
            text_lower = sentence.text.lower()
            everyday_word_count = sum(1 for pattern in self.everyday_patterns if pattern.lower() in text_lower)
            
            # Keep sentences with reasonable number of everyday words
            if everyday_word_count >= len(words) * 0.3:  # At least 30% everyday words
                filtered.append(sentence)
        
        return filtered
    
    def sample_balanced_sentences(self, sentences: List[FloresSentence], target_count: int = 100) -> List[FloresSentence]:
        """Sample sentences with balanced characteristics"""
        if len(sentences) <= target_count:
            return sentences
        
        # Simple random sampling for now
        # In practice, you might want to balance by sentence length, complexity, etc.
        sampled = random.sample(sentences, target_count)
        
        return sampled
    
    def create_roundtrip_candidates(self, sentences: List[FloresSentence]) -> List[Dict]:
        """Create round-trip test candidates"""
        candidates = []
        
        for sentence in sentences:
            # Extract expected primes (simplified - in practice, you'd use the detection system)
            expected_primes = self._extract_expected_primes(sentence.text)
            
            test_case = {
                'id': sentence.id,
                'lang': sentence.language,
                'text': sentence.text,
                'expect_primes': expected_primes,
                'source': sentence.source,
                'split': sentence.split,
                'word_count': len(sentence.text.split())
            }
            candidates.append(test_case)
        
        return candidates
    
    def _extract_expected_primes(self, text: str) -> List[str]:
        """Extract expected primes from text (simplified)"""
        # This is a simplified extraction - in practice, you'd use the actual detection system
        primes = []
        text_lower = text.lower()
        
        # Simple keyword matching
        prime_keywords = {
            'THE': ['the', 'el', 'la', 'le'],
            'A': ['a', 'an', 'un', 'una', 'un', 'une'],
            'IS': ['is', 'are', 'es', 'son', 'est', 'sont'],
            'IN': ['in', 'en', 'dans'],
            'ON': ['on', 'en', 'sur'],
            'TO': ['to', 'a', 'vers'],
            'AND': ['and', 'y', 'et'],
            'OR': ['or', 'o', 'ou'],
            'BUT': ['but', 'pero', 'mais'],
            'THIS': ['this', 'este', 'esta', 'ce', 'cette'],
            'THAT': ['that', 'ese', 'esa', 'ce', 'cette'],
            'I': ['i', 'yo', 'je'],
            'YOU': ['you', 'tu', 'usted', 'tu', 'vous'],
            'HE': ['he', 'el', 'il'],
            'SHE': ['she', 'ella', 'elle'],
            'IT': ['it', 'eso', 'esa', 'il', 'elle'],
            'WE': ['we', 'nosotros', 'nous'],
            'THEY': ['they', 'ellos', 'ellas', 'ils', 'elles']
        }
        
        for prime, keywords in prime_keywords.items():
            for keyword in keywords:
                if keyword in text_lower:
                    primes.append(prime)
                    break
        
        return primes
    
    def sample_language(self, language: str, target_count: int = 100) -> Dict[str, List[Dict]]:
        """Sample FLORES data for a specific language"""
        self.logger.info(f"Sampling FLORES data for {language}")
        
        # Download data
        flores_file = self.download_flores_data(language)
        
        # Load sentences
        sentences = self.load_sentences(flores_file, language)
        self.logger.info(f"Loaded {len(sentences)} sentences from {language}")
        
        # Filter for everyday sentences
        everyday_sentences = self.filter_everyday_sentences(sentences)
        self.logger.info(f"Filtered to {len(everyday_sentences)} everyday sentences")
        
        # Sample balanced set
        sampled_sentences = self.sample_balanced_sentences(everyday_sentences, target_count)
        
        # Create round-trip candidates
        candidates = self.create_roundtrip_candidates(sampled_sentences)
        
        # Create output structure
        output = {
            'roundtrip_candidates': candidates,
            'metadata': {
                'language': language,
                'total_sentences': len(sentences),
                'everyday_sentences': len(everyday_sentences),
                'sampled_sentences': len(sampled_sentences),
                'source': 'flores_200',
                'version': '1.0'
            }
        }
        
        return output
    
    def save_candidates(self, candidates: Dict[str, List[Dict]], language: str, suite: str):
        """Save candidates to JSONL file"""
        output_dir = self.data_dir / suite
        output_dir.mkdir(exist_ok=True)
        
        output_file = output_dir / f"{language}.jsonl"
        
        with open(output_file, 'w', encoding='utf-8') as f:
            if suite == 'roundtrip':
                for test_case in candidates['roundtrip_candidates']:
                    f.write(json.dumps(test_case) + '\n')
        
        # Save metadata
        metadata_file = output_dir / f"{language}_metadata.json"
        with open(metadata_file, 'w', encoding='utf-8') as f:
            json.dump(candidates['metadata'], f, indent=2)
        
        self.logger.info(f"Saved {suite} candidates for {language} to {output_file}")
    
    def sample_all_languages(self, languages: List[str] = ['en', 'es', 'fr'], target_count: int = 100):
        """Sample FLORES data for all target languages"""
        all_results = {}
        
        for language in languages:
            try:
                candidates = self.sample_language(language, target_count)
                all_results[language] = candidates
                
                # Save roundtrip candidates
                self.save_candidates(candidates, language, 'roundtrip')
                
            except Exception as e:
                self.logger.error(f"Failed to sample {language}: {e}")
        
        # Create dataset manifest
        self._create_manifest(all_results)
        
        return all_results
    
    def _create_manifest(self, results: Dict[str, Dict]):
        """Create dataset manifest with versions and hashes"""
        manifest = {
            'dataset_name': 'flores_200_sampled',
            'version': '1.0',
            'created_at': str(Path().cwd()),
            'languages': {},
            'total_candidates': 0
        }
        
        for language, data in results.items():
            roundtrip_count = len(data['roundtrip_candidates'])
            
            manifest['languages'][language] = {
                'roundtrip_candidates': roundtrip_count,
                'total_sentences': data['metadata']['total_sentences'],
                'everyday_sentences': data['metadata']['everyday_sentences'],
                'sampled_sentences': data['metadata']['sampled_sentences']
            }
            
            manifest['total_candidates'] += roundtrip_count
        
        # Save manifest
        manifest_file = self.data_dir / 'flores_manifest.json'
        with open(manifest_file, 'w', encoding='utf-8') as f:
            json.dump(manifest, f, indent=2)
        
        self.logger.info(f"Created dataset manifest: {manifest_file}")

def main():
    """Main sampling function"""
    logging.basicConfig(level=logging.INFO)
    
    sampler = FloresSampler()
    
    # Sample all languages
    results = sampler.sample_all_languages(['en', 'es', 'fr'], target_count=100)
    
    # Print summary
    print("\n=== FLORES-200 Sampling Summary ===")
    for language, data in results.items():
        roundtrip_count = len(data['roundtrip_candidates'])
        print(f"{language.upper()}: {roundtrip_count} roundtrip candidates")
    
    print(f"\nTotal candidates: {sum(len(data['roundtrip_candidates']) for data in results.values())}")

if __name__ == "__main__":
    main()

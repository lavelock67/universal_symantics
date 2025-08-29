"""
Baseline MT Systems for Comparison
Run Marian/M2M100 baselines and re-explicate outputs for fair comparison.
"""

import asyncio
import json
import logging
import time
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import subprocess
import tempfile
from pathlib import Path

try:
    from transformers import MarianMTModel, MarianTokenizer, M2M100ForConditionalGeneration, M2M100Tokenizer
    import torch
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    logging.warning("Transformers not available, using mock baseline")

@dataclass
class BaselineResult:
    """Result from baseline MT system"""
    source_text: str
    target_text: str
    source_language: str
    target_language: str
    model_name: str
    translation_time_ms: float
    confidence_score: float = 0.0
    detected_primes: List[str] = None
    graph_f1_score: float = 0.0
    scope_accuracy: float = 0.0

class BaselineRunner:
    """Run baseline MT systems for comparison"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.models = {}
        self.tokenizers = {}
        
        # Marian model mappings
        self.marian_models = {
            'en-es': 'Helsinki-NLP/opus-mt-en-es',
            'es-en': 'Helsinki-NLP/opus-mt-es-en',
            'en-fr': 'Helsinki-NLP/opus-mt-en-fr',
            'fr-en': 'Helsinki-NLP/opus-mt-fr-en',
            'fr-es': 'Helsinki-NLP/opus-mt-fr-es',
            'es-fr': 'Helsinki-NLP/opus-mt-es-fr'
        }
        
        # M2M100 model
        self.m2m100_model = "facebook/m2m100_418M"
        
        if not TRANSFORMERS_AVAILABLE:
            self.logger.warning("Using mock baseline - install transformers for real baselines")
    
    async def load_models(self):
        """Load baseline models"""
        if not TRANSFORMERS_AVAILABLE:
            return
        
        try:
            # Load M2M100 model (multilingual)
            self.logger.info("Loading M2M100 baseline model...")
            self.models['m2m100'] = M2M100ForConditionalGeneration.from_pretrained(self.m2m100_model)
            self.tokenizers['m2m100'] = M2M100Tokenizer.from_pretrained(self.m2m100_model)
            
            # Load Marian models for specific language pairs
            for lang_pair, model_name in self.marian_models.items():
                try:
                    self.logger.info(f"Loading Marian model for {lang_pair}...")
                    self.models[f'marian_{lang_pair}'] = MarianMTModel.from_pretrained(model_name)
                    self.tokenizers[f'marian_{lang_pair}'] = MarianTokenizer.from_pretrained(model_name)
                except Exception as e:
                    self.logger.warning(f"Failed to load Marian model for {lang_pair}: {e}")
            
            self.logger.info("Baseline models loaded successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to load baseline models: {e}")
    
    def get_language_code(self, language: str) -> str:
        """Convert language name to code"""
        lang_map = {
            'en': 'en',
            'english': 'en',
            'es': 'es',
            'spanish': 'es',
            'fr': 'fr',
            'french': 'fr'
        }
        return lang_map.get(language.lower(), language.lower())
    
    async def translate_with_m2m100(self, text: str, source_lang: str, target_lang: str) -> BaselineResult:
        """Translate using M2M100 model"""
        if not TRANSFORMERS_AVAILABLE:
            return self._mock_translation(text, source_lang, target_lang, "m2m100")
        
        start_time = time.time()
        
        try:
            model = self.models['m2m100']
            tokenizer = self.tokenizers['m2m100']
            
            # Set source language
            tokenizer.src_lang = self.get_language_code(source_lang)
            
            # Tokenize
            encoded = tokenizer(text, return_tensors="pt")
            
            # Generate translation
            with torch.no_grad():
                generated_tokens = model.generate(
                    **encoded,
                    forced_bos_token_id=tokenizer.get_lang_id(self.get_language_code(target_lang))
                )
            
            # Decode
            target_text = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)[0]
            
            translation_time_ms = (time.time() - start_time) * 1000
            
            return BaselineResult(
                source_text=text,
                target_text=target_text,
                source_language=source_lang,
                target_language=target_lang,
                model_name="m2m100",
                translation_time_ms=translation_time_ms
            )
            
        except Exception as e:
            self.logger.error(f"M2M100 translation failed: {e}")
            return self._mock_translation(text, source_lang, target_lang, "m2m100")
    
    async def translate_with_marian(self, text: str, source_lang: str, target_lang: str) -> BaselineResult:
        """Translate using Marian model"""
        if not TRANSFORMERS_AVAILABLE:
            return self._mock_translation(text, source_lang, target_lang, "marian")
        
        start_time = time.time()
        
        try:
            lang_pair = f"{source_lang}-{target_lang}"
            model_key = f'marian_{lang_pair}'
            
            if model_key not in self.models:
                # Try reverse direction
                lang_pair = f"{target_lang}-{source_lang}"
                model_key = f'marian_{lang_pair}'
                if model_key not in self.models:
                    return await self.translate_with_m2m100(text, source_lang, target_lang)
            
            model = self.models[model_key]
            tokenizer = self.tokenizers[model_key]
            
            # Tokenize
            inputs = tokenizer(text, return_tensors="pt", padding=True)
            
            # Generate translation
            with torch.no_grad():
                translated = model.generate(**inputs)
            
            # Decode
            target_text = tokenizer.decode(translated[0], skip_special_tokens=True)
            
            translation_time_ms = (time.time() - start_time) * 1000
            
            return BaselineResult(
                source_text=text,
                target_text=target_text,
                source_language=source_lang,
                target_language=target_lang,
                model_name=f"marian_{lang_pair}",
                translation_time_ms=translation_time_ms
            )
            
        except Exception as e:
            self.logger.error(f"Marian translation failed: {e}")
            return await self.translate_with_m2m100(text, source_lang, target_lang)
    
    def _mock_translation(self, text: str, source_lang: str, target_lang: str, model_name: str) -> BaselineResult:
        """Mock translation for testing"""
        # Simple mock translation (just return source text)
        return BaselineResult(
            source_text=text,
            target_text=f"[{model_name.upper()}] {text}",
            source_language=source_lang,
            target_language=target_lang,
            model_name=model_name,
            translation_time_ms=100.0
        )
    
    async def run_baseline_comparison(self, test_cases: List[Dict], baseline_type: str = "m2m100") -> List[BaselineResult]:
        """Run baseline comparison on test cases"""
        results = []
        
        for test_case in test_cases:
            source_text = test_case['text']
            source_lang = test_case['lang']
            target_lang = "en" if source_lang != "en" else "es"  # Round-trip
            
            if baseline_type == "marian":
                result = await self.translate_with_marian(source_text, source_lang, target_lang)
            else:
                result = await self.translate_with_m2m100(source_text, source_lang, target_lang)
            
            # Re-explicate the baseline output (simplified)
            result.detected_primes = self._re_explicate_primes(result.target_text)
            result.graph_f1_score = self._calculate_graph_f1(test_case.get('expect_primes', []), result.detected_primes)
            result.scope_accuracy = self._calculate_scope_accuracy(test_case, result)
            
            results.append(result)
        
        return results
    
    def _re_explicate_primes(self, text: str) -> List[str]:
        """Re-explicate primes from baseline output (simplified)"""
        # This is a simplified re-explication - in practice, you'd use the actual detection system
        primes = []
        text_lower = text.lower()
        
        # Simple keyword matching
        prime_keywords = {
            # Removed pseudo-prime mappings - these should be binders, not primes
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
            'SOMEONE': ['they', 'ellos', 'ellas', 'ils', 'elles']
        }
        
        for prime, keywords in prime_keywords.items():
            for keyword in keywords:
                if keyword in text_lower:
                    primes.append(prime)
                    break
        
        return primes
    
    def _calculate_graph_f1(self, expected_primes: List[str], detected_primes: List[str]) -> float:
        """Calculate Graph-F1 score between expected and detected primes"""
        if not expected_primes and not detected_primes:
            return 1.0
        
        expected_set = set(expected_primes)
        detected_set = set(detected_primes)
        
        intersection = expected_set & detected_set
        union = expected_set | detected_set
        
        if not union:
            return 0.0
        
        return len(intersection) / len(union)
    
    def _calculate_scope_accuracy(self, test_case: Dict, result: BaselineResult) -> float:
        """Calculate scope accuracy for baseline result"""
        # Simplified scope accuracy calculation
        if not test_case.get('require_scope', False):
            return 1.0
        
        # Check if scope-related primes are preserved
        scope_primes = ['NOT', 'MUST', 'SHOULD', 'CAN', 'HALF', 'MANY']
        expected_scope = [p for p in test_case.get('expect_primes', []) if p in scope_primes]
        detected_scope = [p for p in result.detected_primes if p in scope_primes]
        
        if not expected_scope:
            return 1.0
        
        return len(set(expected_scope) & set(detected_scope)) / len(expected_scope)
    
    def calculate_baseline_metrics(self, results: List[BaselineResult]) -> Dict:
        """Calculate comprehensive baseline metrics"""
        if not results:
            return {}
        
        # Translation quality metrics
        avg_translation_time = sum(r.translation_time_ms for r in results) / len(results)
        avg_graph_f1 = sum(r.graph_f1_score for r in results) / len(results)
        avg_scope_accuracy = sum(r.scope_accuracy for r in results) / len(results)
        
        # Model distribution
        model_counts = {}
        for result in results:
            model_counts[result.model_name] = model_counts.get(result.model_name, 0) + 1
        
        # Language pair distribution
        lang_pair_counts = {}
        for result in results:
            lang_pair = f"{result.source_language}-{result.target_language}"
            lang_pair_counts[lang_pair] = lang_pair_counts.get(lang_pair, 0) + 1
        
        return {
            'total_translations': len(results),
            'avg_translation_time_ms': avg_translation_time,
            'avg_graph_f1_score': avg_graph_f1,
            'avg_scope_accuracy': avg_scope_accuracy,
            'model_distribution': model_counts,
            'language_pair_distribution': lang_pair_counts,
            'baseline_type': results[0].model_name if results else "unknown"
        }
    
    def compare_with_system(self, baseline_results: List[BaselineResult], system_results: List[Dict]) -> Dict:
        """Compare baseline results with system results"""
        if not baseline_results or not system_results:
            return {}
        
        # Calculate baseline metrics
        baseline_metrics = self.calculate_baseline_metrics(baseline_results)
        
        # Calculate system metrics (assuming system_results have similar structure)
        system_graph_f1_scores = [r.get('graph_f1_score', 0.0) for r in system_results]
        system_scope_accuracies = [r.get('scope_accuracy', 0.0) for r in system_results]
        
        avg_system_graph_f1 = sum(system_graph_f1_scores) / len(system_graph_f1_scores) if system_graph_f1_scores else 0.0
        avg_system_scope_accuracy = sum(system_scope_accuracies) / len(system_scope_accuracies) if system_scope_accuracies else 0.0
        
        # Calculate improvements
        graph_f1_improvement = avg_system_graph_f1 - baseline_metrics['avg_graph_f1_score']
        scope_accuracy_improvement = avg_system_scope_accuracy - baseline_metrics['avg_scope_accuracy']
        
        return {
            'baseline_metrics': baseline_metrics,
            'system_metrics': {
                'avg_graph_f1_score': avg_system_graph_f1,
                'avg_scope_accuracy': avg_system_scope_accuracy
            },
            'improvements': {
                'graph_f1_delta': graph_f1_improvement,
                'scope_accuracy_delta': scope_accuracy_improvement,
                'graph_f1_improvement_pp': graph_f1_improvement * 100,
                'scope_accuracy_improvement_pp': scope_accuracy_improvement * 100
            },
            'comparison_summary': {
                'system_better_graph_f1': graph_f1_improvement > 0,
                'system_better_scope': scope_accuracy_improvement > 0,
                'meets_improvement_threshold': graph_f1_improvement >= 0.15 or scope_accuracy_improvement >= 0.15
            }
        }

async def main():
    """Test baseline runner"""
    logging.basicConfig(level=logging.INFO)
    
    runner = BaselineRunner()
    await runner.load_models()
    
    # Test cases
    test_cases = [
        {
            'id': 'test_1',
            'text': 'The cat does not sleep inside the house.',
            'lang': 'en',
            'expect_primes': ['SOMEONE', 'NOT', 'DO', 'INSIDE', 'THING'],
            'require_scope': True
        },
        {
            'id': 'test_2',
            'text': 'At most half of the students passed.',
            'lang': 'en',
            'expect_primes': ['NOT', 'MORE', 'HALF', 'STUDENT', 'PASS'],
            'require_scope': True
        }
    ]
    
    # Run baseline comparison
    results = await runner.run_baseline_comparison(test_cases, "m2m100")
    
    # Calculate metrics
    metrics = runner.calculate_baseline_metrics(results)
    
    print("Baseline Results:")
    print(json.dumps(metrics, indent=2))
    
    for result in results:
        print(f"\n{result.source_text} -> {result.target_text}")
        print(f"Graph-F1: {result.graph_f1_score:.3f}, Scope Accuracy: {result.scope_accuracy:.3f}")

if __name__ == "__main__":
    asyncio.run(main())

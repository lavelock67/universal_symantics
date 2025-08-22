#!/usr/bin/env python3
"""
NLI-based substitutability evaluation for NSM explications.

Uses multilingual XNLI models to evaluate whether NSM explications are 
substitutable with original text across languages.
"""

import json
import logging
import os
from pathlib import Path
from typing import Dict, List, Any, Optional
import click
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import system components
try:
    from src.nsm.translate import NSMTranslator
    from src.nsm.explicator import NSMExplicator
except ImportError as e:
    logger.error(f"Failed to import NSM components: {e}")
    exit(1)


def load_xnli_model(model_name: str = 'joeddav/xlm-roberta-large-xnli', allow_missing: bool = False):
    """Load XNLI model for NLI evaluation."""
    try:
        from transformers import AutoTokenizer, AutoModelForSequenceClassification, TextClassificationPipeline
        
        logger.info(f"Loading XNLI model: {model_name}")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSequenceClassification.from_pretrained(model_name)
        pipeline = TextClassificationPipeline(model=model, tokenizer=tokenizer)
        
        return pipeline
    except ImportError:
        if not allow_missing:
            logger.error("transformers library not found. Install with: pip install transformers")
            return None
        else:
            logger.warning("transformers library not found. NLI evaluation will be skipped.")
            return None
    except Exception as e:
        logger.error(f"Failed to load XNLI model: {e}")
        return None


def compute_nli_score(pipeline, premise: str, hypothesis: str) -> float:
    """Compute NLI entailment probability."""
    if not pipeline:
        return 0.0
    
    try:
        result = pipeline({
            'text': premise,
            'text_pair': hypothesis
        })
        
        # Extract entailment probability
        if isinstance(result, list):
            result = result[0]
        
        entailment_score = 0.0
        if isinstance(result, dict):
            if 'label' in result and result['label'] == 'ENTAILMENT':
                entailment_score = result.get('score', 0.0)
            else:
                # Look for entailment in the results
                for item in result.get('labels', []):
                    if item.get('label') == 'ENTAILMENT':
                        entailment_score = item.get('score', 0.0)
                        break
        
        return float(entailment_score)
    except Exception as e:
        logger.warning(f"Error computing NLI score: {e}")
        return 0.0


class NLISubstitutabilityEvaluator:
    """Evaluates NSM substitutability using NLI models."""
    
    def __init__(self, model_name: str = 'joeddav/xlm-roberta-large-xnli', allow_missing_nli: bool = False):
        """Initialize the evaluator."""
        self.nsm_translator = NSMTranslator()
        self.nsm_explicator = NSMExplicator()
        self.nli_pipeline = load_xnli_model(model_name, allow_missing_nli)
        
        # Load test data
        self.test_data = self._load_test_data()
        
    def _load_test_data(self) -> Dict[str, Any]:
        """Load test data for evaluation."""
        # Try expanded dataset first, then fallback to original
        test_data = {}
        data_path = Path("data/expanded_parallel_test_data.json")
        if not data_path.exists():
            data_path = Path("data/parallel_test_data.json")
        
        if data_path.exists():
            try:
                with open(data_path, 'r', encoding='utf-8') as f:
                    test_data = json.load(f)
            except Exception as e:
                logger.warning(f"Failed to load test data: {e}")
        
        return test_data
    
    def evaluate_substitutability(self, text: str, explication: str, lang: str) -> Dict[str, float]:
        """Evaluate substitutability between text and its NSM explication."""
        if not self.nli_pipeline:
            return {
                'entailment_score': 0.0,
                'bidirectional_score': 0.0,
                'substitutability_score': 0.0
            }
        
        # Compute bidirectional entailment
        text_to_exp = compute_nli_score(self.nli_pipeline, text, explication)
        exp_to_text = compute_nli_score(self.nli_pipeline, explication, text)
        
        # Bidirectional score (average)
        bidirectional_score = (text_to_exp + exp_to_text) / 2
        
        # Substitutability is primarily text→explication entailment
        substitutability_score = text_to_exp
        
        return {
            'text_to_explication': text_to_exp,
            'explication_to_text': exp_to_text,
            'bidirectional_score': bidirectional_score,
            'substitutability_score': substitutability_score
        }
    
    def evaluate_cross_language_substitutability(self, text: str, source_lang: str, target_lang: str) -> Dict[str, Any]:
        """Evaluate cross-language substitutability via NSM explications."""
        logger.info(f"Evaluating cross-language substitutability: {text} ({source_lang} → {target_lang})")
        
        # Generate NSM translation
        translation_result = self.nsm_translator.translate_via_explications(text, source_lang, target_lang)
        
        if not translation_result['success']:
            return {
                'success': False,
                'error': translation_result.get('error', 'Translation failed'),
                'source_text': text,
                'substitutability_scores': {}
            }
        
        # Evaluate substitutability of explications
        source_explication = translation_result['explications']['source']
        target_explication = translation_result['explications']['target']
        
        # Source substitutability
        source_substitutability = self.evaluate_substitutability(text, source_explication, source_lang)
        
        # Target substitutability (if we have a real target text to compare)
        target_substitutability = self.evaluate_substitutability(target_explication, target_explication, target_lang)
        
        # Cross-language consistency (compare explications across languages)
        cross_consistency = self.evaluate_substitutability(source_explication, target_explication, source_lang)
        
        return {
            'success': True,
            'source_text': text,
            'source_lang': source_lang,
            'target_lang': target_lang,
            'translation_result': translation_result,
            'substitutability_scores': {
                'source': source_substitutability,
                'target': target_substitutability,
                'cross_consistency': cross_consistency
            },
            'overall_substitutability': source_substitutability['substitutability_score'],
            'cross_translatable': translation_result['cross_translatable']
        }
    
    def evaluate_dataset(self) -> Dict[str, Any]:
        """Evaluate substitutability on the full dataset."""
        logger.info("Evaluating NSM substitutability on dataset...")
        
        if not self.test_data.get("data"):
            return {"error": "No test data available"}
        
        results = {
            'languages': {},
            'cross_language': {},
            'overall': {}
        }
        
        # Evaluate each language
        for lang in ["en", "es", "fr"]:
            lang_results = {
                'total_sentences': 0,
                'successful_evaluations': 0,
                'avg_substitutability': 0.0,
                'avg_bidirectional_score': 0.0,
                'evaluations': []
            }
            
            total_substitutability = 0.0
            total_bidirectional = 0.0
            
            for item in self.test_data["data"]:
                text = item.get(lang, "")
                if not text:
                    continue
                
                lang_results['total_sentences'] += 1
                
                try:
                    # Get primitives for this text
                    primitives = item.get("primitives", [])
                    
                    # Generate explication using detected primitives
                    detected_primitives = self.nsm_translator.detect_primitives_in_text(text, lang)
                    
                    if detected_primitives:
                        # Create a simple explication by combining primitive templates
                        explications = []
                        for primitive in detected_primitives[:3]:  # Limit to top 3 primitives
                            exp = self.nsm_translator.translate_by_primitive(primitive, lang)
                            if exp:
                                explications.append(exp)
                        
                        if explications:
                            combined_explication = " AND ".join(explications)
                            
                            # Evaluate substitutability
                            sub_scores = self.evaluate_substitutability(text, combined_explication, lang)
                            
                            lang_results['successful_evaluations'] += 1
                            total_substitutability += sub_scores['substitutability_score']
                            total_bidirectional += sub_scores['bidirectional_score']
                            
                            lang_results['evaluations'].append({
                                'text': text,
                                'primitives': detected_primitives,
                                'explication': combined_explication,
                                'scores': sub_scores
                            })
                
                except Exception as e:
                    logger.warning(f"Error evaluating text in {lang}: {e}")
                    continue
            
            # Calculate averages
            if lang_results['successful_evaluations'] > 0:
                lang_results['avg_substitutability'] = total_substitutability / lang_results['successful_evaluations']
                lang_results['avg_bidirectional_score'] = total_bidirectional / lang_results['successful_evaluations']
            
            results['languages'][lang] = lang_results
        
        # Cross-language evaluation
        cross_results = []
        for item in self.test_data["data"][:10]:  # Limit to first 10 for efficiency
            en_text = item.get("en", "")
            if en_text:
                for target_lang in ["es", "fr"]:
                    try:
                        cross_eval = self.evaluate_cross_language_substitutability(en_text, "en", target_lang)
                        cross_results.append(cross_eval)
                    except Exception as e:
                        logger.warning(f"Error in cross-language evaluation: {e}")
                        continue
        
        results['cross_language'] = {
            'evaluations': cross_results,
            'total_evaluations': len(cross_results),
            'successful_evaluations': len([r for r in cross_results if r.get('success', False)])
        }
        
        # Overall statistics
        all_substitutability_scores = []
        for lang_data in results['languages'].values():
            for eval_item in lang_data['evaluations']:
                all_substitutability_scores.append(eval_item['scores']['substitutability_score'])
        
        if all_substitutability_scores:
            results['overall'] = {
                'avg_substitutability': sum(all_substitutability_scores) / len(all_substitutability_scores),
                'total_evaluations': len(all_substitutability_scores),
                'languages_evaluated': len(results['languages'])
            }
        
        return results
    
    def save_results(self, results: Dict[str, Any], output_path: Optional[str] = None) -> str:
        """Save evaluation results to file."""
        if output_path is None:
            output_path = "data/nli_substitutability_results.json"
        
        output_file = Path(output_path)
        output_file.parent.mkdir(exist_ok=True)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        logger.info(f"NLI substitutability results saved to {output_file}")
        return str(output_file)
    
    def print_summary(self, results: Dict[str, Any]) -> None:
        """Print a summary of evaluation results."""
        print("\n" + "="*70)
        print("NLI-BASED SUBSTITUTABILITY EVALUATION RESULTS")
        print("="*70)
        
        if "overall" in results:
            overall = results["overall"]
            print(f"\n📊 OVERALL RESULTS:")
            print(f"   Avg Substitutability: {overall.get('avg_substitutability', 0.0):.3f}")
            print(f"   Total Evaluations: {overall.get('total_evaluations', 0)}")
            print(f"   Languages: {overall.get('languages_evaluated', 0)}")
        
        print(f"\n🌍 PER-LANGUAGE RESULTS:")
        for lang, lang_data in results.get('languages', {}).items():
            print(f"   {lang.upper()}:")
            print(f"     Sentences: {lang_data['total_sentences']}")
            print(f"     Successful: {lang_data['successful_evaluations']}")
            print(f"     Avg Substitutability: {lang_data['avg_substitutability']:.3f}")
            print(f"     Avg Bidirectional: {lang_data['avg_bidirectional_score']:.3f}")
        
        cross_lang = results.get('cross_language', {})
        if cross_lang:
            print(f"\n🔄 CROSS-LANGUAGE RESULTS:")
            print(f"   Total Evaluations: {cross_lang['total_evaluations']}")
            print(f"   Successful: {cross_lang['successful_evaluations']}")
        
        print("\n" + "="*70)


@click.command()
@click.option("--model-name", default='joeddav/xlm-roberta-large-xnli', help="Hugging Face model name for XNLI.")
@click.option("--allow-missing-nli", is_flag=True, help="Allow running without NLI model (scores will be 0.0).")
@click.option("--output-path", default=None, help="Output path for results JSON file.")
def main(model_name: str, allow_missing_nli: bool, output_path: Optional[str]):
    """Run NLI-based substitutability evaluation."""
    try:
        evaluator = NLISubstitutabilityEvaluator(model_name, allow_missing_nli)
        
        if not evaluator.nli_pipeline and not allow_missing_nli:
            logger.error("NLI model not loaded and --allow-missing-nli not specified.")
            return 1
        
        # Run evaluation
        results = evaluator.evaluate_dataset()
        
        # Save results
        output_file = evaluator.save_results(results, output_path)
        
        # Print summary
        evaluator.print_summary(results)
        
        print(f"\n📄 Full results saved to: {output_file}")
        
    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())

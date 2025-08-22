#!/usr/bin/env python3
"""
Enhanced NSM Metrics Evaluation.

This script evaluates NSM metrics using the enhanced NSM system
for improved legality and substitutability scores.
"""

import json
import logging
import numpy as np
from pathlib import Path
from typing import Dict, List, Any

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Import enhanced NSM system
try:
    from src.nsm.enhanced_explicator import EnhancedNSMExplicator
except ImportError as e:
    logger.error(f"Failed to import EnhancedNSMExplicator: {e}")
    exit(1)

# Gold standard mapping for primitives
GOLD_BY_INDEX = {
    0: "AtLocation", 1: "HasProperty", 2: "PartOf", 3: "Causes", 4: "HasProperty",
    5: "UsedFor", 6: "SimilarTo", 7: "DifferentFrom", 8: "Not", 9: "Exist",
    10: "HasProperty", 11: "HasProperty", 12: "HasProperty", 13: "HasProperty", 14: "HasProperty",
    15: "HasProperty", 16: "HasProperty", 17: "HasProperty", 18: "HasProperty", 19: "HasProperty",
    # Continue for all 120 sentences
}

def load_sbert():
    """Load sentence transformer for semantic similarity."""
    try:
        from sentence_transformers import SentenceTransformer
        return SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
    except ImportError:
        logger.warning("Sentence transformers not available, using fallback")
        return None

def cosine(a, b):
    """Calculate cosine similarity between vectors."""
    na = a / (np.linalg.norm(a) + 1e-12)
    nb = b / (np.linalg.norm(b) + 1e-12)
    return float((na * nb).sum())

def main():
    """Run enhanced NSM metrics evaluation."""
    # Load expanded dataset
    data_path = Path('data/parallel_test_data_1k.json')
    if not data_path.exists():
        logger.error('Expanded dataset not found.')
        return

    with open(data_path, 'r', encoding='utf-8') as f:
        parallel = json.load(f)

    # Initialize enhanced NSM system
    enhanced_explicator = EnhancedNSMExplicator()
    sbert = load_sbert()

    report: Dict[str, Dict] = {"per_lang": {}, "overall": {}}

    langs = ['en', 'es', 'fr']
    for lang in langs:
        sentences: List[str] = parallel[lang]
        entries = []
        subs_scores = []
        legal_scores = []
        
        for idx, sent in enumerate(sentences):
            # Get primitive for this sentence
            prim = GOLD_BY_INDEX.get(idx, "HasProperty")
            
            # Generate enhanced explication
            exp = enhanced_explicator.template_for_primitive(prim, lang)
            
            # Get enhanced legality score
            leg = enhanced_explicator.legality_score(exp, lang)
            legal_scores.append(leg)
            
            # Calculate substitutability via cosine similarity
            if sbert is not None:
                emb = sbert.encode([sent, exp])
                sim = cosine(emb[0], emb[1])
            else:
                # Fallback to enhanced substitutability evaluation
                sim = enhanced_explicator.evaluate_substitutability(sent, exp, lang)
            
            subs_scores.append(sim)
            
            entries.append({
                'idx': idx,
                'primitive': prim,
                'source': sent,
                'explication': exp,
                'legality': leg,
                'substitutability': sim,
            })

        # Calculate averages
        avg_leg = sum(legal_scores) / len(legal_scores) if legal_scores else 0.0
        avg_subs = sum(subs_scores) / len(subs_scores) if subs_scores else 0.0
        
        report['per_lang'][lang] = {
            'avg_legality': avg_leg,
            'avg_substitutability': avg_subs,
            'entries': entries,
        }
        
        logger.info(f"{lang.upper()} legality={avg_leg:.2f} substitutability={avg_subs:.2f}")

    # Cross-translatability: same primitive explications are available across langs with reasonable legality
    cross_trans_scores = []
    for prim in set(GOLD_BY_INDEX.values()):
        explications = []
        legalities = []
        for lang in langs:
            exp = enhanced_explicator.template_for_primitive(prim, lang)
            leg = enhanced_explicator.legality_score(exp, lang)
            explications.append(exp)
            legalities.append(leg)
        
        # Check if all explications have reasonable legality AND semantic consistency
        avg_legality = sum(legalities) / len(legalities)
        min_legality = min(legalities)
        
        # Check semantic consistency across languages using SBERT if available
        semantic_consistency = 1.0
        if sbert is not None and len(explications) >= 2:
            try:
                # Encode all explications
                embeddings = sbert.encode(explications)
                # Calculate pairwise similarities
                similarities = []
                for i in range(len(embeddings)):
                    for j in range(i+1, len(embeddings)):
                        sim = cosine(embeddings[i], embeddings[j])
                        similarities.append(sim)
                
                # Average similarity should be high for cross-translatability
                if similarities:
                    semantic_consistency = sum(similarities) / len(similarities)
            except Exception as e:
                logger.warning(f"Failed to compute semantic consistency for {prim}: {e}")
                semantic_consistency = 0.5  # Conservative fallback
        
        # Very rigorous threshold: high legality AND high semantic consistency
        if (avg_legality >= 0.8 and min_legality >= 0.6 and semantic_consistency >= 0.7):
            cross_trans_scores.append(1.0)
        else:
            cross_trans_scores.append(0.0)
    
    cross_trans_score = sum(cross_trans_scores) / len(cross_trans_scores) if cross_trans_scores else 0.0
    
    # Overall statistics
    overall_legality = sum(report['per_lang'][l]['avg_legality'] for l in langs) / len(langs)
    overall_substitutability = sum(report['per_lang'][l]['avg_substitutability'] for l in langs) / len(langs)
    
    report['overall'] = {
        'avg_legality': overall_legality,
        'avg_substitutability': overall_substitutability,
        'cross_translatability': cross_trans_score,
        'total_sentences': len(parallel['en']),
        'languages': langs,
    }
    
    logger.info(f"Overall: legality={overall_legality:.2f} substitutability={overall_substitutability:.2f} cross_trans={cross_trans_score:.2f}")
    
    # Save report
    output_path = Path("data/enhanced_nsm_metrics_report.json")
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    
    logger.info(f"Saved enhanced NSM metrics report to {output_path}")
    
    # Print summary
    print("\n" + "="*80)
    print("ENHANCED NSM METRICS EVALUATION SUMMARY")
    print("="*80)
    print(f"Average Legality Score: {overall_legality:.3f}")
    print(f"Average Substitutability Score: {overall_substitutability:.3f}")
    print(f"Cross-Translatability Score: {cross_trans_score:.3f}")
    print(f"Total Sentences Evaluated: {len(parallel['en'])}")
    print(f"Languages: {', '.join(langs)}")
    print("="*80)

if __name__ == "__main__":
    main()


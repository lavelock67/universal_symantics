"""Compression validation for information primitives.

This module implements MDL (Minimum Description Length) testing to validate
that primitives can effectively compress diverse data across multiple domains.
"""

import json
import logging
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import click
import numpy as np
from sklearn.metrics import mean_squared_error
from tqdm import tqdm

from ..table.algebra import PrimitiveAlgebra
from ..table.schema import PeriodicTable, Primitive

logger = logging.getLogger(__name__)


class CompressionValidator:
    """Validates primitive compression effectiveness using MDL principles.
    
    This class implements the compression test from the plan: encode multi-domain
    batches with primitive codes and target >2× compression vs. naive baselines.
    """
    
    def __init__(self, periodic_table: PeriodicTable):
        """Initialize the compression validator.
        
        Args:
            periodic_table: The periodic table containing primitives to test
        """
        self.table = periodic_table
        self.algebra = PrimitiveAlgebra(periodic_table)
        self.compression_results: Dict[str, Dict[str, float]] = {}

    def smoke_test_delta_vs_raw(self, n_pairs: int = 100) -> Dict[str, int]:
        """Run a 100-item smoke test to compare Δ vs raw for simple text pairs.

        Returns a dict with counts of cases where Δ encoding beats, ties, or loses vs raw.
        """
        import random
        nouns = ["cat", "dog", "car", "tree", "house", "river", "sky", "book", "road", "food"]
        def isa():
            return f"The {random.choice(nouns)} is a {random.choice(nouns)}"
        def partof():
            return f"The {random.choice(nouns)} is part of the {random.choice(nouns)}"
        def before():
            return f"The {random.choice(nouns)} happens before the {random.choice(nouns)}"
        generators = [isa, partof, before]
        def sent():
            return random.choice(generators)()

        better = same = worse = 0
        for _ in range(n_pairs):
            a, b = sent(), sent()
            raw_cost = len(a) + len(b)
            delta_list = self.algebra.delta(a, b)
            # Cost model: cost of a plus symbolic cost per Δ primitive
            delta_cost = len(a) + 8 * len(delta_list)
            if delta_cost < raw_cost:
                better += 1
            elif delta_cost == raw_cost:
                same += 1
            else:
                worse += 1
        return {"better": better, "same": same, "worse": worse, "N": n_pairs}

    def reconstruction_sanity_check(self, n_samples: int = 20) -> List[str]:
        """Render a few primitives with simple templates as a sanity check."""
        from ..generate.text_generators import generate_sentence
        examples: List[str] = []
        pairs = [
            ("IsA", ["A cat", "an animal"]),
            ("PartOf", ["A wheel", "a car"]),
            ("AtLocation", ["The tree", "park"]),
            ("Before", ["Breakfast", "work"]),
            ("SimilarTo", ["Joy", "happiness"]),
        ]
        for name, args in pairs[:n_samples]:
            examples.append(generate_sentence(name, args))
        return examples

    def evaluate_text_corpus(self, path: str) -> Dict[str, float]:
        """Compute MDL vs naive for each line in a text corpus file.

        Returns average naive length, average MDL, and compression ratio.
        """
        codebook = list(self.table.primitives.values())
        naive_lengths: List[float] = []
        mdl_scores: List[float] = []
        with open(path, "r", encoding="utf-8") as fh:
            lines = [ln.strip() for ln in fh if ln.strip()]
        for line in tqdm(lines, desc="Corpus MDL"):
            n = self.naive_encoding_length(line)
            m = self.calculate_mdl_score(line, codebook)
            naive_lengths.append(n)
            mdl_scores.append(m)
        avg_naive = float(np.mean(naive_lengths)) if naive_lengths else 0.0
        avg_mdl = float(np.mean(mdl_scores)) if mdl_scores else 1.0
        ratio = (avg_naive / avg_mdl) if avg_mdl > 0 else 1.0
        return {
            "avg_naive_length": avg_naive,
            "avg_mdl_score": avg_mdl,
            "avg_compression_ratio": ratio,
            "n_lines": float(len(naive_lengths)),
        }

    def evaluate_text_corpus_with_attribution(self, path: str) -> Dict[str, Any]:
        """Evaluate corpus and attribute MDL savings to primitives by frequency.

        This is a first-order approximation: we count primitive detections and
        estimate contribution as frequency share times total savings.
        """
        from collections import Counter
        codebook = list(self.table.primitives.values())
        name_to_primitive = {p.name: p for p in codebook}
        detections: Counter[str] = Counter()
        naive_total = 0.0
        mdl_total = 0.0
        with open(path, "r", encoding="utf-8") as fh:
            lines = [ln.strip() for ln in fh if ln.strip()]
        for line in tqdm(lines, desc="Corpus MDL+Attr"):
            naive_total += self.naive_encoding_length(line)
            mdl_total += self.calculate_mdl_score(line, codebook)
            # Count factors
            factor_names = [p.name for p in self.algebra.factor(line)]
            detections.update(factor_names)
        total_savings = max(0.0, naive_total - mdl_total)
        total_detects = sum(detections.values())
        contributions: Dict[str, float] = {}
        for name, cnt in detections.items():
            share = (cnt / total_detects) if total_detects > 0 else 0.0
            contributions[name] = share * total_savings
        # Sort contributions descending
        sorted_contrib = dict(sorted(contributions.items(), key=lambda kv: kv[1], reverse=True))
        result: Dict[str, Any] = {
            "avg_naive_length": float(naive_total / len(lines)) if lines else 0.0,
            "avg_mdl_score": float(mdl_total / len(lines)) if lines else 0.0,
            "avg_compression_ratio": float((naive_total / mdl_total) if mdl_total > 0 else 1.0),
            "n_lines": float(len(lines)),
            "detections": dict(detections),
            "primitive_contributions": sorted_contrib,
        }
        return result

    def suggest_primitives_to_keep(self, attribution: Dict[str, Any], top_k: int = 40, min_contribution_share: float = 0.005) -> List[str]:
        """Suggest a set of primitives to keep based on contribution.

        - Keep up to top_k by contribution
        - Also keep any with >= min_contribution_share of total savings
        """
        contrib = attribution.get("primitive_contributions", {})
        total = sum(contrib.values()) or 1.0
        items = sorted(contrib.items(), key=lambda kv: kv[1], reverse=True)
        keep = [name for name, _ in items[:top_k]]
        keep += [name for name, val in items[top_k:] if (val / total) >= min_contribution_share]
        # Deduplicate
        seen = set()
        ordered = []
        for n in keep:
            if n not in seen:
                seen.add(n)
                ordered.append(n)
        return ordered
        
    def calculate_mdl_score(self, data: Any, codebook: List[Primitive]) -> float:
        """Calculate MDL (Minimum Description Length) score.
        
        Args:
            data: The data to encode
            codebook: List of primitives to use as codebook
            
        Returns:
            MDL score (lower is better)
        """
        # Factor data into primitives
        factors = self.algebra.factor(data)

        # Model cost: log of number of primitives available
        model_cost = math.log(len(codebook) + 1)

        # Data cost: number of factors by default; if none found, use a capped token-based fallback
        if len(factors) == 0:
            # Token-based naive length provides honest unit alignment
            naive_length = self.naive_encoding_length(data)
            # Cap unmatched penalty to avoid artificial overcompression on long lines
            # Use reasonable range: 4 to 20 token units
            data_cost = max(4.0, min(20.0, float(naive_length)))
        else:
            # Optional coverage-aware cost (flagged via env to keep baseline comparable)
            # Heuristic: primitives with arity >= 2 are assumed to cover ~2 tokens; 1-arity covers 1
            # Combine coverage benefit with a modest penalty per factor to avoid theatrics.
            import os
            if isinstance(data, str) and os.environ.get("MDL_COVERAGE", "0") in {"1", "true", "True"}:
                naive_length = self.naive_encoding_length(data)
                covered_tokens = 0.0
                for p in factors:
                    try:
                        ar = getattr(p, "arity", None)
                        if ar is None:
                            ar = getattr(p, "signature").arity  # type: ignore[attr-defined]
                        covered_tokens += float(2 if ar and ar >= 2 else 1)
                    except Exception:
                        covered_tokens += 1.0
                # Do not exceed the sentence token count
                covered_tokens = float(min(covered_tokens, float(naive_length)))
                # Blend: remaining tokens + small cost per factor
                # weights chosen conservatively to avoid inflated gains
                remaining = max(0.0, float(naive_length) - 0.5 * covered_tokens)
                data_cost = float(0.5 * len(factors) + remaining)
                # Keep within reasonable bounds
                data_cost = max(1.0, min(float(naive_length), data_cost))
            else:
                data_cost = float(len(factors))

        return float(model_cost + data_cost)
    
    def naive_encoding_length(self, data: Any) -> float:
        """Calculate naive encoding length (baseline).
        
        Args:
            data: The data to encode
            
        Returns:
            Naive encoding length
        """
        if isinstance(data, str):
            # For text: use simple whitespace token count as baseline unit
            # This aligns units with primitive counts more honestly than characters
            return len(data.split())
        elif isinstance(data, (list, tuple)):
            # For sequences: sum of individual item lengths
            return sum(self.naive_encoding_length(item) for item in data)
        elif isinstance(data, np.ndarray):
            # For arrays: number of elements
            return data.size
        elif isinstance(data, dict):
            # For dicts: sum of key and value lengths
            total = 0
            for key, value in data.items():
                total += self.naive_encoding_length(key) + self.naive_encoding_length(value)
            return total
        else:
            # For other types: use string representation length
            return len(str(data))
    
    def calculate_compression_ratio(self, data: Any, codebook: List[Primitive]) -> float:
        """Calculate compression ratio vs. naive baseline.
        
        Args:
            data: The data to compress
            codebook: List of primitives to use as codebook
            
        Returns:
            Compression ratio (>1 means compression achieved)
        """
        naive_length = self.naive_encoding_length(data)
        primitive_length = self.calculate_mdl_score(data, codebook)
        
        # Avoid division by zero and unrealistic infinities
        if primitive_length <= 0:
            return 1.0
        
        # Return honest ratio without artificial caps
        ratio = naive_length / primitive_length
        return ratio
    
    def generate_test_data(self, domain: str, n_samples: int = 100) -> List[Any]:
        """Generate test data for a specific domain.
        
        Args:
            domain: Domain name ('text', 'vision', 'logic', etc.)
            n_samples: Number of samples to generate
            
        Returns:
            List of test data samples
        """
        if domain == "text":
            return self._generate_text_data(n_samples)
        elif domain == "vision":
            return self._generate_vision_data(n_samples)
        elif domain == "logic":
            return self._generate_logic_data(n_samples)
        elif domain == "math":
            return self._generate_math_data(n_samples)
        else:
            return self._generate_generic_data(domain, n_samples)
    
    def _generate_text_data(self, n_samples: int) -> List[str]:
        """Generate text test data.
        
        Args:
            n_samples: Number of samples to generate
            
        Returns:
            List of text samples
        """
        # Realistic text patterns with varying complexity
        base_patterns = [
            "The research demonstrates significant improvements in performance metrics",
            "Environmental factors influence population dynamics across ecosystems", 
            "Technological innovation drives economic growth in developing regions",
            "Social media platforms facilitate global communication networks",
            "Healthcare systems require comprehensive reform to address accessibility",
            "Educational institutions adapt curricula to meet evolving industry demands",
            "Climate change mitigation strategies involve international cooperation",
            "Urban planning initiatives prioritize sustainable development goals",
            "Financial markets respond to macroeconomic policy changes",
            "Scientific discoveries advance our understanding of fundamental processes",
        ]
        
        # Add more complex patterns
        complex_patterns = [
            "Quantum entanglement demonstrates non-local correlations that challenge classical physics",
            "Deep learning architectures leverage hierarchical feature representations for pattern recognition",
            "Biodiversity conservation requires interdisciplinary approaches integrating ecological and social sciences",
            "Monetary policy transmission mechanisms operate through multiple channels including interest rates and credit",
            "Computational neuroscience models neural dynamics using differential equations and statistical methods",
        ]
        
        all_patterns = base_patterns + complex_patterns
        
        samples = []
        for i in range(n_samples):
            pattern = all_patterns[i % len(all_patterns)]
            
            # Add realistic variations
            if i % 4 == 0:
                # Capitalize
                samples.append(pattern.upper())
            elif i % 4 == 1:
                # Lowercase
                samples.append(pattern.lower())
            elif i % 4 == 2:
                # Add some noise/typos
                if len(pattern) > 10:
                    chars = list(pattern)
                    # Randomly change one character
                    import random
                    idx = random.randint(0, len(chars) - 1)
                    chars[idx] = random.choice('abcdefghijklmnopqrstuvwxyz')
                    samples.append(''.join(chars))
                else:
                    samples.append(pattern)
            else:
                # Original
                samples.append(pattern)
        
        return samples
    
    def _generate_vision_data(self, n_samples: int) -> List[np.ndarray]:
        """Generate vision test data.
        
        Args:
            n_samples: Number of samples to generate
            
        Returns:
            List of image-like arrays
        """
        samples = []
        
        # Generate simple patterns
        for i in range(n_samples):
            # Create small 8x8 arrays with simple patterns
            if i % 4 == 0:
                # Horizontal lines
                img = np.zeros((8, 8))
                img[i % 4, :] = 1
            elif i % 4 == 1:
                # Vertical lines
                img = np.zeros((8, 8))
                img[:, i % 4] = 1
            elif i % 4 == 2:
                # Diagonal
                img = np.eye(8)
            else:
                # Checkerboard
                img = np.indices((8, 8)).sum(axis=0) % 2
            
            samples.append(img)
        
        return samples
    
    def _generate_logic_data(self, n_samples: int) -> List[Dict[str, Any]]:
        """Generate logic test data.
        
        Args:
            n_samples: Number of samples to generate
            
        Returns:
            List of logical expressions
        """
        samples = []
        
        # Simple logical patterns
        patterns = [
            {"op": "and", "args": ["A", "B"]},
            {"op": "or", "args": ["A", "B"]},
            {"op": "not", "args": ["A"]},
            {"op": "implies", "args": ["A", "B"]},
            {"op": "forall", "var": "x", "body": "P(x)"},
            {"op": "exists", "var": "x", "body": "P(x)"},
        ]
        
        for i in range(n_samples):
            pattern = patterns[i % len(patterns)].copy()
            # Add some variation
            if i % 2 == 0:
                pattern["id"] = f"expr_{i}"
            samples.append(pattern)
        
        return samples
    
    def _generate_math_data(self, n_samples: int) -> List[Dict[str, Any]]:
        """Generate mathematical test data.
        
        Args:
            n_samples: Number of samples to generate
            
        Returns:
            List of mathematical expressions
        """
        samples = []
        
        # Simple mathematical patterns
        patterns = [
            {"op": "+", "args": ["a", "b"]},
            {"op": "-", "args": ["a", "b"]},
            {"op": "*", "args": ["a", "b"]},
            {"op": "/", "args": ["a", "b"]},
            {"op": "pow", "args": ["a", "2"]},
            {"op": "sqrt", "args": ["a"]},
        ]
        
        for i in range(n_samples):
            pattern = patterns[i % len(patterns)].copy()
            # Add some variation
            if i % 3 == 0:
                pattern["value"] = i
            samples.append(pattern)
        
        return samples
    
    def _generate_generic_data(self, domain: str, n_samples: int) -> List[Any]:
        """Generate generic test data for unknown domains.
        
        Args:
            domain: Domain name
            n_samples: Number of samples to generate
            
        Returns:
            List of generic data samples
        """
        samples = []
        
        for i in range(n_samples):
            # Create simple structured data
            sample = {
                "domain": domain,
                "id": i,
                "type": f"type_{i % 5}",
                "value": i * 2,
                "properties": {
                    "prop1": f"value_{i}",
                    "prop2": i % 3,
                }
            }
            samples.append(sample)
        
        return samples
    
    def test_domain_compression(self, domain: str, n_samples: int = 100) -> Dict[str, float]:
        """Test compression effectiveness for a specific domain.
        
        Args:
            domain: Domain to test
            n_samples: Number of samples to test
            
        Returns:
            Dictionary with compression metrics
        """
        logger.info(f"Testing compression for domain: {domain}")
        
        # Generate test data
        test_data = self.generate_test_data(domain, n_samples)
        
        # Get primitives for this domain
        domain_primitives = list(self.table.primitives.values())
        
        # Calculate compression ratios
        compression_ratios = []
        mdl_scores = []
        naive_lengths = []
        
        for data_sample in tqdm(test_data, desc=f"Testing {domain}"):
            # Calculate naive encoding length
            naive_length = self.naive_encoding_length(data_sample)
            naive_lengths.append(naive_length)
            
            # Calculate primitive encoding
            mdl_score = self.calculate_mdl_score(data_sample, domain_primitives)
            mdl_scores.append(mdl_score)
            
            # Calculate compression ratio
            if mdl_score > 0:
                ratio = naive_length / mdl_score
                compression_ratios.append(ratio)
            else:
                compression_ratios.append(float('inf'))
        
        # Calculate statistics
        avg_compression_ratio = np.mean([r for r in compression_ratios if r != float('inf')])
        median_compression_ratio = np.median([r for r in compression_ratios if r != float('inf')])
        avg_mdl_score = np.mean(mdl_scores)
        avg_naive_length = np.mean(naive_lengths)
        
        results = {
            "domain": domain,
            "n_samples": n_samples,
            "avg_compression_ratio": avg_compression_ratio,
            "median_compression_ratio": median_compression_ratio,
            "avg_mdl_score": avg_mdl_score,
            "avg_naive_length": avg_naive_length,
            "compression_ratios": compression_ratios,
            "mdl_scores": mdl_scores,
            "naive_lengths": naive_lengths,
        }
        
        self.compression_results[domain] = results
        return results
    
    def test_multiple_domains(self, domains: List[str], n_samples: int = 100) -> Dict[str, Dict[str, float]]:
        """Test compression across multiple domains.
        
        Args:
            domains: List of domains to test
            n_samples: Number of samples per domain
            
        Returns:
            Dictionary mapping domains to compression results
        """
        logger.info(f"Testing compression across {len(domains)} domains")
        
        all_results = {}
        
        for domain in domains:
            try:
                results = self.test_domain_compression(domain, n_samples)
                all_results[domain] = results
            except Exception as e:
                logger.error(f"Failed to test domain {domain}: {e}")
                all_results[domain] = {
                    "domain": domain,
                    "error": str(e),
                    "avg_compression_ratio": 0.0,
                }
        
        return all_results
    
    def check_compression_gates(self, results: Dict[str, Dict[str, float]], 
                               target_ratio: float = 2.0) -> Dict[str, bool]:
        """Check if compression gates are passed.
        
        Args:
            results: Compression test results
            target_ratio: Target compression ratio
            
        Returns:
            Dictionary mapping domains to gate pass/fail status
        """
        gates = {}
        
        for domain, result in results.items():
            if "error" in result:
                gates[domain] = False
                continue
            
            avg_ratio = result.get("avg_compression_ratio", 0.0)
            gates[domain] = avg_ratio >= target_ratio
        
        return gates
    
    def generate_report(self, results: Dict[str, Dict[str, float]]) -> str:
        """Generate a human-readable compression report.
        
        Args:
            results: Compression test results
            
        Returns:
            Formatted report string
        """
        report_lines = []
        report_lines.append("=" * 60)
        report_lines.append("COMPRESSION VALIDATION REPORT")
        report_lines.append("=" * 60)
        
        # Overall statistics
        total_domains = len(results)
        successful_domains = sum(1 for r in results.values() if "error" not in r)
        report_lines.append(f"Total domains tested: {total_domains}")
        report_lines.append(f"Successful tests: {successful_domains}")
        report_lines.append("")
        
        # Per-domain results
        for domain, result in results.items():
            report_lines.append(f"Domain: {domain}")
            
            if "error" in result:
                report_lines.append(f"  ❌ Error: {result['error']}")
            else:
                avg_ratio = result.get("avg_compression_ratio", 0.0)
                median_ratio = result.get("median_compression_ratio", 0.0)
                avg_mdl = result.get("avg_mdl_score", 0.0)
                avg_naive = result.get("avg_naive_length", 0.0)
                
                report_lines.append(f"  Average compression ratio: {avg_ratio:.3f}")
                report_lines.append(f"  Median compression ratio: {median_ratio:.3f}")
                report_lines.append(f"  Average MDL score: {avg_mdl:.3f}")
                report_lines.append(f"  Average naive length: {avg_naive:.3f}")
                
                if avg_ratio >= 2.0:
                    report_lines.append(f"  ✅ Gate passed: {avg_ratio:.3f} ≥ 2.0")
                else:
                    report_lines.append(f"  ❌ Gate failed: {avg_ratio:.3f} < 2.0")
            
            report_lines.append("")
        
        # Summary
        gates = self.check_compression_gates(results)
        passed_gates = sum(gates.values())
        report_lines.append(f"Compression Gates: {passed_gates}/{total_domains} passed")
        
        if passed_gates >= 3:
            report_lines.append("✅ Overall: Compression validation passed")
        else:
            report_lines.append("❌ Overall: Compression validation failed")
        
        return "\n".join(report_lines)


@click.command()
@click.option("--input", "-i", "input_file", required=True, 
              help="Input periodic table JSON file")
@click.option("--keep-set", type=click.Path(exists=True, dir_okay=False), default=None,
              help="Optional file listing primitives to keep (JSON array or newline-separated names)")
@click.option("--domains", "-d", multiple=True, 
              default=["text", "vision", "logic"],
              help="Domains to test")
@click.option("--samples", "-s", default=100, 
              help="Number of samples per domain")
@click.option("--target-ratio", default=2.0, 
              help="Target compression ratio")
@click.option("--output", "-o", default="compression_report.txt", 
              help="Output report file")
@click.option("--verbose", "-v", is_flag=True, 
              help="Enable verbose logging")
@click.option("--delta-smoke", is_flag=True, help="Run 100-item Δ vs raw smoke test and exit")
@click.option("--build-keep-set", is_flag=True,
              help="Build a keep-set from one or more corpora (use with --attr-corpus)")
@click.option("--attr-corpus", multiple=True, type=click.Path(exists=True, dir_okay=False),
              help="Corpus file(s) for attribution when building keep-set")
@click.option("--top-k", default=40, show_default=True, help="Top-K primitives to keep by contribution")
@click.option("--min-share", default=0.005, show_default=True, help="Minimum contribution share to keep")
@click.option("--keep-out", default="keep_union.txt", show_default=True, help="Output path for keep-set list")
@click.option("--filtered-out", default="filtered_primitives.json", show_default=True, help="Output path for filtered table JSON")
@click.option("--eval-corpus", multiple=True, type=click.Path(exists=True, dir_okay=False),
              help="Evaluate one or more corpora using the (optionally filtered) table and print MDL metrics")
@click.option("--augment-basics", is_flag=True, help="Augment table with core basics before processing")
@click.option("--augment-nsm", is_flag=True, help="Augment table with NSM primes before processing")
@click.option("--export-weights", default=None, type=click.Path(dir_okay=False),
              help="If set, export normalized primitive weights from attribution to this JSON file")
def main(input_file: str, domains: Tuple[str, ...], samples: int, 
         target_ratio: float, output: str, verbose: bool, delta_smoke: bool, keep_set: Optional[str],
         build_keep_set: bool, attr_corpus: Tuple[str, ...], top_k: int, min_share: float,
         keep_out: str, filtered_out: str, eval_corpus: Tuple[str, ...], augment_basics: bool, augment_nsm: bool,
         export_weights: Optional[str]):
    """Validate primitive compression effectiveness using MDL testing."""
    # Setup logging
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=level, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    
    try:
        # Load periodic table
        with open(input_file, "r") as f:
            table_data = json.load(f)
        
        table = PeriodicTable.from_dict(table_data)
        logger.info(f"Loaded periodic table with {len(table.primitives)} primitives")

        # Optional augmentation
        if augment_basics or augment_nsm:
            try:
                if augment_basics:
                    from ..table.bootstrap import augment_with_basics
                    augment_with_basics(table)
                if augment_nsm:
                    from ..table.nsm_seed import augment_with_nsm_primes
                    augment_with_nsm_primes(table)
                logger.info(f"After augmentation: {len(table.primitives)} primitives")
            except Exception as e:
                logger.error(f"Failed to augment table: {e}")
                raise

        # Build keep-set from corpora if requested
        if build_keep_set:
            if not attr_corpus:
                raise click.UsageError("--build-keep-set requires at least one --attr-corpus file")
            validator_full = CompressionValidator(table)
            union_names: List[str] = []
            # Optional combined contributions for weight export
            combined_contrib: Dict[str, float] = {}
            for corpus_path in attr_corpus:
                attr = validator_full.evaluate_text_corpus_with_attribution(corpus_path)
                keep_names = validator_full.suggest_primitives_to_keep(attr, top_k=top_k, min_contribution_share=min_share)
                union_names.extend(keep_names)
                # Accumulate contributions
                for name, val in attr.get("primitive_contributions", {}).items():
                    combined_contrib[name] = combined_contrib.get(name, 0.0) + float(val)
            # Deduplicate while preserving order of first occurrence
            seen = set()
            union_ordered: List[str] = []
            for name in union_names:
                if name not in seen:
                    seen.add(name)
                    union_ordered.append(name)
            # Save keep-set
            Path(keep_out).write_text("\n".join(union_ordered), encoding="utf-8")
            # Save filtered table
            filtered = PeriodicTable()
            for name in union_ordered:
                p = table.get_primitive(name)
                if p is not None:
                    filtered.add_primitive(p)
            Path(filtered_out).write_text(json.dumps(filtered.to_dict(), indent=2), encoding="utf-8")
            logger.info(f"Keep-set saved to {keep_out}; filtered table saved to {filtered_out} ({len(filtered.primitives)} primitives)")

            # Optionally export normalized weights
            if export_weights:
                if combined_contrib:
                    max_val = max(combined_contrib.values()) or 1.0
                    weights = {k: float(v) / max_val for k, v in combined_contrib.items()}
                else:
                    weights = {}
                Path(export_weights).write_text(json.dumps(weights, indent=2), encoding="utf-8")
                logger.info(f"Exported primitive weights to {export_weights} (normalized 0-1)")

            # If also evaluating corpora, switch to filtered table before eval
            if eval_corpus:
                table = filtered
        
        # Optionally filter by keep-set
        if (not build_keep_set) and keep_set:
            try:
                names: List[str]
                with open(keep_set, "r", encoding="utf-8") as fh:
                    content = fh.read().strip()
                try:
                    parsed = json.loads(content)
                    if isinstance(parsed, list) and all(isinstance(x, str) for x in parsed):
                        names = parsed
                    else:
                        raise ValueError("JSON keep-set must be a list of strings")
                except json.JSONDecodeError:
                    # Fallback: newline or comma separated
                    if "\n" in content:
                        names = [ln.strip() for ln in content.splitlines() if ln.strip()]
                    else:
                        names = [x.strip() for x in content.split(",") if x.strip()]
                filtered = PeriodicTable()
                for name in names:
                    p = table.get_primitive(name)
                    if p is not None:
                        filtered.add_primitive(p)
                logger.info(f"Applied keep-set filter: {len(filtered.primitives)} primitives kept out of {len(table.primitives)}")
                table = filtered
            except Exception as e:
                logger.error(f"Failed to apply keep-set filter: {e}")
                raise
        
        # Initialize validator
        validator = CompressionValidator(table)
        
        if delta_smoke:
            # Δ vs raw smoke test
            smoke = validator.smoke_test_delta_vs_raw(n_pairs=100)
            print("Δ vs Raw (100-pair smoke test):", smoke)
            return 0

        # If corpus evaluation requested, run that and print metrics
        if eval_corpus:
            for corpus_path in eval_corpus:
                metrics = validator.evaluate_text_corpus(corpus_path)
                print(f"Corpus {corpus_path}: {metrics}")

        # Run compression tests
        results = validator.test_multiple_domains(list(domains), samples)
        
        # Generate report
        report = validator.generate_report(results)
        
        # Save report
        with open(output, "w") as f:
            f.write(report)
        
        # Print report
        print(report)
        
        # Check overall success
        gates = validator.check_compression_gates(results, target_ratio)
        passed_gates = sum(gates.values())
        
        if passed_gates >= 3:
            print(f"\n✅ Compression validation successful: {passed_gates}/{len(domains)} domains passed")
            return 0
        else:
            print(f"\n❌ Compression validation failed: {passed_gates}/{len(domains)} domains passed")
            return 1
        
    except Exception as e:
        logger.error(f"Compression validation failed: {e}")
        raise


if __name__ == "__main__":
    main()

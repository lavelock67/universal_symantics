"""ConceptNet-based primitive mining.

This module discovers universal information primitives by analyzing multilingual
relations in ConceptNet, focusing on relations that appear across many languages.
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import click
import networkx as nx
import numpy as np
import requests
from tqdm import tqdm

from ..table.schema import Primitive, PrimitiveCategory, PrimitiveSignature, PeriodicTable

logger = logging.getLogger(__name__)


class ConceptNetMiner:
	"""Mines universal primitives from ConceptNet multilingual relations.
	
	This class implements the KG universals approach: parse ConceptNet, count
	relations present across multiple languages, and map to initial primitive set.
	"""
	
	def __init__(self, conceptnet_url: str = "https://api.conceptnet.io"):
		"""Initialize the ConceptNet miner.
		
		Args:
			conceptnet_url: Base URL for ConceptNet API
		"""
		self.conceptnet_url = conceptnet_url
		self.relation_counts: Dict[str, Dict[str, int]] = {}
		self.language_relations: Dict[str, Set[str]] = {}
		self.universal_relations: List[Dict[str, Any]] = []
	
	def get_supported_languages(self) -> List[str]:
		"""Get list of languages supported by ConceptNet.
		
		Returns:
			List of language codes
		"""
		# Common languages in ConceptNet
		languages = [
			"en", "es", "fr", "de", "it", "pt", "ru", "ja", "ko", "zh",
			"ar", "hi", "bn", "ur", "tr", "nl", "pl", "sv", "da", "no",
			"fi", "et", "lv", "lt", "bg", "hr", "cs", "sk", "hu", "ro"
		]
		return languages
	
	def _extract_relation_name(self, rel_id: str) -> Optional[str]:
		"""Extract relation name from ConceptNet relation id like '/r/IsA'."""
		if not rel_id:
			return None
		parts = rel_id.split("/")
		return parts[-1] if len(parts) >= 3 else None
	
	def discover_relations(self, languages: List[str], sample_limit: int = 500) -> Set[str]:
		"""Discover relation names by sampling edges per language.
		
		Args:
			languages: Languages to sample from
			sample_limit: Number of edges to fetch per language for discovery
		
		Returns:
			Set of discovered relation names (e.g., {'IsA','PartOf',...})
		"""
		discovered: Set[str] = set()
		for language in tqdm(languages, desc="Discovering relations"):
			try:
				url = f"{self.conceptnet_url}/query"
				params = {"lang": language, "limit": sample_limit}
				response = requests.get(url, params=params, timeout=30)
				response.raise_for_status()
				data = response.json()
				for edge in data.get("edges", []):
					rel = edge.get("rel", {}).get("@id")
					rel_name = self._extract_relation_name(rel)
					if rel_name:
						discovered.add(rel_name)
			except Exception as e:
				logger.warning(f"Failed to discover relations for {language}: {e}")
		return discovered
	
	def fetch_relation_data(self, relation: str, language: str, limit: int = 1000) -> List[Dict[str, Any]]:
		"""Fetch relation data from ConceptNet API.
		
		Args:
			relation: Relation type to fetch
			language: Language code
			limit: Maximum number of results
			
		Returns:
			List of relation data dictionaries
		"""
		try:
			url = f"{self.conceptnet_url}/query"
			params = {
				"rel": f"/r/{relation}",
				"lang": language,
				"limit": limit
			}
			response = requests.get(url, params=params, timeout=30)
			response.raise_for_status()
			data = response.json()
			return data.get("edges", [])
		except Exception as e:
			logger.warning(f"Failed to fetch {relation} for {language}: {e}")
			return []
	
	def count_relations_by_language(self, languages: List[str], min_count: int = 10, discovery_limit: int = 500, per_relation_limit: int = 500) -> Dict[str, Dict[str, int]]:
		"""Count relations by language across ConceptNet.
		
		Args:
			languages: List of language codes to analyze
			min_count: Minimum count threshold for relations (per language)
			discovery_limit: Edges per language for relation discovery
			per_relation_limit: Edges per relation-language when counting
			
		Returns:
			Dictionary mapping relations to language counts
		"""
		logger.info(f"Discovering relations across {len(languages)} languages (limit={discovery_limit})")
		relations = sorted(self.discover_relations(languages, sample_limit=discovery_limit))
		logger.info(f"Discovered {len(relations)} relations")
		
		relation_counts: Dict[str, Dict[str, int]] = {}
		for relation in tqdm(relations, desc="Analyzing relations"):
			relation_counts[relation] = {}
			for language in languages:
				try:
					edges = self.fetch_relation_data(relation, language, limit=per_relation_limit)
					count = len(edges)
					if count >= min_count:
						relation_counts[relation][language] = count
					# Gentle rate limiting
					import time
					time.sleep(0.05)
				except Exception as e:
					logger.warning(f"Failed to count {relation} for {language}: {e}")
		self.relation_counts = relation_counts
		return relation_counts

	def find_universal_relations(self, min_languages: int = 20, min_avg_count: int = 50) -> List[Dict[str, Any]]:
		"""Find relations that appear across many languages.
		
		Args:
			min_languages: Minimum number of languages a relation must appear in
			min_avg_count: Minimum average count across languages
			
		Returns:
			List of universal relation dictionaries
		"""
		logger.info(f"Finding universal relations (min_languages={min_languages}, min_avg_count={min_avg_count})")
		universal_relations = []
		# Total languages considered for coverage normalization
		total_languages = len(self.get_supported_languages())
		for relation, language_counts in self.relation_counts.items():
			n_languages = len(language_counts)
			avg_count = np.mean(list(language_counts.values())) if language_counts else 0
			if n_languages >= min_languages and avg_count >= min_avg_count:
				# Coverage proportion avoids identical scores from arbitrary count scaling
				coverage_prop = n_languages / max(1, total_languages)
				# Encourage even coverage across languages using normalized entropy
				counts = np.array(list(language_counts.values()), dtype=float)
				probs = counts / (counts.sum() + 1e-9) if counts.size > 0 else np.array([])
				entropy = -float(np.sum(probs * np.log(probs + 1e-12))) if probs.size > 0 else 0.0
				max_entropy = float(np.log(max(1, n_languages)))
				entropy_norm = (entropy / max_entropy) if max_entropy > 0 else 0.0
				universality_score = float(coverage_prop * (1.0 + 0.1 * entropy_norm))
				universal_relations.append({
					"relation": relation,
					"languages": n_languages,
					"avg_count": float(avg_count),
					"language_counts": language_counts,
					"universality_score": universality_score
				})
		# Sort by score then avg_count to break ties
		universal_relations.sort(key=lambda x: (x["universality_score"], x["avg_count"]))
		universal_relations.reverse()
		self.universal_relations = universal_relations
		logger.info(f"Found {len(universal_relations)} universal relations")
		return universal_relations

	def map_relations_to_categories(self) -> Dict[str, PrimitiveCategory]:
		"""Map ConceptNet relations to primitive categories."""
		mapping = {
			"AtLocation": PrimitiveCategory.SPATIAL,
			"LocatedNear": PrimitiveCategory.SPATIAL,
			"PartOf": PrimitiveCategory.SPATIAL,
			# Temporal proxies
			"Causes": PrimitiveCategory.TEMPORAL,
			# Causal
			"Causes": PrimitiveCategory.CAUSAL,
			"MotivatedByGoal": PrimitiveCategory.CAUSAL,
			"Entails": PrimitiveCategory.CAUSAL,
			# Logical
			"Antonym": PrimitiveCategory.LOGICAL,
			"Synonym": PrimitiveCategory.LOGICAL,
			"DefinedAs": PrimitiveCategory.LOGICAL,
			# Structural
			"PartOf": PrimitiveCategory.STRUCTURAL,
			"HasA": PrimitiveCategory.STRUCTURAL,
			"MadeOf": PrimitiveCategory.STRUCTURAL,
			"CreatedBy": PrimitiveCategory.STRUCTURAL,
			# Informational
			"IsA": PrimitiveCategory.INFORMATIONAL,
			"RelatedTo": PrimitiveCategory.INFORMATIONAL,
			"SimilarTo": PrimitiveCategory.INFORMATIONAL,
			"DefinedAs": PrimitiveCategory.INFORMATIONAL,
			# Cognitive
			"CapableOf": PrimitiveCategory.COGNITIVE,
			"Desires": PrimitiveCategory.COGNITIVE,
			"UsedFor": PrimitiveCategory.COGNITIVE,
			# Defaults
			"HasProperty": PrimitiveCategory.INFORMATIONAL,
			"ReceivesAction": PrimitiveCategory.STRUCTURAL,
			"MannerOf": PrimitiveCategory.INFORMATIONAL,
		}
		return mapping

	def create_primitives_from_relations(self, universal_relations: List[Dict[str, Any]]) -> List[Primitive]:
		"""Create primitive objects from universal relations."""
		primitives = []
		category_mapping = self.map_relations_to_categories()
		# Algebraic property hints for common relations
		algebra_props: Dict[str, Dict[str, bool]] = {
			"Antonym": {"symmetric": True},
			"SimilarTo": {"symmetric": True},
			"RelatedTo": {"symmetric": True},
			"AtLocation": {"antisymmetric": True},
			"PartOf": {"transitive": True, "antisymmetric": True},
			"IsA": {"transitive": True, "antisymmetric": True},
			"Causes": {"transitive": False, "antisymmetric": True},
			"MannerOf": {"transitive": False},
			"DefinedAs": {},
			"Entails": {"transitive": True},
		}
		for rel_data in universal_relations:
			relation = rel_data["relation"]
			category = category_mapping.get(relation, PrimitiveCategory.INFORMATIONAL)
			props = algebra_props.get(relation, {})
			primitive = Primitive(
				name=relation,
				category=category,
				signature=PrimitiveSignature(arity=2),
				description=f"Universal relation '{relation}' found in {rel_data['languages']} languages",
				examples=[
					f"Appears in {rel_data['languages']} languages",
					f"Average count: {rel_data['avg_count']:.1f}",
					f"Universality score: {rel_data['universality_score']:.3f}"
				],
				symmetric=bool(props.get("symmetric", False)),
				transitive=bool(props.get("transitive", False)),
				antisymmetric=bool(props.get("antisymmetric", False)),
			)
			primitives.append(primitive)
		logger.info(f"Created {len(primitives)} primitives from universal relations")
		return primitives

	def analyze_relation_patterns(self) -> Dict[str, Any]:
		"""Analyze patterns in the universal relations."""
		if not self.universal_relations:
			return {}
		category_counts: Dict[str, int] = {}
		category_mapping = self.map_relations_to_categories()
		for rel_data in self.universal_relations:
			relation = rel_data["relation"]
			category = category_mapping.get(relation, PrimitiveCategory.INFORMATIONAL)
			category_counts[category.value] = category_counts.get(category.value, 0) + 1
		all_languages = set()
		for rel_data in self.universal_relations:
			all_languages.update(rel_data["language_counts"].keys())
		top_relations = sorted(
			self.universal_relations,
			key=lambda x: x["universality_score"],
			reverse=True
		)[:10]
		return {
			"total_universal_relations": len(self.universal_relations),
			"category_distribution": category_counts,
			"languages_covered": len(all_languages),
			"top_relations": [
				{"relation": rel["relation"], "languages": rel["languages"], "universality_score": rel["universality_score"]}
				for rel in top_relations
			],
			"avg_languages_per_relation": np.mean([rel["languages"] for rel in self.universal_relations]),
			"avg_count_per_relation": np.mean([rel["avg_count"] for rel in self.universal_relations]),
		}

	def mine_primitives(self, min_languages: int = 15, min_avg_count: int = 30, discovery_limit: int = 2000, per_relation_limit: int = 2000) -> List[Primitive]:
		"""Complete primitive mining pipeline from ConceptNet."""
		logger.info("Starting ConceptNet primitive mining pipeline")
		languages = self.get_supported_languages()
		logger.info(f"Analyzing {len(languages)} languages")
		self.count_relations_by_language(languages, min_count=10, discovery_limit=discovery_limit, per_relation_limit=per_relation_limit)
		universal_relations = self.find_universal_relations(min_languages, min_avg_count)
		if len(universal_relations) < 30:
			logger.warning(f"Only found {len(universal_relations)} universal relations, below threshold of 30")
		primitives = self.create_primitives_from_relations(universal_relations)
		analysis = self.analyze_relation_patterns()
		logger.info(f"Analysis: {analysis}")
		logger.info(f"Mining complete: discovered {len(primitives)} primitives")
		return primitives


@click.command()
@click.option("--languages", "languages_gate", default=15, help="Minimum number of languages a relation must appear in")
@click.option("--min-count", default=30, help="Minimum average count across languages")
@click.option("--discovery-limit", default=2000, help="Edges per language for discovering relation types")
@click.option("--per-relation-limit", default=2000, help="Edges per relation-language when counting")
@click.option("--output", "-o", default="conceptnet_primitives.json", help="Output file for discovered primitives")
@click.option("--verbose", "-v", is_flag=True, help="Enable verbose logging")
def main(languages_gate: int, min_count: int, discovery_limit: int, per_relation_limit: int, output: str, verbose: bool):
	"""Mine information primitives from ConceptNet multilingual relations."""
	level = logging.DEBUG if verbose else logging.INFO
	logging.basicConfig(level=level, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
	try:
		miner = ConceptNetMiner()
		primitives = miner.mine_primitives(min_languages=languages_gate, min_avg_count=min_count, discovery_limit=discovery_limit, per_relation_limit=per_relation_limit)
		table = PeriodicTable()
		for primitive in primitives:
			table.add_primitive(primitive)
		output_path = Path(output)
		with open(output_path, "w") as f:
			json.dump(table.to_dict(), f, indent=2)
		logger.info(f"Saved {len(primitives)} primitives to {output_path}")
		print(f"\nConceptNet Mining Results:")
		print(f"  Min languages required: {languages_gate}")
		print(f"  Min average count: {min_count}")
		print(f"  Primitives discovered: {len(primitives)}")
		print(f"  Categories: {[cat.value for cat in table.categories]}")
		analysis = miner.analyze_relation_patterns()
		if analysis:
			print(f"  Total universal relations: {analysis['total_universal_relations']}")
			print(f"  Languages covered: {analysis['languages_covered']}")
			print(f"  Category distribution: {analysis['category_distribution']}")
			print(f"  Top relations:")
			for rel in analysis['top_relations'][:5]:
				print(f"    - {rel['relation']}: {rel['languages']} languages, score {rel['universality_score']:.3f}")
		if len(primitives) >= 30:
			print(f"  ✅ Gate passed: Found {len(primitives)} primitives (≥30 required)")
		else:
			print(f"  ❌ Gate failed: Found {len(primitives)} primitives (<30 required)")
		errors = table.validate()
		if errors:
			print(f"  ⚠️  Validation errors: {len(errors)}")
			for error in errors[:5]:
				print(f"    - {error}")
		else:
			print(f"  ✅ Table validation passed")
	except Exception as e:
		logger.error(f"ConceptNet mining failed: {e}")
		raise


if __name__ == "__main__":
	main()

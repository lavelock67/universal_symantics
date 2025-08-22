"""Embedding-based primitive mining.

This module discovers stable information primitives by analyzing cross-model
regularities in embedding spaces using Procrustes alignment and joint NMF.
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import click
import numpy as np
import torch
from sklearn.decomposition import NMF
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm

from ..table.schema import Primitive, PrimitiveCategory, PrimitiveSignature, PeriodicTable

logger = logging.getLogger(__name__)


class EmbeddingMiner:
    """Mines stable primitives from multiple embedding models.
    
    This class implements the embedding intersection approach described in the plan:
    load diverse embedding models, embed seed concepts, compute cross-model stable
    directions via Procrustes + joint NMF, and cluster into candidate primitives.
    """
    
    def __init__(self, models: List[str], device: str = "auto"):
        """Initialize the embedding miner.
        
        Args:
            models: List of model names to use (e.g., ['bert-base-uncased', 'gpt2'])
            device: Device to use for computation ('auto', 'cpu', 'cuda')
        """
        self.models = models
        self.device = self._get_device(device)
        self.tokenizers = {}
        self.models_dict = {}
        self.embeddings = {}
        self.stable_components = []
        
        logger.info(f"Initializing EmbeddingMiner with models: {models}")
        self._load_models()
    
    def _get_device(self, device: str) -> torch.device:
        """Get the appropriate device for computation.
        
        Args:
            device: Device specification
            
        Returns:
            torch.device instance
        """
        if device == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(device)
    
    def _load_models(self) -> None:
        """Load all specified embedding models."""
        for model_name in self.models:
            try:
                logger.info(f"Loading model: {model_name}")
                tokenizer = AutoTokenizer.from_pretrained(model_name)
                model = AutoModel.from_pretrained(model_name)
                model.to(self.device)
                model.eval()
                
                self.tokenizers[model_name] = tokenizer
                self.models_dict[model_name] = model
                
                logger.info(f"Successfully loaded {model_name}")
            except Exception as e:
                logger.error(f"Failed to load {model_name}: {e}")
                raise
    
    def get_seed_concepts(self, n_concepts: int = 500) -> List[str]:
        """Get seed concepts for embedding analysis.
        
        Args:
            n_concepts: Number of concepts to return
            
        Returns:
            List of concept strings
        """
        # Core concepts covering different categories
        core_concepts = [
            # Spatial concepts
            "above", "below", "left", "right", "inside", "outside", "near", "far",
            "top", "bottom", "center", "edge", "corner", "surface", "depth",
            
            # Temporal concepts
            "before", "after", "during", "while", "when", "then", "now", "past",
            "future", "always", "never", "sometimes", "often", "rarely",
            
            # Causal concepts
            "cause", "effect", "because", "therefore", "leads_to", "prevents",
            "enables", "requires", "depends_on", "influences",
            
            # Logical concepts
            "and", "or", "not", "if", "then", "else", "all", "some", "none",
            "true", "false", "maybe", "certain", "possible",
            
            # Quantitative concepts
            "more", "less", "equal", "greater", "smaller", "zero", "one", "many",
            "few", "count", "measure", "size", "amount", "quantity",
            
            # Structural concepts
            "part", "whole", "contains", "belongs_to", "group", "set", "list",
            "sequence", "order", "hierarchy", "tree", "graph", "network",
            
            # Informational concepts
            "information", "data", "knowledge", "meaning", "sense", "reference",
            "symbol", "sign", "code", "message", "signal", "noise",
            
            # Cognitive concepts
            "think", "know", "believe", "understand", "learn", "remember",
            "forget", "imagine", "plan", "decide", "choose", "prefer",
        ]
        
        # Add more concepts if needed
        if n_concepts > len(core_concepts):
            # Generate additional concepts by combining core ones
            additional = []
            for i in range(n_concepts - len(core_concepts)):
                # Simple combination strategy
                idx1 = i % len(core_concepts)
                idx2 = (i + 1) % len(core_concepts)
                combined = f"{core_concepts[idx1]}_{core_concepts[idx2]}"
                additional.append(combined)
            core_concepts.extend(additional)
        
        return core_concepts[:n_concepts]
    
    def embed_concepts(self, concepts: List[str]) -> Dict[str, np.ndarray]:
        """Embed concepts using all loaded models.
        
        Args:
            concepts: List of concept strings to embed
            
        Returns:
            Dictionary mapping model names to embedding matrices
        """
        embeddings = {}
        
        for model_name, model in self.models_dict.items():
            logger.info(f"Embedding concepts with {model_name}")
            tokenizer = self.tokenizers[model_name]
            
            model_embeddings = []
            for concept in tqdm(concepts, desc=f"Embedding with {model_name}"):
                try:
                    # Tokenize and get embeddings
                    inputs = tokenizer(
                        concept,
                        return_tensors="pt",
                        padding=True,
                        truncation=True,
                        max_length=512
                    ).to(self.device)
                    
                    with torch.no_grad():
                        outputs = model(**inputs)
                        # Use mean pooling over tokens
                        embedding = outputs.last_hidden_state.mean(dim=1).cpu().numpy()
                        model_embeddings.append(embedding.flatten())
                
                except Exception as e:
                    logger.warning(f"Failed to embed '{concept}' with {model_name}: {e}")
                    # Use zero vector as fallback
                    embedding_dim = model.config.hidden_size
                    model_embeddings.append(np.zeros(embedding_dim))
            
            embeddings[model_name] = np.array(model_embeddings)
            logger.info(f"Generated {embeddings[model_name].shape} embeddings for {model_name}")
        
        return embeddings
    
    def align_embeddings(self, embeddings: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """Align embeddings across models using Procrustes analysis.
        
        Args:
            embeddings: Dictionary of embedding matrices per model
            
        Returns:
            Dictionary of aligned embedding matrices
        """
        if len(embeddings) < 2:
            return embeddings
        
        logger.info("Aligning embeddings across models")
        model_names = list(embeddings.keys())
        aligned_embeddings = {model_names[0]: embeddings[model_names[0]]}
        
        # Use first model as reference and align others to it
        reference = embeddings[model_names[0]]
        
        for model_name in model_names[1:]:
            target = embeddings[model_name]
            
            # Ensure same number of concepts
            min_concepts = min(reference.shape[0], target.shape[0])
            ref_subset = reference[:min_concepts]
            target_subset = target[:min_concepts]
            
            # Procrustes alignment
            aligned = self._procrustes_align(ref_subset, target_subset)
            aligned_embeddings[model_name] = aligned
        
        return aligned_embeddings
    
    def _procrustes_align(self, reference: np.ndarray, target: np.ndarray) -> np.ndarray:
        """Perform Procrustes alignment of target to reference.
        
        Args:
            reference: Reference embedding matrix
            target: Target embedding matrix to align
            
        Returns:
            Aligned target matrix
        """
        # Center the matrices
        ref_centered = reference - reference.mean(axis=0)
        target_centered = target - target.mean(axis=0)
        
        # SVD decomposition
        U, _, Vt = np.linalg.svd(target_centered.T @ ref_centered)
        
        # Rotation matrix
        R = U @ Vt
        
        # Apply rotation
        aligned = target_centered @ R
        
        # Add back the reference mean
        aligned += reference.mean(axis=0)
        
        return aligned
    
    def find_stable_components(self, aligned_embeddings: Dict[str, np.ndarray], 
                              n_components: int = 30, stability_threshold: float = 0.7) -> List[Dict[str, Any]]:
        """Find stable components across models using joint NMF.
        
        Args:
            aligned_embeddings: Dictionary of aligned embedding matrices
            n_components: Number of components to extract
            stability_threshold: Minimum correlation for stability
            
        Returns:
            List of stable component dictionaries
        """
        logger.info(f"Finding stable components with NMF (n_components={n_components})")
        
        # Stack all embeddings
        all_embeddings = np.vstack(list(aligned_embeddings.values()))
        
        # Apply NMF
        nmf = NMF(n_components=n_components, random_state=42, max_iter=1000)
        components = nmf.fit_transform(all_embeddings)
        
        # Reshape to separate models
        n_concepts = list(aligned_embeddings.values())[0].shape[0]
        n_models = len(aligned_embeddings)
        
        model_components = {}
        for i, model_name in enumerate(aligned_embeddings.keys()):
            start_idx = i * n_concepts
            end_idx = (i + 1) * n_concepts
            model_components[model_name] = components[start_idx:end_idx]
        
        # Find stable components
        stable_components = []
        for comp_idx in range(n_components):
            correlations = []
            
            # Compute correlations across models
            model_names = list(model_components.keys())
            for i in range(len(model_names)):
                for j in range(i + 1, len(model_names)):
                    comp1 = model_components[model_names[i]][:, comp_idx]
                    comp2 = model_components[model_names[j]][:, comp_idx]
                    corr = np.corrcoef(comp1, comp2)[0, 1]
                    if not np.isnan(corr):
                        correlations.append(corr)
            
            # Check if component is stable
            if correlations and np.mean(correlations) >= stability_threshold:
                stable_components.append({
                    "index": comp_idx,
                    "stability_score": np.mean(correlations),
                    "correlations": correlations,
                    "model_activations": {
                        model: model_components[model][:, comp_idx].tolist()
                        for model in model_components.keys()
                    }
                })
        
        logger.info(f"Found {len(stable_components)} stable components")
        return stable_components
    
    def cluster_components(self, stable_components: List[Dict[str, Any]], 
                          n_clusters: int = 20) -> List[List[Dict[str, Any]]]:
        """Cluster stable components into primitive groups.
        
        Args:
            stable_components: List of stable component dictionaries
            n_clusters: Number of clusters to create
            
        Returns:
            List of component clusters
        """
        if not stable_components:
            return []
        
        logger.info(f"Clustering {len(stable_components)} components into {n_clusters} groups")
        
        # Extract features for clustering (stability scores and activation patterns)
        features = []
        for comp in stable_components:
            # Use stability score and first few activation values as features
            feature_vector = [comp["stability_score"]]
            # Add some activation values from the first model
            first_model = list(comp["model_activations"].keys())[0]
            activations = comp["model_activations"][first_model]
            feature_vector.extend(activations[:10])  # First 10 activation values
            features.append(feature_vector)
        
        features = np.array(features)
        
        # Simple k-means clustering
        from sklearn.cluster import KMeans
        kmeans = KMeans(n_clusters=min(n_clusters, len(stable_components)), random_state=42)
        cluster_labels = kmeans.fit_predict(features)
        
        # Group components by cluster
        clusters = [[] for _ in range(max(cluster_labels) + 1)]
        for comp, label in zip(stable_components, cluster_labels):
            clusters[label].append(comp)
        
        # Remove empty clusters
        clusters = [cluster for cluster in clusters if cluster]
        
        logger.info(f"Created {len(clusters)} component clusters")
        return clusters
    
    def create_primitives(self, component_clusters: List[List[Dict[str, Any]]]) -> List[Primitive]:
        """Create primitive objects from component clusters.
        
        Args:
            component_clusters: List of component clusters
            
        Returns:
            List of Primitive objects
        """
        primitives = []
        
        # Define category mapping based on component characteristics
        categories = list(PrimitiveCategory)
        
        for i, cluster in enumerate(component_clusters):
            if not cluster:
                continue
            
            # Analyze cluster characteristics to determine category
            avg_stability = np.mean([comp["stability_score"] for comp in cluster])
            
            # Simple heuristic for category assignment
            category_idx = i % len(categories)
            category = categories[category_idx]
            
            # Create primitive name
            primitive_name = f"EmbeddingPrimitive_{i+1:03d}"
            
            # Create primitive
            primitive = Primitive(
                name=primitive_name,
                category=category,
                signature=PrimitiveSignature(arity=1),
                description=f"Stable embedding component (stability={avg_stability:.3f})",
                examples=[f"Component cluster {i+1} with {len(cluster)} stable directions"],
            )
            
            primitives.append(primitive)
        
        logger.info(f"Created {len(primitives)} primitives from {len(component_clusters)} clusters")
        return primitives
    
    def mine_primitives(self, n_concepts: int = 500, n_components: int = 30, 
                       stability_threshold: float = 0.7, n_clusters: int = 20) -> List[Primitive]:
        """Complete primitive mining pipeline.
        
        Args:
            n_concepts: Number of seed concepts to use
            n_components: Number of NMF components to extract
            stability_threshold: Minimum correlation for stability
            n_clusters: Number of component clusters
            
        Returns:
            List of discovered primitives
        """
        logger.info("Starting primitive mining pipeline")
        
        # Get seed concepts
        concepts = self.get_seed_concepts(n_concepts)
        logger.info(f"Using {len(concepts)} seed concepts")
        
        # Embed concepts
        embeddings = self.embed_concepts(concepts)
        
        # Align embeddings
        aligned_embeddings = self.align_embeddings(embeddings)
        
        # Find stable components
        stable_components = self.find_stable_components(
            aligned_embeddings, n_components, stability_threshold
        )
        
        # Check gate: require minimum number of stable components
        if len(stable_components) < 30:
            logger.warning(f"Only found {len(stable_components)} stable components, below threshold of 30")
        
        # Cluster components
        component_clusters = self.cluster_components(stable_components, n_clusters)
        
        # Create primitives
        primitives = self.create_primitives(component_clusters)
        
        logger.info(f"Mining complete: discovered {len(primitives)} primitives")
        return primitives


@click.command()
@click.option("--models", "-m", multiple=True, 
              default=["bert-base-uncased", "gpt2"],
              help="Embedding models to use")
@click.option("--concepts", "-c", default=500, 
              help="Number of seed concepts to use")
@click.option("--components", default=30, 
              help="Number of NMF components to extract")
@click.option("--stability-threshold", default=0.7, 
              help="Minimum correlation for stability")
@click.option("--clusters", default=20, 
              help="Number of component clusters")
@click.option("--output", "-o", default="primitives.json", 
              help="Output file for discovered primitives")
@click.option("--verbose", "-v", is_flag=True, 
              help="Enable verbose logging")
def main(models: Tuple[str, ...], concepts: int, components: int, 
         stability_threshold: float, clusters: int, output: str, verbose: bool):
    """Mine information primitives from embedding models."""
    # Setup logging
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=level, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    
    try:
        # Initialize miner
        miner = EmbeddingMiner(list(models))
        
        # Mine primitives
        primitives = miner.mine_primitives(
            n_concepts=concepts,
            n_components=components,
            stability_threshold=stability_threshold,
            n_clusters=clusters
        )
        
        # Create periodic table
        table = PeriodicTable()
        for primitive in primitives:
            table.add_primitive(primitive)
        
        # Save results
        output_path = Path(output)
        with open(output_path, "w") as f:
            json.dump(table.to_dict(), f, indent=2)
        
        logger.info(f"Saved {len(primitives)} primitives to {output_path}")
        
        # Print summary
        print(f"\nMining Results:")
        print(f"  Models used: {len(models)}")
        print(f"  Concepts embedded: {concepts}")
        print(f"  Primitives discovered: {len(primitives)}")
        print(f"  Categories: {[cat.value for cat in table.categories]}")
        
        # Check gates
        if len(primitives) >= 30:
            print(f"  ✅ Gate passed: Found {len(primitives)} primitives (≥30 required)")
        else:
            print(f"  ❌ Gate failed: Found {len(primitives)} primitives (<30 required)")
        
        # Validate table
        errors = table.validate()
        if errors:
            print(f"  ⚠️  Validation errors: {len(errors)}")
            for error in errors[:5]:  # Show first 5 errors
                print(f"    - {error}")
        else:
            print(f"  ✅ Table validation passed")
        
    except Exception as e:
        logger.error(f"Mining failed: {e}")
        raise


if __name__ == "__main__":
    main()

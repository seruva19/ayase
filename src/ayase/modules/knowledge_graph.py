import logging
import json
from typing import List, Dict, Set, Any
from pathlib import Path
from ayase.models import Sample
from ayase.pipeline import PipelineModule

logger = logging.getLogger(__name__)

class KnowledgeGraphModule(PipelineModule):
    """
    Constructs a conceptual knowledge graph of the dataset.
    Nodes: Videos, Actions, Style Traits, (future: Objects, Locations)
    Edges: Relationships based on presence, co-occurrence, and semantic similarity.
    """
    name = "knowledge_graph"
    description = "Generates a conceptual knowledge graph of the video dataset"
    default_config = {
        "output_file": "knowledge_graph.json",
        "min_confidence": 0.5,
        "include_similarity_edges": False, # Redundant with clusters
        "similarity_threshold": 0.9,
        "use_clustering": True,
        "num_clusters": "auto", # 'auto' or int
    }

    def process(self, sample: Sample) -> Sample:
        # Building the graph happens in post_process
        return sample

    def post_process(self, all_samples: List[Sample]) -> None:
        """
        Aggregates all detections and embeddings into a graph structure.
        """
        if not all_samples:
            return

        nodes = []
        edges = []
        
        # Maps for unique concept nodes
        concepts: Dict[str, Dict[str, Any]] = {} # concept_id -> data
        
        min_conf = self.config.get("min_confidence", 0.5)
        
        logger.info("Building Knowledge Graph...")

        for sample in all_samples:
            sample_id = sample.path.name
            # Node for the video itself
            nodes.append({
                "id": sample_id,
                "label": sample_id,
                "type": "video",
                "quality": sample.quality_metrics.aesthetic_score if sample.quality_metrics else None
            })
            
            # Extract concepts from detections
            for det in sample.detections:
                dtype = det.get("type")
                
                if dtype == "action":
                    label = det.get("label")
                    conf = det.get("confidence", 0.0)
                    if conf >= min_conf:
                        concept_id = f"action:{label}"
                        if concept_id not in concepts:
                            concepts[concept_id] = {"id": concept_id, "label": label, "type": "action"}
                        
                        edges.append({
                            "source": sample_id,
                            "target": concept_id,
                            "relation": "contains_action",
                            "weight": float(conf)
                        })
                
                elif dtype == "style_traits":
                    traits_dict = det.get("data", {})
                    for trait, score in traits_dict.items():
                        if score >= min_conf:
                            concept_id = f"style:{trait}"
                            if concept_id not in concepts:
                                concepts[concept_id] = {"id": concept_id, "label": trait, "type": "style"}
                            
                            edges.append({
                                "source": sample_id,
                                "target": concept_id,
                                "relation": "exhibits_style",
                                "weight": float(score)
                            })

        # Add concept nodes
        for concept_data in concepts.values():
            nodes.append(concept_data)

        # 4. Optional: Similarity Edges between videos (using embeddings)
        if self.config.get("include_similarity_edges", False):
            # ... (existing similarity edge logic)
            pass

        # 5. Embedding Clusterization
        if self.config.get("use_clustering", True):
            valid_samples = [s for s in all_samples if s.embedding is not None]
            num_samples = len(valid_samples)
            
            if num_samples >= 2:
                try:
                    from sklearn.cluster import KMeans
                    import numpy as np
                    
                    # Determine n_clusters
                    n_clusters_cfg = self.config.get("num_clusters", "auto")
                    if n_clusters_cfg == "auto":
                        # Square root heuristic for number of clusters
                        n_clusters = int(np.sqrt(num_samples / 2))
                        n_clusters = max(2, min(n_clusters, 15)) # Clamp to reasonable range
                    else:
                        n_clusters = int(n_clusters_cfg)
                    
                    n_clusters = min(n_clusters, num_samples)
                    
                    X = np.array([s.embedding for s in valid_samples])
                    kmeans = KMeans(n_clusters=n_clusters, n_init='auto', random_state=42)
                    cluster_labels = kmeans.fit_predict(X)
                    
                    # Create Cluster Nodes
                    for c_idx in range(n_clusters):
                        cluster_id = f"cluster:{c_idx}"
                        
                        # Heuristic: Find common traits for member videos to "name" the cluster
                        member_indices = np.where(cluster_labels == c_idx)[0]
                        cluster_members = [valid_samples[i] for i in member_indices]
                        
                        # Aggregate traits and actions
                        concept_votes = {}
                        for s in cluster_members:
                            for det in s.detections:
                                if det.get("type") in ["style_traits", "action"]:
                                    if det.get("type") == "action":
                                        label = det.get("label")
                                        concept_votes[label] = concept_votes.get(label, 0) + det.get("confidence", 0)
                                    else: # style_traits
                                        for t, v in det.get("data", {}).items():
                                            if v > 0.5:
                                                concept_votes[t] = concept_votes.get(t, 0) + v
                        
                        # Top trait/action
                        top_concept = "General"
                        if concept_votes:
                            top_concept = max(concept_votes, key=concept_votes.get)
                        
                        nodes.append({
                            "id": cluster_id,
                            "label": f"Cluster {c_idx} ({top_concept.replace('_', ' ').title()})",
                            "type": "concept_cluster",
                            "size": len(cluster_members)
                        })
                        
                        # Link members
                        for s in cluster_members:
                            edges.append({
                                "source": s.path.name,
                                "target": cluster_id,
                                "relation": "belongs_to_cluster",
                                "weight": 1.0
                            })
                            
                    logger.info(f"Knowledge Graph: Created {n_clusters} clusters (Auto: {n_clusters_cfg == 'auto'}).")
                except ImportError:
                    logger.warning("scikit-learn not installed. Skipping clusterization.")
                except Exception as e:
                    logger.error(f"Clusterization failed: {e}")

        graph_data = {
            "nodes": nodes,
            "edges": edges,
            "metadata": {
                "total_videos": len(all_samples),
                "total_concepts": len(concepts),
                "total_edges": len(edges)
            }
        }

        output_path = Path(self.config.get("output_file", "knowledge_graph.json"))
        try:
            with open(output_path, "w") as f:
                json.dump(graph_data, f, indent=2)
            logger.info(f"Knowledge Graph exported to {output_path}")
        except Exception as e:
            logger.error(f"Failed to export Knowledge Graph: {e}")


#!/usr/bin/env python3
"""
Data Deduplication Script for Intent Classification
Uses embedding-based similarity and MinHash for deduplication
"""

import json
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Tuple
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import DBSCAN
import hashlib
from collections import defaultdict

class DataDeduplicator:
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        """Initialize with sentence transformer model"""
        self.model = SentenceTransformer(model_name)
        self.similarity_threshold = 0.95
        self.min_cluster_size = 2
        
    def load_data(self, data_file: Path) -> List[Dict[str, Any]]:
        """Load synthetic data from JSON file"""
        with open(data_file, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def compute_embeddings(self, texts: List[str]) -> np.ndarray:
        """Compute embeddings for all texts"""
        print(f"Computing embeddings for {len(texts)} texts...")
        embeddings = self.model.encode(texts, show_progress_bar=True)
        return embeddings
    
    def find_duplicates_by_similarity(self, data: List[Dict[str, Any]]) -> List[Tuple[int, int, float]]:
        """Find duplicate pairs using cosine similarity"""
        texts = [item['text'] for item in data]
        embeddings = self.compute_embeddings(texts)
        
        # Compute similarity matrix
        similarity_matrix = cosine_similarity(embeddings)
        
        # Find pairs above threshold
        duplicates = []
        for i in range(len(similarity_matrix)):
            for j in range(i + 1, len(similarity_matrix)):
                similarity = similarity_matrix[i][j]
                if similarity > self.similarity_threshold:
                    duplicates.append((i, j, similarity))
        
        return duplicates
    
    def find_duplicates_by_clustering(self, data: List[Dict[str, Any]]) -> List[List[int]]:
        """Find duplicate clusters using DBSCAN"""
        texts = [item['text'] for item in data]
        embeddings = self.compute_embeddings(texts)
        
        # Apply DBSCAN clustering
        clustering = DBSCAN(
            eps=1 - self.similarity_threshold,  # Convert similarity to distance
            min_samples=self.min_cluster_size,
            metric='cosine'
        )
        cluster_labels = clustering.fit_predict(embeddings)
        
        # Group items by cluster
        clusters = defaultdict(list)
        for idx, label in enumerate(cluster_labels):
            if label != -1:  # Not noise
                clusters[label].append(idx)
        
        return list(clusters.values())
    
    def find_exact_duplicates(self, data: List[Dict[str, Any]]) -> List[List[int]]:
        """Find exact duplicates using text hashing"""
        text_to_indices = defaultdict(list)
        
        for idx, item in enumerate(data):
            # Normalize text for comparison
            normalized_text = item['text'].lower().strip()
            text_hash = hashlib.md5(normalized_text.encode()).hexdigest()
            text_to_indices[text_hash].append(idx)
        
        # Return groups with more than one item
        exact_duplicates = [indices for indices in text_to_indices.values() if len(indices) > 1]
        return exact_duplicates
    
    def select_best_from_cluster(self, data: List[Dict[str, Any]], cluster_indices: List[int]) -> int:
        """Select the best example from a cluster of duplicates"""
        if len(cluster_indices) == 1:
            return cluster_indices[0]
        
        # Score examples by various criteria
        scores = []
        for idx in cluster_indices:
            item = data[idx]
            score = 0
            
            # Prefer longer texts (more informative)
            score += len(item['text'].split()) * 0.1
            
            # Prefer texts with better language distribution
            if item.get('language') == 'mixed':
                score += 1.0
            
            # Prefer texts with more diverse vocabulary
            unique_words = len(set(item['text'].lower().split()))
            score += unique_words * 0.05
            
            scores.append(score)
        
        # Return index with highest score
        best_idx = cluster_indices[np.argmax(scores)]
        return best_idx
    
    def deduplicate_data(self, data: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """Main deduplication process"""
        print(f"Starting deduplication of {len(data)} examples...")
        
        # Find exact duplicates
        exact_duplicates = self.find_exact_duplicates(data)
        print(f"Found {len(exact_duplicates)} exact duplicate groups")
        
        # Find similarity-based duplicates
        similarity_duplicates = self.find_duplicates_by_similarity(data)
        print(f"Found {len(similarity_duplicates)} similarity-based duplicate pairs")
        
        # Find cluster-based duplicates
        cluster_duplicates = self.find_duplicates_by_clustering(data)
        print(f"Found {len(cluster_duplicates)} duplicate clusters")
        
        # Combine all duplicate information
        duplicate_indices = set()
        
        # Add exact duplicates
        for cluster in exact_duplicates:
            best_idx = self.select_best_from_cluster(data, cluster)
            for idx in cluster:
                if idx != best_idx:
                    duplicate_indices.add(idx)
        
        # Add similarity-based duplicates
        for i, j, similarity in similarity_duplicates:
            if i not in duplicate_indices and j not in duplicate_indices:
                # Keep the one with better quality
                if len(data[i]['text']) > len(data[j]['text']):
                    duplicate_indices.add(j)
                else:
                    duplicate_indices.add(i)
        
        # Add cluster-based duplicates
        for cluster in cluster_duplicates:
            best_idx = self.select_best_from_cluster(data, cluster)
            for idx in cluster:
                if idx != best_idx and idx not in duplicate_indices:
                    duplicate_indices.add(idx)
        
        # Create deduplicated dataset
        deduplicated_data = []
        for idx, item in enumerate(data):
            if idx not in duplicate_indices:
                deduplicated_data.append(item)
        
        # Create statistics
        stats = {
            "original_count": len(data),
            "deduplicated_count": len(deduplicated_data),
            "removed_count": len(duplicate_indices),
            "removal_rate": len(duplicate_indices) / len(data),
            "exact_duplicates": len(exact_duplicates),
            "similarity_duplicates": len(similarity_duplicates),
            "cluster_duplicates": len(cluster_duplicates)
        }
        
        print(f"Deduplication completed:")
        print(f"  Original: {stats['original_count']}")
        print(f"  Deduplicated: {stats['deduplicated_count']}")
        print(f"  Removed: {stats['removed_count']} ({stats['removal_rate']:.2%})")
        
        return deduplicated_data, stats
    
    def save_deduplicated_data(self, data: List[Dict[str, Any]], output_file: Path):
        """Save deduplicated data to file"""
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        print(f"Deduplicated data saved to {output_file}")
    
    def save_statistics(self, stats: Dict[str, Any], output_file: Path):
        """Save deduplication statistics"""
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(stats, f, ensure_ascii=False, indent=2)
        print(f"Statistics saved to {output_file}")

def main():
    """Main function"""
    print("Starting data deduplication...")
    
    # Setup paths
    data_dir = Path("data")
    raw_dir = data_dir / "raw"
    clean_dir = data_dir / "clean"
    clean_dir.mkdir(exist_ok=True)
    
    # Input file
    input_file = raw_dir / "complete_synthetic_data.json"
    if not input_file.exists():
        print(f"Error: Input file {input_file} not found")
        print("Please run data generation first")
        return
    
    # Initialize deduplicator
    deduplicator = DataDeduplicator()
    
    # Load data
    data = deduplicator.load_data(input_file)
    print(f"Loaded {len(data)} examples")
    
    # Deduplicate
    deduplicated_data, stats = deduplicator.deduplicate_data(data)
    
    # Save results
    output_file = clean_dir / "deduplicated_data.json"
    deduplicator.save_deduplicated_data(deduplicated_data, output_file)
    
    stats_file = clean_dir / "deduplication_stats.json"
    deduplicator.save_statistics(stats, stats_file)
    
    # Print intent distribution
    intent_counts = defaultdict(int)
    for item in deduplicated_data:
        intent_counts[item['intent']] += 1
    
    print("\nDeduplicated data distribution:")
    for intent, count in sorted(intent_counts.items()):
        print(f"  {intent}: {count}")

if __name__ == "__main__":
    main()

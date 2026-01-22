"""
Combined Edge Predictor: Geometry + Semantics.

This algorithm combines:
1. Geometric Structural Proposals (DBSCAN + MST + Convex Hull) -> High Recall
2. Semantic Filtering (Descriptor Similarity) -> High Precision

It trusts geometry for strong shapes (wheels, long lines) but requires 
texture confirmation for loose connections.
"""

import numpy as np
import cv2
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
from scipy.spatial import ConvexHull
from scipy.spatial.distance import pdist, squareform
from scipy.sparse.csgraph import minimum_spanning_tree
from sklearn.neighbors import NearestNeighbors
import warnings
warnings.filterwarnings('ignore')

class EdgePredictor:
    def __init__(self, eps=20, min_samples=3, 
                 desc_thresh=0.5, # Descriptor similarity threshold (0-1)
                 max_dist=20,
                 gap_dist=60):
        self.eps = eps
        self.min_samples = min_samples
        self.desc_thresh = desc_thresh
        self.max_dist = max_dist
        self.gap_dist = gap_dist

    def predict(self, keypoints, img_shape, scores=None, descriptors=None):
        h, w = img_shape[:2]
        edge_map = np.zeros((h, w), dtype=np.uint8)

        if len(keypoints) < 2:
            return edge_map
            
        if descriptors is None:
            descriptors = np.zeros((256, len(keypoints)))
            descriptors[0, :] = 1.0

        # --- STAGE 1: Generate Geometric Proposals ---
        # We use DBSCAN to find structure groups
        candidate_edges = self._generate_geometric_proposals(keypoints)
        
        # --- STAGE 2: Semantic Filtering & Merging ---
        # We validate edges based on descriptor similarity + geometry type
        final_edges = []
        
        for idx1, idx2, type_tag in candidate_edges:
            # 1. Geometry Trust:
            # If it's a 'hull' (circular object), we implicitly trust the geometry
            # because random noise rarely forms a clean convex hull.
            if type_tag == 'hull':
                final_edges.append((idx1, idx2))
                continue
                
            # 2. Descriptor Verification:
            # For linear (mst) or loose (prox) connections, we check texture
            d1 = descriptors[:, idx1]
            d2 = descriptors[:, idx2]
            
            # Cosine similarity (1.0 = same, 0.0 = orthogonal)
            sim = np.dot(d1, d2)
            
            # Independent dynamic thresholds
            if type_tag == 'mst':
                # Lines are fairly trustworthy, lenient threshold
                if sim > 0.4: 
                    final_edges.append((idx1, idx2))
            
            elif type_tag == 'proximity':
                # Dispersed points need STRONG evidence to connect
                if sim > 0.75:
                    final_edges.append((idx1, idx2))
            
            elif type_tag == 'gap_fill':
                # Gaps are hallucinations, need support
                if sim > 0.6:
                    final_edges.append((idx1, idx2))

        # --- Draw ---
        for p1, p2 in final_edges:
            pt1 = tuple(keypoints[p1].astype(int))
            pt2 = tuple(keypoints[p2].astype(int))
            cv2.line(edge_map, pt1, pt2, 255, 1)

        return edge_map

    def _generate_geometric_proposals(self, points):
        """
        Run DBSCAN + PCA to propose MST/Hull/Proximity edges.
        Returns list of (idx1, idx2, type_string)
        """
        proposals = []
        
        # 1. Cluster
        dbscan = DBSCAN(eps=self.eps, min_samples=self.min_samples)
        labels = dbscan.fit_predict(points)
        unique_labels = set(labels)
        
        # 2. Per-Cluster Strategy
        for label in unique_labels:
            mask = labels == label
            cluster_pts = points[mask]
            indices = np.where(mask)[0]
            
            if label == -1 or len(cluster_pts) < 3:
                # Noise -> Proximity
                edges = self._proximity_edges(cluster_pts, indices)
                for p1, p2 in edges:
                    proposals.append((p1, p2, 'proximity'))
                continue

            dims = self._get_pca_dims(cluster_pts)

            # STRATEGY SELECTION
            if dims['ratio'] > 0.85:  # Linear (Roads, Poles)
                edges = self._mst_edges(cluster_pts, indices)
                for p1, p2 in edges:
                    proposals.append((p1, p2, 'mst'))

            elif dims['ratio'] < 0.3 and dims['density'] > 0.5:  # Small/Dense Circular (Wheels)
                edges = self._convex_hull_edges(cluster_pts, indices)
                for p1, p2 in edges:
                    proposals.append((p1, p2, 'hull'))

            else:  # Generic Blob/Texture
                edges = self._proximity_edges(cluster_pts, indices)
                for p1, p2 in edges:
                    proposals.append((p1, p2, 'proximity'))

        # 3. Gap Filling (Global)
        mst_only = [(p1, p2) for p1, p2, t in proposals if t == 'mst']
        gaps = self._connect_gaps(mst_only, points)
        for p1, p2 in gaps:
            proposals.append((p1, p2, 'gap_fill'))
            
        return proposals

    def _get_pca_dims(self, points):
        pca = PCA(n_components=2).fit(points)
        ratio = pca.explained_variance_ratio_[0]
        
        min_pt = np.min(points, axis=0)
        max_pt = np.max(points, axis=0)
        area = (max_pt[0] - min_pt[0]) * (max_pt[1] - min_pt[1])
        density = len(points) / (area + 1e-5)
        return {'ratio': ratio, 'density': density}

    def _convex_hull_edges(self, points, indices):
        try:
            hull = ConvexHull(points)
            edges = []
            for i in range(len(hull.vertices)):
                v1 = hull.vertices[i]
                v2 = hull.vertices[(i + 1) % len(hull.vertices)]
                if np.linalg.norm(points[v1] - points[v2]) <= self.max_dist * 2.5:
                    edges.append((indices[v1], indices[v2]))
            return edges
        except Exception:
            return self._proximity_edges(points, indices)

    def _mst_edges(self, points, indices):
        dist_matrix = squareform(pdist(points))
        mst = minimum_spanning_tree(dist_matrix)
        rows, cols = mst.nonzero()
        edges = []
        for r, c in zip(rows, cols):
            if dist_matrix[r, c] <= self.max_dist:
                edges.append((indices[r], indices[c]))
        return edges

    def _proximity_edges(self, points, indices, k=2):
        n_neighbors = min(k + 1, len(points))
        nbrs = NearestNeighbors(n_neighbors=n_neighbors).fit(points)
        dists, idxs = nbrs.kneighbors(points)
        edges = set()
        for i in range(len(points)):
            for j in range(1, n_neighbors):
                if dists[i][j] <= self.max_dist:
                    p1, p2 = min(indices[i], indices[idxs[i][j]]), max(indices[i], indices[idxs[i][j]])
                    edges.add((p1, p2))
        return list(edges)
        
    def _connect_gaps(self, edges, keypoints):
        """Connect gap endpoints within alignment threshold."""
        adj = {i: [] for i in range(len(keypoints))}
        for p1, p2 in edges:
            adj[p1].append(p2)
            adj[p2].append(p1)
        endpoints = [i for i, n in adj.items() if len(n) == 1]

        gap_edges = []
        for i in range(len(endpoints)):
            u = endpoints[i]
            v = adj[u][0]
            dir_vec = (keypoints[u] - keypoints[v]).astype(float)
            norm = np.linalg.norm(dir_vec)
            if norm == 0:
                continue
            dir_vec /= norm
            
            for j in range(i + 1, len(endpoints)):
                target = endpoints[j]
                gap_vec = keypoints[target] - keypoints[u]
                dist = np.linalg.norm(gap_vec)
                
                if dist > 0 and dist <= self.gap_dist:
                    gap_dir = gap_vec / dist
                    if np.dot(dir_vec, gap_dir) > 0.866: # cos(30)
                        gap_edges.append((u, target))
                        
        return gap_edges

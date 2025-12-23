import numpy as np

class WassersteinClustering:
    def __init__(self, n_regimes=2, max_iter=15, n_init=10):
        self.k = n_regimes
        self.max_iter = max_iter
        self.n_init = n_init
        self.centroids = None
        self.best_inertia = float('inf')

    def _w1_distance(self, dist1, dist2):
        # Wasserstein-1 distance for sorted 1D arrays
        return np.mean(np.abs(dist1 - dist2))

    def fit(self, windows):
        best_labels = None
        
        for seed in range(self.n_init):
            np.random.seed(seed) # Ensures variety in initializations
            # Initialize: Pick k random windows
            idx = np.random.choice(len(windows), self.k, replace=False)
            current_centroids = windows[idx].copy()
            current_labels = np.zeros(len(windows))
            
            for _ in range(self.max_iter):
                # 1. Assignment Step
                # We calculate distance from every window to every centroid
                distances = np.array([[self._w1_distance(w, c) for c in current_centroids] for w in windows])
                current_labels = np.argmin(distances, axis=1)
                
                # 2. Update Step (Barycenter)
                for j in range(self.k):
                    mask = (current_labels == j)
                    if np.any(mask):
                        # Median of each rank is the W1 Barycenter
                        current_centroids[j] = np.median(windows[mask], axis=0)
            
            # Calculate Inertia (Total W1 Distance) to check quality
            inertia = sum(self._w1_distance(windows[i], current_centroids[current_labels[i]]) 
                          for i in range(len(windows)))
            
            if inertia < self.best_inertia:
                self.best_inertia = inertia
                self.centroids = current_centroids
                best_labels = current_labels
                
        return best_labels
    
    def predict(self, new_windows):
        """Assigns new windows to the closest existing centroid."""
        if self.centroids is None:
            raise ValueError("Model must be fitted before predicting.")
            
        labels = []
        for w in new_windows:
            dists = [self._w1_distance(w, c) for c in self.centroids]
            labels.append(np.argmin(dists))
        return np.array(labels)
    
    def get_centroids(self):
        return self.centroids    

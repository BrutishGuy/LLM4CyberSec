from sklearn.feature_extraction.text import CountVectorizer
from scipy.spatial.distance import cosine
from scipy.stats import ks_2samp

class DataDriftDetector:
    """
    A class to detect data drift in text data.
    
    Attributes:
        vectorizer (CountVectorizer): The vectorizer for extracting text features.
        training_data_features (array): Features of the training data.
    """
    
    def __init__(self, training_data):
        """
        Initializes the DataDriftDetector with training data.
        
        Args:
            training_data (list): A list of training data text.
        """
        self.vectorizer = CountVectorizer()
        self.training_data_features = self.vectorizer.fit_transform(training_data).toarray()

    def detect_drift(self, new_data):
        """
        Detects data drift in new data compared to the training data.
        
        Args:
            new_data (list): A list of new data text.
        
        Returns:
            dict: Drift detection metrics.
        """
        new_data_features = self.vectorizer.transform(new_data).toarray()
        
        drift_metrics = {}
        
        # Compare distributions using KS test for each feature
        ks_statistics = [ks_2samp(self.training_data_features[:, i], new_data_features[:, i]).statistic for i in range(self.training_data_features.shape[1])]
        drift_metrics['ks_statistics'] = ks_statistics
        
        # Compare distributions using cosine similarity
        training_centroid = self.training_data_features.mean(axis=0)
        new_centroid = new_data_features.mean(axis=0)
        cosine_sim = 1 - cosine(training_centroid, new_centroid)
        drift_metrics['cosine_similarity'] = cosine_sim
        
        return drift_metrics

# Example usage
training_data = [d["instruction"] for d in data]
new_data = [
    "Explain this EDR alert: unusual network activity detected.",
    "What does this alert mean for system security?"
]

drift_detector = DataDriftDetector(training_data)
drift_metrics = drift_detector.detect_drift(new_data)

print(f"Data Drift Metrics: {drift_metrics}")

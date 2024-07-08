from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from datasets import Dataset, load_metric
from scipy.spatial.distance import cosine
from scipy.stats import ks_2samp
import argparse 

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

    @staticmethod
    def load_data_from_s3(self, s3_path: str):
        bucket, key = self._parse_s3_path(s3_path)
        obj = self.s3_client.get_object(Bucket=bucket, Key=key)
        data = json.loads(obj['Body'].read().decode('utf-8'))
        return self.load_data(data)

    def _parse_s3_path(self, s3_path: str):
        path_parts = s3_path.replace("s3://", "").split("/", 1)
        return path_parts[0], path_parts[1]
    
    @staticmethod
    def load_data(self, data: list):
        """
        Loads and preprocesses the data.
        
        Args:
            data (list): A list of dictionaries containing 'instruction' and 'response' pairs.
        
        Returns:
            tuple: Tokenized training, validation, and test datasets.
        """
        train_data, temp_data = train_test_split(data, test_size=0.3, random_state=42)
        val_data, test_data = train_test_split(temp_data, test_size=0.5, random_state=42)
        
        train_dataset = Dataset.from_dict({"instruction": [d["instruction"] for d in train_data], "response": [d["response"] for d in train_data]})
        val_dataset = Dataset.from_dict({"instruction": [d["instruction"] for d in val_data], "response": [d["response"] for d in val_data]})
        test_dataset = Dataset.from_dict({"instruction": [d["instruction"] for d in test_data], "response": [d["response"] for d in test_data]})
        
        train_dataset = train_dataset.map(self.tokenize_function, batched=True)
        val_dataset = val_dataset.map(self.tokenize_function, batched=True)
        test_dataset = test_dataset.map(self.tokenize_function, batched=True)
        
        return train_dataset, val_dataset, test_dataset

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

def validate_drift(user_validation_s3_path):
    # Example usage
    train_dataset, _, _ = DataDriftDetector.load_data_from_s3(user_validation_s3_path)

    train_dataset = [d["instruction"] for d in train_dataset]
    new_data = [
        "Explain this EDR alert: unusual network activity detected.",
        "What does this alert mean for system security?"
    ]

    drift_detector = DataDriftDetector(train_dataset)
    drift_metrics = drift_detector.detect_drift(new_data)

    print(f"Data Drift Metrics: {drift_metrics}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Detect dataset drift in data used for LLMs.")
    parser.add_argument('--user_validation_s3_path', type=str, required=True, help="S3 path to the user validation data, in the S3 bucket.")
    args = parser.parse_args()

    validate_drift(args.user_validation_s3_path)

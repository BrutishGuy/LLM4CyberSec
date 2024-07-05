import os
import json
import boto3
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments
from sklearn.model_selection import train_test_split
from datasets import Dataset, load_metric
import logging

# Initialize logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModelTrainer:
    """
    A class to manage the training, validation, and deployment of a large language model.
    
    Attributes:
        model_name (str): The name of the model to be loaded.
        tokenizer (AutoTokenizer): The tokenizer associated with the model.
        model (AutoModelForCausalLM): The model to be trained and deployed.
        s3_client (boto3.client): The S3 client for interacting with AWS S3.
        metrics (list): A list of metrics to evaluate model performance.
    """
    
    def __init__(self, model_name: str):
        """
        Initializes the ModelTrainer with a specified model name.
        
        Args:
            model_name (str): The name of the model to be loaded.
        """
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        self.s3_client = boto3.client('s3')
        self.metrics = ["accuracy", "f1", "bleu", "rouge"]

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

    def tokenize_function(self, examples):
        """
        Tokenizes the input data.
        
        Args:
            examples (dict): A dictionary of examples to be tokenized.
        
        Returns:
            dict: Tokenized examples.
        """
        return self.tokenizer(examples['instruction'], padding="max_length", truncation=True)

    def train_model(self, train_dataset, val_dataset, output_dir='/opt/ml/model'):
        """
        Trains the model using the provided datasets.
        
        Args:
            train_dataset (Dataset): The training dataset.
            val_dataset (Dataset): The validation dataset.
            output_dir (str): The directory where the trained model will be saved.
        """
        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=3,
            per_device_train_batch_size=16,
            per_device_eval_batch_size=16,
            warmup_steps=500,
            weight_decay=0.01,
            logging_dir='/opt/ml/logs',
            logging_steps=10,
            evaluation_strategy="steps",
            eval_steps=100,
            save_total_limit=2,
            save_steps=200,
        )

        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
        )

        trainer.train()

    def validate_model(self, val_dataset):
        """
        Validates the model using the provided validation dataset and logs metrics.
        
        Args:
            val_dataset (Dataset): The validation dataset.
        
        Returns:
            dict: Validation metrics.
        """
        trainer = Trainer(
            model=self.model,
            args=TrainingArguments(per_device_eval_batch_size=16),
            eval_dataset=val_dataset,
        )

        metrics = trainer.evaluate()
        
        logger.info(f"Validation metrics: {metrics}")
        self.save_metrics_to_s3(metrics, 'validation_metrics.json')
        
        return metrics

    def save_metrics_to_s3(self, metrics, filename):
        """
        Saves metrics to an S3 bucket.
        
        Args:
            metrics (dict): A dictionary of metrics to be saved.
            filename (str): The filename to save the metrics as in the S3 bucket.
        """
        with open(f'/tmp/{filename}', 'w') as f:
            json.dump(metrics, f)
        
        self.s3_client.upload_file(f'/tmp/{filename}', 'your-bucket', filename)

    def deploy_model(self, model_dir, bucket, model_name):
        """
        Deploys the model to S3 for use in SageMaker.
        
        Args:
            model_dir (str): The directory where the model is saved.
            bucket (str): The S3 bucket to upload the model to.
            model_name (str): The name of the model file.
        
        Returns:
            str: The S3 URI of the deployed model.
        """
        model_tar_path = os.path.join(model_dir, model_name)
        self.s3_client.upload_file(model_tar_path, bucket, model_name)
        
        return f"s3://{bucket}/{model_name}"

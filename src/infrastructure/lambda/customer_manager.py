import boto3
import requests
import json

dynamodb = boto3.resource('dynamodb')
table = dynamodb.Table('CustomerManagement')

def lambda_handler(event, context):
    customers = table.scan()['Items']
    
    tf_config = generate_terraform_config(customers)
    
    # Save the configuration to S3
    s3 = boto3.client('s3')
    s3.put_object(Bucket='terraform-config-bucket', Key='customer_config.tf', Body=tf_config)
    
    # Trigger Terraform Cloud run
    headers = {
        'Authorization': 'Bearer YOUR_TERRAFORM_CLOUD_API_TOKEN',
        'Content-Type': 'application/vnd.api+json'
    }
    
    payload = {
        "data": {
            "attributes": {
                "is-destroy": False,
                "message": "Triggered by Lambda",
                "variables": {}
            },
            "type": "runs",
            "relationships": {
                "workspace": {
                    "data": {
                        "type": "workspaces",
                        "id": "YOUR_WORKSPACE_ID"
                    }
                }
            }
        }
    }
    
    response = requests.post('https://app.terraform.io/api/v2/runs', headers=headers, data=json.dumps(payload))
    print(response.text)
    
    return {
        'statusCode': 200,
        'body': json.dumps('Terraform configuration generated and Terraform Cloud run triggered successfully')
    }

def generate_terraform_config(customers):
    header = """
    provider "aws" {
      region = "us-west-2"
    }
    """
    
    resources = ""
    
    for customer in customers:
        customer_id = customer['CustomerId']
        
        # Add VPC, S3, and other resources as needed
        resources += f"""
        resource "aws_s3_bucket" "{customer_id}_data_storage" {{
          bucket = "{customer_id}-data-storage"
          versioning {{
            enabled = true
          }}
          server_side_encryption_configuration {{
            rule {{
              apply_server_side_encryption_by_default {{
                sse_algorithm = "AES256"
              }}
            }}
          }}
        }}
        
        resource "aws_sagemaker_model" "{customer_id}_model" {{
          name               = "{customer_id}-model"
          execution_role_arn = aws_iam_role.sagemaker_execution_role.arn

          primary_container {{
            image   = "763104351884.dkr.ecr.us-west-2.amazonaws.com/huggingface-pytorch-inference:1.6.0-transformers4.11.0-cpu"
            mode    = "SingleModel"
            model_data_url = "s3://customer-model-storage/{customer_id}/model.tar.gz"
          }}
        }}
        """
    
    return header + resources

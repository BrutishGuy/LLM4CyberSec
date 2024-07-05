import json
import boto3

sagemaker_runtime = boto3.client('sagemaker-runtime')

def lambda_handler(event, context):
    # Extract user input from Lex event
    user_input = event['currentIntent']['slots']['UserInput']
    
    # Call SageMaker endpoint
    response = sagemaker_runtime.invoke_endpoint(
        EndpointName='your-sagemaker-endpoint',
        ContentType='application/json',
        Body=json.dumps({"inputs": user_input})
    )
    
    result = json.loads(response['Body'].read().decode())
    
    # Formulate response for Lex
    return {
        "dialogAction": {
            "type": "Close",
            "fulfillmentState": "Fulfilled",
            "message": {
                "contentType": "PlainText",
                "content": result['generated_text']
            }
        }
    }

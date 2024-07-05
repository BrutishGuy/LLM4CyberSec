resource "aws_lambda_function" "rag_orchestrator" {
  filename         = "lambda_function_payload.zip"
  function_name    = "cybersecurity-bot-prod-us-east-1-rag-orchestrator-lambda-function"
  role             = aws_iam_role.lambda_execution_role.arn
  handler          = "lambda_function.lambda_handler"
  source_code_hash = filebase64sha256("lambda_function_payload.zip")
  runtime          = "python3.8"

  environment {
    variables = {
      RETRIEVER_ENDPOINT = "retriever-endpoint"
      GENERATOR_ENDPOINT = "generator-endpoint"
    }
  }
}

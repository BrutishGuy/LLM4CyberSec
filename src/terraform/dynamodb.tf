resource "aws_dynamodb_table" "customer_table" {
  name           = "CustomerManagement"
  billing_mode   = "PAY_PER_REQUEST"
  hash_key       = "CustomerId"

  attribute {
    name = "CustomerId"
    type = "S"
  }

  stream_enabled = true
  stream_view_type = "NEW_AND_OLD_IMAGES"
}

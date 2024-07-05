resource "aws_s3_bucket" "data_storage" {
  bucket = "customer-data-storage"

  versioning {
    enabled = true
  }

  server_side_encryption_configuration {
    rule {
      apply_server_side_encryption_by_default {
        sse_algorithm = "AES256"
      }
    }
  }
}

resource "aws_s3_bucket" "model_storage" {
  bucket = "customer-model-storage"

  versioning {
    enabled = true
  }

  server_side_encryption_configuration {
    rule {
      apply_server_side_encryption_by_default {
        sse_algorithm = "AES256"
      }
    }
  }
}

resource "aws_s3_bucket" "terraform_config_bucket" {
  bucket = "terraform-config-bucket"
  versioning {
    enabled = true
  }
}

resource "aws_s3_bucket_object" "terraform_config" {
  bucket = aws_s3_bucket.terraform_config_bucket.bucket
  key    = "customer_config.tf"
  source = "/path/to/generated/terraform/config"
}

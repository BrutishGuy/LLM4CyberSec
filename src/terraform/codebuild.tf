resource "aws_codebuild_project" "terraform_build_project" {
  name          = "terraform-build-project"
  service_role  = aws_iam_role.codepipeline_role.arn
  source {
    type            = "S3"
    location        = "terraform-config-bucket/customer_config.tf"
    buildspec       = <<BUILD_SPEC
version: 0.2

phases:
  install:
    runtime-versions:
      python: 3.x
  build:
    commands:
      - pip install terraform
      - terraform init
      - terraform apply -auto-approve
artifacts:
  files:
    - '**/*'
BUILD_SPEC
  }

  artifacts {
    type = "NO_ARTIFACTS"
  }

  environment {
    compute_type                = "BUILD_GENERAL1_SMALL"
    image                       = "aws/codebuild/standard:5.0"
    type                        = "LINUX_CONTAINER"
    privileged_mode             = true
  }
}

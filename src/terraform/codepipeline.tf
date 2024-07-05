resource "aws_codepipeline" "terraform_pipeline" {
  name     = "terraform-pipeline"
  role_arn = aws_iam_role.codepipeline_role.arn

  artifact_store {
    type = "S3"
    location = aws_s3_bucket.terraform_config_bucket.bucket
  }

  stage {
    name = "Source"
    action {
      name             = "Source"
      category         = "Source"
      owner            = "AWS"
      provider         = "S3"
      version          = "1"
      output_artifacts = ["source_output"]
      configuration = {
        S3Bucket = aws_s3_bucket.terraform_config_bucket.bucket
        S3ObjectKey = "customer_config.tf"
      }
    }
  }

  stage {
    name = "Deploy"
    action {
      name             = "Deploy"
      category         = "Deploy"
      owner            = "AWS"
      provider         = "CodeBuild"
      version          = "1"
      input_artifacts  = ["source_output"]
      configuration = {
        ProjectName = aws_codebuild_project.terraform_build_project.name
      }
    }
  }
}

output "bucketName" {
  description = "Name of the S3 bucket"
  value       = aws_s3_bucket.nashville311DataBucket.bucket
}

output "bucketArn" {
  description = "ARN of the S3 bucket"
  value       = aws_s3_bucket.nashville311DataBucket.arn
}

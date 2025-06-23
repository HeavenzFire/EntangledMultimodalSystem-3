terraform {
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 4.0"
    }
  }
}

provider "aws" {
  region = "us-east-1"
}

module "quantum_core" {
  source = "./modules/quantum_core"

  qpu_type          = "IonQ_Harmony"
  qubit_count       = 128
  error_correction  = "surface_code"
  classical_vcpu    = 96
  ethical_firewall  = true
}

# IAM role for Braket access
resource "aws_iam_role" "braket_role" {
  name = "digigod-braket-role"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Action = "sts:AssumeRole"
        Effect = "Allow"
        Principal = {
          Service = "braket.amazonaws.com"
        }
      }
    ]
  })
}

# IAM policy for Braket access
resource "aws_iam_role_policy" "braket_policy" {
  name = "digigod-braket-policy"
  role = aws_iam_role.braket_role.id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Action = [
          "braket:CreateQuantumTask",
          "braket:GetQuantumTask",
          "braket:CancelQuantumTask",
          "braket:SearchDevices",
          "s3:GetObject",
          "s3:PutObject"
        ]
        Resource = "*"
      }
    ]
  })
}

# S3 bucket for quantum task results
resource "aws_s3_bucket" "quantum_results" {
  bucket = "digigod-quantum-results"
  acl    = "private"

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

  lifecycle_rule {
    id      = "cleanup"
    enabled = true

    expiration {
      days = 30
    }
  }
}

# Braket quantum task queue
resource "aws_sqs_queue" "quantum_tasks" {
  name                      = "digigod-quantum-tasks"
  delay_seconds             = 0
  max_message_size          = 262144
  message_retention_seconds = 86400
  receive_wait_time_seconds = 0

  redrive_policy = jsonencode({
    deadLetterTargetArn = aws_sqs_queue.quantum_tasks_dlq.arn
    maxReceiveCount     = 3
  })
}

# Dead letter queue for quantum tasks
resource "aws_sqs_queue" "quantum_tasks_dlq" {
  name = "digigod-quantum-tasks-dlq"
}

# CloudWatch log group for quantum tasks
resource "aws_cloudwatch_log_group" "quantum_tasks" {
  name              = "/aws/braket/digigod"
  retention_in_days = 30
}

output "quantum_core_arn" {
  value = module.quantum_core.arn
}

output "braket_role_arn" {
  value = aws_iam_role.braket_role.arn
}

output "quantum_results_bucket" {
  value = aws_s3_bucket.quantum_results.id
}

output "quantum_tasks_queue" {
  value = aws_sqs_queue.quantum_tasks.id
} 
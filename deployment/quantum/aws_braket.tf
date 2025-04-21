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

resource "aws_iam_role" "quantum_execution_role" {
  name = "quantum_execution_role"

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

resource "aws_iam_role_policy" "quantum_policy" {
  name = "quantum_policy"
  role = aws_iam_role.quantum_execution_role.id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Action = [
          "braket:CreateQuantumTask",
          "braket:GetQuantumTask",
          "braket:CancelQuantumTask",
          "s3:PutObject",
          "s3:GetObject"
        ]
        Resource = "*"
      }
    ]
  })
}

resource "aws_s3_bucket" "quantum_results" {
  bucket = "quantum-results-${var.environment}"
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
}

output "quantum_core_arn" {
  value = module.quantum_core.arn
}

output "execution_role_arn" {
  value = aws_iam_role.quantum_execution_role.arn
}

output "results_bucket" {
  value = aws_s3_bucket.quantum_results.bucket
} 
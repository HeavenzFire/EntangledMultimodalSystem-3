import os
from azure.identity import DefaultAzureCredential
from azure.storage.blob import BlobServiceClient
import boto3

# Azure Integration Example
def azure_integration_example():
    # Set the Azure storage account name and container name
    account_name = os.environ["AZURE_STORAGE_ACCOUNT"]
    container_name = os.environ["AZURE_STORAGE_CONTAINER"]

    # Create a credential object
    credential = DefaultAzureCredential()

    # Create a blob service client
    blob_service_client = BlobServiceClient(
        account_url=f"https://{account_name}.blob.core.windows.net",
        credential=credential
    )

    # Get a reference to the container
    container_client = blob_service_client.get_container_client(container_name)

    # Upload a file to the container
    with open("example.txt", "rb") as data:
        container_client.upload_blob(name="example.txt", data=data)

# AWS Integration Example
def aws_integration_example():
    # Create an S3 client
    s3_client = boto3.client("s3")

    # Upload a file to an S3 bucket
    s3_client.upload_file("example.txt", "my-bucket", "example.txt")

if __name__ == "__main__":
    azure_integration_example()
    aws_integration_example() 
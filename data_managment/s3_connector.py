import json
import logging
import os

import boto3
from botocore.exceptions import ClientError
from tqdm import tqdm


def upload_dataset_version(
    path="data/",
    date="2022-06-30",
    aws_secret_path="s3_secret.json",
    bucket="datasets.ml.labs.aws.intellij.net",
):
    client, resource = aws_connect(aws_secret_path)
    data_files = os.listdir(f"{path}/{date}")
    for data_file in tqdm(data_files):
        obj = f"jupyter-kaggle/{date}/{data_file}"
        file = f"{path}/{date}/{data_file}"
        upload_file(client, file, bucket, obj)


def file_exist(client, bucket, key):
    try:
        client.head_object(Bucket=bucket, Key=key)
    except ClientError as e:
        return False
    return True


def upload_file(client, file_name, bucket, object_name):
    if not file_exist(client, bucket, object_name):
        try:
            response = client.upload_file(file_name, bucket, object_name)
        except ClientError as e:
            logging.error(e)
            return False
    return True


def download_dataset_version(
    date,
    dataset="jupyter-kaggle",
    path_to_secret="s3_secret.json",
    bucketName="datasets.ml.labs.aws.intellij.net",
):
    client, resource = aws_connect(path_to_secret)

    if not os.path.exists(f"data/{date}"):
        os.makedirs(f"data/{date}")

    bucket = resource.Bucket(bucketName)
    for obj in tqdm(bucket.objects.filter(Prefix=f"{dataset}/{date}")):
        if not os.path.exists(f"data/{date}/{obj.key.split('/')[-1]}"):
            bucket.download_file(obj.key, f"data/{date}/{obj.key.split('/')[-1]}")


def aws_connect(path):
    with open(path) as file:
        key = json.load(file)

    client = boto3.client(
        "s3",
        aws_access_key_id=key["aws_key_id"],
        aws_secret_access_key=key["aws_key"],
        region_name=key["region"],
    )

    resource = boto3.resource(
        "s3",
        aws_access_key_id=key["aws_key_id"],
        aws_secret_access_key=key["aws_key"],
        region_name=key["region"],
    )

    return client, resource

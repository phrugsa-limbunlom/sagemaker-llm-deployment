import argparse
import os
from datetime import datetime
from typing import Optional

import boto3
import sagemaker
import yaml
from sagemaker import image_uris
from sagemaker.model import Model

from sagemaker.huggingface import HuggingFaceModel
from dotenv import find_dotenv, load_dotenv


def get_LMI_image():
    """Get the latest LMI container image URI.
    This is the pre-built AWS container with vLLM support."""

    # image_uri = image_uris.retrieve(
    #     framework="djl-lmi",  # or "djl-deepspeed"
    #     region=sagemaker_session.boto_session.region_name,
    #     version= "0.30.0"
    # )

    REGION = sagemaker_session.boto_session.region_name

    CONTAINER_VERSION = '0.33.0-lmi15.0.0-cu128'

    image_uri = f'763104351884.dkr.ecr.{REGION}.amazonaws.com/djl-inference:{CONTAINER_VERSION}'

    return image_uri

def create_model(model_name: str, model_id : Optional[str], role: str, image_uri: str):
    """Create HuggingFace model from Hub or model artifacts."""

    load_dotenv(find_dotenv())

    HF_TOKEN = os.getenv("HF_TOKEN")

    if isinstance(model_id,str):

        env = {
                'HF_MODEL_ID': model_id,
                'HF_TOKEN': HF_TOKEN,
                'HF_TASK': 'text-generation',
                'SAGEMAKER_CONTAINER_LOG_LEVEL': '20',
                'OPTION_ASYNC_MODE' : 'true',
                'OPTION_ROLLING_BATCH' : 'disable',
                'OPTION_ENTRYPOINT' : 'djl_python.lmi_vllm.vllm_async_service',
                'TENSOR_PARALLEL_DEGREE' : 'max'
        }

        assert env['HF_TOKEN'] == HF_TOKEN, "You have to provide a token."

        return HuggingFaceModel(
            env=env,
            role=role,
            name=model_name,
            sagemaker_session=sagemaker_session,
            image_uri=image_uri,
        )
    else:
        return Model(
        image_uri=image_uri,
        role=role,
        name=model_name,
        sagemaker_session=sagemaker_session,
        model_data=f"s3://{data["bucket"]}/{data['model_artifacts']}"
    )


def deploy_model(endpoint_name : str, model : Model):
    model.deploy(
        initial_instance_count=1,
        instance_type=data['instance_type'],
        endpoint_name=endpoint_name,
        container_startup_health_check_timeout=1800,  # 30 minutes
        model_data_download_timeout=1200,  # 20 minutes for model download
    )

def read_config(file_path : str):

    with open(file_path, 'r') as file:
        data = yaml.safe_load(file)

    return data

if __name__ == "__main__":

    data = read_config('config.yaml')

    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", type=str)
    parser.add_argument("--endpoint-name", type=str)
    args = parser.parse_args()

    timestamp = datetime.now().strftime('%Y-%m-%d-%H-%M')

    model_name = args.model_name + "-" + timestamp
    endpoint_name = args.endpoint_name + "-" + timestamp

    # Initialize SageMaker session
    sagemaker_session = sagemaker.Session(boto_session=boto3.Session(region_name=data['region']))

    image_uri = get_LMI_image()
    print(f"Using LMI container: {image_uri}")

    print(f"Creating model: {model_name}")
    model = create_model(model_name=model_name, model_id=data['hf_model_id'], role=data['role'], image_uri=image_uri)
    print(f"Model created: {model_name}")

    print(f"Deploying model: {model_name}")
    deploy_model(endpoint_name=endpoint_name, model=model)
    print(f"Model deployed to endpoint: {endpoint_name}")

    print(f"Endpoint URL: {endpoint_name}")
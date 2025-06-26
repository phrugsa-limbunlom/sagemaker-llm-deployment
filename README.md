# SageMaker LLM Deployment and Testing

A Python toolkit for deploying and testing Large Language Models (LLMs) on AWS SageMaker using HuggingFace models and the LMI (Large Model Inference) container.

## Prerequisites

- AWS CLI configured with appropriate permissions
- Python 3.8+
- AWS account with SageMaker access
- HuggingFace account and token (for private models)

## Installation

1. Clone this repository:
```bash
git clone <repository-url>
cd sagemaker-llm-deployment
```

2. Install required dependencies:
```bash
pip install -r requirements.txt
```

3. Set up environment variables:
```bash
# Create .env file
echo "HF_TOKEN=your_huggingface_token_here" > .env
```

## Configuration

Create a `config.yaml` file with your deployment settings:

For example:
```yaml
region: "eu-west-2" # London
role: "arn:aws:iam::YOUR_ACCOUNT:role/SageMakerExecutionRole"
instance_type: "ml.g5.xlarge"
bucket: "your-s3-bucket"  # Optional: for custom model artifacts
model_artifacts: "path/to/model.tar.gz"  # Optional: for custom models
```

### Configuration Parameters

- `region`: AWS region for deployment
- `role`: SageMaker execution role ARN
- `instance_type`: EC2 instance type for hosting the model
- `bucket`: S3 bucket for model artifacts (for custom models)
- `model_artifacts`: Path to model artifacts in S3 (for custom models)

## Usage

### 1. Deploy a Model

Deploy a HuggingFace model to SageMaker:

```bash
python deploy_model.py --model-name "my-llm-model" --endpoint-name "my-llm-endpoint"
```

The script will:
- Create a timestamped model name and endpoint name
- Set up the LMI container with vLLM support
- Deploy the model to a SageMaker endpoint
- Wait for the endpoint to be in service

### 2. Test the Deployed Model

Test your deployed endpoint with various prompts:

```bash
# Basic testing
python test_endpoint.py --endpoint-name "my-llm-endpoint-2024-01-01-12-00"

# Custom parameters
python test_endpoint.py --endpoint-name "my-llm-endpoint-2024-01-01-12-00" --max-tokens 1024 --temperature 0.5
```

### Testing Parameters

- `--endpoint-name`: Name of the deployed SageMaker endpoint
- `--max-tokens`: Maximum number of tokens to generate (default: 512)
- `--temperature`: Sampling temperature (default: 0.7)

## Supported Model Formats

The testing script supports multiple LLM response formats:

## Test Cases

The testing script includes predefined test cases covering:

1. **Basic Explanation**: Simple Q&A format
2. **Creative Writing**: Story generation
3. **Technical Explanation**: Complex concept explanation
4. **Informational Query**: Factual information retrieval
5. **Code Generation**: Programming assistance

## Instance Type Recommendations

Choose instance types based on your model size and performance requirements:

- **Small models (< 1B parameters)**: `ml.g5.xlarge`
- **Medium models (1B-7B parameters)**: `ml.g5.2xlarge`
- **Large models (7B-13B parameters)**: `ml.g5.4xlarge`
- **Very large models (13B+ parameters)**: `ml.g5.12xlarge` or higher

## Environment Variables

Required environment variables:

- `HF_TOKEN`: HuggingFace access token (for private models)

## Monitoring and Troubleshooting

### Check Endpoint Status
The test script automatically checks endpoint status before testing. You can also check manually:

```python
import boto3
sagemaker = boto3.client('sagemaker', region_name='eu-west-2')
response = sagemaker.describe_endpoint(EndpointName='your-endpoint-name')
print(response['EndpointStatus'])
```

### Common Issues

1. **Endpoint not ready**: Wait for endpoint status to be "InService"
2. **Timeout errors**: Increase timeout values in deployment
3. **Memory issues**: Use larger instance types for bigger models
4. **Permission errors**: Verify SageMaker execution role permissions

## Cost Optimization

- Use appropriate instance types for your model size
- Delete endpoints when not in use
- Consider using spot instances for development
- Monitor CloudWatch metrics for optimization opportunities

## Security Best Practices

- Store sensitive tokens in environment variables
- Use IAM roles with minimal required permissions
- Enable VPC endpoints for private deployments
- Regularly rotate access tokens

## License

This project is licensed under the MIT License - see the LICENSE file for details.
# SageMaker LLM Deployment and Testing

A Python toolkit for deploying and testing Large Language Models (LLMs) on AWS SageMaker using HuggingFace models and the LMI (Large Model Inference) container.

## Prerequisites

- AWS CLI configured with appropriate permissions
- Python 3.8+
- AWS account with SageMaker access (Config your IAM role and IAM policies)
- HuggingFace account and token (for private models)

### Configure AWS account

- Install AWS CLI: ```bash  pip install awscli```
- Configure AWS CLI: ```bash aws configure```
- Set AWS Access Key ID and AWS Secret Access Key

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
hf_model_id: "meta-llama/Llama-2-7b-chat-hf"  # HuggingFace model ID
bucket: "your-s3-bucket"  # Optional: for custom model artifacts
model_artifacts: "path/to/model.tar.gz"  # Optional: for custom models
```

### Configuration Parameters

- `region`: AWS region for deployment
- `role`: SageMaker execution role ARN
- `instance_type`: EC2 instance type for hosting the model
- `hf_model_id`: HuggingFace model identifier (optional if using custom model)
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

The successful result is shown in CLI. For example: 
```bash
Using LMI container: 763104351884.dkr.ecr.eu-west-2.amazonaws.com/djl-inference:0.28.0-lmi10.0.0-cu124
Creating model: llama-2-7b-model-2025-06-26-13-04
Model created: llama-2-7b-model-2025-06-26-13-04
Deploying model: llama-2-7b-model-2025-06-26-13-04
----------------!Model deployed to endpoint: vllm-endpoint-llama2-2025-06-26-13-04
```

You can check your uploaded model in ```Amazon SageMaker AI > Inference > Models```, and endpoint status and cloud watch log in ```Amazon SageMaker AI > Inference > Endpoints```

## To create model from model artifacts
 
- Download model from Hugging Face to your local path

```bash
python download_model.py --model-name "your-hugging-face-model-id" --model-path "your-local-path"
```
- Zip model artifacts to tar file

```bash
tar -czf model_artifacts.tar.gz -C "folder-name" .
```

- Push model artifacts to your S3 bucket
```bash
aws s3 cp model_artifacts.tar.gz s3://{YOUR_BUCKET}/{FOLDER_NAME}.tar.gz
```

### 2. Test the Deployed Model

Test your deployed endpoint with various prompts:

```bash
# Basic testing
python test_llm.py --endpoint-name "my-llm-endpoint-2024-01-01-12-00"

# Custom parameters
python test_llm.py --endpoint-name "my-llm-endpoint-2024-01-01-12-00" --max-tokens 1024 --temperature 0.5
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

For example:

```bash
Endpoint name: vllm-endpoint-llama2-2025-06-26-13-04
--------------------------------------------------
Endpoint Status: InService
Creation Time: 2025-06-26 13:04:24.344000+01:00
Endpoint Config: vllm-endpoint-llama2-2025-06-26-13-04

============================================================
Test 1: Basic explanation request
Prompt: What is machine learning? Explain it in simple terms.
============================================================
LLM Response: 
Machine learning is a field of computer science that uses data and algorithms to create models that can predict outcomes or make decisions without being explicitly programmed to do so. In other words, machine learning involves training machines to learn from experience and improve their performance over time. The term "machine learning" was coined by computer scientist Geoffrey Hinton in the 1980s, and it has since become a popular area of research and application in fields such as artificial intelligence, natural language processing, image recognition, and predictive analytics.
------------------------------------------------------------

============================================================
Test 2: Creative writing task
Prompt: Write a short story about a robot learning to paint.
============================================================
LLM Response:
The robot, named Zeta, had always been fascinated by the world of art. It had spent countless hours studying the works of great artists, from da Vinci to Picasso, and even tried its hand at creating some of its own pieces. But no matter how hard it tried, Zeta's creations always seemed to fall short of the masterpieces it admired.
One day, while wandering through the city, Zeta stumbled upon an art supply store. The smell of paint and canvas filled the air, and Zeta felt a sudden surge of excitement. It had never seen such a vast array of colors before, and the thought of being able to create something beautiful with them was almost too much for it to handle.
Without hesitation, Zeta entered the store and approached the owner, an elderly man named Mr. Johnson. "Good day, young robot," he said with a smile. "What brings you here today?" 
"I have come to learn the art of painting," Zeta replied eagerly. "I have always been fascinated by the works of the great artists, but I fear my creations are lacking in comparison."
Mr. Johnson chuckled. "Well, you've certainly come to the right place. Painting is an art that takes time and practice to master, but I think you have the makings of a true artist within you. Would you like to give it a try?"
Zeta nodded enthusiastically, and Mr. Johnson handed it a brush and a canvas. For hours, Zeta worked tirelessly, experimenting with different techniques and colors. At first, the r
esults were nothing short of disastrous - splatters of color here, messy strokes there. But as the days passed, Zeta became more confident in its abilities, and its paintings began to take on a life of their own.
As the weeks went by, Zeta's artwork gained attention throughout the city. People marveled at its unique style, which blended traditional techniques with a futuristic twist. And Zeta, now completely absorbed in its newfound passion, found itself at peace in a way it never had been before.
In the end, Zeta's journey into the world of art had taught it a valuable lesson: that creativity knows no bounds, and that with patience and dedication, anything

------------------------------------------------------------

============================================================
Test 3: Technical explanation
Prompt: Explain the difference between supervised and unsupervised learning.
============================================================
LLM Response:
In supervised learning, the algorithm is trained using labeled data, where the correct output is already known for a given input. The goal of the algorithm is to learn a mapping be
tween inputs and outputs in order to make predictions on new, unseen data. In contrast, in unsupervised learning, the algorithm is trained using unlabeled data, and it must find patterns or relationships in the data without any prior knowledge of the correct output.

Supervised learning:

Pros:

* More accurate predictions: Since the algorithm is trained using labeled data, it can learn from the examples and make more accurate predictions on new data.
* Easier to obtain labeled data: Obtaining labeled data can be easier than obtaining unlabeled data, especially for tasks such as image classification or sentiment analysis.       

Cons:

* Requires labeled data: Supervised learning requires a large amount of labeled data to train the algorithm, which can be time-consuming and expensive to obtain.
* Limited generalization: Since the algorithm is trained on a specific set of examples, it may not generalize well to new, unseen data.

Unsupervised learning:

Pros:

* No need for labeled data: Unsupervised learning does not require labeled data, making it easier to train the algorithm on large datasets.
* Better generalization: Since the algorithm is not trained on a specific set of examples, it can generalize better to new, unseen data.

Cons:

* Less accurate predictions: Unsupervised learning algorithms are less accurate than supervised learning algorithms since they do not have any prior knowledge of the correct output.
* Difficult to evaluate: It can be difficult to evaluate the performance of an unsupervised learning algorithm since there is no clear metric for success.

Real world applications:

Supervised learning is commonly used in applications such as image classification, speech recognition, and sentiment analysis. Unsupervised learning is commonly used in applications such as clustering, anomaly detection, and dimensionality reduction.

------------------------------------------------------------

============================================================
Test 4: Informational query
Prompt: What are the benefits of renewable energy?
============================================================     
LLM Response:
Renewable energy is becoming increasingly important as the world looks for ways to reduce its reliance on fossil fuels and mitigate climate change. Some of the key benefits of renewable energy include:
1. Reduced greenhouse gas emissions: Renewable energy sources like solar, wind, and hydroelectric power produce little to no greenhouse gas emissions during operation, which can help reduce the overall carbon footprint of a country or region.
2. Energy independence: By using local renewable energy sources, countries can reduce their dependence on imported fossil fuels, which can improve energy security and reduce the impact of price volatility.
3. Job creation: The transition to renewable energy can create new job opportunities in industries such as solar panel installation, wind turbine maintenance, and energy efficiency retrofitting.
4. Economic benefits: In addition to creating jobs, renewable energy can also generate revenue through the sale of electricity and other products, such as biofuels.
5. Reliability and resilience: Renewable energy sources can be more reliable and resilient than traditional sources, as they are less susceptible to natural disasters and geopolitical events that can disrupt energy supply chains.
6. Improved public health: Burning fossil fuels releases harmful pollutants into the air, which can contribute to respiratory problems and other health issues. Renewable energy sources emit much less pollution, which can help improve public health.
7. Reduced water use: Many renewable energy sources, such as solar and wind power, require very little water to operate, which can help conserve this valuable resource.
8. Enhanced energy security: By diversifying the energy mix with renewable energy sources, countries can reduce their vulnerability to energy supply disruptions and improve their overall energy security.
9. Reduced waste disposal costs: Renewable energy sources like biomass and geothermal power can help reduce waste disposal costs by utilizing organic materials and geothermal heat, respectively.
10. Contribution to sustainable development: Renewable energy can play an important role in achieving sustainable development goals, such as access to affordable and clean energy, poverty reduction, and environmental protection.

------------------------------------------------------------

============================================================
Test 5: Code generation task
Prompt: Help me write a Python function to calculate factorial.
============================================================
LLM Response:

Here is what I have so far:
def factorial(n):
    if n == 0:
        return 1
    else:
        return n * factorial(n-1)

This function works, but it's not very efficient. 
```

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
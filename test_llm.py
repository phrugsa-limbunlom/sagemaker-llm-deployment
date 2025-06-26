import argparse
import json
import boto3


def test_llm_endpoint(endpoint_name: str, prompt: str, **kwargs):
    """Test LLM endpoint with text prompt"""

    runtime = boto3.client('sagemaker-runtime', region_name='eu-west-2')

    payload = {
        "inputs": prompt,
        "parameters": {
            "max_new_tokens": kwargs.get("max_new_tokens", 512),
            "temperature": kwargs.get("temperature", 0.7),
            "top_p": kwargs.get("top_p", 0.9),
            "top_k": kwargs.get("top_k", 50),
            "do_sample": kwargs.get("do_sample", True),
            "repetition_penalty": kwargs.get("repetition_penalty", 1.1),
            "return_full_text": kwargs.get("return_full_text", False)
        }
    }

    try:
        response = runtime.invoke_endpoint(
            EndpointName=endpoint_name,
            ContentType='application/json',
            Body=json.dumps(payload)
        )

        result_str = response['Body'].read().decode('utf-8')

        result_dict = json.loads(result_str)

        generated_text = result_dict["generated_text"]

        return generated_text

    except Exception as e:
        print(f"Error calling endpoint: {e}")
        return None

def check_endpoint_status(endpoint_name: str):
    """Check the status of the SageMaker endpoint"""

    sagemaker = boto3.client('sagemaker', region_name='eu-west-2')

    try:
        response = sagemaker.describe_endpoint(EndpointName=endpoint_name)
        print(f"Endpoint Status: {response['EndpointStatus']}")
        print(f"Creation Time: {response['CreationTime']}")
        print(f"Endpoint Config: {response['EndpointConfigName']}")

        if response['EndpointStatus'] != 'InService':
            print(f"⚠️  Endpoint is not ready. Current status: {response['EndpointStatus']}")
            return False
        return True

    except Exception as e:
        print(f"Error describing endpoint: {e}")
        return False


# Example usage
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test LLM SageMaker endpoint")
    parser.add_argument("--endpoint-name", type=str, required=True, help="SageMaker endpoint name")
    parser.add_argument("--max-tokens", type=int, default=512, help="Maximum new tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature")

    args = parser.parse_args()

    ENDPOINT_NAME = args.endpoint_name

    print(f"Endpoint name: {ENDPOINT_NAME}")
    print("-" * 50)

    # Check endpoint status
    if not check_endpoint_status(ENDPOINT_NAME):
        print("Endpoint is not ready. Exiting...")
        exit(1)

    # Test cases for LLM
    test_cases = [
        {
            "prompt": "What is machine learning? Explain it in simple terms.",
            "description": "Basic explanation request"
        },
        {
            "prompt": "Write a short story about a robot learning to paint.",
            "description": "Creative writing task"
        },
        {
            "prompt": "Explain the difference between supervised and unsupervised learning.",
            "description": "Technical explanation"
        },
        {
            "prompt": "What are the benefits of renewable energy?",
            "description": "Informational query"
        },
        {
            "prompt": "Help me write a Python function to calculate factorial.",
            "description": "Code generation task"
        }
    ]

    for i, test in enumerate(test_cases):
        print(f"\n{'=' * 60}")
        print(f"Test {i + 1}: {test['description']}")
        print(f"Prompt: {test['prompt']}")
        print(f"{'=' * 60}")

        generated_text = test_llm_endpoint(
            ENDPOINT_NAME,
            test['prompt'],
            max_new_tokens=args.max_tokens,
            temperature=args.temperature
        )

        print(f"LLM Response: {generated_text}")

        print("\n" + "-" * 60)
import json
import boto3
import os

ENDPOINT_NAME = os.environ['ENDPOINT_NAME']
runtime = boto3.client('runtime.sagemaker')


def format_prompt(message, history, system_prompt):
    prompt = ""
    if system_prompt:
        prompt += f"System: {system_prompt}\n"
    for user_prompt, bot_response in history:
        prompt += f"User: {user_prompt}\n"
        # Response already contains "Falcon: "
        prompt += f"Falcon: {bot_response}\n"
    prompt += f"""User: {message}
Falcon:"""
    return prompt


def lambda_handler(event, context):

    data = json.loads(json.dumps(event))
    print("data = ", data)

    parameters = {
        "do_sample": True,
        "top_p": 0.9,
        "temperature": 0.8,
        "max_new_tokens": 1024,
        "repetition_penalty": 1.03,
        "stop": ["\nUser:", "<|endoftext|>", " User:", "###"],
    }

    formatted_prompt = format_prompt(
        data['query'], [], "You are a helpful assistant.")
    payload = {"inputs": formatted_prompt, "parameters": parameters}

    response = runtime.invoke_endpoint(
        EndpointName=ENDPOINT_NAME,
        ContentType="application/json",
        Body=json.dumps(payload))

    print(response)

    generation = json.loads(response['Body'].read().decode('utf-8'))
    final_response = generation[0]['generated_text']

    return {
        'statusCode': 200,
        'headers': {"Access-Control-Allow-Origin": "*", },
        'body': json.dumps(final_response)
    }

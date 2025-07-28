"""Functions for sending prompts to GPT and OLLAMA models, with retry logic and model selection."""

import importlib
import urllib.request
import requests
import json
import ssl
import regex as re
from tenacity import (
    retry,
    stop_after_attempt,  # Used to limit retry attempts
    wait_exponential     # Exponential backoff for retries
)

gpt_250 = 'fd141762ad904a91b170781fcb428b04'  # GPT-4 enabled API key

def endUrl(deployment, api,
           base='https://apigw.rand.org/openai/RAND/inference/deployments/', 
           method='/chat/completions?api-version='):
    """Constructs the full API endpoint URL.

    Args:
        deployment (str): Model deployment name.
        api (str): API version.
        base (str): Base URL for the API.
        method (str): Method path for the API.

    Returns:
        str: Full endpoint URL.
    """
    return base + deployment + method + api

def sendRequest(url, hdr, data):
    """Sends a POST request to the specified URL with headers and data.

    Args:
        url (str): Endpoint URL.
        hdr (dict): HTTP headers.
        data (dict): Data payload.

    Returns:
        dict: Decoded JSON response.

    Raises:
        URLError: If a network error occurs.
        Exception: For any other error.
    """
    data = json.dumps(data)
    context = ssl._create_unverified_context()  # Disable SSL verification
    req = urllib.request.Request(url, headers=hdr, data=bytes(data.encode("utf-8")))
    req.get_method = lambda: 'POST'
    
    try:
        response = urllib.request.urlopen(req, context=context, timeout=10)  # 10s timeout
        content = bytes.decode(response.read(), 'utf-8')
        return json.loads(content)
    except urllib.error.URLError as e:
        print(f"Network error: {e}")
        raise
    except Exception as e:
        print(f"Unexpected error: {e}")
        raise

@retry(wait=wait_exponential(multiplier=1, min=2, max=7))
def gptRespond(prompt, context='', t=1, c=1, model='4om', n=1, print_rslt=False):
    """Makes text calls to RAND's internal GPT.

    Args:
        prompt (str): User prompt.
        context (str): System context.
        t (float): Temperature.
        c (float): Top-p.
        model (str): Model key.
        n (int): Number of completions.
        print_rslt (bool): Print results if True.

    Returns:
        list: List of response strings.

    Raises:
        Exception: If the request fails.
    """
    key = gpt_250
    try:
        api = '2024-06-01'  # API version
        
        Deployment = {
            '3': 'gpt-35-turbo-v0125-base',
            '4': 'gpt-4-v0613-base',
            '4o': 'gpt-4o-2024-08-06',
            '4om': 'gpt-4o-mini-2024-07-18',
        }
    
        Model = {
            '3': 'gpt-35-turbo',
            '4': 'gpt-4',
            '4o': 'gpt-4o',
            '4om': 'gpt-4o-mini',
        }
        
        deployment = Deployment[model]
        model = Model[model]
        
        url = endUrl(deployment, api)
        
        hdr = {
            'Content-Type': 'application/json',
            'Cache-Control': 'no-cache',
            'Ocp-Apim-Subscription-Key': key,
        }
        
        data = {
            'model': model,
            'messages': [
                {'role': 'system', 'content': context},
                {'role': 'user', 'content': prompt}],
            'temperature': t,
            'top_p': c,
            'n': n,
        }
        
        res = sendRequest(url, hdr, data)
        
        Results = [res['choices'][i]['message']['content'] for i in range(n)]
        if print_rslt:
            for answer in Results:
                print('#---------------------------------#')
                print(answer)
        return Results
    except Exception as e:
        print(e)
        raise

@retry(wait=wait_exponential(multiplier=1, min=2, max=5))
def olmRespond(prompt, context='', t=1, c=1, model='r1:8', n=1,
               url = f'http://127.0.0.1:11434/api/chat',
               print_rslt = False, filter_think = True):
    """Makes text calls to a local OLLAMA model.

    Args:
        prompt (str): User prompt.
        context (str): System context.
        t (float): Temperature.
        c (float): Top-p.
        model (str): Model key.
        n (int): Number of completions.
        url (str): OLLAMA API endpoint.
        print_rslt (bool): Print results if True.
        filter_think (bool): Remove <think> tags if True.

    Returns:
        list: List of response strings.

    Raises:
        Exception: If the request fails.
    """
    try:
        Model = {
            'l3.2': 'llama3.2',
            'r1:8': 'deepseek-r1:8b',
            'g3:12': 'gemma3:12b'
        }
        
        hdr = {
            'Content-Type': 'application/json',
        }
        
        mdl = Model[model]
        data = {
            'model': mdl,
            'messages': [
                {'role': 'system', 'content': context},
                {'role': 'user', 'content': prompt}],
            'temperature': t,
            'top_p': c,
            'stream': False
        }
        Results = []
        for i in range(n):
            completion_response = requests.post(url, json=data)
            Result = completion_response.json()['message']['content']
            if filter_think:
                Result = think_masking(Result)
            Results.append(Result)
            if print_rslt:
                print(Result)
                print('#---------------------------------#')
        if print_rslt:
            print("DONE.")
        return Results
    except Exception as e:
        print(e)
        raise 

def think_masking(json_string):
    """Removes <think>...</think> tags from a string.

    Args:
        json_string (str): Input string.

    Returns:
        str: Cleaned string.
    """
    cleaned_string = re.sub(r'<think>.*?</think>', '', json_string, flags=re.DOTALL).strip()
    return cleaned_string
        
def Respond(*args, model='4om', **kwargs):
    """Dispatches to the correct response function based on model.

    Args:
        *args: Arguments for the response function.
        model (str): Model key.
        **kwargs: Keyword arguments for the response function.

    Returns:
        list: List of response strings.

    Raises:
        ValueError: If the model is not recognized.
    """
    Model_Choice = {
        '3': 'gpt',
        '4': 'gpt',
        '4o': 'gpt',
        '4om': 'gpt',
        '3.2': 'olm',
        'r1:8': 'olm'
    }

    choice = Model_Choice.get(model)
    if choice == 'gpt':
        return gptRespond(*args, model=model, **kwargs)
    elif choice == 'olm':
        return olmRespond(*args, model=model, **kwargs)
    else:
        print("Model not recognized")
        raise ValueError("Invalid model choice")

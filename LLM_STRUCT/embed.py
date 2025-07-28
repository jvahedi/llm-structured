"""Embedding utilities for GPT and OLLAMA models."""

from openai import AzureOpenAI
import numpy as np
import ollama
from tenacity import retry, wait_fixed

gpt_250 = 'fd141762ad904a91b170781fcb428b04'
rand_endpoint = 'https://apigw.rand.org/openai/RAND/inference'
api_ver = '2024-02-01'

model_dict = {
    '2': 'text-embedding-ada-002-v2-base',
    '3': 'text-embedding-3-large-v1-base'
}
KEY = gpt_250

client = AzureOpenAI(
  api_key=KEY,
  api_version=api_ver,
  azure_endpoint=rand_endpoint,
  default_headers={"Ocp-Apim-Subscription-Key": KEY},
)

@retry(wait=wait_fixed(3))
def gpt_embed(text, ver='3', dims=3072):
    """Get embeddings from GPT model.

    Args:
        text (str or list): Text(s) to embed.
        ver (str): Model version.
        dims (int): Embedding dimensions.

    Returns:
        Embedding object: OpenAI embedding response.
    """
    mod_emb = model_dict[ver]
    args = {
        'input': text if type(text) != 'str' else [text],
        'model': mod_emb,
    } | ({
        'dimensions': dims } if mod_emb == 'text-embedding-3-large-v1-base'
                             else {})
    return client.embeddings.create(**args)

def strip(documents):
    """Extracts embedding matrix from OpenAI embedding response.

    Args:
        documents: OpenAI embedding response.

    Returns:
        np.ndarray: Matrix of embeddings.
    """
    matrix = np.array([documents.data[i].embedding for i in range(len(documents.data))])
    return matrix

def gptEmbed(text, ver='3', dims=3072, chunk=10000, prnt_prg=False):
    """Batch embedding for large text lists using GPT.

    Args:
        text (str or list): Text(s) to embed.
        ver (str): Model version.
        dims (int): Embedding dimensions.
        chunk (int): Chunk size for batching.
        prnt_prg (bool): Print progress if True.

    Returns:
        np.ndarray: Matrix of embeddings.
    """
    if type(text) == 'str':
        text = [text]
        
    L = len(text)
    embeddings = np.array([])
    points = [i for i in range(0, L, chunk)] + [L]
    for j in range(len(points) - 1):
        if prnt_prg:
            print((points[j], points[j+1]))
        data = gpt_embed(text[points[j]:points[j+1]], ver=ver, dims=dims)
        embeddings_chunk = strip(data)
        embeddings = np.vstack([embeddings, embeddings_chunk]) if embeddings.size else embeddings_chunk
    return embeddings
    
def olm_embed(text, ver='nomic-embed-text', dims=768):
    """Get embedding from OLLAMA model.

    Args:
        text (str): Text to embed.
        ver (str): Model version.
        dims (int): Embedding dimensions.

    Returns:
        list: Embedding vector.
    """
    response = ollama.embeddings(model=ver, prompt=text)
    embedding = response.embedding
    return embedding

def olmEmbed(text, ver='nomic-embed-text', dims=768, chunk=1, prnt_prg=False):
    """Batch embedding for large text lists using OLLAMA.

    Args:
        text (str or list): Text(s) to embed.
        ver (str): Model version.
        dims (int): Embedding dimensions.
        chunk (int): Chunk size for batching.
        prnt_prg (bool): Print progress if True.

    Returns:
        np.ndarray: Matrix of embeddings.
    """
    if type(text) == 'str':
        text = [text]
    L = len(text)
    embeddings = []
    for j in range(L):
        if prnt_prg:
            print(j)
        data = olm_embed(text[j], ver=ver, dims=dims)
        embeddings.append(data)
    return np.array(embeddings)

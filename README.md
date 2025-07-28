# LLM Structured Response Toolkit

A Python package for **structured, robust, and scalable interaction with Large Language Models (LLMs)**, including OpenAI GPT (via Azure) and local OLLAMA models. This toolkit enables users to send prompts, receive structured responses, validate outputs, and process results into tabular formats for downstream analysis.

---

## Features

- **Unified LLM Interface:**  
  Seamlessly interact with both cloud-based (OpenAI/Azure) and local (OLLAMA) LLMs using a consistent API.

- **Structured Output Validation:**  
  Enforce and validate JSON schema for LLM responses using Pydantic models.

- **Batch Embedding Utilities:**  
  Generate and process text embeddings in bulk from both GPT and OLLAMA models.

- **Tabular Data Processing:**  
  Convert unstructured LLM outputs into structured pandas DataFrames, with support for voting, aggregation, and error handling.

- **Automated Retry and Error Handling:**  
  Built-in retry logic for robust API calls and validation.

- **Customizable Prompting and Schema:**  
  Easily define custom instructions and output formats for a wide range of tasks.

---

## Installation Instructions

1. **Clone the repository:**
   ```bash
   git clone https://github.com/yourusername/llm-structured-response-toolkit.git
   cd llm-structured-response-toolkit
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

   *Typical dependencies include:*
   - `openai`
   - `ollama`
   - `tenacity`
   - `pandas`
   - `numpy`
   - `pydantic`
   - `tqdm`
   - `requests`
   - `regex`

3. **(Optional) Set up OLLAMA for local LLM inference:**  
   See [OLLAMA documentation](https://ollama.com/) for installation and model setup.

---

## Basic Usage

### 1. Sending a Prompt and Getting a Structured Response

```python
from respond import Respond

prompt = "Summarize the following text in one sentence."
context = "You are a helpful assistant."
result = Respond(prompt, context, model='4om')
print(result)
```

### 2. Embedding Text

```python
from embed import gptEmbed, olmEmbed

texts = ["This is a test.", "Another sentence."]
embeddings = gptEmbed(texts)
# or for OLLAMA
embeddings_ollama = olmEmbed(texts)
```

### 3. Structuring Unstructured Data

```python
from structure import Structure, collapse

import pandas as pd

unstructured_text = pd.Series([
    "The attack resulted in 3 fatalities and 5 wounded.",
    "No casualties were reported."
])

format_dict = {
    "Fatalities": int,
    "Wounded": int
}
instructions = "Extract the number of fatalities and wounded from the text."

Outcomes, Reasoning = Structure(unstructured_text, instructions, format_dict, vote_num=3)
df = collapse(Outcomes, format_dict, unstructured_text)
print(df)
```

### 4. Validating and Processing QA Tasks

```python
from validate import qaProcess

items = ['hospital', 'school', 'park']
terms = {
    'a medical facility': 'Medical',
    'an educational institution': 'School',
    'a public space': 'Public'
}

qa_table = qaProcess(items, terms)
print(qa_table)
```

---

## Screenshots or Diagrams

*Under Construction*

---

## License

Licensed under the Apache License, Version 2.0. See the [LICENSE](LICENSE) file for more information.

## Contact

Feel free to reach out at [Vahedi.john@columbia.edu](mailto:Vahedi.john@columbia.edu) for queries or support.

---

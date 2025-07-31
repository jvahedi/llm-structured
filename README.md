# LLM Structured Response Toolkit

A Python package for **structured, robust, and scalable interaction with Large Language Models (LLMs)**, including OpenAI GPT (via Azure) and local OLLAMA models. This toolkit enables users to send prompts, receive validated structured responses, and process outputs into tabular formats for downstream analysis.

---

## Features

* **Unified LLM Interface**
  Seamlessly interact with both cloud-based (OpenAI/Azure) and local (OLLAMA) LLMs using a consistent API.

* **Structured Output Validation**
  Enforce and validate JSON-style responses using Pydantic models.

* **Tabular Data Processing**
  Convert unstructured LLM outputs into structured `pandas` DataFrames for easy analysis.

* **Automated Retry and Error Handling**
  Robust built-in retry logic with error handling for API calls and validation.

* **Voting and Aggregation**
  Support for odd-number voting via high-temperature sampling, with downstream aggregation and table highlighting for disagreement resolution.

* **Customizable Prompting and Schema**
  Define custom instructions and output formats tailored to a wide range of tasks.

* **Batch Embedding Utilities**
  Generate and process text embeddings in bulk using GPT or OLLAMA models.

---

## Installation

1. **Clone the repository**

   ```bash
   git clone https://github.com/yourusername/llm-structured-response-toolkit.git
   cd llm-structured-response-toolkit
   ```

2. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

   *Key dependencies include:*

   * `openai`
   * `ollama`
   * `tenacity`
   * `pandas`
   * `numpy`
   * `pydantic`
   * `tqdm`
   * `requests`
   * `regex`

3. **(Optional) Set up OLLAMA for local inference**
   See the [OLLAMA documentation](https://ollama.com/) for installation and model setup.

---

## Usage

### 1. Send a Prompt and Receive a Structured Response

```python
from respond import Respond

prompt = "Summarize the following text in one sentence."
context = "You are a helpful assistant."
result = Respond(prompt, context, model='4om')
print(result)
```

### 2. Embed Text

```python
from embed import gptEmbed, olmEmbed

texts = ["This is a test.", "Another sentence."]
embeddings = gptEmbed(texts)
# or using OLLAMA:
embeddings_ollama = olmEmbed(texts)
```

### 3. Extract Structured Data from Text

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

### 4. Validate and Process QA Tasks

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

## Visuals

**Screenshots and diagrams coming soon.**

---

## License

Licensed under the Apache License, Version 2.0.
See the [LICENSE](LICENSE) file for full terms.

---

## Contact

For questions or support, contact [vahedi.john@columbia.edu](mailto:vahedi.john@columbia.edu).

---

Let me know if you want a `docx` or markdown file of this, or want to generate a GitHub-friendly `README.md` with badges and a table of contents.

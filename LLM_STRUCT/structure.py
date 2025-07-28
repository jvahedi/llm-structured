"""Functions for structuring, validating, and processing LLM outputs into tabular form."""

import pandas as pd
import numpy as np
from tenacity import retry, stop_after_attempt, wait_exponential, wait_fixed
from typing import Literal
from pydantic import BaseModel, ValidationError
from tqdm import tqdm
import json
import regex as re
import pickle 
import os
import timeit

from .respond import Respond

def Structure(unstructured_text, instructions, format_dict, vote_num, reasoning=True, verbose=False, debug=False, timing=False,
              model='4om', max_retry=7, save_freq=None):
    """Structures unstructured text into a tabular format using LLMs.

    Args:
        unstructured_text (pd.Series): Input text.
        instructions (str): Prompt instructions.
        format_dict (dict or class): Output schema.
        vote_num (int): Number of LLM completions per item.
        reasoning (bool): Include reasoning in output.
        verbose (bool): Print responses if True.
        debug (bool): Store errors if True.
        timing (bool): Store timing if True.
        model (str): Model key.
        max_retry (int): Max retries for validation.
        save_freq (int): Save frequency for intermediate results.

    Returns:
        tuple: Arrays of outcomes, (optionally) reasoning, errors, and times.
    """
    E = []
    class_annotations = {}

    # Prepare class annotations from format_dict
    if isinstance(format_dict, dict):
        class_annotations = {key: (value if isinstance(value, type) else str) for key, value in format_dict.items()}
    elif isinstance(format_dict, type):
        class_annotations = format_dict.__annotations__.copy()
    else:
        raise ValueError("format_dict must be either a dictionary or a class.")

    # Add Reasoning field if needed
    if reasoning:
        class_annotations = {'Reasoning': str, **class_annotations}

    # Dynamically create a Pydantic model for validation
    ResponseModel = type('ResponseModel', (BaseModel,), {
        '__annotations__': class_annotations
    })

    keys = list(class_annotations.keys())
    item_num = len(unstructured_text)
    qust_num = len(keys)
    
    # Build schema string for prompt
    schema_components = []
    schema_components.extend(
        f'  "{key}": "{ResponseModel.__annotations__[key].__name__ if isinstance(ResponseModel.__annotations__[key], type) else str(ResponseModel.__annotations__[key])}"' for key in keys
    )
    schema_components.extend(
        f'  "{key}": "{ResponseModel.__annotations__[key].__name__}"' for key in keys
    )
    schema = '{\n' + ',\n'.join(schema_components) + '\n}'
    
    # Add reasoning instructions if needed
    if reasoning:
        instructions = '''
        Make sure to include a short explanation in the "'Reasoning': 'reasoning_text'," entry of the json, for why you give the answers you provide for the other entries. Do not make use of quotes from the text inside your explanations.\n
        ''' + instructions

    context = f'''
    Please provide only the following information strictly in a structured JSON format, and place no extraneous text, such explainations, anywhere outside of that structure.

    {instructions} Your response should strictly be structured in the following JSON format as delimited by the triple backticks, and ensure that no comments or annotations are included in the JSON output, such as an answers that follow with a "#" or "//". 'Literal' and 'Optional' keywords are meant to instruct what kinds of responses are allowed but do not use those keywords themselves as responses.
    
    ```json
    {schema}
    ``` 
    '''
    
    L = len(unstructured_text)
    Outcomes = np.array([[['']*vote_num]*qust_num]*item_num, dtype=object)
    Reasoning = np.array([[['']*vote_num]*qust_num]*item_num, dtype=object)
    Times = np.array([['']]*item_num, dtype=object)
    
    for i in tqdm(range(item_num)):
        if pd.isna(unstructured_text.iloc[i]):
            # Fill with NaN if input is missing
            for k in range(vote_num):
                for j in range(qust_num):
                    Outcomes[i,j,k] = float('nan')
                    Reasoning[i,j,k] = float('nan')
            Times[i] = 0.0 
            continue

        r = 0
        while True:
            if timing: start = timeit.default_timer() 
            responses = Respond(prompt=unstructured_text.iloc[i],
                                   context=context, model=model,
                                   t=1.6, c=.7,
                                   print_rslt=False, 
                                   n=vote_num
                                  )
            if timing: 
                end = timeit.default_timer()
                timed = end - start        
            try:
                for k in range(vote_num):
                    response = cleaning(responses[k])
                    if verbose:
                        print(response)
                    validated_response = ResponseModel.model_validate_json(response)
                    for j in range(qust_num):
                        name = keys[j]
                        Outcomes[i,j,k] = getattr(validated_response, name)
                        if reasoning:
                            Reasoning[i,j,k] = getattr(validated_response, 'Reasoning')
                if timing: 
                    Times[i] = timed
                break
            except ValidationError as e:
                r += 1
                if debug:
                    print(repr(response))
                    E.append((i, response, e))
                    print('####################################')
                print("Unable to validate LLM response:", e)
                if r > max_retry:
                    for k in range(vote_num):
                        for j in range(qust_num):
                            Outcomes[i,j,k] = "ERR"
                            Reasoning[i,j,k] = "ERR"
                    if timing: 
                        Times[i] = timed
                    break
        if save_freq: 
            conditional_save(count=i, max_count=L, threshold=save_freq,
                             Outcomes=Outcomes,
                             Reasoning=Reasoning,
                             Error=E,
                             Times=Times)

    Out = (Outcomes,)
    if reasoning:
        Out += (Reasoning,)
    if debug:
        Out += (E,)
    if timing:
        Out += (Times,)
    return Out

def conditional_save(count, max_count, threshold=10, **kwargs):
    """Saves intermediate results to disk.

    Args:
        count (int): Current index.
        max_count (int): Total number of items.
        threshold (int): Save every `threshold` items.
        **kwargs: Arrays to save.
    """
    drct = r'Saves' + r'_'
    if (count % threshold == threshold - 1) or (count == max_count - 1):
        os.makedirs(drct, exist_ok=True)
        for key, value in kwargs.items():
            filepath = drct + r'/' + key + r'.pkl'
            pickle.dump(value, open(filepath, 'wb'))
        filepath = drct + r'/count.pkl'
        pickle.dump(count, open(filepath, 'wb'))

def collapse(Outcomes, format_dict, unstructured_text):
    """Collapses vote arrays into a DataFrame.

    Args:
        Outcomes (np.ndarray): Array of votes.
        format_dict (dict or class): Output schema.
        unstructured_text (pd.Series): Input text.

    Returns:
        pd.DataFrame: Collapsed table.
    """
    if isinstance(format_dict, dict):
        keys = list(format_dict.keys())
    elif isinstance(format_dict, type):
        keys = [field for field in format_dict.__annotations__.keys()]
    else:
        raise ValueError("format_dict must be either a dictionary or a class.")

    qust_num = len(keys)
    item_num = len(unstructured_text)

    try:
        Outcomes = Outcomes.astype(float)
    except ValueError:
        pass

    reasoning_included = Outcomes.shape[1] > qust_num

    if reasoning_included:
        Votes = mode(Outcomes[:, 1:1 + qust_num, :], axis=2)[0][:, :, 0]
    else:
        Votes = mode(Outcomes[:, :qust_num, :], axis=2)[0][:, :, 0]
    
    Table = pd.DataFrame(Votes, columns=keys, index=unstructured_text[:item_num]).reset_index()
    return Table

def Process(Outcomes, format_dict, unstructured_text,
              thresh_dis=0.75, thresh_dif=2.0,
              verbose=False,
              color=True, save=False):
    """Processes and colors the QA table.

    Args:
        Outcomes (np.ndarray): Array of votes.
        format_dict (dict or class): Output schema.
        unstructured_text (pd.Series): Input text.
        thresh_dis (float): Disagreement threshold.
        thresh_dif (float): Difficulty threshold.
        verbose (bool): Print debug info.
        color (bool): Color output if True.
        save (bool): Save to Excel if True.

    Returns:
        pd.DataFrame or Styler: QA table.
    """
    items, questions, votes = np.shape(Outcomes)
    if isinstance(format_dict, dict):
        keys = list(format_dict.keys())
    elif isinstance(format_dict, type):
        keys = [field for field in format_dict.__annotations__.keys()]
    else:
        raise ValueError("format_dict must be either a dictionary or a class.")
    
    qust_num = len(keys)
    reasoning_included = questions > qust_num
    QA_Table = collapse(Outcomes, format_dict, unstructured_text)

    if color:
        if reasoning_included:
            x = [[[answers for answers in questions] for questions in objct[1:, : qust_num]] for objct in Outcomes]
        else:
            x = [[[answers for answers in questions] for questions in objct[:, :qust_num]] for objct in Outcomes]

        C = mode(np.array(x).T)[1][0] / votes
        C = pd.DataFrame(C.T, index=QA_Table.index, columns=QA_Table.columns[1:])

        B = C.copy().astype(object)
        mask_dis = (C <= thresh_dis)
        mask_both = mask_dis
        B[mask_dis] = 'dis'
        B[mask_both] = 'both'
        B[B.columns[0]] = 1.0

        QA_Table = QA_Table.style.apply(lambda x: B.applymap(color_cells_bool), axis=None)

    if save:
        QA_Table.to_excel('Categories.xlsx', engine='openpyxl')

    return QA_Table

def color_cells_bool(s):
    """Returns background color for cell based on value."""
    if s == 'dis':
        return 'background-color: #FFB6C1'
    elif s == 'dif':
        return 'background-color: #DCDCDC'
    elif s == 'both':
        return 'background-color: #F08080'
    else:
        return ''

def mode(a, axis=0):
    """Computes the mode and counts along an axis.

    Args:
        a (np.ndarray): Input array.
        axis (int): Axis to compute mode.

    Returns:
        tuple: (mode array, count array)
    """
    a_str = a.astype(str)
    scores = np.unique(np.ravel(a_str))
    testshape = list(a.shape)
    testshape[axis] = 1
    oldmostfreq = np.empty(testshape, dtype=object)
    oldcounts = np.zeros(testshape, dtype=int)

    for score in scores:
        template = (a_str == score)
        counts = np.expand_dims(np.sum(template, axis), axis)
        mostfrequent = np.where(counts > oldcounts, score, oldmostfreq)
        oldcounts = np.maximum(counts, oldcounts)
        oldmostfreq = mostfrequent

    shape = oldmostfreq.shape
    for i in range(shape[0]):
        for j in range(shape[1]):
            for k in range(shape[2]):
                slice_idx = [i, j, k]
                slice_idx[axis] = slice(None)
                original_slice_data = a[tuple(slice_idx)]
                mode_value_str = oldmostfreq[i, j, k]
                mode_value = next((x for x in original_slice_data if str(x) == mode_value_str), mode_value_str)
                oldmostfreq[i, j, k] = mode_value

    return oldmostfreq, oldcounts

def cleaning(json_string):
    """Cleans a JSON string by removing formatting artifacts and ensuring valid JSON.

    Args:
        json_string (str): The JSON string to clean.

    Returns:
        str: A cleaned JSON string.
    """
    cleaned_string = re.sub(r'<think>.*?</think>', '', json_string, flags=re.DOTALL).strip()
    cleaned_string = re.sub(r'```json\n', '', cleaned_string)
    cleaned_string = re.sub(r'\n```', '', cleaned_string)
    cleaned_string = re.sub(r'```', '', cleaned_string)
    pattern = r'\{(?:[^{}]|(?R))*\}'
    match = re.search(pattern, cleaned_string, flags=re.DOTALL)
    if match:
        cleaned_string = match.group(0).strip()
    cleaned_string = re.sub(r',(\s*[}\]])', r'\1', cleaned_string)
    cleaned_string = re.sub(r'\s+', ' ', cleaned_string)
    return cleaned_string

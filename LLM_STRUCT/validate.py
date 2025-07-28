"""Validation and QA utilities for LLM outputs."""

import pandas as pd
import numpy as np
from tenacity import retry, stop_after_attempt, wait_exponential, wait_fixed
from typing import Literal
from pydantic import BaseModel, ValidationError
from tqdm import tqdm
import json

from .respond import Respond

def unwrap(txt):
    """Removes code block markers from a string.

    Args:
        txt (str): Input string.

    Returns:
        str: Unwrapped string.
    """
    return txt.replace('```', '').replace('json', '')
    
def mode(a, axis=0):
    """Computes the mode and counts along an axis.

    Args:
        a (np.ndarray): Input array.
        axis (int): Axis to compute mode.

    Returns:
        tuple: (mode array, count array)
    """
    scores = np.unique(np.ravel(a))
    testshape = list(a.shape)
    testshape[axis] = 1
    oldmostfreq = np.zeros(testshape)
    oldcounts = np.zeros(testshape)

    for score in scores:
        template = (a == score)
        counts = np.expand_dims(np.sum(template, axis), axis)
        mostfrequent = np.where(counts > oldcounts, score, oldmostfreq)
        oldcounts = np.maximum(counts, oldcounts)
        oldmostfreq = mostfrequent

    return mostfrequent, oldcounts

def Uhrr_Thum_Respond(items, qust_terms, 
                  qust='Question {q_num}: Is "{{item}}" any of the following: {terms}?',
                  votes=3, reason=False, verbose=False):
    """Runs a QA process for a list of items and question terms.

    Args:
        items (list): List of items to query.
        qust_terms (list): List of question terms.
        qust (str): Question template.
        votes (int): Number of completions per item.
        reason (bool): Return reasoning if True.
        verbose (bool): Print responses if True.

    Returns:
        np.ndarray or tuple: Category array, optionally with Reasoning array.
    """
    item_num = len(items)
    qust_num = len(qust_terms)
    vote_num = votes
    
    preform = 'Question {q_num}: '
    qust_form = preform + qust + ' \n'
    
    questions = ''
    for t in range(qust_num):
        questions += qust_form.format(q_num=str(t+1), terms=qust_terms[t])
    
    schema = '''
    {
    "1": {
        "thought": "How you reason or think about the first question.",
        "answer": "The answer to the question. Use a one word lowercase answer, in the form of yes or no.",
        "difficulty": "How difficult the question was. One of easy, medium or hard."
       },
    "2": {
        "thought": "How you reason or think about the second question.",
        "answer": "The answer to the question. Use a one word lowercase answer, in the form of yes or no.",
        "difficulty": "How difficult the question was. One of easy, medium or hard."
       },
       ...
    }
    '''
    
    context = f'''
    I will ask you questions and you will respond. Your response should be in the following format:
    ```json
    {schema}
    ``` 
    '''
    Binary = Literal["yes", "no"]
    Difficulty = Literal["easy", "medium", "hard"]
    class ThoughtAnswerResponse(BaseModel):
        thought: str
        answer: Binary
        difficulty: Difficulty
    
    Category = np.array([[['']*vote_num]*qust_num]*item_num, dtype=object)
    Reasoning = np.array([[['']*vote_num]*qust_num]*item_num, dtype=object)
    
    for i in tqdm(range(item_num)):
        item_ = items[i]
        prompt = questions.format(item=item_)
        while True:
            responses = Respond(prompt, context, t=1.9, c=.75, model='4o', n=vote_num)
            try:
                for k in range(vote_num):
                    response = responses[k]
                    for j in range(qust_num):
                        q = j+1
                        subresponse = json.loads(unwrap(response))[str(q)]
                        validated_subresponse = ThoughtAnswerResponse.model_validate_json(json.dumps(subresponse))
                        Category[i, j, k] = validated_subresponse.answer
                        Reasoning[i, j, k] = validated_subresponse
                        if verbose:
                            print(json.dumps(subresponse, indent=4))
                break
            except ValidationError as e:
                print("Unable to validate LLM response.")
    return Category if reason == False else (Category, Reasoning)

def collapse(Category, names, items, qust_terms):
    """Collapses vote arrays into a DataFrame.

    Args:
        Category (np.ndarray): Array of votes.
        names (list): Column names.
        items (list): Row index.
        qust_terms (list): Question terms.

    Returns:
        pd.DataFrame: Collapsed table.
    """
    item_num = len(items)
    qust_num = len(qust_terms)
    Votes = mode(Category, axis=2)[0][:, :, 0]
    return pd.DataFrame(Votes, columns=names, index=items[:item_num])

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
        
def qaProcess(items, terms,
              qust='Is "{{item}}" any of the following: {terms}?',
              thresh_dis=0.75, thresh_dif=2.0,
              votes=3, reason=True, verbose=False,
              color=True, save=False):
    """Runs the full QA process and returns a colored table.

    Args:
        items (list): List of items to query.
        terms (dict): Mapping of question terms to column names.
        qust (str): Question template.
        thresh_dis (float): Disagreement threshold.
        thresh_dif (float): Difficulty threshold.
        votes (int): Number of completions per item.
        reason (bool): Return reasoning if True.
        verbose (bool): Print responses if True.
        color (bool): Color output if True.
        save (bool): Save to Excel if True.

    Returns:
        pd.DataFrame or Styler: QA table.
    """
    qust_terms = list(terms.keys())
    qust_names = list(terms.values())
    
    Category, Reasoning = Uhrr_Thum_Respond(
        items, qust_terms,
        qust,
        votes=votes, reason=True,
        verbose=False)
    
    QA_Table = collapse(Category, qust_names, items, qust_terms)

    if color:
        x = [[[answers.answer for answers in questions] for questions in objct] for objct in Reasoning]
        C = mode(np.array(x).T)[1][0] / votes
        C = pd.DataFrame(C.T, index=QA_Table.index, columns=QA_Table.columns)
        dif_metric = {'easy': 1, 'medium': 2, 'hard': 3}
        y = [[[dif_metric[answers.difficulty] for answers in questions] for questions in objct] for objct in Reasoning]
        D = np.mean(y, axis=2)
        D = pd.DataFrame(D, index=QA_Table.index, columns=QA_Table.columns)
    
        B = C.copy().astype(object)
        mask_dis = (C <= thresh_dis)
        mask_dif = (D >= thresh_dif)
        mask_both = (mask_dis & mask_dif)
        B[mask_dis] = 'dis'
        B[mask_dif] = 'dif'
        B[mask_both] = 'both'
        QA_Table = QA_Table.style.apply(lambda x: B.map(color_cells_bool), axis=None)
        
    if save:
        QA_Table.to_excel('Categories.xlsx', engine='openpyxl')
        
    return QA_Table

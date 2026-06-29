import os

import base64
import requests
from io import BytesIO

import argparse, json, datetime, time, asyncio
import aiohttp
import ssl
import certifi

import openai  

# Need your API key here
client = openai.AsyncOpenAI(
    # api_key="",
    # base_url="https://idealab.alibaba-inc.com/api/openai/v1",  # Example of a custom URL
    # api_key="0",
    # base_url="http://0.0.0.0:8000/v1",  # Example of a custom URL
    api_key="",
    base_url=""
)

import os
import argparse
import json
import datetime
import time
import asyncio

# MODEL = "qwen2.5-vl-7b-instruct"
# MODEL ="Llama-3.3-70b-instruct"
MODEL = "deepseek-v3-0324"
COMMON_KWARGS = {"temperature": 0.0, "max_tokens": 1024}

def genquery(question, answer, ground_truth, prompt):
    return prompt + f"""
Question: {question.replace('<image>', ' ')}
Standard answer: {ground_truth}
Model's answer: {answer}
"""
async def call_chatgpt_async(key, question, answer, ground_truth, prompt):
    query = genquery(question, answer, ground_truth, prompt)
    messages = [{"role": "user", 
                 "content": [{"type": "text", "text": query},]}]

    try:
        # use the OpenAI async client
        resp = await client.chat.completions.create(
            model=MODEL,  # Use the correct model name
            messages=messages,  # Provide the prompt directly
            temperature=COMMON_KWARGS["temperature"],
            max_tokens=COMMON_KWARGS["max_tokens"]
        )
        print(resp)
        result = resp.choices[0].message.content
    except Exception as e:
        print(f"Error: {e}")
        result = ""
    return key, result

async def call_chatgpt_bulk(keys, questions, answers, ground_truths, prompt):
    # gather a bunch of acreate() calls concurrently
    tasks = [
        call_chatgpt_async(k, q, a, g, prompt)
        for k, q, a, g in zip(keys, questions, answers, ground_truths)
    ]
    return await asyncio.gather(*tasks)

def bulk_evaluate(data, batch_size, prompt_item):
    output = []
    for start in range(0, len(data), batch_size):  ## BUG lx
        batch = data[start : start + batch_size]
        keys  = [d["index"]        for d in batch]
        qs    = [d["question"]     for d in batch]
        ans   = [d["answer"]       for d in batch]
        gts   = [d["ground_truth"] for d in batch]

        responses = asyncio.run(call_chatgpt_bulk(keys, qs, ans, gts, prompt_item))

        for key, res in responses:
            output.append(tuple([key, res]))

    return output

#########################  QUALITATIVE  ##############################
import re
from typing import List, Dict, Tuple
DELTA = 2
# DELTA = 1.25
_UNIT_TO_M = {
    "m":           1.0,
    "meter":       1.0,     "meters":      1.0,
    "cm":          0.01,
    "centimeter":  0.01,    "centimeters": 0.01, "centimetres": 0.01,
    "mm":          0.001,
    "millimeter":  0.001,   "millimeters": 0.001, "millimetres": 0.001,
    "km":          1000.0,
    "kilometer":   1000.0,  "kilometers": 1000.0, "kilometres": 1000.0,
    "in":          0.0254,
    "inch":        0.0254,  "inches":      0.0254,
    "ft":          0.3048,
    "foot":        0.3048,  "feet":        0.3048,
    "unknown":    -1.0,
}

def to_meters(value: float, unit: str) -> float:
    if value is None or unit is None:
        return 0.0
    u = unit.strip().lower()
    if u not in _UNIT_TO_M:
        u = re.sub(r"[^a-z]+", "", u)
    factor = _UNIT_TO_M.get(u)
    if factor is None:
        # raise ValueError(f"Unknown unit: {unit}")
        print(f"Unknown unit: {unit}")
        return 0.0
    if factor < 0:
        return 0.0
    return value * factor


def evaluate_pairs(pairs: List[Tuple[float, float]]) -> Dict[str, float]:
    """
    Given a list of (gt_meters, pred_meters), compute:
      - mean_abs_error
      - success_rate: fraction where pred in [0.5*gt, 2*gt]
    """
    errors = []
    output = []
    successes = 0
    successes_25 = 0
    n = len(pairs)
    for key, gt, pred in pairs:
        errors.append(abs(pred - gt))
        if (1/2) * gt <= pred <= 2 * gt:
            successes += 1
            output.append((key, "score: 1"))
            if 0.75 * gt <= pred <= 1.25 * gt:
                successes_25 +=1
        else:
            output.append((key, "score: 0"))

    mean_abs_error = sum(errors) / n if n else 0.0
    success_rate   = successes / n if n else 0.0
    success_rate_25   = successes_25 / n if n else 0.0

    print( "\033[91m" ,{
        "mean_abs_error_m": mean_abs_error,
        "success_rate": success_rate,
        "success_rate_25": success_rate_25
    }, "\033[0m")
    return output


def evaluate_dataset(examples) -> Dict[str, float]:
    """
    examples: [
      {"question": "...", "answer": "..."},
      ...
    ]
    Returns overall metrics.
    """
    converted: List[Tuple[float,float]] = []
    mismatch = 0
    for item in examples:
        key, ex = item
        try:
            match = re.search(r"\{[\s\S]*\}", ex)
            ex = match.group()
            j =  json.loads(ex)
        except json.JSONDecodeError as e:
            print(f"Error decoding JSON for key {key}: {e}")
            mismatch+=1
            continue
        gt_m   = to_meters(j["gt"]["value"],   j["gt"]["unit"])
        pred_m = to_meters(j["pred"]["value"], j["pred"]["unit"])
        converted.append((key, gt_m, pred_m))
    print(f"mismatch: {mismatch}")
    return evaluate_pairs(converted)


def bulk_evaluate_qualitative(data, batch_size, prompt_item):
    output = []
    
    for start in range(0, len(data), batch_size):  ## BUG lx
        batch = data[start : start + batch_size]
        keys  = [d["index"]        for d in batch]
        qs    = [d["question"]     for d in batch]
        ans   = [d["answer"]       for d in batch]  # BUG 取最后20个字符，因为qwen系列回答太长了，容易失败. [-20: ]
        gts   = [d["ground_truth"] for d in batch]

        responses = asyncio.run(call_chatgpt_bulk(keys, qs, ans, gts, prompt_item))

        for key, res in responses:
            output.append(tuple([key, res]))

    output = evaluate_dataset(output)

    return output
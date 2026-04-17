import json
import pandas as pd

from Agents.Components.SemaFlex_Cube import SemaFlex_Cube
import datetime
from Utils.jsonfy_result import jsonfy_llm_response
import re

PROMPT_FILTER_ORIGINAL_DATA = """
You are a keyword generation assistant. Your task is to build a rich list of keywords for preliminary text filtering, based on a user's intent description.

Instructions:
- First, identify 1 to 3 core keywords that best capture the intent.
- Then, for each core keyword, generate multiple variations or alternative expressions, including:
    - Formal or technical terms
    - Informal or colloquial phrases

Constraints:
- Each keyword expression must be usable on its own for matching.
- Prefer one-word expressions; two-word phrases are allowed if necessary.
- Do not return any full sentences or explanations.

Output format:
Return a flat Python list of strings:
["keyword1", "keyword2", "keyword3", ...]

User intent:
%s
"""


def remove_super_keywords(keywords):
    keywords = sorted(set(keywords), key=lambda x: (len(x), x))
    result = []
    for i, kw in enumerate(keywords):
        if not any(kw != shorter and shorter in kw for shorter in result):
            result.append(kw)
    return result

class Slice_Agent:
    def __init__(self, llm):
        self.llm = llm

    def initial_filter(self, query):
        prompt = PROMPT_FILTER_ORIGINAL_DATA % (query)
        result = self.llm.predict(prompt)
        result = re.sub("'", '"', result)
        ret_json = jsonfy_llm_response(result, [])
        ret_json = remove_super_keywords(ret_json)
        return ret_json

    def fields_choose(self, cube: SemaFlex_Cube, query: str):
        cols = cube.get_dimensions()

        prompt = """You are a field selection assistant.

    Given a user query and a list of candidate fields, identify the fields that are potentially relevant to the query.

    Guidelines:
    - Select fields that may help interpret, filter, or respond to the query.
    - If you're uncertain whether a field is related, include it.
    - Be conservative — do not exclude fields unless you're sure they are irrelevant.
    - Return only a list of field names, no explanation or extra output.

    Format:
    ["field_name_1", "field_name_2", ...]

    User query:
    %s

    Candidate fields:
    %s
    """ % (query, cols)

        raw_response = self.llm.predict(prompt).strip()

        try:
            # raw_response = re.sub("'", '"', raw_response)
            selected_fields = jsonfy_llm_response(raw_response, [])
            # Ensure it returns only field names that exist in cols
            filtered_fields = [f for f in selected_fields if f in cols]
        except Exception:
            # Fallback: return all fields if parsing fails
            filtered_fields = cols

        return filtered_fields

    def run(self, query, cube, ids):

        ret_json = []

        filtered_fields = self.fields_choose(cube, query)
        if(filtered_fields == []):
            filtered_fields = cube.get_dimensions()

        query["action"]= query["action"].replace("in any field", "")
        result = {
            "type": "slice",
            "optimize":{
                "initial_filter": ret_json,
                "filtered_fields": filtered_fields
            },
            "query": query
        }

        return result





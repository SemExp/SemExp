from typing import Any, Dict, Tuple, Union, Optional, List, Annotated, Sequence, TypedDict
import json
from Utils.jsonfy_result import jsonfy_llm_response
from Agents.Components.SemaFlex_Cube import SemaFlex_Cube

PROMPT_STRATEGY = """
You are a data filtering optimization expert. The system will provide you with three pieces of information:

1. A list of optimization strategies  
2. Field information: includes field names and corresponding sample values  
3. A user query

[Task]  
- Read the "Field Information" and the "User Query" to determine which field the query most likely targets.  
- Based on the meaning of the field and its sample values, identify all potentially applicable optimization strategies.  
- Note: This is only a preliminary inference. Actual feasibility will be validated in later stages.

[Optimization Strategy Descriptions]  
1. pattern_extraction  
   - Applicable when:  
     - Field values typically follow the format of "fixed anchors + variable segments" (e.g., instance IDs like `Task#123`, dates like `yyyy-mm-dd`).  
     - Variable segments can be extracted into new columns via regex or parsing.  
   - Determination basis:  
     - Your commonsense understanding of the field name  
     - Sample values  

2. enumerable  
   - Applicable when:  
     - Field values is finite — typical cases include status enums, labels, or boolean flags. 
     - Note that the value is list or set ['value1', 'value2',...] do not fit this. 
   - Determination basis:
     - The field name suggests enumeration semantics  
     - Sample values suggest a discrete set

3. compare_str_or_num  
   - Applicable when:  
     - The user explicitly refers to comparing purely numeric fields (e.g., price, quantity, rating) using magnitude.  
     - The query explicitly requests a containment condition on a specific field, such as finding records where a given field contains a specified value (e.g., "tags contains 'data'").
   - Determination basis:  
     - The user query 
     - Whether the sample values support the above use cases  
     - If the field is clearly structured (e.g., date or structured instance IDs), use pattern_extraction instead.

[Output Format]  
Only output a dictionary (dict) with the following fields:
- thought: Describe your reasoning behind choosing the strategy or strategies.  
- strategy: A list of strategy names you believe might apply. If you believe no strategies apply, return an empty list [].  
- field: The field that the user query most likely applies to.  
- Example:
{
    "thought": "",
    "field": "",
    "strategy": ["", "", ...]
}

[Field Information]  
%s

[User Query]  
%s
"""


def is_highly_patternized(col, top_n=3, top_ratio_threshold=0.8, unique_ratio_threshold=0.01):
    import re
    patterns = col.apply(lambda s: re.sub(r"\d+", "<*>", s))
    pattern_counts = patterns.value_counts()

    total = pattern_counts.sum()
    top_n_ratio = pattern_counts.head(top_n).sum() / total
    unique_pattern_ratio = len(pattern_counts) / total

    return (top_n_ratio > top_ratio_threshold and unique_pattern_ratio < unique_ratio_threshold) or (len(pattern_counts)<5)

def is_enumerable(
    col,
    max_unique=20,
    max_ratio=0.05,
    max_avg_length=30,
    top_coverage_threshold=0.9,
    top_n=5
):
    if col.isnull().all():
        return False

    n_unique = col.nunique(dropna=True)
    n_total = len(col)
    unique_ratio = n_unique / n_total

    try:
        avg_len = col.dropna().astype(str).str.len().mean()
    except:
        avg_len = float("inf")

    top_coverage = col.value_counts(normalize=True).head(top_n).sum()

    return (
        (n_unique <= max_unique or unique_ratio <= max_ratio)
        and avg_len <= max_avg_length
        and top_coverage >= top_coverage_threshold
    )

def llm_has_simple_filter(query: str, llm) -> bool:
    prompt = """
You are an assistant for judging filtering intent on structured data.

Given a user input, determine whether the sentence clearly expresses a simple filtering condition on a field.

Judgment criteria:

1. For string fields:
   - The sentence explicitly uses expressions like "contains", "equals", "starts with", or "ends with";
   - It conveys a requirement to match a specific string value.

2. For numeric fields:
   - The sentence explicitly uses expressions like "greater than", "less than", "≥", "≤", or "= some number";
   - It conveys a threshold or range condition involving a specific number.

Instructions:
- You do NOT need to identify the field name or type;
- You do NOT need to execute anything — just determine if the sentence expresses a valid filtering condition;
- Strictly respond with only YES or NO.

User input:  
%s

Your answer (respond with only YES or NO):

""" % (query)
    response = llm.predict(prompt).strip().upper()
    return response == "YES"




class Dice_Agent:
    def __init__(self, llm: Any):
        self.llm = llm

    def strategy_choose(self, query: str, field: str, cube: SemaFlex_Cube):
        description = ""
        granularities = cube.get_granularities(field)
        if(len(granularities)==1):
            description += f"There are just one field '{granularities[0]}', please just output it."
        else:
            for granularity in granularities:
                example_df = cube.sample_granularity(field, granularity, 3)
                examples = (example_df[granularity].dropna().astype(str).tolist())
                if((sum(len(s) for s in examples) / len(examples)) > 100 ):
                    description += f"{granularity}: This field is too long and do not able to sample. This is an unstructured field. \n"
                else:
                    description += f"{granularity}: {examples} \n"

        prompt = PROMPT_STRATEGY % (description, query)

        result = self.llm.predict(prompt)
        ret_json = jsonfy_llm_response(result)
        return ret_json

    def strategy_check(self, plan: dict, query: str, cube: SemaFlex_Cube, ids):
        import copy
        checked = copy.deepcopy(plan)


        field_name   = checked.get("field")
        strat_list   = checked.get("strategy", []) or []

        data = cube.read_raw(ids, [field_name])

        if field_name not in data.columns:
            checked["strategy"] = "None"
            return checked

        col = data[field_name]


        PRIORITY = ["pattern_extraction", "enumerable", "compare_str_or_num"]

        final_strat = "None"

        for strat in PRIORITY:
            if strat not in strat_list:
                continue
            if strat == "pattern_extraction":
                if is_highly_patternized(col):
                    final_strat = strat
                    break

            elif strat == "enumerable":
                if is_enumerable(col):
                    final_strat = strat
                    break

            elif strat == "compare_str_or_num":
                if llm_has_simple_filter(query, self.llm):
                    final_strat = strat
                    break

        checked["strategy"] = final_strat
        return checked

    def run(self, query, cube, ids):

        action = query['action']
        filed = query['field']
        
        ret_json = self.strategy_choose(action, filed, cube)

        plan = self.strategy_check(ret_json, query, cube, ids)


        slim_strategy_plan = {"strategy": plan.get("strategy", "None")}


        result = {
            "type": "dice",
            "optimize": slim_strategy_plan,
            "field": plan.get("field", None),
            "query": query
        }

        return result

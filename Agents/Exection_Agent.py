import copy, json, itertools
import pandas as pd

from Agents.Components.Operaters import filter_LLM, filter_enum_llm, filter_numeric, filter_keywords, filter_pattern
from typing import Dict, Any, List, Union
from Agents.Components.SemaFlex_Cube import SemaFlex_Cube
Logic = Union[int, List]
from Utils.jsonfy_result import jsonfy_llm_response
from typing import Dict, Any
import copy

PROMPT_DIRECT = """
You are a structured data filter plan assistant. Your task is to convert the user's natural language query into a structured filter plan that can be executed by the system.

The system currently supports the following basic filtering functions:

1. filter_keywords:
   - Used to check whether specific keywords appear in one or more columns.
   - Parameters include:
     - keywords: A list of keywords, in the format ["key", "key", ...]
     - relation: One of "contains", "contained_in", "equal", or "irrelevant"
     - logic: Logical relation between keywords, either "or" or "and"

2. filter_numeric:
   - Used to apply numerical comparisons on a specific field.
   - Parameters include:
     - operator: One of the comparison operators "==", "!=", ">", "<", ">=", "<="
     - threshold: The numeric threshold to compare against (int or float)

Based on the user's natural language query, determine which filter function(s) should be used and extract the corresponding parameters. Return a complete filter plan.

Output format:

{
  "operations": [
    {
      "id": 1,
      "type": "<function name, e.g. filter_keywords or filter_numeric>",
      "params": {
        ... parameter structure ...
      }
    }
  ],
  "logic": ["AND", 1, ["OR", 2, 3]]
}

Do not output any explanation or extra text. Only return the JSON structure.

[User Query]  
%s
"""

PRIORITY_MAP = {
    "filter_numeric": 0,
    "filter_keywords": 0,
    "pattern_based": 1,
    "enumerable": 1,
    "filter_LLM": 2,
}


def _substitute(expr, mapping):
    if isinstance(expr, list):
        if expr and isinstance(expr[0], str) and expr[0] in ("AND", "OR"):
            return [expr[0]] + [_substitute(e, mapping) for e in expr[1:]]
        else:
            return [_substitute(e, mapping) for e in expr]
    if isinstance(expr, int):
        return mapping.get(expr, expr)
    return expr


def _safe_copy_operator(op: Dict[str, Any], new_id: int) -> Dict[str, Any]:
    return {
        "id": new_id,
        "type": op["type"],
        "params": {k: v for k, v in op.get("params", {}).items() if k != "llm"}
    }


def flatten_plan(plan: Dict[str, Any]) -> Dict[str, Any]:
    new_ops: List[Dict[str, Any]] = []
    new_id = 1

    def _flatten(node: Dict[str, Any]) -> Logic:
        nonlocal new_id
        id_map: Dict[int, Logic] = {}

        for op in node["operations"]:
            if "sub_plan" in op:
                id_map[op["id"]] = _flatten(op["sub_plan"])
            else:
                op_flat = _safe_copy_operator(op, new_id)
                new_ops.append(op_flat)
                id_map[op["id"]] = new_id
                new_id += 1

        return _substitute(node["logic"], id_map)

    flat_logic = _flatten(plan)
    return {"operations": new_ops, "logic": flat_logic}


def reorder_by_priority(flat_plan: Dict[str, Any]) -> Dict[str, Any]:
    ops = flat_plan["operations"]
    ops_sorted = sorted(ops, key=lambda op: PRIORITY_MAP.get(op["type"], 1))
    id_map = {}
    for new_id, op in enumerate(ops_sorted, start=1):
        id_map[op["id"]] = new_id
        op["id"] = new_id
    new_logic = _substitute(flat_plan["logic"], id_map)
    return {"operations": ops_sorted, "logic": new_logic}


def make_flat_sorted_plan(nested_plan: Dict[str, Any]) -> Dict[str, Any]:
    flat_plan = flatten_plan(nested_plan)
    sorted_plan = reorder_by_priority(flat_plan)
    return sorted_plan


def exec_filter_keywords(df_sub, params, llm):
    return filter_keywords(
        keywords=params["keywords"],
        relation=params["relation"],
        logic=params["logic"],
        df=df_sub,
        cols=params["cols"]
    )


def exec_filter_numeric(df_sub, params, llm):
    return filter_numeric(
        df=df_sub,
        col=params["col"],
        operator=params["operator"],
        threshold=params["threshold"]
    )


def exec_filter_LLM(df_sub, params, llm):
    return filter_LLM(
        llm=llm,
        description=params["query"],
        df=df_sub,
        cols=params["cols"]
    )


def exec_filter_pattern(df_sub, params, llm):
    return filter_pattern(
        df=df_sub,
        col=params["col"],
        query=params["query"],
        llm=llm
    )


def exec_filter_enum_llm(df_sub, params, llm):
    return filter_enum_llm(
        df=df_sub,
        col=params["col"],
        query=params["query"],
        llm=llm
    )


OP_EXECUTOR_MAP = {
    "filter_keywords": exec_filter_keywords,
    "filter_numeric": exec_filter_numeric,
    "filter_LLM": exec_filter_LLM,
    "pattern_based": exec_filter_pattern,
    "filter_pattern": exec_filter_pattern,
    "enumerable": exec_filter_enum_llm,
}


LogicExpr = Union[int, List]

def _and_values(*vals):
    if any(v is False for v in vals):
        return False
    if all(v is True for v in vals):
        return True
    return pd.NA


def _or_values(*vals):
    if any(v is True for v in vals):
        return True
    if all(v is False for v in vals):
        return False
    return pd.NA


def _merge_logic(expr, leaf):
    if isinstance(expr, int):
        return leaf.get(expr, pd.Series(pd.NA, index=next(iter(leaf.values())).index))

    op, *subs = expr
    series_list = [_merge_logic(sub, leaf) for sub in subs]
    df = pd.concat(series_list, axis=1)

    if op == "AND":
        return df.apply(lambda row: _and_values(*row), axis=1)
    elif op == "OR":
        return df.apply(lambda row: _or_values(*row), axis=1)
    else:
        raise ValueError(f"Unsupported logical op {op}")


def normalize_logic(expr):

    if isinstance(expr, int):
        return expr

    if isinstance(expr, list):

        if len(expr) == 1:
            return normalize_logic(expr[0])

        op = expr[0]
        if op in {"AND", "OR"}:
            normalized = [op]
            for sub in expr[1:]:
                norm_sub = normalize_logic(sub)

                if isinstance(norm_sub, list) and norm_sub and norm_sub[0] == op:
                    normalized.extend(norm_sub[1:])
                else:
                    normalized.append(norm_sub)

            return normalized
        else:
            return [normalize_logic(e) for e in expr]

    return expr


class Execution_Agent:
    def __init__(self, llm):
        self.llm = llm

    def operation_execution(self, opt):
        if (opt["type"] == "slice"):
            operations_list = []
            if(opt["optimize"]["initial_filter"]!=[]):
                operations_list.append({"id": 1, "type": "filter_keywords",
                                        "params": {"keywords": opt["optimize"]["initial_filter"], "relation": "contains",
                                                   "logic": "or",
                                                   "cols": opt["optimize"]["filtered_fields"]}})
            operations_list.append({"id": 2, "type": "filter_LLM",
                                    "params": {"llm": self.llm, "query": opt["query"],
                                               "cols": opt["optimize"]["filtered_fields"]}})
            logic = ["AND", 1, 2]
            return {"operations": operations_list, "logic": logic}
        else:
            if (opt["optimize"]["strategy"] == "pattern_extraction"):
                operations_list = []
                operations_list.append({"id": 1, "type": "pattern_based",
                                        "params": {"llm": self.llm, "query": opt["query"],
                                                   "col": opt["field"]}})
                logic = [1]
                return {"operations": operations_list, "logic": logic}
            elif (opt["optimize"]["strategy"] == "enumerable"):
                operations_list = []
                operations_list.append({"id": 1, "type": "enumerable",
                                        "params": {"llm": self.llm, "query": opt["query"],
                                                   "col": opt["field"]}})
                logic = [1]
                return {"operations": operations_list, "logic": logic}
            elif (opt["optimize"]["strategy"] == "compare_str_or_num"):
                prompt = PROMPT_DIRECT % (opt["query"])
                result = self.llm.predict(prompt)
                ret_json = jsonfy_llm_response(result)
                filtered_fields = opt["field"]
                for op in ret_json.get("operations", []):
                    if op["type"] in {"filter_keywords", "filter_LLM"}:
                        if "params" in op:
                            op["params"]["cols"] = filtered_fields
                    else:
                        if "params" in op:
                            op["params"]["col"] = filtered_fields
                return ret_json
            else:
                operations_list = []
                operations_list.append({"id": 1, "type": "filter_LLM",
                                        "params": {"llm": self.llm, "query": opt["query"],
                                                   "cols": opt["field"]}})
                logic = [1]
                return {"operations": operations_list, "logic": logic}

    def inject_sub_plans(self, plan: dict) -> dict:
        ops_new = [{
            "id": op["id"],
            "sub_plan": self.operation_execution(op["plan"])
        } for op in plan["operations"]]
        return {"operations": ops_new, "logic": plan["logic"]}

    def run_filter(self, plan_dict: Dict, cube: SemaFlex_Cube, ids):
        plan_finer = self.inject_sub_plans(copy.deepcopy(plan_dict))

        flat_plan = make_flat_sorted_plan(plan_finer)
        flat_plan["logic"] = normalize_logic(flat_plan["logic"])

        tmp_plan = []

        for item in flat_plan["operations"]:
            if item["type"] == "filter_numeric":
                tmp_plan.append({
                    "operator_name": "num_filter",
                    "parameters": {
                        "conditions": [
                            {
                                "column": item["params"]["col"],
                                "threshold": item["params"]["threshold"],
                                "operator": item["params"]["operator"]
                            }
                        ]
                    }
                })
            elif item["type"] == "filter_keywords":
                tmp_plan.append({
                    "operator_name": "keyword_filter",
                    "parameters": {
                        "columns": [
                            item["params"]["cols"]
                        ],
                        "keywords": [
                            item["params"]["keywords"]
                        ],
                        "logic": item["params"]["logic"],
                        "relation": item["params"]["relation"]
                    }
                })
            elif item["type"] == "pattern_based":
                tmp_plan.append({
                    "operator_name": "sem_filter",
                    "parameters": {
                        "columns": [
                            [item["params"]["col"]]
                        ],
                        "condition": item["params"]["query"]["action"]
                    }
                })
            elif item["type"] == "enumerable":
                tmp_plan.append({
                    "operator_name": "sem_filter",
                    "parameters": {
                        "columns": [
                            [item["params"]["col"]]
                        ],
                        "condition": item["params"]["query"]["action"]
                    }
                })
            elif item["type"] == "filter_LLM":
                tmp_plan.append({
                    "operator_name": "sem_filter",
                    "parameters": {
                        "columns": [
                            item["params"]["cols"]
                        ],
                        "condition": item["params"]["query"]["action"]
                    }
                })

        df = cube.read_raw(ids)

        logic = flat_plan["logic"]
        op_list = sorted(flat_plan["operations"], key=lambda x: x["id"])

        leaves: Dict[int, pd.Series] = {}
        undecided = pd.Series(pd.NA, index=df.index)

        for op in op_list:
            idx_pending = undecided[undecided.isna()].index
            if idx_pending.empty:
                break

            df_sub = df.loc[idx_pending]
            exec_fn = OP_EXECUTOR_MAP[op["type"]]
            mask_sub = exec_fn(df_sub, op["params"], self.llm)

            leaves[op["id"]] = mask_sub.reindex(df.index).astype(object)

            undecided = _merge_logic(logic, leaves)

        final_mask = undecided.fillna(False).astype(bool)

        return df[final_mask], tmp_plan

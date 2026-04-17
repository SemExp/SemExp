
import random
import re, json
import pandas as pd
from typing import Dict, Any, List, Union

from Utils.jsonfy_result import jsonfy_llm_response


def filter_keywords(keywords, relation, logic, df, cols):

    if isinstance(cols, str):
        cols = [cols]

    series = df[cols].astype(str).agg(" ".join, axis=1).str.lower()
    keywords = [k.lower() for k in keywords]


    if relation in ["contains", "irrelevant"]:
        conditions = [series.str.contains(k, na=False) for k in keywords]
    elif relation == "contained_in":
        conditions = [series.apply(lambda val, kk=k: kk in val for k in keywords)]
    elif relation == "equal":
        conditions = [series == k for k in keywords]
    else:
        raise ValueError(f"Unsupported relation: {relation}")

    if not conditions:
        mask = pd.Series([True] * len(df))
    elif logic == "or":
        mask = conditions[0]
        for cond in conditions[1:]:
            mask |= cond
    elif logic == "and":
        mask = conditions[0]
        for cond in conditions[1:]:
            mask &= cond
    else:
        raise ValueError(f"Unsupported logic: {logic}")

    if relation == "irrelevant":
        mask = ~mask

    return mask


def filter_LLM(llm, description: str, df: pd.DataFrame, cols, retain_if: bool = True):

    if isinstance(cols, str):
        cols = [cols]

    series = df[cols].astype(str).agg(" ".join, axis=1)

    print(len(series))

    def judge(doc: str) -> bool:
        prompt = f"""
Please read the following text and make a judgment based on the given instruction.

[Task Description]  
{description}

[Text Content]  
{doc}

If the content satisfies the task requirement, respond: "Yes".  
Otherwise, respond: "No".

Only respond with "Yes" or "No". Do not provide any explanation.
"""

        result = llm.predict(prompt).strip()
        print(f"【{result}】")
        if 'yes' in result.lower():
            return retain_if
        else:
            return not retain_if

    mask = series.apply(judge)

    return mask

import pandas as pd

def filter_numeric(df: pd.DataFrame, col: str, operator: str, threshold) -> pd.Series:

    series = pd.to_numeric(df[col], errors="coerce")

    if operator == "==":
        mask = series == threshold
    elif operator == "!=":
        mask = series != threshold
    elif operator == ">":
        mask = series > threshold
    elif operator == "<":
        mask = series < threshold
    elif operator == ">=":
        mask = series >= threshold
    elif operator == "<=":
        mask = series <= threshold
    else:
        pass

    return mask.fillna(False)

def num_cmp(value: Union[pd.Series, float, int, str],
            op: str,
            threshold: Union[float, int]) -> pd.Series:
    if not isinstance(value, pd.Series):
        value = pd.Series([value])

    v  = pd.to_numeric(value, errors="coerce")
    th = int(threshold)

    if op == ">":
        mask = v > th
    elif op == "<":
        mask = v < th
    elif op == ">=":
        mask = v >= th
    elif op == "<=":
        mask = v <= th
    elif op == "==":
        mask = v == th
    elif op == "!=":
        mask = v != th
    else:
        raise ValueError(f"Unsupported op: {op}")

    return mask.fillna(False)

def build_named_regex_prompt(query: str, examples) -> str:
    """
    Prompt for LLM: Ask it to derive a regex based on examples and assign names to variable segments
    """
    eg = "\n".join(f"- {e}" for e in examples)
    return """
You are an assistant for structured extraction and numerical filtering.

The task consists of two steps:
1. Derive a regular expression from the examples. Use named capture groups (?P<variable_name>...) to assign names to each numeric or variable segment.
2. Based on the user query, generate a “numerical comparison” filtering plan for the named variables in the regex.

[User Query]
%s

[Field Examples]
%s

Please return in the following format:

{
  "regex": "<single-line regular expression>",
  "filter_plan": {
    "operations": [
      {
        "id": 1,
        "var": "<variable_name>", 
        "op": ">",  
        "threshold": 100
      },
      ...
    ],
    "logic": ["AND", 1, ["OR", 2, 3]]  // or ["AND", ["OR", 2, 3], ["OR", 4, 5]]  / [1], etc.
  }
}

Where:
- `regex` must be directly compilable by Python's `re.compile()`.
- `var` must be a named group defined in the regex.
- `op` must be one of: ">", "<", ">=", "<=", "==", "!=".
- `logic` is a nested array representing logical expressions:
  - The first element is an operator: "AND" or "OR"
  - The rest are operation IDs or sub-expressions
  - Sub-expressions can be nested arrays as well

Do not include any additional explanation. Only return the required dictionary.
""" % (query, eg)


LogicExpr = Union[int, List]

def _eval_logic_vector(expr: LogicExpr, leaf: Dict[int, pd.Series]) -> pd.Series:
    idx = next(iter(leaf.values())).index

    if isinstance(expr, int):
        return leaf.get(expr, pd.Series(False, index=idx))

    if isinstance(expr, list):
        if not expr:
            return pd.Series(True, index=idx)
        if all(isinstance(x, int) for x in expr):
            expr = ["AND"] + expr

        op, *subs = expr
        if op == "AND":
            res = _eval_logic_vector(subs[0], leaf)
            for s in subs[1:]:
                res &= _eval_logic_vector(s, leaf)
            return res
        elif op == "OR":
            res = _eval_logic_vector(subs[0], leaf)
            for s in subs[1:]:
                res |= _eval_logic_vector(s, leaf)
            return res
    return pd.Series(False, index=idx)


def filter_pattern(
    df: pd.DataFrame,
    col: str,
    query: str,
    llm,
    sample_n: int = 3,
    random_state: int = 42
) -> pd.Series:

    series_str = df[col].astype(str)

    pattern_key = series_str.str.replace(r"\d+", "<*>", regex=True)

    global_mask = pd.Series(False, index=df.index)

    for pat, grp_idx in pattern_key.groupby(pattern_key).groups.items():
        rows = series_str.loc[grp_idx]

        examples = rows.sample(n=min(sample_n, len(rows)),
                               random_state=random_state).tolist()

        prompt = build_named_regex_prompt(query, examples)
        raw = llm.predict(prompt).strip()
        print(raw)

        try:
            plan_json = jsonfy_llm_response(raw)
            regex_str = plan_json["regex"].strip()
            compiled = re.compile(regex_str)
            plan = plan_json.get("filter_plan", {})
        except Exception:
            continue
        try:
            extracted = rows.str.extract(compiled)
        except Exception:
            continue
        leaf_res: Dict[int, pd.Series] = {}

        for op_item in plan.get("operations", []):
            rid = op_item["id"]
            var = op_item["var"]
            op  = op_item["op"]
            th  = op_item["threshold"]
            if var not in extracted.columns:
                leaf_res[rid] = pd.Series(False, index=rows.index)
            else:
                leaf_res[rid] = num_cmp(extracted[var], op, th).reindex(rows.index)

        logic_expr = plan.get("logic", [])
        group_mask = _eval_logic_vector(logic_expr, leaf_res)
        global_mask.loc[rows.index] |= group_mask

    return global_mask


ENUM_PROMPT_TEMPLATE = """
You are a data filtering assistant.

[Field Name]  
{col_name}

[Possible Field Values] (total {n_vals})  
{values}

[User Query]  
{query}

Based on the user query, select the **values to keep** from the list of possible field values, and return them as a Python list:
["value1", "value2", ...]

Only return the list itself. Do not include any explanation or extra text.
"""



def filter_enum_llm(
    df: pd.DataFrame,
    col: str,
    query: str,
    llm,
    top_n_preview: int = 100
) -> pd.Series:

    if col not in df.columns:
        raise ValueError(f"列 '{col}' 不存在")

    unique_vals = df[col].dropna().unique().tolist()
    preview_vals = unique_vals[:top_n_preview]
    values_block = "\n".join(f"- {v}" for v in preview_vals)
    if len(unique_vals) > top_n_preview:
        values_block += f"\n... 共 {len(unique_vals)} 个取值，已截前 {top_n_preview} 个"

    prompt = ENUM_PROMPT_TEMPLATE.format(
        col_name=col,
        n_vals=len(unique_vals),
        values=values_block,
        query=query
    )


    raw = llm.predict(prompt).strip()
    try:
        selected = jsonfy_llm_response(raw)
        if not isinstance(selected, list):
            raise ValueError
    except Exception:
        selected = []

    mask = df[col].isin(selected)
    return mask




def sem_map(llm, df: pd.DataFrame, dimension: dict) -> pd.DataFrame:
    title = dimension.get("title", "").strip()
    if not title:
        return df
    if title in df.columns:
        df = df.drop(columns=[title])

    # 逐行调用 LLM：询问该行在新维度下的取值
    def ask_llm(row_text: str) -> str:
        prompt = f"""
You are a data mapping assistant. Your task is to map a value for a given content in specific dimension.

[Dimension name]  
{title}

[Dimension description]
{dimension.get('dimension_desc', '')}

[Output requirements]  
Only output the value itself — no explanation, parentheses, or extra text.  

[Input content]  
{row_text}
"""
        ans = llm.predict(prompt).strip()
        return "NA" if ans.upper() in {"NA", "N/A", "UNKNOWN", ""} else ans

    joined = df.apply(lambda row: '\n'.join(f"{col}: {row[col]}" for col in df.columns), axis=1)
    df[title] = joined.apply(ask_llm)


    return df[["OLAP_ID", title]]




PROMPT_TAG_CLASSIFY = """
You are a concept categorization assistant. Your goal is to assign a tag to the input based on a target granularity.

[Target granularity]  
%s

[Existing tags]  
%s

[Input text]  
%s

[Instructions]  
1. If the input clearly fits into one of the existing tags, return that exact tag.  
2. If it does not match any existing tag, generate a new, concise tag that fits the target granularity.  
3. The new tag should be short (1–3 words), in English, and conceptually aligned with the existing tag style.  
4. Only return the chosen or newly generated tag — do not include explanations, formatting, or punctuation.
"""

def sem_group(llm, df, target, text_col="text", tags = None):
    if(tags is None):
        tags = []
    tag_list = []

    for text in df[text_col]:
        prompt = PROMPT_TAG_CLASSIFY % (
            target,
            ", ".join(tags) if tags else "None",
            text
        )

        response = llm.predict(prompt).strip()
        tag_list.append(response)

        if response not in tags:
            tags.append(response)
            print(tags)

    new_df = df.copy()
    new_df[target] = tag_list
    return new_df[["OLAP_ID", target]]


def group_by(df, group_col: str) -> dict:
    return {val: group_df for val, group_df in df.groupby(group_col)}


def sem_reduce(llm, name, series: pd.Series, desc) -> str:
    unique_texts = pd.Series(series.dropna().astype(str).unique())
    contents = "\n".join(f"- {text}" for text in unique_texts)
    prompt = f"""
You are a semantic aggregator. Summarize the following texts into a single phrase or sentence that captures their common meaning.

[Field name]
{name}

[Input texts]
{contents}

[Reduce description]
{desc}

[Instructions]
- Focus on their shared intent or theme.
- Only output the result. No examples, no explanation.
"""
    return llm.predict(prompt).strip()


def num_reduce(s: pd.Series, agg: str) -> float:
    """
    对 Series 执行数值聚合，仅返回单个数值。
    """
    agg_map = {
        "avg": "mean",
        "sum": "sum",
        "max": "max",
        "min": "min"
    }
    return s.agg(agg_map[agg])


def count(s: pd.Series) -> int:
    """
    返回 Series 的总元素数量。

    Args:
        s (pd.Series): 输入的 Series。

    Returns:
        int: 元素总数。
    """
    return s.size


def num_topk(df: pd.DataFrame, column: str, k: int = 1, order: str = "desc") -> pd.DataFrame:
    """
    Retrieves the top-k rows with the highest or lowest values in a numeric column.

    Args:
        df (pd.DataFrame): The input table.
        column (str): The numeric column to sort by.
        k (int): The number of rows to retain.
        order (str): 'asc' for smallest values, 'desc' for largest values.

    Returns:
        pd.DataFrame: The top-k rows sorted by the given column.
    """
    if column not in df.columns:
        raise ValueError(f"Column '{column}' does not exist in the dataframe.")

    if order not in ("asc", "desc"):
        raise ValueError("Order must be 'asc' or 'desc'.")

    ascending = True if order == "asc" else False

    return df.sort_values(by=column, ascending=ascending).head(k)


def sem_topk(llm, df: pd.DataFrame, columns: list[str], query: str, k: int) -> pd.DataFrame:
    """
    Perform semantic top-k ranking via pairwise LLM comparisons on multiple text columns.

    Args:
        llm: A language model with a .predict(prompt) method that returns 'Document 1' or 'Document 2'.
        df (pd.DataFrame): The input table.
        columns (list of str): Names of the text columns to compare.
        query (str): The natural language prompt describing the ranking criterion.
        k (int): Number of top rows to return.

    Returns:
        pd.DataFrame: Top-k rows ranked by semantic relevance.
    """
    # Drop rows with any NaN in selected columns
    candidates = df.dropna(subset=columns).copy().reset_index(drop=True)
    n = len(candidates)

    if n <= k:
        return candidates

    # Pre-concatenate selected columns into one string per row
    candidates["_merged_text"] = candidates[columns].astype(str).agg(" ".join, axis=1)

    # Initialize win counts
    wins = [0] * n

    # Sample a subset of pairwise comparisons
    comparisons = []
    for i in range(n):
        sampled_js = random.sample([j for j in range(n) if j != i], min(5, n - 1))
        for j in sampled_js:
            comparisons.append((i, j))

    # Shuffle to avoid order bias
    random.shuffle(comparisons)

    # Run LLM comparisons
    for i, j in comparisons:
        text_i = candidates.at[i, "_merged_text"]
        text_j = candidates.at[j, "_merged_text"]

        prompt = f"""
You are comparing two documents based on the following question:
"{query}"

Document 1:
{text_i}

Document 2:
{text_j}

Which document better satisfies the question? Answer with "Document 1" or "Document 2".
"""
        response = llm.predict(prompt).strip().lower()
        if "1" in response:
            wins[i] += 1
        elif "2" in response:
            wins[j] += 1

    topk_indices = sorted(range(n), key=lambda idx: wins[idx], reverse=True)[:k]
    return candidates.drop(columns=["_merged_text"]).iloc[topk_indices].reset_index(drop=True)

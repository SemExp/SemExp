import json

import pandas as pd
from filelock import FileLock
from Agents.Components.Operaters import sem_map
from Utils.jsonfy_result import jsonfy_llm_response
from Agents.Components.SemaFlex_Cube import OLAP_Axis, SemaFlex_Cube
import os


PROMPT_DRILLDOWN_DIMENSION = """You are an expert in data modeling, dimensional modeling, and semantic analysis.

Given:
- A target dimension (Target)
- A list of candidate dimensions (Candidates)

Your task is to determine whether the Target dimension is:
- semantically equivalent to exactly one candidate,
- a direct roll-up of exactly one candidate,
- a direct drill-down of exactly one candidate,
- or orthogonal to all candidates.

Classify the relationship into exactly ONE of the following categories:

1. "same_concept"
   - The Target and one candidate are semantically interchangeable
   - They refer to the same dimension, with no meaningful difference in scope or granularity
   - This applies only when they are true equivalents, such as aliases, near-identical naming variants, or wording differences that do not change meaning
   - Do NOT use this category for concepts that are merely related, adjacent, overlapping, derived from one another, or similar in topic

2. "coarser"
   - The Target is a more abstract, aggregated, or higher-level representation
   - One candidate is more fine-grained
   - The candidate can be rolled up into the Target without introducing a new semantic axis
   - This applies only when both terms lie on the same semantic dimension axis and the hierarchy is direct and stable

3. "finer"
   - The Target is more detailed, specific, or lower-level
   - One candidate is more abstract or aggregated
   - The Target can be rolled up into that candidate without introducing a new semantic axis
   - This applies only when both terms lie on the same semantic dimension axis and the hierarchy is direct and stable

4. "orthogonal"
   - The Target does not form a valid same-concept or hierarchical relationship with any candidate
   - The terms represent different dimensions, different semantic axes, different scopes, or only a loose/related connection

Instructions:
- Always check for "same_concept" FIRST, but apply it very strictly.
- Use "same_concept" only when the Target and candidate are interchangeable in meaning.
- If one term is broader, narrower, derived, summarized, counted, labeled, annotated, mapped, scored, or otherwise differently scoped, they are NOT "same_concept".
- Only classify as "coarser" or "finer" when the relationship is direct, explicit, logically stable, and context-independent.
- A valid hierarchy must stay within the same semantic dimension axis.
- Do NOT infer hierarchy from extraction, annotation, summarization, correlation, causation, workflow dependency, co-occurrence, or typical usage patterns.
- Do NOT treat source/content fields as finer-grained dimensions for extracted concepts.
- Do NOT treat metrics, counts, scores, or frequencies as hierarchical dimensions relative to entities or categories.
- Do NOT treat summaries, labels, maps, annotations, statuses, or derived outputs as hierarchical dimensions unless there is an explicit and stable category/type/subtype or aggregation relationship.
- If the relationship is related-but-not-equivalent, or related-but-not-hierarchical, classify as "orthogonal".
- If there is any ambiguity about equivalence or hierarchy, classify as "orthogonal".
- Be conservative: prefer false negatives over false positives.

Output format (JSON only):
{
  "reasoning": "Concise explanation",
  "classification": "same_concept | coarser | finer | orthogonal",
  "matched_candidate": "<dimension name or null>"
}

Target dimension:
%s

Candidate dimensions:
%s
"""



class Drill_Down_Agent:
    def __init__(self, llm):
        self.llm = llm

    def check(self, query, candidates):
        candidates.remove("question_id")
        candidates.remove("tags")
        prompt = PROMPT_DRILLDOWN_DIMENSION % (query, candidates)
        result = self.llm.predict(prompt)
        ret_json = jsonfy_llm_response(result)

        return ret_json

    def Complete_data(self, params, cube: SemaFlex_Cube, ids):
        dimension_desc = params['desc']
        dimension = params["dimension_name"]

        dimension = "_".join(dimension.strip().split())
        dimension_params = {"title": dimension, "dimension_desc": dimension_desc}

        df = cube.read_raw(ids)

        mask = df[dimension].isna()
        if not mask.any():
            return
        df_missing = df.loc[mask]
        mapped_df = sem_map(self.llm, df_missing, dimension_params)
        df.loc[mask, dimension] = mapped_df[dimension].values
        cube.write_dimension(df, dimension, None, None)



    def fake_run(self, params, axis: OLAP_Axis):
        dimension = params["dimension_name"]
        dimension = "_".join(dimension.strip().split())

        candidates = axis.get_dimensions()
        check_result = self.check(dimension, candidates)
        if check_result["classification"] == "coarser":
            return f"A dimension '{check_result['matched_candidate']}' already exists and is a finer-grained representation of '{dimension}'. Please roll up from '{check_result['matched_candidate']}' if you want '{dimension}'"

        axis.update_dimension(dimension, [{"from": "drill_down", "params": params}])
        return f"New dimension has successfully build: {dimension}"

    def run(self, params, cube: SemaFlex_Cube, ids):
        dimension_desc = params['desc']
        dimension = params["dimension_name"]
        dimension = "_".join(dimension.strip().split())

        if(dimension in cube.get_dimensions()):
            gp = cube.axis.get_dimension(dimension).generate_plan
            if(gp!=[]):
                for p in gp:
                    parameters = p["params"]
                    self.Complete_data(parameters, cube, ids)

        tmp_plan = {"operator_name": "sem_map", "parameters":{"columns": [],
                        "map_description": dimension_desc,
                        "keyword": dimension}}
        dimension_params={"title":dimension, "dimension_desc":dimension_desc}

        df = cube.read_raw(ids)
        docs_df = sem_map(self.llm, df, dimension_params)
        cube.write_dimension(docs_df, dimension, [tmp_plan], generate_plan=[{"from": "drill_down", "params": params}])
        return []

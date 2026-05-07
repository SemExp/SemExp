import pandas as pd
from Agents.Components.SemaFlex_Cube import OLAP_Axis, SemaFlex_Cube

from Agents.Components.Operaters import sem_group, group_by, sem_reduce, count, num_reduce
from Utils.jsonfy_result import jsonfy_llm_response
import json



def get_check_prompt(term_a, term_b):
    prompt = f"""
You are an expert in data modeling, ontology design, and semantic analysis.

Given two terms, Term A and Term B, classify their relationship into exactly ONE of the following categories.
Use the category names EXACTLY as defined below.

1. "a_coarser"
   - Term A is more abstract or aggregated
   - Term B is more fine-grained
   - B can be safely rolled up into A without introducing new semantic dimensions
   - We say A is coarser than B when A is month and B is day; A is concept category, and B is concept.
   - For example, the error_category or error_type is coarser than error_details or reported_error.

2. "a_finer"
   - Term A is more detailed or fine-grained
   - Term B is more abstract or aggregated
   - A can be safely rolled up into B without introducing new semantic dimensions
   - For example, we say A is finer than B, when B is month and A is day; B is concept category, and A is concept.

3. "related_detail"
   - The terms are semantically related or interacting
   - They do NOT form a valid roll-up / drill-down hierarchy
   - Any aggregation would require interpretation, evaluation rules, or business logic
   - The relationship is better suited for diagnostic or explanatory analysis, not aggregation
   - For example, the tags are related to category but not suited for diagnostic or explanatory analysis

4. "unrelated"
   - The terms describe unrelated or independent concepts
   - There is no meaningful semantic dependency or analytical relationship
   - They should not be modeled together in a granularity or analytical context

Instructions:
- Do NOT infer granularity from correlation, causation, or workflow dependency.
- A valid granularity relationship must be logically stable and context-independent.
- Choose exactly ONE category.
- Do NOT invent new category names.

Output format (JSON only):
{{
  "classification": "a_coarser | a_finer | related_detail | unrelated",
  "reasoning": "Concise justification (2–4 sentences)"
}}

Term A: {term_a}
Term B: {term_b}
"""
    return prompt



def query_dimension_exist(llm, granularities, query_granularity, thought):
    if (query_granularity in granularities):
        return query_granularity

    prompt = """
Please select the granularity from the list that is most closely aligned with the query granularity in meaning.
A match should only be made if they refer to the same conceptual level of granularity detail. That is, we can directly use it instead of perform roll-up.
If there is no such match, return "None".

Return your answer as a JSON object with the following keys:
- "chosen": the chosen granularity exactly as it appears in the list, or "None" if none match in both meaning and level of detail.
- "thought": a short explanation of your reasoning.

The output should follow the following format:
{
  "thought": "<Your thought>",
  "chosen": "<Your choice>"
}

You can also refer to the existing thoughts to adjust your determination:
%s

[Available granularities]
%s

[Query granularity]
%s

Only return a valid JSON object. Do not return any other text.
""" % (thought, granularities, query_granularity)

    result_str = llm.predict(prompt).strip()

    try:
        result_json = jsonfy_llm_response(result_str)
    except json.JSONDecodeError:
        return None
    chosen = result_json.get("chosen")

    if chosen in granularities:
        return chosen
    elif chosen == "None":
        return None
    else:
        return None


class Roll_Up_Agent:
    def __init__(self, llm):
        self.llm = llm


    def Complete_data(self, params, cube: SemaFlex_Cube, ids):
        dimension = params["dimension"]
        target_granularity = params["target_granularity"]

        if (target_granularity is not None and target_granularity != "None"):
            exist_granularity = query_dimension_exist(self.llm, cube.axis.get_granularities(dimension),
                                                      target_granularity, params["thought"])
            print(exist_granularity)
            if (exist_granularity is not None):

                df_now = cube.read_raw(ids, [dimension])

                if not df_now[dimension].isna().any():
                    return

                df_missing = df_now.loc[df_now[dimension].isna()]
                df_all = cube.read_granularity(None, dimension, exist_granularity, [])
                granularity_values = (
                    df_all[exist_granularity]
                    .dropna()
                    .drop_duplicates()
                    .tolist()
                )
                new_docs_df = sem_group(
                    llm=self.llm,
                    df=df_missing,
                    target=target_granularity,
                    text_col=dimension,
                    tags=granularity_values
                )
                cube.write_granularity(
                    new_docs_df,
                    dimension,
                    target_granularity,
                    None,
                    None
                )

            else:
                now_granularity = dimension
                # TODO: choose now granularity
                df_now = cube.read_raw(ids, [dimension])
                new_docs_df = sem_group(llm=self.llm, df=df_now, target=target_granularity, text_col=dimension)
                cube.write_granularity(new_docs_df, dimension, target_granularity, None, None)
        return

    def fake_run(self, params, axis: OLAP_Axis):
        print("=====roll up=====")
        print(params)
        dimension = params["dimension"]
        target_granularity = params["target_granularity"]

        if(dimension!=target_granularity):
            prompt_check = get_check_prompt(dimension, target_granularity)

            result = self.llm.predict(prompt_check)
            ret_json = jsonfy_llm_response(result)
            ret = ret_json["classification"].lower()
            if("a_coarser" in ret):
                return f"'{dimension}' is coarser than target granularity '{target_granularity}', it can not be roll-up. No operation performed."
            elif("unrelated" in ret):
                return f"'{dimension}' is unrelated to '{target_granularity}', it can not be roll-up. No operation performed."
            elif("related_detail" in ret):
                return f"'{dimension}' is related to '{target_granularity}' but not aggregatable, roll-up is not performed. Please try drill-down."

        if(dimension in ["question_id", "tags", "score"]):
            if(target_granularity != dimension):
                return f"{dimension} can not perform roll-up operation, please try to drill-down for '{target_granularity}'."
            else:
                return f"{dimension} can not perform roll-up operation, please try other operations."

        exist_granularity = None
        if (target_granularity is not None and target_granularity!="None"):
            exist_granularity = query_dimension_exist(self.llm, axis.get_granularities(dimension), target_granularity, params["thought"])
            print(exist_granularity)
            if (exist_granularity is not None):
                actual_used_granularity = exist_granularity
            else:
                actual_used_granularity = target_granularity
        else:
            actual_used_granularity = dimension

        tmp_plan=[]
        dimension_plan = axis.get_dimension(dimension).plan.copy()
        if (dimension_plan == [] or len(dimension_plan) > 1 or "from" not in dimension_plan[0]):
            pass
        else:
            tmp_plan.append(dimension_plan[0])
        tmp_plan.append({"from": "roll_up", "params": params})
        axis.update_granularity(dimension, actual_used_granularity, tmp_plan)

        if actual_used_granularity == dimension:
            return f"No group performed. Using the existing granularity '{dimension}' directly for analysis."

        elif exist_granularity is not None and actual_used_granularity == exist_granularity:
            return f"The target granularity '{target_granularity}' already exists in '{dimension}' as '{exist_granularity}', using it directly."

        else:
            return f"Created new granularity '{target_granularity}' for '{dimension}' and grouped data accordingly."

    #
    def run(self, params, cube: SemaFlex_Cube, ids):

        print("=====roll up=====")
        print(params)
        dimension = params["dimension"]
        target_granularity = params["target_granularity"]

        if(dimension in cube.get_dimensions() and target_granularity in cube.get_granularities(dimension) and target_granularity != dimension):
            gp = cube.axis.get_granularity(dimension, target_granularity).generate_plan
            if (gp != []):
                for p in gp:
                    parameters = p["params"]
                    self.Complete_data(parameters, cube, ids)

        exist_granularity = None
        if (target_granularity is not None and target_granularity!="None"):
            exist_granularity = query_dimension_exist(self.llm, cube.axis.get_granularities(dimension), target_granularity, params["thought"])
            print(exist_granularity)
            if (exist_granularity is not None):
                actual_used_granularity = exist_granularity
                if(actual_used_granularity == dimension):
                    new_docs_df = cube.read_raw(ids, [dimension])
                else:
                    new_docs_df = cube.read_granularity(ids, dimension, actual_used_granularity)
            else:
                actual_used_granularity = target_granularity
                now_granularity = dimension

                df_now = cube.read_raw(ids, [dimension])

                new_docs_df = sem_group(llm=self.llm, df=df_now, target=actual_used_granularity, text_col=dimension)

                plan = []
                plan.extend(cube.axis.get_dimension(dimension).plan.copy())
                plan.append({
                    "operator_name": "sem_group",
                    "parameters": {
                        "columns": [
                            now_granularity
                        ],
                        "group_description": f"group into granularity {actual_used_granularity}",
                        "keyword": actual_used_granularity
                    }
                })

                cube.write_granularity(new_docs_df, dimension, actual_used_granularity, plan, generate_plan=[{"from": "roll_up", "params": params}])

        else:
            actual_used_granularity = dimension
            new_docs_df = cube.read_raw(ids, [dimension])

        olap_to_group = {}
        for _, row in new_docs_df.iterrows():
            olap_id = row["OLAP_ID"]
            group_value = row[actual_used_granularity]
            olap_to_group[olap_id] = group_value

        return olap_to_group

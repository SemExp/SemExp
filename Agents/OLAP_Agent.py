from Agents.Dice_Agent import Dice_Agent
from Agents.Slice_Agent import Slice_Agent
from Agents.Drill_Down_Agent import Drill_Down_Agent
from Agents.Roll_Up_Agent import Roll_Up_Agent
from Agents.Exection_Agent import Execution_Agent
from typing import Annotated, Any, Dict, List, Optional, Sequence, TypedDict, Union
import os
import pandas as pd
from Utils.send_logs import debug_log
import re
from Utils.jsonfy_result import jsonfy_llm_response, llm_retry
from Agents.Components.Operaters import sem_topk, num_topk
from Agents.Components.SemaFlex_Cube import SemaFlex_Cube, OLAP_Axis
from concurrent.futures import ThreadPoolExecutor


import json
import os

def append_time_record(file_path, record):
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


class State(TypedDict):
    query: str
    bot_view: str


PLAN_OLAP_AGENT_PROMPT_SLICE_DICE = """
You are a query planner responsible for breaking down a user's natural language query intent into a set of atomic filtering steps, and assigning them to one of two agents in the system:

1. dice agent: Performs filtering on a clearly identified single field.  
   Appropriate when: the query explicitly mentions a specific field, or the filter value clearly maps to one known field.

2. slice agent: Performs filtering across all fields.  
   Appropriate when: the filter condition cannot be mapped to a specific field, or it may involve multiple fields and requires cross-field or full-text search.

[Input Information]  
- List of structured fields (these are all directly available to the dice agent):  
%s  
- Historical Query (the user’s previous query):  
%s  
- Current Query:  
%s  

[Task Instructions]  
- The current query builds upon the historical query and represents a refinement. Start by identifying what new filtering conditions have been added compared to the historical query.
    Step 1: Compare historical query and current query meaning-by-meaning, and extract only the new filter conditions that are NOT already covered in the historical query.
    Step 2: Use only these new conditions in the planning below.
- Analyze the semantic intent of the current query and determine which filtering conditions can be handled by the dice agent (targeting a known single field), and which require the slice agent (cross-field or ambiguous field).
- Each operation must include only one filtering condition. Do not merge multiple conditions into a single operation.
- Break down the query intent into a sequence of atomic single-condition filtering steps, and define their logical combination.

[Output Requirements]  
The output must be strictly valid JSON, with no extra text. It should include:
- `filter_condition`: The filtering conditions added in the current query (compared to the historical query)
- `reasoning`: A brief explanation of how the operations and logical structure were determined from the added conditions
- `operations`: A list of filtering steps, each including:  
  - `id`: A unique integer ID starting from 1  
  - `agent`: Either `"dice"` or `"slice"`  
  - `instruction`: A brief human-readable description of the filtering step (must clearly indicate this one condition) 
  - `field`: If the agent is `"dice"`, this should be the corresponding field name from the structured field list. If the agent is `"slice"`, set this to `null`.  
  - `filtering_condition`: A natural language instruction describing what kind of content the agent should keep. In other words, specify what should be retained after filtering.

- `logic`: A nested array representing the logical structure of the operations  
  - The first element is a logical operator `"AND"` or `"OR"`  
  - The remaining elements are either operation IDs or nested sub-expressions (arrays)

Note: In each `action`, focus on clearly stating what needs to be filtered according to the user query. Do not specify how the agent should perform the task — simply restate the filtering condition in full natural language.


[Example Output Format]  
{   
  "historical_conditions": "The filtering conditions already applied from the historical query.",
  "current_conditions": "All filtering conditions implied by the current query, including any carried over from the historical query.",
  "filter_condition": "The additional filter(s) found by comparing current_conditions to historical_conditions — these are the new constraints to apply on top of the existing filters.",
  "reasoning": "Explain how the identified filter_condition (difference between current_conditions and historical_conditions) is used to plan the operations. Ensure that historical_conditions are not repeated in the operations.",
  "operations": [
    {
      "id": 1,
      "agent": "dice",
      "instruction": "Describe step 1",
      "field": "Choosed field name if dice, or possible fields if slice (or all fields)",
      "filtering_condition": "Describe the filtering condition only, without mentioning field, agent. Use expressions consistent with the query as much as possible."
    },
    ...
  ],
  "logic": ["OR", ["AND", 1, 2], 3]
}
"""


def make_stepwise_ReAct_prompt(query: str, history: str = "") -> str:
    return """You are assisting in building a knowledge organization system that operates on a structured data object.
Your are given some questions from stackoverflow, these questions are stored in the following structure. (note that each question is a data instance)

[Data Structure]
The data is organized across multiple dimensions and granularities:
- A dimension represents a semantic axis used to describe data from a specific angle.
- Each dimension contains one or more granularities, and these granularities represent different abstraction levels of that dimension. 
  - Finer granularities correspond to more detailed expressions.
  - Coarser granularities correspond to more abstract or generalized expressions.
- A granularity determines:
  1. How this dimension is represented at that abstraction level (i.e., what concrete value describes this dimension for each record).
  2. How data from other dimensions should be aggregated at this level of abstraction.
Thus, for each record, every dimension–granularity pair stores:
  - The value of that dimension at that granularity.
  - Aggregated summaries of all other dimensions, computed according to the granularity’s aggregation rules.
Each dimension always begins with a granularity of the same name, representing its initial and most detailed form.


[Your Task]
Your task is to gradually plan or refine the structure so that it can support the user’s query intent. Ignore any filtering or sorting (top-k) requirements, as they are handled elsewhere.
Using the available actions, construct the semantic structure needed to answer the query.
You should follow a ReAct-style process: Thought → Action → Observation, and only output the next step at each round.
If the current structure is already sufficient for the query, return "action": null to end the reasoning loop.
Default principle: Make the minimal structural change necessary to answer the query. 
After every Observation, FIRST check sufficiency: 
- If the current views already let you directly answer the query, return "action": null. 
- Note that do NOT perform additional analytical aggregation or add new granularities unless required by the user’s query.



[Available Actions]
- get_dimension
This is a lookup action that returns all existing dimensions in the data structure.
"params": {} # No parameters required

- get_granularity
This is a lookup action that, given a specific dimension, returns all granularities available under that dimension.
"params": { "dimension": "<the name of the dimension>" }


- drill_down
Drill-down is used to create a new dimension that:
 Is orthogonal to all existing dimensions — meaning it introduces a completely new semantic axis not covered by the current structure.
Note:
Treat the new dimension as orthogonal by default.
Only consider it related to an existing dimension if there is a clear coarse-to-fine granularity relationship, and the existing dimension’s values can be obtained by grouping the new dimension’s values into those categories.
Execution rules:
1. Create a completely new dimension with the given name and description.
2. This dimension is independent of all existing dimensions, so no changes are made to other dimensions or their granularities.
"params":
{
  "desc": "<A natural language description of the new dimension and what it represents>",
  "dimension_name": "<The name of the new dimension to be created>"
}

roll_up
The roll_up operation aggregates a dimension to a coarser granularity and optionally performs analysis on the aggregated results.
You only need to provide the name of the target granularity. The system will automatically create the new granularity, group records, generate representative labels, and populate values. You do NOT need to manually create the granularity via drill_down or any other operation.
There are 3 cases for roll-up

Case 1: Aggregate to a coarser granularity and perform analysis
Conditions:
- target_granularity is different with existing ones
- analyze_dimension is provided and non-empty
Behavior:
1. The system aggregates the specified dimension to a coarser target granularity and creates a new granularity named <target_granularity>.
2. Within each aggregated group, the system assigns a target granularity label that represents the shared characteristics of the grouped records.
3. After aggregation, the system performs analysis as specified in analyze_dimension:
   - If the analysis target refers to another dimension, the system performs the specified reduction.
   - If the analysis target is "self", only aggregate-level analysis is allowed, such as:
     - counting records in each group, or
     - applying numeric reductions (sum, average, minimum, maximum) if the grouped values are numeric.

Case 2: Aggregate to a coarser granularity only (no analysis)
Conditions:
- target_granularity is different with existing ones
- analyze_dimension is empty
Behavior:
1. The system aggregates the specified dimension to a coarser target granularity and creates a new granularity named <target_granularity>.
2. The system completes grouping, label assignment, and data population without producing any analysis results.

Case 3: Perform analysis only (no aggregation)
Conditions:
- target_granularity is an existing one
- analyze_dimension is provided and non-empty
Behavior:
1. The system preserves the current granularity of the specified dimension and performs no aggregation.
2. The system directly performs the requested analysis on the existing granularity:
   - For other dimensions, cross-dimension analysis or the specified reduction is performed.
   - For "self", only aggregate-level operations such as record counting or numeric reduction are allowed.

params:
{
  "dimension": "<The dimension to be aggregated to a coarser granularity>",
  "target_granularity": "<Name of the target granularity, or an existing one if no aggregation is performed>",
  "analyze_dimension": [
    {
      "thought": "<Why this analysis is needed or what insight is expected>",
      "dimension": "<Name of another dimension to analyze, or 'self'>",
      "reduce_target": "<Type of reduction or analysis to perform; set to None if not explicitly specified>"
    }
  ]
}

[Output Format]

If a next step is needed:
{
  "thought": "A brief explain of your choice",
  "action": {
    "type": "<one of: drill_down, roll_up, get_dimension, get_granularity>",
    "params": { ... }
  }
}

If the structure is already sufficient:
{
  "thought": "No further refinement needed. The current views are sufficient.",
  "action": null
}


Input:
- User query:
%s

- History:
%s
""" % (query, (history or ""))



def choose_dg(query: str, history: str = "") -> str:
    return """You are assisting in a knowledge organization system that operates on a structured, multidimensional semantic data object.

[Data Structure]
The data is organized across multiple dimensions and granularities:
- A dimension represents a semantic axis used to describe data from a specific angle.
- Each dimension contains one or more granularities, and these granularities represent different abstraction levels of that dimension. 
  - Finer granularities correspond to more detailed expressions.
  - Coarser granularities correspond to more abstract or generalized expressions.
- A granularity determines:
  1. How this dimension is represented at that abstraction level (i.e., what concrete value describes this dimension for each record).
  2. How data from other dimensions should be aggregated at this level of abstraction.
Thus, for each record, every dimension–granularity pair stores:
  - The value of that dimension at that granularity.
  - Aggregated summaries of all other dimensions, computed according to the granularity’s aggregation rules.
Each dimension always begins with a granularity of the same name, representing its initial and most detailed form.


[Your Task]
You are invoked after a full ReAct reasoning process has already been executed.  
You now receive:
- a user query, and  
- the prior ReAct history (which includes Thought → Action → Observation sequences)
Your task is: Based on the user query and the completed ReAct history, determine which single dimension and single granularity are the most appropriate to use as the final output view for answering the query.
You should analyze the ReAct steps, understand what semantic perspective the system has been converging toward, and choose the dimension and granularity that best represent the user’s required abstraction level and viewing angle.
Your output is not another ReAct step. You do not perform actions or generate new structure.  
You simply select the best existing dimension and granularity.

[Output Format]

Return exactly one JSON object containing:
{
  "thought": "<your reasoning for selecting this pair>",
  "dimension": "<the selected dimension name>",
  "granularity": "<the selected granularity name>"
}
The `thought` should concisely explain why this dimension and granularity best align with:
- the user query intention, and  
- the completed ReAct reasoning trajectory.
Do not output anything else.

User Query:
%s

ReAct History:
%s
""" % (query, (history or ""))



def determine_pass(query: str, description: str) -> str:
    return """You are assisting in a knowledge organization system that operates on a structured, multidimensional semantic data object.

[Data Structure]
The data is organized across multiple dimensions and granularities:

- A dimension represents a semantic axis used to describe or analyze data.
- Each dimension contains one or more granularities, representing different abstraction levels of that dimension:
  - Finer granularities provide more detailed expressions.
  - Coarser granularities provide more abstract or aggregated expressions.
- A granularity determines:
  1. How the dimension is represented at that abstraction level.
  2. How values from other dimensions should be aggregated at this level.

Therefore, for each record, every dimension–granularity pair stores:
- The value of that dimension at the selected granularity.
- Aggregated summaries of all other dimensions, computed according to the granularity’s rules.

Each dimension always begins with a granularity of the same name, representing its initial and most detailed form.


[Your Task]
You are invoked after a full ReAct reasoning process has been completed.
You now receive:
- A user query, and
- A description of the generated table that corresponds to a specific dimension–granularity pair.

Your goal is to determine whether the information contained in this table is sufficient to answer the user query.
Each piece of information required by the query must be explicitly represented by a corresponding field in the table, and must not rely on inference, assumptions, or derived interpretation.

- If the table contains enough information to answer the query (i.e., all needed columns have been built, including needed dimensions, count for some dimension, or specific granularity), return: sufficient: true
- If the table does not contain enough information, return: sufficient: false, and explain what information is missing or why the table cannot support the query.


[Output Format]

Return exactly one JSON object in the following form:

{
  "thought": "<your reasoning>",
  "sufficient": true or false,
  "reason": "<explanation only if sufficient is false>"
}

User Query:
%s

Description:
%s
""" % (query, description)


UNDERSTAND_TOPK_PROMPT = """
You are a query analyzer responsible for determining whether a user's natural language query contains a top-k intent, and if so, standardizing the extracted information.

[Task Instructions]
1. Determine whether the query expresses top-k intent.
- Top-k intent is typically indicated by phrases like: top 5, first, most, highest, etc.
- Ignore general sorting words if they do not explicitly indicate a top-N selection.

2. If top-k intent is found:
- topk_type: Choose "num" when the ranking is based on a measurable or computed quantity from the structured field candidates.  
  Choose "sem" when the ranking is based on qualitative, descriptive, or subjective characteristics.
- sort_field: The field to rank by. Must be one of the structured field candidates above.
- sort_order: 'desc' if the query implies largest/highest/most; 'asc' if it implies smallest/lowest/least.
- top_k: The numeric value of N (e.g., 5, 10).
- sort_basis: A natural language description explaining the basis of sorting as implied by the query.

3. If no top-k intent is found, return a simple flag.

[Output Requirements]
Return strictly valid JSON with no extra text.

If top-k intent is found:
{
  "topk_type": "<choose 'num' for numeric top-k, or 'sem' for semantic top-k>",
  "sort_field": "<name of the field from structured list>",
  "sort_order": "desc",
  "top_k": <number>,
  "sort_basis": "<natural language description of the sorting basis>"
}

If not found:
{
  "topk_type": "not applicable"
}


[Input Information]
- Structured field candidates that the user may want to rank by:  
%s  

- User Query:  
%s  
"""

def Preprocess_query(query, llm):
    prompt = """
    You are given a natural language query.
Your task is to determine whether the query contains any expressions related to sorting, ranking, or top-k semantics (for example: "most frequent", "top", "highest", "largest", "minimum", "best", "top k", "sorted by", etc.).

If such expressions exist:
- Remove all sorting / ranking / top-k related expressions from the query.
- Return the result in the following format:
{
  "has_sort_or_topk": true,
  "stripped_query": "<the query after removing sort/top-k related expressions>"
}

If no such expressions exist:
- Do not modify the query.
- Return the result in the following format:
{
  "has_sort_or_topk": false,
  "stripped_query": ""
}

Rules:
- Only remove phrases that explicitly imply sorting, ranking, comparison, or top-k selection.
- Do not rewrite, paraphrase, or otherwise alter the remaining content of the query.
- Ensure the output is valid JSON and strictly follows the specified schema.

Query:
%s
""" % (query)
    response = llm.predict(prompt)
    ret = jsonfy_llm_response(response)
    return ret

def merge_slice_operations(plan):
    logic = plan.get("logic", [])
    if not logic or logic[0] != "AND":
        return plan

    operations = plan.get("operations", [])
    slice_ops = [op for op in operations if op.get("agent") == "slice"]
    other_ops = [op for op in operations if op.get("agent") != "slice"]

    if len(slice_ops) <= 1:
        return plan

    merged_filter = ", and ".join(
        op["filtering_condition"] for op in slice_ops if op.get("filtering_condition")
    )

    new_id = max(op["id"] for op in operations) + 1

    merged_slice_op = {
        "id": new_id,
        "agent": "slice",
        "field": None,
        "filtering_condition": merged_filter
    }

    plan["operations"] = other_ops + [merged_slice_op]

    kept_ids = [op["id"] for op in other_ops] + [new_id]
    plan["logic"] = ["AND"] + kept_ids

    return plan



class OLAP_Agent:
    def __init__(self, llm: Any, data: pd.DataFrame, Cube: SemaFlex_Cube = None):
        self.llm = llm
        self.slice_agent = None
        self.data = data

        if (Cube is not None):
            self.Cube = Cube
        else:
            self.Cube = SemaFlex_Cube(self.data, llm)

        self.dice_agent = Dice_Agent(llm)
        self.slice_agent = Slice_Agent(llm)
        self.rollup_agent = Roll_Up_Agent(llm)
        self.drilldown_agent = Drill_Down_Agent(llm)
        self.Execution_Agent = Execution_Agent(llm)

        self.history = []

        self.agent_map = {"dice": self.dice_agent,
                          "slice": self.slice_agent,
                          "roll_up": self.rollup_agent,
                          "drill_down": self.drilldown_agent}

    def decompose_query_intent(self, query):
        prompt = """
You are a query decomposition assistant. Your task is to break down a user's natural language query into the following three semantically distinct sub-queries:

1. filter_query: Describes the subset of data the user wants to select. This typically includes constraints on time, entities, status, numerical ranges, etc. The essence is: “which data points are of interest.” This part narrows down the dataset by reducing the number of rows but does not change the structure of the data.

2. analysis_query: Describes how the user wants to organize, aggregate, or transform the selected data. This includes exploring a new dimension or group some dimensions into coarser granularity. The essence is: “how to process the selected data structurally.” This part may change the structure or granularity of the data, but does not affect the filtered scope.

Decomposition rules:
- If the original query does not contain a certain type of intent, return an empty string "" for that part.
- Use the same expression as the original query as much as possible
- Do not provide any explanation — only return the output in the following format:

{
  "filter_query": "...",
  "analysis_query": "..."
}

Query: %s
""" % query

        response = self.llm.predict(prompt)

        try:
            parsed = jsonfy_llm_response(response)
            return parsed.get("filter_query", "").strip(), parsed.get("analysis_query", "").strip()
        except Exception:
            return "", ""

    def plan_generate_filter(self, history_query, now_query):
        col_str = self.Cube.get_dimensions()
        col_str.remove("question_id")
        prompt = PLAN_OLAP_AGENT_PROMPT_SLICE_DICE % (col_str, history_query, now_query)
        result_str = self.llm.predict(prompt)
        result = jsonfy_llm_response(result_str)
        return result

    def execute_plan_filter(self, plan, ids):
        new_operations = []

        plan = merge_slice_operations(plan)
        for op in plan["operations"]:
            agent = op['agent']
            action = op['filtering_condition']
            field = op['field']
            agent_now = self.agent_map[agent]

            query = {'action': action, 'field': field}
            sub_plan = agent_now.run(query, self.Cube, ids)

            op_with_plan = {
                "id": op["id"],
                "plan": sub_plan
            }

            new_operations.append(op_with_plan)

        new_plan = {
            "operations": new_operations,
            "logic": plan["logic"]
        }

        return new_plan


    def ReAct_Reflect_singlepass(self, query_view, history, tmp_axis: OLAP_Axis):
        END_TAG = 0
        while END_TAG < 10:
            END_TAG += 1

            prompt = make_stepwise_ReAct_prompt(query_view, history)
            result = self.llm.predict(prompt)

            ret_json = jsonfy_llm_response(result, "failed")

            if(isinstance(ret_json, str) or "thought" not in ret_json):
                result = llm_retry(result, self.llm)
                ret_json = jsonfy_llm_response(result)


            if not ret_json or ret_json.get("action") is None or ret_json.get("action")=="null":

                history += (
                    f"thought: {ret_json['thought']}\n"
                    f"action: END"
                )
                break
            action = ret_json["action"]

            action_type = action["type"]
            params = action.get("params", {})
            params["thought"] = ret_json["thought"]
            if action_type == "drill_down":
                observation = self.drilldown_agent.fake_run(params, tmp_axis)
            elif action_type == "roll_up":
                observation = self.rollup_agent.fake_run(params, tmp_axis)
            elif action_type == "get_dimension":
                observation = tmp_axis.get_dimensions()
                observation.remove("question_id")
            elif action_type == "get_granularity":
                dimension = params["dimension"]
                try:
                    observation = tmp_axis.get_granularities(dimension)
                except:
                    observation = f"Dimension '{dimension}' does not exist"
            else:
                break  # Unknown or unsupported action

            history_now = {
                "thought": ret_json['thought'],
                "action": json.dumps(action, ensure_ascii=False),
                "observation": observation
            }

            history += (
                f"thought: {ret_json['thought']}\n"
                f"action: {json.dumps(action, ensure_ascii=False)}\n"
                f"observation: {observation}\n"
            )

            debug_log({
                "type": "log",
                "message": f"OLAP Agent: 内层迭代记录\n\n ```json\n{json.dumps(history_now, indent=4, ensure_ascii=False)}\n```"
            })
        postfix = ""

        for i in range(3):
            choose_prompt = choose_dg(query_view, history + "\n\n" + postfix)
            choose_result = self.llm.predict(choose_prompt)
            choose_ret = jsonfy_llm_response(choose_result)

            try:
                if (choose_ret["dimension"] != choose_ret["granularity"]):
                    cols_list = tmp_axis.get_granularity(choose_ret["dimension"], choose_ret["granularity"]).values
                    if ("OLAP_ID" in cols_list):
                        cols_list.remove("OLAP_ID")
                    desc = f"The data contains the following columns: {cols_list}. All values in this table are grouped and aggregated according to the {choose_ret['granularity']} granularity."
                else:
                    if(len(tmp_axis.get_granularity(choose_ret["dimension"], choose_ret["dimension"]).values)>1):
                        append_col_list = tmp_axis.get_granularity(choose_ret["dimension"], choose_ret["granularity"]).values
                    else:
                        append_col_list = []
                    cols_list = tmp_axis.get_dimensions()
                    while ("OLAP_ID" in cols_list):
                        cols_list.remove("OLAP_ID")
                    for item in append_col_list:
                        if(item not in cols_list):
                            cols_list.append(item)
                    desc = f"The table contains the following columns: {cols_list}. These columns represent raw values from a single dimension without any cross-dimensional aggregation or grouping applied."
                break
            except:
                postfix += f"You just return {choose_result}, but it doesn't exist, please retry and use another dimension-granularity tuple."

        pass_prompt = determine_pass(query_view, desc)
        pass_result = self.llm.predict(pass_prompt)

        pass_result = jsonfy_llm_response(pass_result)

        return history, tmp_axis, pass_result, choose_ret

    def run_roll_up_and_drill_down(self, query_view):

        history = ""
        tmp_axis = self.Cube.axis.copy()

        pass_result = {"sufficient": False}
        COUNT_TAG = 0
        while (pass_result["sufficient"] == False):
            COUNT_TAG += 1
            if (COUNT_TAG >= 3):
                break
            history, tmp_axis, pass_result, choose_ret = self.ReAct_Reflect_singlepass(query_view, history, tmp_axis)
            if(pass_result["sufficient"]==False):
                history += f"After check your result, user need you continue. Now the user get the table from dimension-granularity : '{choose_ret['dimension']}-{choose_ret['granularity']}'. In this table: {pass_result['reason']}"
        if(choose_ret["dimension"] == choose_ret["granularity"] and len(tmp_axis.get_granularity(choose_ret["dimension"], choose_ret["dimension"]).values)<=1):
            plan = tmp_axis.get_dimension(choose_ret["dimension"]).plan.copy()
        else:
            plan = tmp_axis.get_granularity(choose_ret["dimension"], choose_ret["granularity"]).plan.copy()
        plan.append({"from": "END", "params": choose_ret})
        return plan

    def merge_history(self, query):
        history_text = "\n".join(self.history)

        prompt = """You are a contextual query rewriting assistant.  
The user is conducting a multi-step data analysis task. Each query may depend on previous ones and may omit important context stated earlier.

Your task is to rewrite the current query into a fully self-contained version by incorporating any relevant information from the query history.

Instructions:
- The current query expresses the user's main intent. Preserve its original meaning exactly.
- Only add missing filters, data scopes, or topics that are clearly established in the history.
- Do not change the core intent or rephrase the query.
- If the current query is already complete, return it unchanged.

Output format (in strict JSON):
{
  "thought": "<brief explanation of what context you used or why no change was needed>",
  "rewritten_query": "<the rewritten query with context completed>"
}

[Query history]
%s

[Current query]
%s
""" % (history_text, query)

        result = self.llm.predict(prompt).strip()

        try:
            parsed = jsonfy_llm_response(result)
            return parsed["rewritten_query"]
        except Exception:
            return result

    def understand_presentation(self, query, fields):
        prompt = """
You are an information extraction assistant.

The user will input a natural language query describing a top-k selection from a dataset.

Your task is to extract the following three elements from the query:
1. The value of k (how many top items the user wants).
2. The field to sort by.
3. A natural language description of how the sorting should be done, using the user’s own phrasing or a close paraphrase.

Only include a field if it clearly matches one from the list. If no value or field is found, return null for that item.

Respond in the following JSON format:
{
  "k": <integer or null>,
  "field": "<matched field or null>",
  "description": "<sorting description or null>"
}

[Field list]  
%s

[User query]  
%s
""" % (fields, query)

        result = self.llm.predict(prompt).strip()

        try:
            parsed = jsonfy_llm_response(result)
            return parsed
        except Exception:
            # fallback: return original result as-is
            return {"k": None, "field": None, "description": None}

    def execute_enrichment(self, plan, ids):
        aps = []
        for p in plan:
            if p["from"] == "drill_down":
                self.drilldown_agent.run(p["params"], self.Cube, ids)

            elif p["from"] == "roll_up":
                ap = self.rollup_agent.run(p["params"], self.Cube, ids)
                aps.extend(ap)

            else:
                dim = p["params"]["dimension"]
                gra = p["params"]["granularity"]

                if dim == gra:
                    now_plan = self.Cube.axis.get_dimension(dim).plan.copy()
                    now_plan.extend(aps)

                    raw_df = self.Cube.read_raw(ids)

                    has_count = any(
                        step.get("operator_name") == "count"
                        for step in aps
                    )

                    if has_count:
                        try:
                            gra_df = self.Cube.read_granularity(ids, dim, gra)
                            count_cols = [
                                c for c in gra_df.columns
                                if c.startswith("count_of_") and c not in raw_df.columns
                            ]
                            if count_cols:
                                patch_df = gra_df[["OLAP_ID"] + count_cols].drop_duplicates(subset=["OLAP_ID"])
                                raw_df = raw_df.merge(patch_df, on="OLAP_ID", how="left")
                        except Exception:
                            pass

                    return raw_df, now_plan

                now_plan = self.Cube.axis.get_granularity(dim, gra).plan.copy()
                now_plan.extend(aps)
                return self.Cube.read_granularity(ids, dim, gra), now_plan

    def show_translation(self, show_table):
        if show_table is None or show_table.empty:
            return show_table

        df = show_table.drop(columns=["OLAP_ID"], errors="ignore")

        df = df.drop_duplicates()

        return df

    def apply_topk_from_analysis(self, show_table, analysis_query, show_plan):
        descriptions = []
        for col in show_table.columns:
            if col == "OLAP_ID":
                continue

            sample_vals = show_table[col].dropna().head(2).tolist()
            total_chars = sum(len(s) for s in sample_vals if isinstance(s, str))

            if total_chars > 100:
                descriptions.append(f"Field: {col} | Samples: content too long to display")
            else:
                descriptions.append(f"Field: {col} | Samples: [{', '.join(map(str, sample_vals))}]")

        table_str = "\n".join(descriptions)

        prompt_topk = UNDERSTAND_TOPK_PROMPT % (table_str, analysis_query)
        result = self.llm.predict(prompt_topk)


        topk_params = jsonfy_llm_response(result)

        if topk_params["topk_type"] in ["num", "sem"]:
            if topk_params["top_k"] is None:
                topk_params["top_k"] = 1

            if topk_params["topk_type"] == "sem":
                show_table = sem_topk(
                    llm=self.llm,
                    df=show_table,
                    columns=[topk_params["sort_field"]],
                    query=topk_params["sort_basis"] + f"\n Order: {topk_params['sort_order']}",
                    k=topk_params["top_k"]
                )
                show_plan.append({
                    "operator_name": "sem_topk",
                    "parameters": {
                        "column": topk_params["sort_field"],
                        "query": topk_params["sort_basis"] + f"\n Order: {topk_params['sort_order']}",
                        "k": topk_params["top_k"]
                    }
                })
            else:
                show_plan.append({
                    "operator_name": "num_topk",
                    "parameters": {
                        "column": topk_params["sort_field"],
                        "k": topk_params["top_k"],
                        "order": topk_params["sort_order"]
                    }
                })

                if is_column_numeric(show_table, topk_params["sort_field"]):
                    show_table = num_topk(
                        df=show_table,
                        column=topk_params["sort_field"],
                        k=topk_params["top_k"],
                        order=topk_params["sort_order"]
                    )
                else:

                    pass


        return show_table, show_plan

    def save_query_plan(self, query, show_plan, record_message):
        if record_message is None:
            return

        id = record_message["id"]
        Q_id = record_message["Q_id"]

        write_message = {
            "Query_Self_Contained": query,
            "plan": show_plan
        }

        file_path = "results_202604/Semantic_OLAP_doubao.json"

        if not os.path.exists(file_path):
            data_list = []
        else:
            with open(file_path, "r", encoding="utf-8") as f:
                try:
                    data_list = json.load(f)
                except json.JSONDecodeError:
                    data_list = []

        found = False
        for item in data_list:
            if item.get("id") == id:
                item[Q_id] = write_message
                found = True
                break

        if not found:
            new_item = {"id": id, Q_id: write_message}
            data_list.append(new_item)

        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(data_list, f, ensure_ascii=False, indent=4)

    def run_filter_stage(self, filter_query):
        show_plan = []

        if filter_query == "":
            state = "Equal"
            nodes = [0]
            olap_id_list = self.Cube.dag.get_node(0).olap_ids
        else:
            nodes, olap_id_list, state = self.Cube.locate_from_dag(filter_query)

        if state != "Equal":
            if nodes == []:
                nodes = [0]

            history_query = self.Cube.dag.get_node(nodes[0]).query
            filt_plan = []
            filt_plan.extend(self.Cube.dag.get_node(nodes[0]).plan.copy())

            plan_filter = self.plan_generate_filter(history_query, filter_query)
            operations_filter = self.execute_plan_filter(plan_filter, olap_id_list)
            result_df, f_plan = self.Execution_Agent.run_filter(
                operations_filter, self.Cube, olap_id_list
            )

            filt_plan.extend(f_plan)
            olap_id_list = result_df["OLAP_ID"].astype(int).tolist()
            self.Cube.update_dag(nodes, olap_id_list, filter_query, filt_plan)
        else:
            result_df = self.Cube.read_raw(olap_id_list)
            filt_plan = self.Cube.dag.get_node(nodes[0]).plan.copy()

        show_plan.extend(filt_plan)

        return result_df, show_plan, olap_id_list

    def run_roll_up_and_drill_down_on_axis(self, query_view, axis_snapshot):
        history = ""
        tmp_axis = axis_snapshot

        pass_result = {"sufficient": False}
        COUNT_TAG = 0
        while pass_result["sufficient"] is False:
            COUNT_TAG += 1
            if COUNT_TAG >= 3:
                break
            history, tmp_axis, pass_result, choose_ret = self.ReAct_Reflect_singlepass(
                query_view, history, tmp_axis
            )
            if pass_result["sufficient"] is False:
                history += (
                    f"After check your result, user need you continue. "
                    f"Now the user get the table from dimension-granularity : "
                    f"'{choose_ret['dimension']}-{choose_ret['granularity']}'. "
                    f"In this table: {pass_result['reason']}"
                )

        if (
                choose_ret["dimension"] == choose_ret["granularity"]
                and len(tmp_axis.get_granularity(choose_ret["dimension"], choose_ret["dimension"]).values) <= 1
        ):
            plan = tmp_axis.get_dimension(choose_ret["dimension"]).plan.copy()
        else:
            plan = tmp_axis.get_granularity(
                choose_ret["dimension"],
                choose_ret["granularity"]
            ).plan.copy()

        plan.append({"from": "END", "params": choose_ret})
        return plan

    def run_analysis_plan_stage(self, analysis_query, axis_snapshot):
        if analysis_query == "":
            return None
        return self.run_roll_up_and_drill_down_on_axis(analysis_query, axis_snapshot)

    def run(self, query, record_message=None):
        if record_message is None:
            self.record_message = {}
        else:
            self.record_message = record_message
        self.record_message["query"] = query

        filter_query, analysis_query = self.decompose_query_intent(query)


        axis_snapshot = None
        if analysis_query != "":
            axis_snapshot = self.Cube.axis.copy()

        with ThreadPoolExecutor(max_workers=2) as executor:
            future_filter = executor.submit(self.run_filter_stage, filter_query)

            future_analysis = None
            if analysis_query != "":
                future_analysis = executor.submit(
                    self.run_analysis_plan_stage,
                    analysis_query,
                    axis_snapshot
                )

            result_df, show_plan, olap_id_list = future_filter.result()
            enrichment_plan = future_analysis.result() if future_analysis else None

        if analysis_query != "":
            show_table, enrich_plan = self.execute_enrichment(enrichment_plan, olap_id_list)
            show_plan.extend(enrich_plan)

            show_table = self.show_translation(show_table)

            show_table, show_plan = self.apply_topk_from_analysis(
                show_table=show_table,
                analysis_query=analysis_query,
                show_plan=show_plan
            )
        else:
            show_table = result_df



        self.save_query_plan(
            query=query,
            show_plan=show_plan,
            record_message=self.record_message
        )

        return show_table


def is_column_numeric(df, col_name):
    series = df[col_name].dropna()  #
    numeric_series = pd.to_numeric(series, errors='coerce')
    return not numeric_series.isna().any()


import json


def df_to_str_json(df: pd.DataFrame) -> str:
    return df.to_json(orient='records', force_ascii=False)

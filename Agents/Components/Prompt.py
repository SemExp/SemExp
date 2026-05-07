

def get_filtering_prompt(col_str, history_query, now_query):
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
    return PLAN_OLAP_AGENT_PROMPT_SLICE_DICE % (col_str, history_query, now_query)

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
The roll_up operation performs semantic grouping on an existing dimension to create or use a coarser granularity.
The `dimension` parameter must be an existing dimension in the current data structure. The `target_granularity` parameter specifies the coarser granularity level you want to group the data into.
You only need to provide the name of the target granularity. The system will automatically create the new granularity (if it doesn't exist), group records by semantic similarity, generate representative labels for each group, and populate values. You do NOT need to manually create the granularity via drill_down or any other operation.

params:
{
  "dimension": "<An existing dimension name that you want to group>",
  "target_granularity": "<The coarser granularity level to aggregate into>"
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

def choose_view(query: str, history: str = "") -> str:
    return """You are assisting in a knowledge organization system that operates on a structured, multidimensional semantic data object.

[Data Structure]
The data is organized across multiple dimensions and granularities:
- A dimension represents a semantic axis used to describe data from a specific angle.
- Each dimension contains one or more granularities, and these granularities represent different abstraction levels of that dimension.
  - Finer granularities correspond to more detailed expressions.
  - Coarser granularities correspond to more abstract or generalized expressions.
- A granularity determines:
  - How this dimension is represented at that abstraction level (i.e., what concrete value describes this dimension for each record).
Thus, for each record, every dimension–granularity pair stores:
  - The value of that dimension at that granularity.
Each dimension always begins with a granularity of the same name, representing its initial and most detailed form.

[Your Task]
You are invoked after a full ReAct reasoning process has already been executed.
You now receive:
- a user query, and
- the prior ReAct history (which includes Thought → Action → Observation sequences).

Your task is to determine how the final output view should be organized in order to answer the query.

Specifically, based on the user query and the completed ReAct history, you must decide:
1. which dimension–granularity pairs the query requires for grouping the data in the final output view; and
2. after grouping by those pairs, which dimensions need to be further aggregated and analyzed within the resulting groups.

The selected dimension–granularity pairs define the grouping structure of the final result.
The aggregation_dimensions define which dimensions should be summarized, analyzed, or reported within each group formed by the selected grouping pairs.

The grouping structure may contain multiple dimension–granularity pairs, although a single pair is the most common case.
You should choose as many grouping pairs as are truly required by the query, but no more.

You should analyze the completed ReAct history and infer what final result organization the reasoning process has converged toward.
Focus on the final analytical structure required to answer the query, rather than intermediate temporary structures created during reasoning.

Important:
- Do not output a single pair by default; output all grouping pairs required by the query. In the common case, the grouping structure will contain only one dimension–granularity pair.
- Multiple grouping pairs should be used only when the query truly requires multi-dimensional grouping in the final result. 
- A dimension and its selected granularity may be identical; this is valid and corresponds to using the base granularity of that dimension.
- aggregation_dimensions should contain the dimensions that need to be aggregated or analyzed after grouping, not simply every dimension that is not selected for grouping.
- A dimension should not appear in aggregation_dimensions unless the query requires aggregation, summarization, or analysis over that dimension.
- Prefer the final analytical organization implied by the completed ReAct process, not intermediate temporary structures.
- Do not generate new dimensions or granularities unless they are clearly already established in the completed reasoning trajectory.
- If a dimension is used for grouping in the final view, include it in view_pairs.
- If a dimension is to be analyzed within groups, include it in aggregation_dimensions.
- aggregation_dimensions may be empty if the query only requires grouping and does not ask for any further aggregation or analysis within groups.
- aggregation_dimensions may include "count" when the query requires counting the number of records in each group.


[Output Format]
Return exactly one JSON object containing:
{
  "thought": "<concise reasoning for the selected grouping structure and aggregation targets>",
  "view_pairs": [
    {"dimension": "<dimension name>", "granularity": "<granularity name>"}
  ],
  "aggregation_dimensions": ["<dimension name>"]
}

Requirements:
- view_pairs must be a list of dimension–granularity pairs used for grouping.
- aggregation_dimensions must be a list of dimensions to be aggregated or analyzed after grouping.
- Include all and only the grouping pairs necessary to answer the query.
- Include all and only the aggregation dimensions necessary to answer the query.
- Do not output anything else.

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




def get_topk_prompt(table_str, analysis_query):
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
    prompt_topk = UNDERSTAND_TOPK_PROMPT % (table_str, analysis_query)
    return prompt_topk

def get_decompose_prompt(query):
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
    return prompt
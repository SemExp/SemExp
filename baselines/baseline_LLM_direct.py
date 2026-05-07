import json
import re




def get_operators(mode = "all"):

    sem_filter_str = """
    Operator Name: sem_filter
    Description:
        - Used when you want to keep only the rows that match a user-defined natural language condition, based on the semantic meaning of the data.
    Behavior:
        - This operator removes rows that do not satisfy the condition, returning a subset of the original table. Only the rows that semantically match the predicate will be retained.
    Parameters:
        - columns: The column or columns to use when evaluating the condition. These should contain the relevant textual information.
        - condition: A natural language filtering expression that specifies the desired semantics.
    """

    keyword_filter_str = """
    Operator Name: keyword_filter
    Description:
        Used when you are certain that semantic interpretation is not needed, and you want to filter rows based on whether specific keywords appear in one or more columns, using literal string matching rather than semantic reasoning.
    Behavior:
        - This operator removes rows that do not satisfy the keyword matching criteria. It allows for configurable logical relationships between multiple keywords and supports different string comparison relations.
    Parameters:
        - columns: The column or columns in which to search for the keywords.
        - keywords: A keyword or list of keywords used to evaluate each row.
        - logic: The logical relationship applied across multiple keywords. Options are 'and' or 'or'.
        - relation: The type of string comparison to apply. Options include 'contains', 'contained_in', 'equal', and 'irrelevant'.
    """

    num_filter_str = """
    Operator Name: num_filter
    Description:
        - Used when you are certain that semantic interpretation is not needed, and you want to filter rows based on whether specific keywords appear in one or more columns, using literal string matching rather than semantic reasoning.
    Behavior:
        - This operator evaluates multiple numeric conditions on the table. Each condition specifies a column, a threshold, and a comparison operator. The overall filter logic is controlled by a logic parameter ('and' or 'or'). Only rows that satisfy the combined condition will be retained.
    Parameters:
        - conditions: A list of conditions, where each condition includes:
            - column: The numeric column to evaluate.
            - threshold: The numeric value used for comparison.
            - operator: The comparison operator to use. Options include '==', '!=', '>', '<', '>=', and '<='.
        - logic: The logical relationship used to combine multiple conditions. Options are 'and' or 'or'.
    """

    sem_map_str = """
    Operator Name: sem_map
    Description:
        - Used when you want to generate new content or values for each row based on existing column values and a natural language generation instruction.
    Behavior:
        - This operator adds a new column to the table, named according to the provided keyword. For each row, the model uses the specified columns as input context and follows the natural language instruction given in the description to generate a new value. The result of this generation is stored under the column named by the keyword.
    Parameters:
        - columns: The column or columns to provide context for the generation task.
        - map_description: A natural language instruction describing what should be generated.
        - keyword: The name of the new column to be created, representing the target of the generation.
    """

    sem_group_str = """
    Operator Name: sem_group
    Description:
        - Used when you want to group rows by a specific dimension described in natural language, allowing the model to assign each row to a semantically coherent group.
    Behavior:
        - This operator performs semantic grouping on the specified columns using the provided dimension description as the basis for clustering. It adds a new column to the table, named according to the given keyword, where each row is labeled with the group it belongs to under that dimension.
    Parameters:
        - columns: The column or columns whose content will be used for grouping.
        - group_description: A natural language description of the dimension along which the grouping should occur.
        - keyword: The name of the new column to be created, representing the grouping label for each row.
    """

    reduce_str = """
    Operator Name: sem_reduce
    Description:
        - Used when you want to reduce the entire table or each group within the table to a condensed form using a language model. This operator performs global or grouped reduction using selected columns as input.
    Behavior:
        - This operator applies a language model to summarize or compress the specified columns of the table. If a group_by column is provided, the table is first grouped by that column, and each group is reduced to a single row using the selected columns. If no group_by is specified, the entire table is reduced into a single-row table. Only the selected columns will be preserved in the output; all other columns will be discarded.
    Parameters:
        - columns: The column or columns to include in the reduction. Only these will be retained in the output.
        - group_by: An optional column name used to group rows before applying the reduction. If provided, reduction is performed per group. If omitted, the entire table is reduced as one unit.
    """

    num_reduce_str = """
    Operator Name: num_reduce
    Description:
        - Used when you want to reduce numeric columns by applying standard aggregation functions such as sum, average, or maximum, either across the entire table or within groups.
    Behavior:
        - This operator performs aggregation over the specified numeric columns using the specified aggregation functions. If a group_by column is provided, the table is grouped accordingly and each group is reduced to a single row. If no group_by is specified, the entire table is reduced into one row. The resulting table replaces the original one.
    Parameters:
        - columns: The numeric column or columns to aggregate. Only these will be retained in the output.
        - agg: The aggregation function or list of functions to apply. Options include 'sum', 'avg', 'max', and 'min'.
        - group_by: An optional column used to group rows before reduction. If provided, aggregation is applied within each group separately.
    """

    count_str = """
    Operator Name: count
    Description:
        - Used when you want to perform grouped counting over a table. This includes counting the number of rows per group, as well as the number of unique values in specified columns.
    Behavior:
        - This operator groups the table by the specified group_by column. If the columns list is empty, it counts the number of rows in each group and outputs that as a 'count' column. If one or more columns are specified, it computes the number of unique values in each of those columns for every group. The resulting table has one row per group and includes the count of rows and/or unique values.
    Parameters:
        - group_by: The column used to group the table before counting.
        - columns: A list of columns to compute the number of unique values per group. If this list is empty, only the number of rows per group will be computed.
    """

    num_topk_str = """
    Operator Name: num_topk
    Description:
        - Used when you want to retrieve the top-k rows with the highest or lowest values in a specific numeric column.
    Behavior:
        - This operator sorts the table by the specified numeric column in the given order and returns only the top-k rows. It is useful for selecting the best, worst, or most extreme values based on quantitative criteria.
    Parameters:
        - column: The numeric column to sort by.
        - k: The number of top rows to retain.
        - order: The sort order. Options are 'asc' for smallest values or 'desc' for largest values.
    """

    sem_topk_str = """
    Operator Name: sem_topk
    Description:
        - Used when you want to retrieve the top-k rows that are most semantically relevant to a given natural language query, based on a language model's interpretation.
    Behavior:
        - This operator performs semantic ranking over a specified column using a natural language query. It returns the top-k rows that are most aligned with the query's meaning.
    Parameters:
        - column: The text column to search within.
        - query: A natural language expression representing the semantic intent.
        - k: The number of top rows to return based on semantic relevance.
    """

    if(mode == "all"):
        return [
            sem_filter_str,
            keyword_filter_str,
            num_filter_str,
            sem_map_str,
            sem_group_str,
            reduce_str,
            num_reduce_str,
            count_str,
            num_topk_str,
            sem_topk_str
        ]
    else:
        return [
            sem_filter_str,
            sem_map_str,
            sem_group_str,
            reduce_str,
            num_reduce_str,
            count_str,
            sem_topk_str
        ]



def get_prompt(query, mode = "all"):

    operations = get_operators(mode = mode)
    operations_str = "\n\n---------\n\n".join(operations)
    prompt_template = """
    You are a query planner that converts user requests written in natural language into a sequence of structured operations over a single table.

    The table has the following columns:
    ["question_id", "title", "body", "tags", "score", "comments", "answers", "creation_date"]

    You can choose from the following available operators:
    %s

    Your task is to:
    - Decompose the user's request into a sequence of data transformation steps
    - For each step, choose the most appropriate operator and extract the required parameters
    - Use only the available operators and only the given columns
    - All operations should be applied on a single table and may involve adding new columns, filtering rows, or updating values.
    - Return the result as a List [{...}, {...}, ...], where each element has this format:
    [{
      "operator_name": "...",
      "parameters": {
        "...": "..."
      }
    },
    ......
    ]

    User request:
    "%s"

    Just output the plan in format, and never output any other information.
    """ % (operations_str, query)

    return prompt_template


def get_plan_LLM_direct(llm, query, mode = "all"):
    prompt = get_prompt(query, mode)
    result = llm.predict(prompt)
    return result













import os
from archytas.react import ReActAgent
from archytas.tool_utils import tool, is_tool, get_tool_names
from archytas.tool_utils import collect_tools_from_object
import json


class MyAgent:
    # ------------------------------------------------------------------ tools

    @tool()
    def sem_filter(self, columns: list[str], condition: str) -> str:
        """
        Used when you want to retain only the rows that match a user-defined natural
        language condition based on the semantic meaning of the data. This operator
        evaluates the specified condition using a language model, and removes rows
        that do not semantically satisfy it, returning a subset of the original table.

        Args:
            columns (list[str]): The column or columns to use when evaluating the
                                 condition. These should contain the relevant textual
                                 information.
            condition (str): A natural language filtering expression that specifies
                             the desired semantics.

        Returns:
            A data table containing only the rows that semantically match the given
            condition.
        """
        self.plan.append(("sem_filter", {"columns": columns, "condition": condition}))
        return "sem_filter step has been executed."

    @tool()
    def keyword_filter(self, columns: list[str], keywords: list[str],
                       logic: str, relation: str) -> str:
        """
        Used when you are certain that semantic interpretation is not needed, and you
        want to filter rows based on whether specific keywords appear in one or more
        columns, using literal string matching. This operator removes rows that do not
        satisfy the keyword matching criteria, supports logical relations across
        multiple keywords, and allows for configurable string comparison methods.

        Args:
            columns (list[str]): The column or columns in which to search for the
                                 keywords.
            keywords (list[str]): A list of keywords used to evaluate each row.
            logic (str): The logical relationship applied across multiple keywords.
                         Options are 'and' or 'or'.
            relation (str): The type of string comparison to apply. Options include
                            'contains', 'contained_in', 'equal', and 'irrelevant'.

        Returns:
            A data table containing rows that satisfy the keyword-based filter.
        """
        self.plan.append(("keyword_filter", {
            "columns": columns, "keywords": keywords,
            "logic": logic, "relation": relation
        }))
        return "keyword_filter step has been executed."

    @tool()
    def num_filter(self, conditions: list[dict], logic: str) -> str:
        """
        Used when you want to filter rows based on numeric conditions without semantic
        interpretation. Each condition includes a column, a numeric threshold, and a
        comparison operator. The final filter result is determined by applying logical
        operators across all conditions.

        Args:
            conditions (list[dict]): A list of conditions, each containing:
                - columns (str): The numeric column to evaluate.
                - threshold (float): The numeric value used for comparison.
                - operator (str): The comparison operator to use. Options include
                                  '==', '!=', '>', '<', '>=', and '<='.
            logic (str): The logical relationship used to combine multiple conditions.
                         Options are 'and' or 'or'.

        Returns:
            A data table containing only the rows that match the numeric filter
            conditions.
        """
        self.plan.append(("num_filter", {"conditions": conditions, "logic": logic}))
        return "num_filter step has been executed."

    @tool()
    def sem_map(self, columns: list[str], map_description: str, keyword: str) -> str:
        """
        Used when you want to generate new content or values for each row based on
        existing column values and a natural language generation instruction. This
        operator creates a new column named with the provided keyword and uses the
        language model to produce values for each row according to the specified
        instruction.

        Args:
            columns (list[str]): The column or columns to provide context for the
                                 generation task.
            map_description (str): A natural language instruction describing what
                                   should be generated.
            keyword (str): The name of the new column to be created, representing the
                           target of the generation.

        Returns:
            A data table with a new column generated from the given description.
        """
        self.plan.append(("sem_map", {
            "columns": columns, "map_description": map_description, "keyword": keyword
        }))
        return "sem_map step has been executed."

    @tool()
    def sem_group(self, columns: list[str], group_description: str,
                  keyword: str) -> str:
        """
        Used when you want to group rows based on a natural language description of a
        dimension. This operator clusters rows into semantically coherent groups using
        the specified columns, and appends a new column with group labels.

        Args:
            columns (list[str]): The column or columns whose content will be used for
                                 grouping.
            group_description (str): A natural language description of the dimension
                                     along which the grouping should occur.
            keyword (str): The name of the new column to be created, representing the
                           grouping label for each row.

        Returns:
            A data table with an added column indicating semantic group assignments.
        """
        self.plan.append(("sem_group", {
            "columns": columns, "group_description": group_description,
            "keyword": keyword
        }))
        return "sem_group step has been executed."

    @tool()
    def sem_reduce(self, columns: list[str], group_by: str | None = None) -> str:
        """
        Used when you want to reduce a table or groups within a table into a more
        condensed form using a language model. When a grouping column is specified,
        each group is summarized separately; otherwise, the entire table is reduced
        into a single row. Only selected columns are preserved in the output.

        Args:
            columns (list[str]): The column or columns to include in the reduction.
            group_by (str, optional): A column name used to group rows before applying
                                      the reduction.

        Returns:
            A reduced data table summarizing the selected content.
        """
        self.plan.append(("sem_reduce", {"columns": columns, "group_by": group_by}))
        return "sem_reduce step has been executed."

    @tool()
    def num_reduce(self, columns: list[str], agg: list[str],
                   group_by: str | None = None) -> str:
        """
        Used when you want to perform numeric aggregation over one or more columns,
        either globally or within groups. Aggregations include functions such as sum,
        average, max, and min. If a group column is specified, aggregation is applied
        per group.

        Args:
            columns (list[str]): The numeric columns to aggregate.
            agg (list[str]): The aggregation function or functions to apply. Options
                             include 'sum', 'avg', 'max', and 'min'.
            group_by (str, optional): A column used to group rows before reduction.

        Returns:
            A data table containing the aggregated numeric results.
        """
        self.plan.append(("num_reduce", {
            "columns": columns, "agg": agg, "group_by": group_by
        }))
        return "num_reduce step has been executed."

    @tool()
    def count(self, group_by: str, columns: list[str] = None) -> str:
        """
        Used when you want to count the number of rows per group or compute the number
        of unique values in specified columns. When no specific columns are provided,
        it outputs the total row count per group; otherwise, it reports counts of
        distinct values per column.

        Args:
            group_by (str): The column used to group the table before counting.
            columns (list[str], optional): A list of columns to compute unique value
                                           counts for each group.

        Returns:
            A data table showing row and/or unique value counts per group.
        """
        columns = columns or []
        self.plan.append(("count", {"group_by": group_by, "columns": columns}))
        return "count step has been executed."

    @tool()
    def num_topk(self, column: str, k: int, order: str) -> str:
        """
        Used when you want to retrieve the top-k rows with the highest or lowest values
        in a numeric column. The table is sorted accordingly and only the top-k results
        are returned.

        Args:
            column (str): The numeric column to sort by.
            k (int): The number of top rows to retain.
            order (str): The sort order. Options are 'asc' for smallest values or
                         'desc' for largest values.

        Returns:
            A data table containing the top-k rows based on numeric sorting.
        """
        self.plan.append(("num_topk", {"column": column, "k": k, "order": order}))
        return "num_topk step has been executed."

    @tool()
    def sem_topk(self, column: str, query: str, k: int) -> str:
        """
        Used when you want to retrieve the top-k rows that are most semantically
        relevant to a given natural language query. This operator uses a language
        model to rank rows based on their alignment with the query’s intent and
        selects the most relevant ones.

        Args:
            column (str): The text column to search within.
            query (str): A natural language expression representing the semantic
                         intent.
            k (int): The number of top rows to return based on semantic relevance.

        Returns:
            A data table containing the top-k semantically relevant rows.
        """
        self.plan.append(("sem_topk", {"column": column, "query": query, "k": k}))
        return "sem_topk step has been executed."

    @tool()
    def display_plan(self) -> str:
        """
        Used when you want to terminate the reasoning process and return the final execution plan.
        This is typically called as the last step in a planning sequence to output the accumulated plan
        in a serialized format. It stops further execution and hands back the complete deferred plan.

        Args:
            This function takes no arguments.

        Returns:
            A serialized representation of the recorded execution plan.
        """

        return "The plan has been completed. You may now safely exit the ReAct loop."

        # if(self.plan):
        #     return json.dumps(self.plan, ensure_ascii=False, indent=2)
        # else:
        #     return "None"


    # ------------------------------------------------------------------ Other utils

    def __init__(self, model: str = "gpt-4o", verbose: bool = True):
        self.plan: list[tuple[str, dict]] = []

        self.tools = collect_tools_from_object(self)
        prefix = """
        You are an intelligent table transformation agent.  
        Your job is to solve user requests by operating on a single tabular dataset using a set of predefined tools.

        The dataset is like a DataFrame and contains the following columns:
        ["question_id", "title", "body", "tags", "score", "comments", "answers", "creation_date"]

        You must use the available tools to process the data step by step to fulfill the user’s intent.  
        You are not allowed to answer directly or explain — your job is to select and invoke tools appropriately.  

        Do not speculate or guess missing data — always rely on the tools, even for simple operations.  
        After you are done with all necessary operations, use the `display_plan` tool as the final step.
        """

        self.agent = ReActAgent(tools=self.tools, verbose=verbose, allow_ask_user=False, model=model, custom_prelude=prefix)
        print(self.show_tool_names())

    def show_plan(self) -> str:
        """
        Display the current deferred-execution plan.
        """
        if not self.plan:
            return "Plan is empty."
        return "\n".join(f"{idx+1}. {name} -> {params}"
                         for idx, (name, params) in enumerate(self.plan))

    def clear_plan(self) -> str:
        """
        Clear all recorded steps in the plan.
        """
        self.plan.clear()
        return "Plan cleared."

    def show_tool_names(self) -> list[str]:
        """
        Returns a list of all registered tool names.
        """
        return get_tool_names(self.tools)


    def run(self, query) -> list[tuple[str, dict]]:
        """
        Convert the current plan into an executable-style script and display it.
        This does NOT actually execute anything.
        """
        self.clear_plan()
        _ = self.agent.react(query)
        print(self.show_plan())
        print("="*50)
        print(self.plan)
        return self.plan


if __name__ == "__main__":
    agent = MyAgent()
    print(agent.run("please find the 2020 data"))

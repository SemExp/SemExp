import os
import json
from pathlib import Path
from typing import Dict, List, Tuple, Any
from archytas.react import ReActAgent
from archytas.tool_utils import tool, collect_tools_from_object


class PalimpzestPlanAgent:
    """
    Palimpzest Plan Agent that integrates the original PalimpZest planning strategy
    with a template-based implementation. This agent focuses on generating logical plans 
    (natural language → tool call chain) without executing them.
    """

    def __init__(self, model: str = "gpt-4o", verbose: bool = True, 
                 procedures_path: str = None):
        # Initialize plan and code storage
        self.plan: List[Tuple[str, Dict[str, Any]]] = []
        self.generated_code: List[str] = []
        self.existing_schemas = {}  # Track created schemas
        
        # Set procedures path
        if procedures_path is None:
            procedures_path = "baselines/procedures/python3"
        self.procedures_path = Path(procedures_path)
        
        # Load procedure templates
        self.templates = self._load_templates()
        
        # Collect tools
        self.tools = collect_tools_from_object(self)
        
        # Initialize agent with PalimpZest strategy prompt
        prefix = """
        You are an intelligent table transformation agent.  
        Your job is to solve user requests by operating on a single tabular dataset using a set of predefined tools.

        The dataset is like a DataFrame and contains the following columns:
        ["question_id", "title", "body", "tags", "score", "comments", "answers", "creation_date"]

        You need to start with such dataset (named stackoverflow_data), and help user complete their query.
        You must use the available tools to process the data step by step to fulfill the user’s intent.  
        You are not allowed to answer directly or explain — your job is to select and invoke tools appropriately.  

        Do not speculate or guess missing data — always rely on the tools, even for simple operations.  
        After you are done with all necessary operations, use the `display_plan` tool as the final step.
        """
        
        self.agent = ReActAgent(
            tools=self.tools, 
            verbose=verbose, 
            allow_ask_user=False, 
            model=model, 
            custom_prelude=prefix
        )

    def _load_templates(self) -> Dict[str, str]:
        """Load procedure templates from the original PalimpZest files."""
        templates = {}
        
        # Load each procedure template from PalimpChat procedures
        procedure_files = [
            'register_dataset.py', 'unregister_dataset.py', 'list_datasets.py',
            'list_schemas.py', 'create_schema.py', 'pick_schema.py',
            'set_input_dataset.py', 'filter_data.py', 'convert_dataset.py',
            'execute_workload.py', 'override_dataset.py', 'print_statistics.py',
            'retrieve_dataset.py',
        ]
        
        for filename in procedure_files:
            file_path = self.procedures_path / filename
            if file_path.exists():
                with open(file_path, 'r') as f:
                    template_name = filename.replace('.py', '')
                    templates[template_name] = f.read()
        
        return templates

    def _render_template(self, template_name: str, params: Dict[str, Any]) -> str:
        """Template rendering system to replace beaker context with proper variable handling."""
        if template_name not in self.templates:
            return f"# Template '{template_name}' not found"
        
        template = self.templates[template_name]
        
        # Template variable replacement handling
        for key, value in params.items():
            placeholder = f"{{{{ {key} }}}}"
            if isinstance(value, str):
                template = template.replace(placeholder, f'"{value}"')
            elif isinstance(value, list):
                # Handle list variables properly for Python code generation
                if all(isinstance(item, str) for item in value):
                    formatted_list = "[" + ", ".join([f'"{item}"' for item in value]) + "]"
                else:
                    formatted_list = str(value)
                template = template.replace(placeholder, formatted_list)
            elif isinstance(value, bool):
                template = template.replace(placeholder, str(value).lower())
            else:
                template = template.replace(placeholder, str(value))
        
        return template

    # ============================================================================
    # Dataset Management Tools
    # ============================================================================

    @tool()
    def register_dataset(self, path: str, name: str) -> str:
        """
        This function registers a dataset with Palimpzest. It takes a path to a file or directory
        and a name for the dataset. The dataset will be registered and made available for use in
        subsequent operations.

        Args:
            path (str): The path to the file or directory to register as a dataset.
            name (str): The name to give to the registered dataset. If not explicitly set, the name of the file or directory will be used.

        Returns:
            str: A message indicating the result of the registration process.
        """
        step = ("register_dataset", {"path": path, "name": name})
        self.plan.append(step)
        
        # Generate code using register_dataset.py template
        code = self._render_template("register_dataset", {"path": path, "name": name})
        self.generated_code.append(code)
        return f"Dataset '{name}' registration recorded in plan."

    @tool()
    def unregister_dataset(self, dataset_name: str) -> str:
        """
        This function unregisters a dataset with Palimpzest. It takes a dataset name and unregisters the dataset. The dataset will be unregistered and made
        unavailable for use in subsequent operations.

        Args:
            dataset_name (str): The name of the dataset to unregister.

        Returns:
            str: A message indicating the result of the unregistration process.
        """
        step = ("unregister_dataset", {"dataset_name": dataset_name})
        self.plan.append(step)
        
        # Generate code using unregister_dataset.py template
        code = self._render_template("unregister_dataset", {"dataset_name": dataset_name})
        self.generated_code.append(code)
        return f"Dataset '{dataset_name}' unregistration recorded in plan."

    @tool()
    def list_datasets(self) -> str:
        """
        This function lists all available datasets in the system. You should use these results to nicely format the output for the user.

        Returns:
            str: A table of the datasets in the system.
        """
        step = ("list_datasets", {})
        self.plan.append(step)
        
        # Generate code using list_datasets.py template
        code = self._render_template("list_datasets", {})
        self.generated_code.append(code)
        return "List datasets operation recorded in plan."

    @tool()
    def retrieve_dataset(self, dataset_path: str) -> str:
        """
        This function lists the available items within a given dataset path. The function prints which records are available
        for the user to use in the given dataset.

        Args:
            dataset_path (str): The path to the dataset to retrieve.

        Returns:
            str: a list of the record identifiers (e.g., filenames, keys, etc...) available to the user in the given dataset.
        """
        step = ("retrieve_dataset", {"dataset_path": dataset_path})
        self.plan.append(step)
        
        # Generate code using retrieve_dataset.py template
        code = self._render_template("retrieve_dataset", {"dataset_path": dataset_path})
        self.generated_code.append(code)
        return f"Dataset retrieval for '{dataset_path}' recorded in plan."

    @tool()
    def set_input_dataset(self, dataset: str, input_schema: str) -> str:
        """
        This function sets the input dataset for the agent to work with when using Palimpzest (pz).
        The input source, also known as the source dataset, or the input dataset, is any dataset that the user will run any workload on.
        If the user does not specify the input schema to use to load the source, you will need to pick the most appropriate schema for the request.
        To do so, you must use one of the existing schemas that have been created within palimpzest.
        This function should be used at the beginning of any workflow to set the input dataset for the agent to work with when using Palimpzest (pz).

        Args:
            dataset (str): The name of the dataset to set as the input source.
            input_schema (str): The schema to use to load the input source. If not specified, a schema will be generated based on the input source.
        Returns:
            str: returns the input source dataset as a palimpzest dataset called `dataset`.
        """
        step = ("set_input_dataset", {"dataset": dataset, "input_schema": input_schema})
        self.plan.append(step)
        
        # Generate code using set_input_dataset.py template
        code = self._render_template("set_input_dataset", {"dataset": dataset, "input_schema": input_schema})
        self.generated_code.append(code)
        return f"Input dataset '{dataset}' with schema '{input_schema}' set in plan."

    # ============================================================================
    # Schema Management Tools
    # ============================================================================

    @tool()
    def generate_extraction_schema(
        self,
        schema_name: str,
        schema_description: str,
        field_names: list,
        field_descriptions: list,
        field_required: list,
    ) -> str:
        """
        This function takes in a set of fields to be used to generate an extraction schema. This should be used when the user is interested in generating a new type of extraction schema. For example, let's say the user is interested
        in extracting parameter values from a set of scientific papers. The user can define the fields of the schema to be used for the extraction.
        In this case the schema name might be `Parameter` and the field information is passed in via three lists which must
        be constructed in proper order. For example, for parameter extractions the fields may be `name`, `value`, `unit`, `source`, etc.
        You should provide a description for each field as well as whether the field is required or not in the same order as you provide the field names. Field names should not have spaces or special characters, but can have underscores.

        Args:
            schema_name (str): the name of the schema to add
            schema_description (str): a description of the schema
            field_names (list): a list of field names
            field_descriptions (list): a list of field descriptions
            field_required (list): a list of whether the field is required or not

        Returns:
            str: the name of the new schema that was created
        """
        step = ("generate_extraction_schema", {
            "schema_name": schema_name,
            "schema_description": schema_description,
            "field_names": field_names,
            "field_descriptions": field_descriptions,
            "field_required": field_required
        })
        self.plan.append(step)
        
        # Track the schema in existing_schemas for later reference
        self.existing_schemas[schema_name] = {
            "description": schema_description,
            "fields": list(zip(field_names, field_descriptions, field_required))
        }
        
        # Generate code using create_schema.py template
        params = {
            "schema_name": schema_name,
            "schema_description": schema_description,
            "field_names": field_names,
            "field_descriptions": field_descriptions,
            "field_required": field_required
        }
        code = self._render_template("create_schema", params)
        self.generated_code.append(code)
        return f"Schema '{schema_name}' creation recorded in plan."

    @tool()
    def pick_schema(self, schema_name: str) -> str:
        """
        This function picks a given schema class given its name.
        If the schema is not found, the function returns None. Provide a message to the user in this case, and proceed with creating a new schema with the given name.
        Args:
            schema_name (str): The name of the schema class to fetch.
        Returns:
            str: returns the schema class object that corresponds to the given schema name.
        """
        step = ("pick_schema", {"schema_name": schema_name})
        self.plan.append(step)
        
        # Generate code using pick_schema.py template
        code = self._render_template("pick_schema", {"schema_name": schema_name})
        self.generated_code.append(code)
        return f"Schema '{schema_name}' selection recorded in plan."

    @tool()
    def list_schemas(self) -> str:
        """
        This function lists all available schemas in the system. You should use these results to nicely format the output for the user.

        Returns:
            str: A table of the schemas in the system.
        """
        step = ("list_schemas", {})
        self.plan.append(step)
        
        # Generate code using list_schemas.py template
        code = self._render_template("list_schemas", {})
        self.generated_code.append(code)
        return "List schemas operation recorded in plan."

    # ============================================================================
    # Data Processing Tools
    # ============================================================================

    @tool()
    def filter_data(self, input_dataset: str, filter_expression: str) -> str:
        """
        This function generates a filtered dataset given an input dataset and a filtering expression. The filter expression is a string that describes a condition that has to be satisfied for each of the data item in the dataset. For example if a user is interested in a dataset of scientific papers and wants to only keep papers that are published in the year 2022, the filter expression might be "The papers is published in 2022".

        Args:
            input_dataset (str): The input Dataset to use for the filtering.
            filter_expression (str): A string that describes a condition in natural language that can be used to filter out data points within a collection.

        Returns:
            str: returns a new dataset corresponding to the filtered input dataset.
        """
        step = ("filter_data", {
            "input_dataset": input_dataset,
            "filter_expression": filter_expression
        })
        self.plan.append(step)
        
        # Generate code using filter_data.py template
        code = self._render_template("filter_data", {"input_dataset": input_dataset, "filter_expression": filter_expression})
        self.generated_code.append(code)
        return f"Data filtering with condition '{filter_expression}' recorded in plan."

    @tool()
    def convert_dataset(self, input_dataset: str, schema: str, cardinality: str) -> str:
        """
        This function converts an input dataset to a new output dataset with a different schema.
        The function can be used for example to extract information from a collection of input documents.
        The function is typically needed to apply a generated schema to an existing dataset, before executing a workload.
        If multiple objects of the new schema can be extracted from a single object of the input dataset, the cardinality should be set to "one_to_many". If only one object of the new schema can be extracted from a single object of the input dataset, the cardinality should be set to "one_to_one".
        For example if a user wants to extract the titles for a dataset of scientific papers, the schema might be a TitleSchema.

        Args:
            input_dataset (str): An existing object of type dataset to use for conversion.
            schema (str): A schema with a name and a set of attributes that describes the object of the new converted dataset.
            cardinality (str): The cardinality of the conversion. Either "one_to_one" or "one_to_many".

        Returns:
            str: returns a new dataset corresponding to the converted input dataset.
        """
        step = ("convert_dataset", {
            "input_dataset": input_dataset,
            "schema": schema,
            "cardinality": cardinality
        })
        self.plan.append(step)
        
        # Generate code using convert_dataset.py template
        code = self._render_template("convert_dataset", {"input_dataset": input_dataset, "schema": schema, "cardinality": cardinality})
        self.generated_code.append(code)
        return f"Dataset conversion to schema '{schema}' with cardinality '{cardinality}' recorded in plan."

    @tool()
    def override_dataset(self, input_dataset: str) -> str:
        """
        The function is required after a workload has been executed, if the user needs to run a new workload with new converts or filters.
        The effect of this function is to reset the working dataset to the input dataset.
        This function deletes an existing dataset and sets the working dataset to a new input dataset.

        Args:
            input_dataset (str): An existing object of type dataset to use for conversion.

        Returns:
            str: returns a new dataset corresponding to the converted input dataset.
        """
        step = ("override_dataset", {"input_dataset": input_dataset})
        self.plan.append(step)
        
        # Generate code using override_dataset.py template
        code = self._render_template("override_dataset", {"input_dataset": input_dataset})
        self.generated_code.append(code)
        return f"Dataset override to '{input_dataset}' recorded in plan."

    # ============================================================================
    # Execution Tools
    # ============================================================================

    @tool()
    def execute_workload(
        self,
        output_dataset: str,
        policy_method: str,
        allow_code_synth: str,
        allow_token_reduction: str,
    ) -> str:
        """
        This function executes a workload starting from a given output dataset.
        If necessary, before executing the workload, any input dataset must be processed to match the schema of the output dataset.
        Processing an input dataset can be composed of several operations such as filtering or converting from one schema to the next. For example, if I want to extract the title of papers with at least 5 authors, I can first filter the papers to only include those with more than 5 authors and then convert the scientific papers to a schema that only includes the title information.
        In this case, the input dataset is the scientific papers dataset and the output dataset would be obtained first with filtering and then with converting the dataset to a schema that only includes the title information.

        The policy method chosen is either to minimize the extraction cost or to maximize the quality
        of the extraction.
        The allow_code_synth and allow_token_reduction are flags that allow the system to use optimization strategies, repsectively to run on synthesized code and to reduce the tokens used when calling LLMs.
        This returns the extractions as a Pandas DataFrame.

        Args:
            output_dataset (str): An output dataset on which to run the workload.
            policy_method (str): Either "min_cost" or "max_quality". Defaults to "max_quality".
            allow_code_synth (str): Whether to allow code synthesis or not. Defaults to "False".
            allow_token_reduction (str): Whether to allow token reduction or not. Defaults to "False".

        Returns:
            str: returns the extracted references as a Pandas DataFrame called `results_df`.

        You should show the user the result after this function runs.
        """
        step = ("execute_workload", {
            "output_dataset": output_dataset,
            "policy_method": policy_method,
            "allow_code_synth": allow_code_synth,
            "allow_token_reduction": allow_token_reduction
        })
        self.plan.append(step)
        
        # Generate code using execute_workload.py template
        params = {
            "output_dataset": output_dataset,
            "policy_method": policy_method,
            "allow_code_synth": allow_code_synth,
            "allow_token_reduction": allow_token_reduction
        }
        code = self._render_template("execute_workload", params)
        self.generated_code.append(code)
        
        return f"Workload execution with policy '{policy_method}' recorded in plan."

    @tool()
    def print_statistics(self) -> str:
        """
        This function shows the runtime statistics after executing a workload.
        The function can be used to check the total cost and total runtime of the pipeline that was run.
        If necessary, before showing the statistics, the workload has to be executed.

        Returns:
            str: returns the statistics objects as it is produced by the execute workflow tool.

        You should show the user the result after this function runs.
        """
        step = ("print_statistics", {})
        self.plan.append(step)
        
        # Generate code using print_statistics.py template
        code = self._render_template("print_statistics", {})
        self.generated_code.append(code)
        return "Print statistics operation recorded in plan."

    # ============================================================================
    # Plan Management Tools
    # ============================================================================

    @tool()
    def display_plan(self) -> str:
        """
        Used when you want to terminate the reasoning process and return the final execution plan.
        This is typically called as the last step in a planning sequence to output the accumulated plan
        in a serialized format. It stops further execution and hands back the complete deferred plan.

        Returns:
            str: A serialized representation of the recorded execution plan and generated code.
        """
        if not self.plan:
            return "No plan generated."
        
        plan_summary = {
            "plan_steps": self.plan,
            "generated_code": "\n".join(self.generated_code),
            "step_count": len(self.plan),
            "schemas_created": list(self.existing_schemas.keys())
        }
        
        return json.dumps(plan_summary, ensure_ascii=False, indent=2)

    # ============================================================================
    # Control Methods
    # ============================================================================

    def clear_plan(self) -> str:
        """Clear all recorded steps in the plan."""
        self.plan.clear()
        self.generated_code.clear()
        self.existing_schemas.clear()
        return "Plan cleared."

    def show_plan_summary(self) -> str:
        """Display a human-readable summary of the current plan."""
        if not self.plan:
            return "Plan is empty."
        
        summary = ["Plan Summary:"]
        for idx, (tool_name, params) in enumerate(self.plan, 1):
            summary.append(f"{idx}. {tool_name}: {params}")
        
        return "\n".join(summary)

    def run(self, query: str) -> Dict[str, Any]:
        """
        Main entry point: generate plan from natural language query.
        
        Args:
            query (str): Natural language query describing the desired workflow
            
        Returns:
            Dict containing the generated plan and code
        """
        self.clear_plan()
        
        # Use ReAct agent to process query
        response = self.agent.react(query)
        
        # Return plan results
        return {
            "query": query,
            "plan_steps": self.plan,
            "generated_code": "\n".join(self.generated_code),
            "step_count": len(self.plan),
            "schemas_created": list(self.existing_schemas.keys()),
            "agent_response": response
        }

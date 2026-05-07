from Agents.Dice_Agent import Dice_Agent
from Agents.Slice_Agent import Slice_Agent
from Agents.Drill_Down_Agent import Drill_Down_Agent
from Agents.Roll_Up_Agent import Roll_Up_Agent
from Agents.Exection_Agent import Execution_Agent
from typing import Annotated, Any, Dict, List, Optional, Sequence, TypedDict, Union
import pandas as pd
from Utils.send_logs import debug_log
from Utils.jsonfy_result import jsonfy_llm_response, llm_retry
from Agents.Components.Operaters import sem_topk, num_topk
from Agents.Components.SemaFlex_Cube import SemaFlex_Cube, OLAP_Axis
from concurrent.futures import ThreadPoolExecutor
from time import time
import os
from Agents.Components.Prompt import get_filtering_prompt, get_topk_prompt, determine_pass, choose_view, make_stepwise_ReAct_prompt, get_decompose_prompt
import json
import threading


_QUERY_PLAN_SAVE_LOCK = threading.Lock()


class FilteringPhase:
    def __init__(self, llm, cube: SemaFlex_Cube, agent_map: dict, execution_agent):
        self.llm = llm
        self.cube = cube
        self.agent_map = agent_map
        self.execution_agent = execution_agent

    def merge_slice_operations(self, plan):
        logic = plan.get("logic", [])
        if not logic or logic[0] != "AND":
            return plan

        operations = plan.get("operations", [])
        slice_ops = [op for op in operations if op.get("agent") == "slice"]
        other_ops = [op for op in operations if op.get("agent") != "slice"]

        if len(slice_ops) <= 1:
            return plan

        merged_filter = ", and ".join(
            op["filtering_condition"]
            for op in slice_ops
            if op.get("filtering_condition")
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

    def plan_generate_filter(self, history_query, now_query):
        col_str = self.cube.get_dimensions()
        if "question_id" in col_str:
            col_str.remove("question_id")

        prompt = get_filtering_prompt(col_str, history_query, now_query)
        result_str = self.llm.predict(prompt)
        return jsonfy_llm_response(result_str)

    def execute_plan_filter(self, plan, ids):
        new_operations = []
        plan = self.merge_slice_operations(plan)

        for op in plan["operations"]:
            agent = op["agent"]
            action = op["filtering_condition"]
            field = op["field"]

            agent_now = self.agent_map[agent]
            query = {"action": action, "field": field}
            sub_plan = agent_now.run(query, self.cube, ids)

            new_operations.append({
                "id": op["id"],
                "plan": sub_plan
            })

        return {
            "operations": new_operations,
            "logic": plan["logic"]
        }

    def run(self, filter_query):
        show_plan = []

        if filter_query == "":
            state = "Equal"
            nodes = [0]
            olap_id_list = self.cube.dag.get_node(0).olap_ids
        else:
            nodes, olap_id_list, state = self.cube.locate_from_dag(filter_query)

        if state != "Equal":
            if not nodes:
                nodes = [0]

            history_query = self.cube.dag.get_node(nodes[0]).query
            filt_plan = self.cube.dag.get_node(nodes[0]).plan.copy()

            plan_filter = self.plan_generate_filter(history_query, filter_query)
            operations_filter = self.execute_plan_filter(plan_filter, olap_id_list)

            result_df, f_plan = self.execution_agent.run_filter(
                operations_filter, self.cube, olap_id_list
            )

            filt_plan.extend(f_plan)
            olap_id_list = result_df["OLAP_ID"].astype(int).tolist()
            self.cube.update_dag(nodes, olap_id_list, filter_query, filt_plan)
        else:
            result_df = self.cube.read_raw(olap_id_list)
            filt_plan = self.cube.dag.get_node(nodes[0]).plan.copy()

        show_plan.extend(filt_plan)
        return result_df, show_plan, olap_id_list

class EnrichmentPhase:
    def __init__(self, llm, cube: SemaFlex_Cube, rollup_agent, drilldown_agent):
        self.llm = llm
        self.cube = cube
        self.rollup_agent = rollup_agent
        self.drilldown_agent = drilldown_agent

    def run_analysis_plan_stage(self, analysis_query, axis_snapshot):
        if analysis_query == "":
            return None
        return self.run_roll_up_and_drill_down_on_axis(analysis_query, axis_snapshot)

    def run_roll_up_and_drill_down_on_axis(self, query_view, axis_snapshot):
        history = ""
        tmp_axis = axis_snapshot

        pass_result = {"sufficient": False}
        count_tag = 0
        choose_ret = None

        while pass_result["sufficient"] is False:
            count_tag += 1
            if count_tag >= 3:
                break

            history, tmp_axis, pass_result, choose_ret = self.ReAct_Reflect_singlepass(
                query_view, history, tmp_axis
            )

            view_pairs = choose_ret.get("view_pairs", [])
            pair_str = [f"{p['dimension']}-{p['granularity']}" for p in view_pairs]

            if pass_result["sufficient"] is False:
                history += (
                    f"After check your result, user need you continue. "
                    f"Now the user get the table from dimension-granularity pairs: "
                    f"{pair_str}. "
                    f"In this table: {pass_result['reason']}"
                )

        if choose_ret is None:
            return [{"from": "END", "params": {}}]

        view_pairs = choose_ret.get("view_pairs", [])
        if not view_pairs:
            raise ValueError("choose_ret must contain a non-empty view_pairs")

        is_raw = self._is_raw_view(view_pairs)

        if is_raw:
            primary_dim = view_pairs[0]["dimension"]
            plan = axis_snapshot.get_dimension(primary_dim).plan.copy()
        else:
            grouped_pair = self._get_first_grouped_pair(view_pairs)
            plan = axis_snapshot.get_granularity(
                grouped_pair["dimension"],
                grouped_pair["granularity"]
            ).plan.copy()

        plan.append({"from": "END", "params": choose_ret})
        return plan

    def ReAct_Reflect_singlepass(self, query_view, history, tmp_axis: OLAP_Axis):
        history = self._run_react_loop(query_view, history, tmp_axis)

        choose_ret, desc = self._choose_valid_view_and_desc(query_view, history, tmp_axis)

        pass_prompt = determine_pass(query_view, desc)
        pass_result = self.llm.predict(pass_prompt)
        pass_result = jsonfy_llm_response(pass_result)

        print(pass_result)
        return history, tmp_axis, pass_result, choose_ret

    def execute_enrichment(self, plan, ids):
        end_params = None

        for p in plan:
            print(p)
            source = p.get("from")
            params = p.get("params", {})

            if source == "drill_down":
                self.drilldown_agent.run(params, self.cube, ids)

            elif source == "roll_up":
                self.rollup_agent.run(params, self.cube, ids)

            else:
                end_params = params

        if end_params is None:
            return self.cube.read_raw(ids), []

        show_table = self._build_final_view_table(end_params, ids)
        now_plan = self._build_final_operator_plan(end_params)
        return show_table, now_plan

    def _build_final_view_table(self, params, ids):
        view_pairs = params.get("view_pairs", [])
        aggregation_dimensions = params.get("aggregation_dimensions", [])

        if not view_pairs:
            return self.cube.read_raw(ids)

        table = self.cube.read_raw(ids, ["OLAP_ID"])

        for pair in view_pairs:
            pair_table = self._read_view_pair_table(pair, ids)
            table = self._merge_new_columns(table, pair_table)

        for dimension in aggregation_dimensions:
            if dimension == "count" or dimension in table.columns:
                continue

            try:
                dim_table = self.cube.read_raw(ids, [dimension])
            except Exception:
                continue

            table = self._merge_new_columns(table, dim_table)

        group_cols = self._resolve_group_columns(view_pairs, table)
        if not group_cols:
            return table

        if not aggregation_dimensions:
            return table.drop(columns=["OLAP_ID"], errors="ignore").drop_duplicates()

        result = table[group_cols].drop_duplicates().reset_index(drop=True)
        grouped = table.groupby(group_cols, dropna=False)

        if "count" in aggregation_dimensions:
            count_col = self._resolve_count_column_name(view_pairs, group_cols)
            count_df = grouped.size().reset_index(name=count_col)
            result = result.merge(count_df, on=group_cols, how="left")

        for dimension in aggregation_dimensions:
            if dimension == "count" or dimension not in table.columns:
                continue

            reduced_df = (
                grouped[dimension]
                .apply(self._summarize_group_values)
                .reset_index(name=dimension)
            )
            result = result.merge(reduced_df, on=group_cols, how="left")

        return result

    def _build_final_operator_plan(self, params):
        view_pairs = params.get("view_pairs", [])
        aggregation_dimensions = params.get("aggregation_dimensions", [])

        operator_plan = []

        for pair in view_pairs:
            operator_plan.extend(self._get_view_pair_dependency_plan(pair))

        for dimension in aggregation_dimensions:
            if dimension == "count":
                continue
            operator_plan.extend(self._get_dimension_dependency_plan(dimension))

        operator_plan = self._dedupe_operator_plan(operator_plan)
        group_cols = self._resolve_view_pair_columns(view_pairs)

        for dimension in aggregation_dimensions:
            if dimension == "count":
                count_col = self._resolve_count_column_name(view_pairs, group_cols)
                operator_plan.append({
                    "operator_name": "count",
                    "parameters": {
                        "group_by": group_cols,
                        "column": []
                    }
                })
            else:
                operator_plan.append({
                    "operator_name": "sem_reduce",
                    "parameters": {
                        "columns": [
                            dimension
                        ],
                        "group_by": group_cols
                    }
                })

        return operator_plan

    def _read_view_pair_table(self, pair, ids):
        dim = pair["dimension"]
        gra = pair["granularity"]

        if dim == gra:
            return self.cube.read_raw(ids, [dim])

        return self.cube.read_granularity(ids, dim, gra)

    def _merge_new_columns(self, base, incoming):
        new_cols = [
            c for c in incoming.columns
            if c != "OLAP_ID" and c not in base.columns
        ]

        if not new_cols:
            return base

        return base.merge(
            incoming[["OLAP_ID"] + new_cols],
            on="OLAP_ID",
            how="left"
        )

    def _resolve_group_columns(self, view_pairs, table):
        return [
            col for col in self._resolve_view_pair_columns(view_pairs)
            if col in table.columns
        ]

    def _resolve_view_pair_columns(self, view_pairs):
        columns = []

        for pair in view_pairs:
            dim = pair["dimension"]
            gra = pair["granularity"]

            if dim == gra:
                pair_cols = [dim]
            else:
                try:
                    pair_cols = list(
                        self.cube.axis.get_granularity(dim, gra).values
                    )
                except Exception:
                    pair_cols = [gra]

            for col in pair_cols:
                if col != "OLAP_ID" and col not in columns:
                    columns.append(col)

        return columns

    def _resolve_count_column_name(self, view_pairs, group_cols=None):
        if group_cols is None:
            group_cols = self._resolve_view_pair_columns(view_pairs)

        if len(group_cols) == 1:
            return f"{group_cols[0]}_count"

        if len(view_pairs) == 1:
            pair = view_pairs[0]
            return f"{pair['granularity']}_count"

        return "count"

    def _get_view_pair_dependency_plan(self, pair):
        dim = pair["dimension"]
        gra = pair["granularity"]

        try:
            if dim == gra:
                return self.cube.axis.get_dimension(dim).plan.copy()
            return self.cube.axis.get_granularity(dim, gra).plan.copy()
        except Exception:
            return []

    def _get_dimension_dependency_plan(self, dimension):
        try:
            return self.cube.axis.get_dimension(dimension).plan.copy()
        except Exception:
            return []

    def _dedupe_operator_plan(self, operator_plan):
        deduped = []
        seen = set()

        for step in operator_plan:
            key = json.dumps(step, ensure_ascii=False, sort_keys=True)
            if key in seen:
                continue
            seen.add(key)
            deduped.append(step)

        return deduped

    def _summarize_group_values(self, series):
        values = []

        for value in series.dropna():
            if value not in values:
                values.append(value)

        return "; ".join(str(value) for value in values)

    def _run_react_loop(self, query_view, history, tmp_axis):
        end_tag = 0

        while end_tag < 10:
            end_tag += 1

            prompt = make_stepwise_ReAct_prompt(query_view, history)
            result = self.llm.predict(prompt)
            ret_json = jsonfy_llm_response(result, "failed")

            if isinstance(ret_json, str) or "thought" not in ret_json:
                result = llm_retry(result, self.llm)
                ret_json = jsonfy_llm_response(result)

            print(ret_json)

            if not ret_json or ret_json.get("action") is None or ret_json.get("action") == "null":
                print(json.dumps(ret_json, indent=4))
                history += (
                    f"thought: {ret_json['thought']}\n"
                    f"action: END"
                )
                break

            action = ret_json["action"]
            thought = ret_json["thought"]
            observation = self._execute_fake_action(action, thought, tmp_axis)

            history_now = {
                "thought": thought,
                "action": json.dumps(action, ensure_ascii=False),
                "observation": observation
            }

            history += (
                f"thought: {thought}\n"
                f"action: {json.dumps(action, ensure_ascii=False)}\n"
                f"observation: {observation}\n"
            )

            debug_log({
                "type": "log",
                "message": f"OLAP Agent: inner iteration record\n\n ```json\n{json.dumps(history_now, indent=4, ensure_ascii=False)}\n```"
            })

        return history

    def _execute_fake_action(self, action, thought, tmp_axis):
        print(action)
        action_type = action["type"]
        params = action.get("params", {})
        params["thought"] = thought

        if action_type == "drill_down":
            return self.drilldown_agent.fake_run(params, tmp_axis)

        elif action_type == "roll_up":
            return self.rollup_agent.fake_run(params, tmp_axis)

        elif action_type == "get_dimension":
            observation = tmp_axis.get_dimensions()
            if "question_id" in observation:
                observation.remove("question_id")
            return observation

        elif action_type == "get_granularity":
            dimension = params["dimension"]
            try:
                return tmp_axis.get_granularities(dimension)
            except Exception:
                return f"Dimension '{dimension}' does not exist"

        return None

    def _choose_valid_view_and_desc(self, query_view, history, tmp_axis):
        postfix = ""
        choose_ret = None
        desc = ""

        for _ in range(3):
            choose_prompt = choose_view(query_view, history + "\n\n" + postfix)
            choose_result = self.llm.predict(choose_prompt)
            choose_ret = jsonfy_llm_response(choose_result)

            print(choose_result)
            print("^" * 50)

            try:
                view_pairs = choose_ret["view_pairs"]
                aggregation_dimensions = choose_ret.get("aggregation_dimensions", [])

                self._validate_view_pairs(view_pairs, aggregation_dimensions, tmp_axis)
                desc = self._build_desc(view_pairs, aggregation_dimensions, tmp_axis)
                return choose_ret, desc

            except Exception as e:
                postfix += (
                    f"\nYou just returned {choose_result}, but it is invalid or does not exist. "
                    f"Please retry and use valid existing dimension-granularity pairs.\n"
                    f"Requirements:\n"
                    f"- view_pairs must be a non-empty list\n"
                    f"- each item in view_pairs must have valid existing 'dimension' and 'granularity'\n"
                    f"- aggregation_dimensions must be a list\n"
                    f"- dimension and granularity may be identical\n"
                    f"Error: {str(e)}"
                )

        raise ValueError("Failed to choose a valid view after 3 retries")

    def _validate_view_pairs(self, view_pairs, aggregation_dimensions, tmp_axis):
        if not isinstance(view_pairs, list) or len(view_pairs) == 0:
            raise ValueError("view_pairs must be a non-empty list")

        if not isinstance(aggregation_dimensions, list):
            raise ValueError("aggregation_dimensions must be a list")

        for pair in view_pairs:
            dim = pair["dimension"]
            gra = pair["granularity"]
            _ = tmp_axis.get_granularity(dim, gra)

    def _build_desc(self, view_pairs, aggregation_dimensions, tmp_axis):
        is_raw = self._is_raw_view(view_pairs)
        aggregation_desc = self._build_aggregation_desc(
            aggregation_dimensions,
            view_pairs
        )
        count_col = self._resolve_count_column_name(view_pairs)

        pair_str = [
            f"({pair['dimension']}, {pair['granularity']})"
            for pair in view_pairs
        ]

        if is_raw:
            cols_list = tmp_axis.get_dimensions()
            while "OLAP_ID" in cols_list:
                cols_list.remove("OLAP_ID")

            for pair in view_pairs:
                dim = pair["dimension"]
                gra = pair["granularity"]
                gran_obj = tmp_axis.get_granularity(dim, gra)

                append_col_list = list(gran_obj.values) if gran_obj.values is not None else []
                while "OLAP_ID" in append_col_list:
                    append_col_list.remove("OLAP_ID")

                for item in append_col_list:
                    if item not in cols_list:
                        cols_list.append(item)

            if "count" in aggregation_dimensions and count_col not in cols_list:
                cols_list.append(count_col)

            if len(aggregation_dimensions) == 0:
                return (
                    f"The table contains the following columns: {cols_list}. "
                    f"The final result uses only base dimension-granularity pairs: {pair_str}. "
                    f"These columns represent raw values without higher-level grouping."
                )
            else:
                return (
                    f"The table contains the following columns: {cols_list}. "
                    f"The final result uses only base dimension-granularity pairs: {pair_str}. "
                    f"These columns represent raw values without higher-level grouping. "
                    f"{aggregation_desc}"
                )

        cols_list = []
        for pair in view_pairs:
            dim = pair["dimension"]
            gra = pair["granularity"]
            gran_obj = tmp_axis.get_granularity(dim, gra)

            cur_cols = list(gran_obj.values) if gran_obj.values is not None else []
            while "OLAP_ID" in cur_cols:
                cur_cols.remove("OLAP_ID")

            for item in cur_cols:
                if item not in cols_list:
                    cols_list.append(item)

        if "count" in aggregation_dimensions and count_col not in cols_list:
            cols_list.append(count_col)

        if len(aggregation_dimensions) == 0:
            return (
                f"The data contains the following columns: {cols_list}. "
                f"The final table is grouped according to the following dimension-granularity pairs: {pair_str}. "
                f"No additional aggregation dimension is required."
            )
        else:
            return (
                f"The data contains the following columns: {cols_list}. "
                f"The final table is grouped according to the following dimension-granularity pairs: {pair_str}. "
                f"{aggregation_desc}"
            )

    def _build_aggregation_desc(self, aggregation_dimensions, view_pairs):
        count_required = "count" in aggregation_dimensions
        count_col = self._resolve_count_column_name(view_pairs)
        non_count_dimensions = [
            dim for dim in aggregation_dimensions
            if dim != "count"
        ]

        desc_parts = []

        if non_count_dimensions:
            desc_parts.append(
                f"After grouping, the following aggregation targets are required: {non_count_dimensions}."
            )

        if count_required:
            desc_parts.append(
                f"The final table must also include a frequency/count column named '{count_col}', representing the number of records or values contained in each group."
            )

        return " ".join(desc_parts)

    def _resolve_end_dimension_granularity(self, params):
        if "dimension" in params and "granularity" in params:
            return params["dimension"], params["granularity"]

        view_pairs = params.get("view_pairs", [])
        if not view_pairs:
            raise ValueError("END params must contain dimension/granularity or non-empty view_pairs")

        if self._is_raw_view(view_pairs):
            primary_dim = view_pairs[0]["dimension"]
            return primary_dim, primary_dim

        grouped_pair = self._get_first_grouped_pair(view_pairs)
        return grouped_pair["dimension"], grouped_pair["granularity"]

    def _is_raw_view(self, view_pairs):
        return all(
            pair["dimension"] == pair["granularity"]
            for pair in view_pairs
        )

    def _get_first_grouped_pair(self, view_pairs):
        return next(
            pair for pair in view_pairs
            if pair["dimension"] != pair["granularity"]
        )


class TopKPhase:
    def __init__(self, llm):
        self.llm = llm

    def run(self, show_table, analysis_query, show_plan):

        if show_table is None or show_table.empty:
            return show_table, show_plan

        if analysis_query == "":
            return show_table, show_plan

        table_str = self._build_table_description(show_table)
        topk_params = self._infer_topk_params(table_str, analysis_query)

        if not isinstance(topk_params, dict):
            return show_table, show_plan

        return self._apply_topk(show_table, topk_params, show_plan)

    def _build_table_description(self, show_table):
        descriptions = []

        for col in show_table.columns:
            if col == "OLAP_ID":
                continue

            sample_vals = show_table[col].dropna().head(2).tolist()
            total_chars = sum(len(s) for s in sample_vals if isinstance(s, str))

            if total_chars > 100:
                descriptions.append(
                    f"Field: {col} | Samples: content too long to display"
                )
            else:
                descriptions.append(
                    f"Field: {col} | Samples: [{', '.join(map(str, sample_vals))}]"
                )

        table_str = "\n".join(descriptions)
        print(table_str)
        return table_str

    def _infer_topk_params(self, table_str, analysis_query):
        prompt_topk = get_topk_prompt(table_str, analysis_query)
        result = self.llm.predict(prompt_topk)
        print(result)

        try:
            return jsonfy_llm_response(result)
        except Exception:
            return {}

    def _apply_topk(self, show_table, topk_params, show_plan):
        topk_type = topk_params.get("topk_type")

        if topk_type not in ["num", "sem"]:
            return show_table, show_plan

        if topk_params.get("top_k") is None:
            topk_params["top_k"] = 1

        if topk_type == "sem":
            return self._apply_semantic_topk(show_table, topk_params, show_plan)

        return self._apply_numeric_topk(show_table, topk_params, show_plan)

    def _apply_semantic_topk(self, show_table, topk_params, show_plan):
        query = topk_params["sort_basis"] + f"\n Order: {topk_params['sort_order']}"

        show_table = sem_topk(
            llm=self.llm,
            df=show_table,
            columns=[topk_params["sort_field"]],
            query=query,
            k=topk_params["top_k"]
        )

        show_plan.append({
            "operator_name": "sem_topk",
            "parameters": {
                "column": topk_params["sort_field"],
                "query": query,
                "k": topk_params["top_k"]
            }
        })

        return show_table, show_plan

    def is_column_numeric(self, df, col_name):
        series = df[col_name].dropna()  #
        numeric_series = pd.to_numeric(series, errors='coerce')
        return not numeric_series.isna().any()

    def _apply_numeric_topk(self, show_table, topk_params, show_plan):
        show_plan.append({
            "operator_name": "num_topk",
            "parameters": {
                "column": topk_params["sort_field"],
                "k": topk_params["top_k"],
                "order": topk_params["sort_order"]
            }
        })

        if self.is_column_numeric(show_table, topk_params["sort_field"]):
            show_table = num_topk(
                df=show_table,
                column=topk_params["sort_field"],
                k=topk_params["top_k"],
                order=topk_params["sort_order"]
            )

        return show_table, show_plan


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
        self.filtering_phase = FilteringPhase(
            llm=self.llm,
            cube=self.Cube,
            agent_map={
                "slice": self.slice_agent,
                "dice": self.dice_agent,
            },
            execution_agent=self.Execution_Agent
        )
        self.enrichment_phase = EnrichmentPhase(
            llm=self.llm,
            cube=self.Cube,
            rollup_agent=self.rollup_agent,
            drilldown_agent=self.drilldown_agent
        )
        self.topk_phase = TopKPhase(self.llm)

    def decompose_query_intent(self, query):
        prompt = get_decompose_prompt(query)

        response = self.llm.predict(prompt)

        try:
            parsed = jsonfy_llm_response(response)
            return parsed.get("filter_query", "").strip(), parsed.get("analysis_query", "").strip()
        except Exception:
            return "", ""

    def show_translation(self, show_table):
        if show_table is None or show_table.empty:
            return show_table

        df = show_table.drop(columns=["OLAP_ID"], errors="ignore")

        df = df.drop_duplicates()

        return df

    def save_query_plan(self, query, show_plan, record_message, file_path):
        if record_message is None:
            return

        id = record_message["id"]
        Q_id = record_message["Q_id"]

        write_message = {
            "Query_Self_Contained": query,
            "plan": show_plan
        }

        dir_path = os.path.dirname(file_path)
        if dir_path:
            os.makedirs(dir_path, exist_ok=True)

        with _QUERY_PLAN_SAVE_LOCK:
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

    def run(self, query, record_message=None, file_path="results/example.json"):
        if record_message is None:
            self.record_message = {}
        else:
            self.record_message = record_message
        self.record_message["query"] = query

        filter_query, analysis_query = self.decompose_query_intent(query)
        print(f"filter_query: {filter_query}")
        print(f"analysis_query: {analysis_query}")
        print("=" * 50)

        axis_snapshot = None
        if analysis_query != "":
            axis_snapshot = self.Cube.axis.copy()

        with ThreadPoolExecutor(max_workers=2) as executor:
            future_filter = executor.submit(self.filtering_phase.run, filter_query)
            future_analysis = None
            if analysis_query != "":
                future_analysis = executor.submit(
                    self.enrichment_phase.run_analysis_plan_stage,
                    analysis_query,
                    axis_snapshot
                )

            result_df, show_plan, olap_id_list = future_filter.result()
            enrichment_plan = future_analysis.result() if future_analysis else None

        if analysis_query != "":
            show_table, enrich_plan = self.enrichment_phase.execute_enrichment(
                enrichment_plan,
                olap_id_list
            )
            show_plan.extend(enrich_plan)
        else:
            show_table = result_df

        show_table = self.show_translation(show_table)

        show_table, show_plan = self.topk_phase.run(
            show_table=show_table,
            analysis_query=analysis_query,
            show_plan=show_plan
        )
        print(show_table)

        self.save_query_plan(
            query=query,
            show_plan=show_plan,
            record_message=self.record_message,
            file_path=file_path
        )

        return show_table

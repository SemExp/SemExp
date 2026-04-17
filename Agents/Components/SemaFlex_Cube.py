import pandas as pd
from typing import Dict, List, Set, Optional, Tuple
import json
import copy
import random
from typing import Set
from Agents.Components.Operaters import group_by
from Utils.jsonfy_result import jsonfy_llm_response


import json
from pathlib import Path


def save_cube(cube, path: str):
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    (path / "granularities").mkdir(exist_ok=True)

    cube.tables.raw_table.to_parquet(
        path / "raw_table.parquet",
        index=False,
        compression="zstd"
    )

    axis_data = {}
    for dim_name, dim_spec in cube.axis._dimensions.items():
        axis_data[dim_name] = {
            "plan": dim_spec.plan,
            "generate_plan": dim_spec.generate_plan,
            "granularities": {}
        }
        for gra_name, gra_spec in dim_spec.granularities.items():
            axis_data[dim_name]["granularities"][gra_name] = {
                "name": gra_spec.name,
                "plan": gra_spec.plan,
                "generate_plan": gra_spec.generate_plan,
                "values": gra_spec.values,
            }

    with open(path / "axis.json", "w", encoding="utf-8") as f:
        json.dump(axis_data, f, ensure_ascii=False)

    granularity_files = {}
    for dim, gmap in cube.tables.dimensions.items():
        granularity_files[dim] = {}
        for gra, df in gmap.items():
            file_name = f"{dim}__{gra}.parquet"
            df.reset_index().to_parquet(
                path / "granularities" / file_name,
                index=False,
                compression="zstd"
            )
            granularity_files[dim][gra] = file_name

    dag_data = {
        "root": cube.dag.root,
        "edges": cube.dag.edges,
        "nodes": {}
    }

    for node_id, node in cube.dag.nodes.items():
        dag_data["nodes"][str(node_id)] = {
            "id": node.id,
            "query": node.query,
            "olap_ids": sorted(node.olap_ids),   # set -> list
            "plan": node.plan,
        }

    with open(path / "dag.json", "w", encoding="utf-8") as f:
        json.dump(dag_data, f, ensure_ascii=False)

    meta = {
        "format_version": 1,
        "class_name": "SemaFlex_Cube",
        "granularity_files": granularity_files,
    }
    with open(path / "meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False)


def load_cube(path: str, llm):
    path = Path(path)

    with open(path / "meta.json", "r", encoding="utf-8") as f:
        meta = json.load(f)

    with open(path / "axis.json", "r", encoding="utf-8") as f:
        axis_data = json.load(f)

    with open(path / "dag.json", "r", encoding="utf-8") as f:
        dag_data = json.load(f)

    raw_table = pd.read_parquet(path / "raw_table.parquet")

    cube = SemaFlex_Cube(raw_table, llm)

    cube.axis = OLAP_Axis()
    for dim_name, dim_info in axis_data.items():
        cube.axis.update_dimension(
            dimension=dim_name,
            plan=dim_info.get("plan", []),
            generate_plan=dim_info.get("generate_plan", [])
        )

        for gra_name, gra_info in dim_info["granularities"].items():
            cube.axis.update_granularity(
                dimension=dim_name,
                granularity=gra_name,
                plan=gra_info.get("plan", []),
                values=gra_info.get("values", []),
                generate_plan=gra_info.get("generate_plan", [])
            )

    cube.tables.dimensions = {}
    for dim, gmap in meta["granularity_files"].items():
        cube.tables.dimensions[dim] = {}
        for gra, file_name in gmap.items():
            df = pd.read_parquet(path / "granularities" / file_name)
            cube.tables.dimensions[dim][gra] = df.set_index(cube.tables.OLAP_KEY)

    cube.dag = OLAP_DAG(llm)
    cube.dag.root = dag_data["root"]
    cube.dag.edges = {int(k): v for k, v in dag_data["edges"].items()}
    cube.dag.nodes = {}

    for node_id_str, node_info in dag_data["nodes"].items():
        node_id = int(node_id_str)
        cube.dag.nodes[node_id] = OLAPNode(
            node_id=node_info["id"],
            query=node_info["query"],
            olap_ids=set(node_info["olap_ids"]),
            plan=node_info["plan"]
        )

    return cube


def sample_olap_ids(
    all_ids: Set[int],
    k: int,
    seed: Optional[int] = None
) -> Set[int]:
    if seed is not None:
        random.seed(seed)

    if k >= len(all_ids):
        return set(all_ids)

    return set(random.sample(list(all_ids), k))


class Tables:
    OLAP_KEY = "OLAP_ID"

    def __init__(self, raw_table: pd.DataFrame):
        if not isinstance(raw_table, pd.DataFrame):
            raise TypeError("raw_table must be a pandas DataFrame")

        raw = raw_table.copy()

        if self.OLAP_KEY not in raw.columns:
            raw.insert(
                0,
                self.OLAP_KEY,
                pd.RangeIndex(len(raw))
            )

        self.raw_table = raw
        self.dimensions = {}
        for col in raw_table.columns:
            self.dimensions[col] = {}

    def set_dimension(self, dimension: str, table: pd.DataFrame):
        if not isinstance(table, pd.DataFrame):
            raise TypeError("table must be a pandas DataFrame")

        if self.OLAP_KEY not in table.columns:
            raise ValueError(f"table must contain column '{self.OLAP_KEY}'")

        invalid_ids = set(table[self.OLAP_KEY]) - set(self.raw_table[self.OLAP_KEY])
        if invalid_ids:
            raise ValueError(
                f"dimension '{dimension}' contains invalid OLAP_IDs: {invalid_ids}"
            )
        incoming = table.copy().set_index(self.OLAP_KEY)
        if dimension not in self.dimensions:
            self.dimensions[dimension] = {}
            self.raw_table = (
                self.raw_table
                .set_index(self.OLAP_KEY)
                .join(incoming, how="left")
                .reset_index()
            )
            return
        raw = self.raw_table.set_index(self.OLAP_KEY)
        all_columns = raw.columns.union(incoming.columns)
        raw_aligned = raw.reindex(columns=all_columns)
        incoming_aligned = incoming.reindex(columns=all_columns)
        updated = raw_aligned.combine_first(incoming_aligned)
        self.raw_table = updated.reset_index()

    def set_granularity(self, dimension: str, granularity: str, table: pd.DataFrame):
        if dimension not in self.dimensions:
            raise ValueError(f"dimension '{dimension}' does not exist")

        if not isinstance(table, pd.DataFrame):
            raise TypeError("table must be a pandas DataFrame")

        if self.OLAP_KEY not in table.columns:
            raise ValueError(f"table must contain column '{self.OLAP_KEY}'")

        invalid_ids = set(table[self.OLAP_KEY]) - set(self.raw_table[self.OLAP_KEY])
        if invalid_ids:
            raise ValueError(
                f"granularity '{granularity}' contains invalid OLAP_IDs: {invalid_ids}"
            )

        incoming = table.copy()
        incoming = incoming.set_index(self.OLAP_KEY)

        if granularity not in self.dimensions[dimension]:
            self.dimensions[dimension][granularity] = incoming
            return

        existing = self.dimensions[dimension][granularity]
        all_index = existing.index.union(incoming.index)
        all_columns = existing.columns.union(incoming.columns)
        existing_aligned = existing.reindex(
            index=all_index,
            columns=all_columns
        )
        incoming_aligned = incoming.reindex(
            index=all_index,
            columns=all_columns
        )
        updated = existing_aligned.combine_first(incoming_aligned)
        self.dimensions[dimension][granularity] = updated

    def get_raw(
            self,
            olap_ids=None,
            list_col=None
    ) -> pd.DataFrame:
        raw = self.raw_table
        if list_col is None or len(list_col) == 0:
            cols = raw.columns
        else:
            missing_cols = set(list_col) - set(raw.columns)
            if missing_cols:
                raise ValueError(f"raw_table does not contain columns: {missing_cols}")
            cols = [self.OLAP_KEY] + [
                c for c in list_col if c != self.OLAP_KEY
            ]

        if olap_ids is None or (
                isinstance(olap_ids, (list, tuple, set)) and len(olap_ids) == 0
        ):
            return raw.loc[:, cols].copy()

        if not isinstance(olap_ids, (list, tuple, set)):
            olap_ids = [olap_ids]

        missing_ids = set(olap_ids) - set(raw[self.OLAP_KEY])
        if missing_ids:
            raise ValueError(f"raw_table does not contain OLAP_IDs: {missing_ids}")

        return (
            raw
            .set_index(self.OLAP_KEY)
            .loc[list(olap_ids), cols if self.OLAP_KEY not in cols else cols[1:]]
            .reset_index()
        )

    def get_granularity(
            self,
            dimension: str,
            granularity: str,
            olap_ids,
            list_col: List[str]
    ) -> pd.DataFrame:
        if list_col is None:
            cols = None
        else:
            cols = list(dict.fromkeys([granularity] + list_col))
        target_columns = None if cols is None else pd.Index(cols)

        if (
                dimension not in self.dimensions
                or granularity not in self.dimensions[dimension]
        ):
            if olap_ids is None:
                return pd.DataFrame(columns=target_columns)
            if not isinstance(olap_ids, (list, tuple, set)):
                olap_ids = [olap_ids]
            target_index = pd.Index(olap_ids, name=self.OLAP_KEY)
            return (
                pd.DataFrame(index=target_index, columns=target_columns)
                .reset_index()
            )

        sparse = self.dimensions[dimension][granularity]

        if olap_ids is None:
            result = sparse.reindex(columns=target_columns)
            return result.reset_index()

        if not isinstance(olap_ids, (list, tuple, set)):
            olap_ids = [olap_ids]

        target_index = pd.Index(olap_ids, name=self.OLAP_KEY)

        result = sparse.reindex(
            index=target_index,
            columns=target_columns
        )

        return result.reset_index()


PROMPT_GET_NODE = """
You are a query semantics analysis assistant. Please compare the data sets described by the "current query" and the "historical query", and output only the specified keyword indicating their relationship.

Available relationship keywords:
- contain  : The data set of the current query strictly contains that of the historical query.  
  - In terms of filtering: The current query has looser conditions or removes some constraints from the historical query. All results returned by the historical query would also be included in the current query, though the current one may return more.

- subset   : The data set of the current query is strictly contained within that of the historical query.  
  - In terms of filtering: The current query has stricter conditions or adds constraints on top of the historical query. All data in the current query also satisfies the historical query, but not vice versa.

- equal    : The two queries describe exactly the same data set.  
  - In terms of filtering: Both queries use the same fields, conditions, and values—even if worded differently, the logical result set is the same.

- intersect: The two data sets definitely have overlap, but neither contains the other.  
  - In terms of filtering: There is partial overlap in fields or values, but the overall conditions don't form a containment relationship. The queries are known to return some common data.

- no_relation: The two data sets definitely do not overlap, or it's impossible to determine whether there is any intersection.  
  - In terms of filtering: The queries contain mutually exclusive conditions (e.g., non-overlapping value ranges for the same field), or there is insufficient information to determine overlap.

Judgment Guidelines:
1. If one query adds conditions or narrows value ranges on top of the other, then its data set is a subset of the other.  
   - If the current query is stricter, it is a subset of the historical query.  
   - If the current query is looser, it contains the historical query.

2. Use intersect only when it's clear the two queries share data but are not subsets. If overlap cannot be determined, use no_relation.

Output format:  
Return a single-line JSON only, with no extra text:  
{"reason": "Your reasoning — list and compare filter conditions of both queries before drawing a conclusion", "relation": "<contain|subset|equal|intersect|no_relation>"}

---

[Current Query]  
%s

[Historical Query]  
%s

Please analyze and determine the relationship:
"""


def parse_relation(ret_str: str) -> str:
    try:
        ret = jsonfy_llm_response(ret_str)
        relation = ret.get("relation", "")
    except Exception:
        relation = ""

    r = relation.lower()
    if "contain" in r:
        return "Contain"
    if "subset" in r:
        return "Subset"
    if "equal" in r:
        return "Equal"
    if "intersect" in r:
        return "Intersect"
    return "No_relation"

class OLAPNode:
    def __init__(
        self,
        node_id: int,
        query: str,
        olap_ids: Set[int],
        plan
    ):
        self.id = node_id
        self.query = query
        self.olap_ids: Set[int] = set(olap_ids)
        self.plan = plan

    def __repr__(self):
        return f"OLAPNode(id={self.id}, size={len(self.olap_ids)})"

class OLAP_DAG:
    def __init__(self, llm):
        self.llm = llm
        self.nodes: Dict[int, OLAPNode] = {}
        self.edges: Dict[int, List[int]] = {}
        self.root: Optional[int] = None

    def init_root(self, all_olap_ids: Set[int], query: str = "All Data"):
        root = OLAPNode(0, query, all_olap_ids, [])
        self.nodes[0] = root
        self.edges[0] = []
        self.root = 0

    def get_node(self, node_id):
        return self.nodes[node_id]

    def add_node(self, query: str, olap_ids: Set[int], plan) -> int:
        node_id = len(self.nodes)
        self.nodes[node_id] = OLAPNode(node_id, query, olap_ids, plan)
        self.edges[node_id] = []
        return node_id

    def add_edge(self, parent: int, child: int):
        self.edges[parent].append(child)

    def judge_relation_by_llm(
        self,
        current_query: str,
        history_query: str
    ) -> str:
        prompt = PROMPT_GET_NODE % (current_query, history_query)
        resp = self.llm.predict(prompt)
        return parse_relation(resp)

    def get_current_nodes(
            self,
            query: str
    ) -> Tuple[List[int], Set[int], str]:

        queue: List[Tuple[int, str]] = []

        queue.append((self.root, "Subset"))

        candidate_subset_nodes: List[int] = []

        while queue:
            node_id, relation = queue.pop(0)
            node = self.nodes[node_id]

            if relation == "Equal":
                return [node_id], node.olap_ids, "Equal"

            if relation != "Subset":
                continue

            children = self.edges.get(node_id, [])
            has_subset_or_equal_child = False

            for child_id in children:
                child = self.nodes[child_id]
                child_relation = self.judge_relation_by_llm(query, child.query)
                if child_relation in ("Subset", "Equal"):
                    has_subset_or_equal_child = True
                    queue.append((child_id, child_relation))

            if not has_subset_or_equal_child:
                candidate_subset_nodes.append(node_id)

        if candidate_subset_nodes:
            intersect_ids = None
            for nid in candidate_subset_nodes:
                ids = self.nodes[nid].olap_ids
                if intersect_ids is None:
                    intersect_ids = set(ids)
                else:
                    intersect_ids &= ids

            if intersect_ids is None:
                intersect_ids = set()

            return candidate_subset_nodes, intersect_ids, "Subset"

        return [self.root], self.nodes[self.root].olap_ids, "Fallback"

class GranularitySpec:
    def __init__(self, name: str):
        self.name: str = name
        self.plan: List[Dict] = []
        self.generate_plan: List[Dict] = []
        self.values: List[str] = [name]

    # ---------- update ----------
    def update(
        self,
        plan: Optional[List[Dict]] = None,
        values: Optional[List[str]] = None,
        generate_plan: Optional[List[Dict]] = None
    ):
        if plan:
            self.plan.extend(plan)
        if generate_plan:
            self.generate_plan.extend(generate_plan)

        if values:
            for v in values:
                if v not in self.values:
                    self.values.append(v)

    # ---------- get ----------
    def get_plan(self) -> List[Dict]:
        return copy.deepcopy(self.plan)

    def get_values(self) -> List[str]:
        return list(self.values)

    # ---------- copy ----------
    def copy(self) -> "GranularitySpec":
        g = GranularitySpec(self.name)
        g.plan = []
        g.values = list(self.values)
        return g

class DimensionSpec:
    def __init__(self, name: str):
        self.name: str = name
        self.plan: List[Dict] = []
        self.generate_plan: List[Dict] = []
        self.granularities: Dict[str, GranularitySpec] = {name: GranularitySpec(name)}

    def update(self, plan: Optional[List[Dict]] = None, generate_plan: Optional[List[Dict]] = None):
        if plan:
            self.plan.extend(plan)
        if generate_plan:
            self.generate_plan.extend(generate_plan)

    def update_granularity(
        self,
        granularity: str,
        plan: Optional[List[Dict]] = None,
        values: Optional[List[str]] = None,
        generate_plan: Optional[List[Dict]] = None
    ) -> GranularitySpec:
        if granularity not in self.granularities:
            self.granularities[granularity] = GranularitySpec(granularity)

        g = self.granularities[granularity]
        g.update(plan=plan, values=values, generate_plan=generate_plan)
        return g

    def get_plan(self) -> List[Dict]:
        return copy.deepcopy(self.plan)

    def get_granularities(self) -> List[str]:
        return list(self.granularities.keys())

    def get_granularity(self, granularity: str) -> GranularitySpec:
        if granularity not in self.granularities:
            raise ValueError(
                f"Granularity '{granularity}' does not exist in dimension '{self.name}'"
            )
        return self.granularities[granularity]

    def copy(self) -> "DimensionSpec":
        d = DimensionSpec(self.name)
        d.plan = []
        d.granularities = {
            k: v.copy() for k, v in self.granularities.items()
        }
        return d

class OLAP_Axis:
    def __init__(self):
        self._dimensions: Dict[str, DimensionSpec] = {}

    def update_dimension(
        self,
        dimension: str,
        plan: Optional[List[Dict]] = None,
        generate_plan: Optional[List[Dict]] = None
    ):
        if dimension not in self._dimensions:
            self._dimensions[dimension] = DimensionSpec(dimension)

        self._dimensions[dimension].update(plan=plan, generate_plan=generate_plan)

    def update_granularity(
        self,
        dimension: str,
        granularity: str,
        plan: Optional[List[Dict]] = None,
        values: Optional[List[str]] = None,
        generate_plan: Optional[List[Dict]] = None
    ):
        if dimension not in self._dimensions:
            self._dimensions[dimension] = DimensionSpec(dimension)

        self._dimensions[dimension].update_granularity(
            granularity=granularity,
            plan=plan,
            values=values,
            generate_plan=generate_plan
        )

    def get_dimensions(self) -> List[str]:
        return list(self._dimensions.keys())

    def get_dimension(self, dimension: str) -> DimensionSpec:
        if dimension not in self._dimensions:
            raise ValueError(f"Dimension '{dimension}' does not exist")
        return self._dimensions[dimension]

    def get_granularities(self, dimension: str) -> List[str]:
        if dimension not in self._dimensions:
            raise ValueError(f"Dimension '{dimension}' does not exist")
        return self._dimensions[dimension].get_granularities()

    def get_granularity(
        self,
        dimension: str,
        granularity: str
    ) -> GranularitySpec:
        if dimension not in self._dimensions:
            raise ValueError(f"Dimension '{dimension}' does not exist")
        return self._dimensions[dimension].get_granularity(granularity)

    def copy(self) -> "OLAP_Axis":
        axis = OLAP_Axis()
        axis._dimensions = {
            k: v.copy() for k, v in self._dimensions.items()
        }
        return axis

class SemaFlex_Cube:
    def __init__(self, df: pd.DataFrame, llm):
        if not isinstance(df, pd.DataFrame):
            raise TypeError("df must be a pandas DataFrame")

        self.tables = Tables(df)

        self.axis = OLAP_Axis()

        for col in df.columns:
            self.axis.update_granularity(
                dimension=col,
                granularity=col,
                values=[col]
            )

        self.dag = OLAP_DAG(llm)
        all_olap_ids = set(self.tables.raw_table[self.tables.OLAP_KEY])
        self.dag.init_root(all_olap_ids, query="All Data")

    def locate_from_dag(self, query: str) -> Tuple[List[int], Set[int], str]:
        return self.dag.get_current_nodes(query)

    def update_dag(
        self,
        parent_node_ids: List[int],
        olap_ids: Set[int],
        query: str,
        plan
    ) -> int:
        new_node_id = self.dag.add_node(query, olap_ids, plan)
        for pid in parent_node_ids:
            self.dag.add_edge(pid, new_node_id)
        return new_node_id

    def get_dimensions(self) -> List[str]:
        return self.axis.get_dimensions()

    def get_granularities(self, dimension: str) -> List[str]:
        return self.axis.get_granularities(dimension)

    def read_granularity(
        self,
        ids: Optional[Set[int]],
        dimension: str,
        granularity: str,
        cols: Optional[List[str]] = None
    ) -> pd.DataFrame:
        return self.tables.get_granularity(
            dimension=dimension,
            granularity=granularity,
            olap_ids=ids,
            list_col=cols
        )

    def read_raw(
        self,
        ids: Set[int],
        dimensions: Optional[List[str]] = None
    ) -> pd.DataFrame:
        return self.tables.get_raw(
            olap_ids=ids,
            list_col=dimensions
        )

    def write_granularity(
        self,
        df: pd.DataFrame,
        dimension: str,
        granularity: str,
        plan: Optional[List[Dict]] = None,
        generate_plan: Optional[List[Dict]] = None
    ):
        if not isinstance(df, pd.DataFrame):
            raise TypeError("df must be a pandas DataFrame")

        if self.tables.OLAP_KEY not in df.columns:
            raise ValueError(
                f"df must contain column '{self.tables.OLAP_KEY}'"
            )

        values = [c for c in df.columns if c != self.tables.OLAP_KEY]

        self.axis.update_granularity(
            dimension=dimension,
            granularity=granularity,
            values=values,
            plan=plan,
            generate_plan=generate_plan
        )

        self.tables.set_granularity(
            dimension=dimension,
            granularity=granularity,
            table=df
        )

    def write_dimension(
        self,
        df: pd.DataFrame,
        dimension: str,
        plan,
        generate_plan: Optional[List[Dict]] = None
    ):
        if not isinstance(df, pd.DataFrame):
            raise TypeError("df must be a pandas DataFrame")

        if self.tables.OLAP_KEY not in df.columns:
            raise ValueError(
                f"df must contain column '{self.tables.OLAP_KEY}'"
            )

        self.axis.update_dimension(
            dimension=dimension,
            plan=plan,
            generate_plan=generate_plan
        )

        self.tables.set_dimension(
            dimension=dimension,
            table=df
        )

    def sample_raw(
        self,
        k: int,
        dimensions: Optional[List[str]] = None,
        seed: Optional[int] = None
    ) -> pd.DataFrame:
        all_ids = set(self.tables.raw_table[self.tables.OLAP_KEY])
        sampled_ids = sample_olap_ids(all_ids, k, seed)

        return self.read_raw(
            ids=sampled_ids,
            dimensions=dimensions
        )

    def sample_granularity(
        self,
        dimension: str,
        granularity: str,
        k: int,
        cols: Optional[List[str]] = None,
        seed: Optional[int] = None
    ) -> pd.DataFrame:
        all_ids = set(self.tables.raw_table[self.tables.OLAP_KEY])
        sampled_ids = sample_olap_ids(all_ids, k, seed)

        return self.read_granularity(
            ids=sampled_ids,
            dimension=dimension,
            granularity=granularity,
            cols=cols
        )


import json

from Utils.LLM import create

from Utils.jsonfy_result import jsonfy_llm_response

llm = None  # Configure your own LLM instance here.

from collections import defaultdict


def extract_and_group_operators(ground_truth_dict):
    grouped = defaultdict(list)
    operator_id = 0

    for source in ["key_operator", "valid_operator"]:
        for op in ground_truth_dict.get(source, []):
            new_op = {
                "id": operator_id,
                "from": "key_operator" if source == "key_operator" else "valid_operator",
                "operator": op["operator"],
                "params": op.get("params", {})
            }
            grouped[op["operator"]].append(new_op)
            operator_id += 1

    key_operator_count = len(ground_truth_dict.get("key_operator", []))

    return dict(grouped), key_operator_count


def choose_from_candidate_prompt(candidate_str, self_str, type_):
    match_rules = {
        "sem_filter": "For sem_filter, the filtering condition must be exactly the same in meaning. Tolerate different ways of expressing the same meaning, such as ‘hold the records with (condition)’ or directly stating the (condition). The columns parameter is optimal and you can ignore it.",
        "keyword_filter": "For keyword_filter, first infer the real-world event or task each keyword list is describing (e.g., keyA -> event_A, keyB -> event_B). Then determine whether event_A and event_B describe broadly the same thing. It is not necessary for the keyword lists to match directly.",
        "num_filter": "For num_filter, each numeric condition (column, operator, threshold) must match exactly, and if there are more than one conditions, the overall logic ('and' or 'or') must be exactly the same. If there is only one numeric condition in num_filter, the logical operator (and/or) can be ignored, since it only matters when there are multiple conditions.",
        "sem_map": "For sem_map, the keyword is automatically generated and may vary a lot in wording, so it only needs to have broadly similar meaning. The map_description should be similar in meaning as well. The columns parameter is optional and can be ignored.",
        "sem_group": "For sem_group, the keyword is automatically generated and may vary a lot in wording, so it only needs to have broadly similar meaning. The columns can also have broadly similar meaning, and the group_description should be similar in meaning.",
        "sem_reduce": "For sem_reduce, the keyword is automatically generated and may vary in wording, so it only needs to have broadly similar meaning. The columns can have broadly similar meaning, the group_by parameter can have broadly similar meaning, and the overall reduce goal should be similar in meaning.",
        "num_reduce": "For num_reduce, the keyword is automatically generated and may vary in wording, so it only needs to have broadly similar meaning. The columns can have broadly similar meaning, the aggregation function(s) must be exactly the same, and the group_by parameter can have broadly similar meaning.",
        "count": "For count, the keyword is automatically generated and may vary in wording, so it only needs to have broadly similar meaning. The group_by column can have broadly similar meaning and the list of columns can have broadly similar meaning as well.",
        "num_topk": "For num_topk, the keyword is automatically generated and may vary in wording, so it only needs to have broadly similar meaning. The value of k must be exactly the same, while the column and order can have broadly similar meaning.",
        "sem_topk": "For sem_topk, the keyword is automatically generated and may vary in wording, so it only needs to have broadly similar meaning. The value of k must be exactly the same, while the column can have broadly similar meaning and the query should be similar in meaning.",
        "date": "For date, the key is whether the filtered time range is semantically the same. Different operators (e.g., sem_filter, keyword_filter) may be used, but as long as they resolve to the same actual date or date range, they should be considered a match."    }

    prompt = """
You are given a list of candidate strings and a target string.
Your task is to determine if the target string matches one of the candidates.

Matching Rule for this operator type (%s):
%s

If a match exists, return the matched candidate's id.
If no match exists, return None.
Always provide a short explanation for your decision.

Return strictly in this format:
{
    "reason": "<short_reason>",
    "match_id": "<matched_candidate_or_None>"
}

Here is the data:
Candidates: 
%s

Target: 
%s
""" % (type_, match_rules.get(type_, "No specific rule provided."), candidate_str, self_str)

    return prompt

def preprocess_params(params, operator_name=None):
    if operator_name == "num_filter":
        conditions = params.get("conditions", [])
        if len(conditions) == 1:
            params = dict(params)
            params.pop("logic", None)
    return params


def require_llm():
    if llm is None:
        print("LLM is not configured. Please create an LLM instance and assign it to `llm` before semantic evaluation.")
        raise SystemExit(1)

def evaluate_semantic_for_key_operators(ground_truth_dict, generated_plan_dict):
    generated_plan = generated_plan_dict["plan"]

    ground_truth_operators, key_operator_count = extract_and_group_operators(ground_truth_dict)

    generated_operator_count = len(generated_plan)
    pairs = []
    matched_generated_ids = set()
    matched_key_ids = set()
    for operator in generated_plan:
        type_ = operator["operator_name"]
        if(type_ in ground_truth_operators.keys()):
            candidate = ground_truth_operators[type_]
        else:
            candidate = []

        if type_ == "keyword_filter":
            cols = operator["parameters"].get("columns")
            if cols == ["creation_date"]:
                extra_candidates = [
                    sem_cand for sem_cand in ground_truth_operators.get("sem_filter", [])
                    if sem_cand["params"].get("columns") == ["creation_date"]
                ]
                candidate.extend(extra_candidates)
                type_ = "date"


        if not candidate:
            continue

        candidate_str = ""
        for c in candidate:
            params = preprocess_params(c['params'], c.get("operator_name"))
            candidate_str += f"(id: {c['id']}, params: {params}) \n\n"

        if candidate_str == "":
            continue

        params = preprocess_params(operator['parameters'], operator.get("operator_name"))
        self_str = f"params: {params}"
        prompt = choose_from_candidate_prompt(candidate_str, self_str, type_)
        require_llm()
        result = llm.predict(prompt)
        result_json = jsonfy_llm_response(result)
        print(result)

        try:
            choosed_id = int(result_json["match_id"])
        except:
            choosed_id = None
            continue

        matched_item = None
        for c in candidate:
            if c["id"] == choosed_id:
                matched_item = c
                break

        if matched_item:
            pairs.append({
                "generated_operator": operator,
                "matched_candidate": matched_item
            })
            matched_generated_ids.add(id(operator))
            candidate.remove(matched_item)
            matched_key_ids.add(matched_item["id"])

    count_key = 0
    count_valid = 0
    for pair in pairs:
        type_ = pair["matched_candidate"]["from"]
        if type_ == "key_operator":
            count_key += 1
        count_valid += 1

    recall = count_key / key_operator_count if key_operator_count > 0 else 0.0
    precision = count_valid / generated_operator_count if generated_operator_count > 0 else 0.0

    unmatched_generated = [
        op for op in generated_plan if id(op) not in matched_generated_ids
    ]

    unmatched_key = []
    for ops in ground_truth_operators.values():
        for op in ops:
            if op["from"] == "key_operator" and op["id"] not in matched_key_ids:
                unmatched_key.append(op)

    return recall, precision, pairs, unmatched_generated, unmatched_key

def evaluate_recall_for_key_operators(ground_truth_dict, generated_plan_dict):
    ground_truth = ground_truth_dict["key_operator"]
    generated_plan = generated_plan_dict["plan"]
    gt_operators = []
    for row in ground_truth:
        if (row["operator"] not in gt_operators):
            gt_operators.append(row["operator"])

    gp_operators = []
    for row in generated_plan:
        if (row["operator_name"] not in gp_operators):
            gp_operators.append(row["operator_name"])
    matched = [op for op in gt_operators if op in gp_operators]
    if len(gt_operators) == 0:
        return 0.0

    recall = len(matched) / len(gt_operators)
    return recall


def evaluate_precision_for_valid_operators(ground_truth_dict, generated_plan_dict):
    ground_truth = ground_truth_dict["key_operator"] + ground_truth_dict["valid_operator"]
    generated_plan = generated_plan_dict["plan"]

    gt_operators = []
    for row in ground_truth:
        if (row["operator"] not in gt_operators):
            gt_operators.append(row["operator"])

    gp_operators = []
    for row in generated_plan:
        if (row["operator_name"] not in gp_operators):
            gp_operators.append(row["operator_name"])

    if len(gp_operators) == 0:
        return 0.0

    matched = [op for op in gp_operators if op in gt_operators]
    precision = len(matched) / len(gp_operators)
    return precision


def evaluate_r_p(ground_trurh, generated):
    result = {"id": ground_trurh["id"]}
    for q_key in ["Q1", "Q2", "Q3", "Q4"]:
        recall = evaluate_recall_for_key_operators(ground_trurh[q_key], generated[q_key])
        precision = evaluate_precision_for_valid_operators(ground_trurh[q_key], generated[q_key])
        result[q_key] = {"recall": recall, "precision": precision}

    return result


def evaluate(ground_trurh, generated):
    results = []
    for row in generated:
        gt = ground_trurh[row["id"]]
        results.append(evaluate_r_p(gt, row))
    return results


def evaluate_r_p_semantic(ground_trurh, generated):
    result = {"id": ground_trurh["id"]}
    for q_key in ["Q1", "Q2", "Q3", "Q4"]:
        recall, precision, pairs, unmatched_generated, unmatched_key = evaluate_semantic_for_key_operators(
            ground_trurh[q_key], generated[q_key])
        print("Ground truth id and question: " + str(ground_trurh["id"]) + ", " + q_key)
        print(f"Recall: {recall}, Precision: {precision}")
        print("="*50)
        result[q_key] = {"recall": recall, "precision": precision, "pairs": pairs,
                         "unmatched_generated": unmatched_generated, "unmatched_key": unmatched_key}

    return result

from concurrent.futures import ThreadPoolExecutor, as_completed


def evaluate_semantic(ground_trurh, generated, max_workers=5):
    def process_one_row(idx, row):
        print(f"Evaluating id: {row['id']}")
        gt = ground_trurh[row["id"]]
        result = evaluate_r_p_semantic(gt, row)
        return idx, result

    results = [None] * len(generated)

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(process_one_row, idx, row)
            for idx, row in enumerate(generated)
        ]

        for future in as_completed(futures):
            idx, result = future.result()
            results[idx] = result

    return results


def load_or_evaluate_missing(eval_result_path, ground_trurh, generated):
    if os.path.exists(eval_result_path):
        print(f"Existing evaluation results found: {eval_result_path}")
        with open(eval_result_path, "r", encoding="utf-8") as f:
            cached_results = json.load(f)
    else:
        print(f"No evaluation result file found: {eval_result_path}")
        cached_results = []

    cached_by_id = {str(row.get("id")): row for row in cached_results}
    missing_generated = [
        row for row in generated
        if str(row.get("id")) not in cached_by_id
    ]

    if missing_generated:
        print(f"Missing evaluation results: {len(missing_generated)} item(s). Evaluating missing items only.")
        new_results = evaluate_semantic(ground_trurh, missing_generated)
        for row in new_results:
            cached_by_id[str(row.get("id"))] = row

        merged_results = [
            cached_by_id[str(row.get("id"))]
            for row in generated
            if str(row.get("id")) in cached_by_id
        ]

        os.makedirs(os.path.dirname(eval_result_path) or ".", exist_ok=True)
        with open(eval_result_path, "w", encoding="utf-8") as f:
            json.dump(merged_results, f, indent=4, ensure_ascii=False)

        print(f"Updated evaluation results saved to: {eval_result_path}")
        return merged_results

    print("All generated items already have evaluation results. Using cached results.")
    return [
        cached_by_id[str(row.get("id"))]
        for row in generated
        if str(row.get("id")) in cached_by_id
    ]


def calc_recall_precision(data):
    total_recall = 0
    total_precision = 0
    count = 0

    details = []

    per_q_totals = {}

    for record in data:
        detail_item = {"id": record.get("id")}
        for key, value in record.items():
            if key.startswith("Q") and isinstance(value, dict):
                recall = value.get("recall", 0)
                precision = value.get("precision", 0)
                detail_item[key] = (recall, precision)

                total_recall += recall
                total_precision += precision
                count += 1

                if key not in per_q_totals:
                    per_q_totals[key] = {"recall": 0, "precision": 0, "count": 0}
                per_q_totals[key]["recall"] += recall
                per_q_totals[key]["precision"] += precision
                per_q_totals[key]["count"] += 1

        details.append(detail_item)

    overall_avg = (
        total_recall / count if count else 0,
        total_precision / count if count else 0
    )

    per_q_avg = {}
    for q, stats in per_q_totals.items():
        per_q_avg[q] = (
            stats["recall"] / stats["count"] if stats["count"] else 0,
            stats["precision"] / stats["count"] if stats["count"] else 0
        )

    return overall_avg, details, per_q_avg

def calc_recall_precision_by_complexity(evaluate_results, complexity_dict):
    level_stats = {
        "Level_1": {"recall_sum": 0, "precision_sum": 0, "count": 0},
        "Level_2": {"recall_sum": 0, "precision_sum": 0, "count": 0},
        "Level_3": {"recall_sum": 0, "precision_sum": 0, "count": 0},
    }

    for record in evaluate_results:
        record_id = str(record["id"])
        if record_id not in complexity_dict:
            continue

        complexity_info = complexity_dict[record_id]

        for q_key in ["Q1", "Q2", "Q3", "Q4"]:
            if q_key not in record or not isinstance(record[q_key], dict):
                continue

            recall = record[q_key].get("recall", 0)
            precision = record[q_key].get("precision", 0)
            level = complexity_info.get(q_key)

            if level not in level_stats:
                continue

            level_stats[level]["recall_sum"] += recall
            level_stats[level]["precision_sum"] += precision
            level_stats[level]["count"] += 1

    level_avg = {}
    for level, stats in level_stats.items():
        count = stats["count"]
        if count == 0:
            avg_recall = 0
            avg_precision = 0
        else:
            avg_recall = stats["recall_sum"] / count
            avg_precision = stats["precision_sum"] / count
        level_avg[level] = {"avg_recall": avg_recall, "avg_precision": avg_precision}

    print("\n=== Average Results by Complexity ===")
    for level, vals in level_avg.items():
        print(f"{level}: Recall={vals['avg_recall']:.4f}, Precision={vals['avg_precision']:.4f}")

    return level_avg


def calc_scores_by_level(evaluate_results, complexity_dict):
    level_scores = {
        "Level_1": [],
        "Level_2": [],
        "Level_3": [],
    }

    for record in evaluate_results:
        record_id = str(record["id"])
        complexity_info = complexity_dict.get(record_id)
        if not complexity_info:
            continue

        for q_key in ["Q1", "Q2", "Q3", "Q4"]:
            q_result = record.get(q_key)
            if not isinstance(q_result, dict):
                continue

            level = complexity_info.get(q_key)
            if level not in level_scores:
                continue

            level_scores[level].append({
                "id": record_id,
                "question": q_key,
                "recall": q_result.get("recall", 0),
                "precision": q_result.get("precision", 0),
            })

    return level_scores


def calc_avg_r_p_per_id(data):
    result = {}

    for record in data:
        cid = str(record["id"])

        recalls = []
        precisions = []

        for q in ["Q1", "Q2", "Q3", "Q4"]:
            if q in record and isinstance(record[q], dict):
                recalls.append(record[q].get("recall", 0))
                precisions.append(record[q].get("precision", 0))

        avg_recall = sum(recalls) / len(recalls) if recalls else 0.0
        avg_precision = sum(precisions) / len(precisions) if precisions else 0.0

        result[cid] = {
            "avg_recall": avg_recall,
            "avg_precision": avg_precision
        }

    return result

def calc_r_p_per_id(results):
    result = {}

    for record in results:
        cid = str(record["id"])
        result[cid] = {}

        for q in ["Q1", "Q2", "Q3", "Q4"]:
            if q in record and isinstance(record[q], dict):
                result[cid][q] = {
                    "recall": record[q].get("recall", 0),
                    "precision": record[q].get("precision", 0)
                }

    return result

import json
import os


if __name__ == "__main__":

    result_path = "results"

    approach = "SemExp"

    test_ground_truth = {}

    with open("datas/evaluate_datas.json", "r", encoding="utf-8") as f:
        test_datas = json.load(f)

    for row in test_datas:
        test_ground_truth[row["id"]] = row

    with open(f"{result_path}/{approach}.json", "r", encoding="utf-8") as f:
        generated = json.load(f)

    eval_result_path = f"{result_path}/{approach}_evaluate.json"

    data = load_or_evaluate_missing(eval_result_path, test_ground_truth, generated)

    overall, details, per_q_avg = calc_recall_precision(data)

    print("\n=== Average by Question ===")
    for q, (rec, prec) in per_q_avg.items():
        print(f"{q}: Average Recall={rec:.4f}, Average Precision={prec:.4f}")

    with open("datas/data_complexity.json", "r", encoding="utf-8") as f:
        complexity = json.load(f)

    level_avg = calc_recall_precision_by_complexity(data, complexity)
    level_question_scores = calc_scores_by_level(data, complexity)

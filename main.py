import json
import pandas as pd

from Agents.OLAP_Agent import OLAP_Agent


def flatten_main_records(data):
    def stringify_nested(obj):
        if isinstance(obj, (dict, list)):
            return json.dumps(obj, ensure_ascii=False)
        return obj

    flat_data = []
    for record in data:
        flat_record = {k: stringify_nested(v) for k, v in record.items()}
        flat_data.append(flat_record)

    return pd.DataFrame(flat_data)


def load_json(path="datas/stackoverflow_database_sample_2000.json"):
    with open(path, encoding="utf-8") as f:
        data_json = json.load(f)
    return flatten_main_records(data_json)


llm = None


def require_llm():
    if llm is None:
        print("LLM is not configured. Please create an LLM instance and assign it to `llm` before running main.")
        raise SystemExit(1)


if __name__ == "__main__":
    require_llm()

    data = load_json("datas/stackoverflow_database_sample_2000.json")
    print("Data loaded.")

    olap_agent = OLAP_Agent(llm=llm, data=data)
    query = "Please find the 2010 data."
    olap_agent.run(query, {"id": "example", "Q_id": "Q1"}, file_path="results/example.json")

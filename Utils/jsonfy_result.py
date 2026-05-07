import re
import json
import ast

def jsonfy_llm_response(response, defalt_result=None):
    response = re.sub(r"```(?:\w+)?\n(.*?)```", r"\1", response, flags=re.DOTALL).strip()

    if response and (response[0].isalnum() or response[-1].isalnum()):
        start = response.find("{")
        end = response.rfind("}")
        if start != -1 and end != -1 and start < end:
            response = response[start:end + 1]

    def strip_json_comments(text: str) -> str:
        text = re.sub(r'//.*?(?=\n)', '', text)
        text = re.sub(r'/\*.*?\*/', '', text, flags=re.DOTALL)
        return text

    try:
        result = json.loads(response)
    except Exception:
        if "//" in response or "/*" in response:
            clean_response = strip_json_comments(response)
            try:
                result = json.loads(clean_response)
            except Exception:
                try:
                    result = ast.literal_eval(clean_response)
                except Exception:
                    print("!" * 50)
                    print(f"ERROR: \n{response}\n can not be parsed into JSON/Python object!")
                    print("!" * 50)
                    result = defalt_result if defalt_result is not None else response
        else:
            try:
                result = ast.literal_eval(response)
            except Exception:
                print("!" * 50)
                print(f"ERROR: \n{response}\n can not be parsed into JSON/Python object!")
                print("!" * 50)
                result = defalt_result if defalt_result is not None else response

    return result



def llm_retry(result, llm):
    prompt = f"""
The following output cannot be parsed by json.loads().
Your task is to minimally fix the output so that it becomes valid JSON.

Rules:
- Preserve the original meaning and content.
- Do NOT add new fields or remove existing information.
- Do NOT infer or invent missing values.
- Fix only formatting issues (such as quotes, commas, brackets, or escaping).
- Output ONLY the corrected JSON, with no additional text.

Original output:
{result}
"""
    return llm.predict(prompt)

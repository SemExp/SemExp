from openai import OpenAI


class LLM:
    def __init__(self, url, key, model):
        self.client = OpenAI(api_key=key, base_url=url)
        self.model = model

    def predict(self, prompt):
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            extra_body={
                "thinking": {"type": "disabled"},
                "enable_thinking": False
            }
        )
        return response.choices[0].message.content.strip()


def create(url, key, model):
    return LLM(url, key, model)

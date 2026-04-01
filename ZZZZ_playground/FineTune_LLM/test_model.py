from openai import OpenAI

client = OpenAI()
MODEL = "ft:gpt-4.1-2025-04-14:lorenz::D88rl5yJ"
MODEL = "gpt-4.1"

SYSTEM = "You classify messages as positive, negative, or neutral. Respond with exactly one word."

def classify(text: str) -> str:
    r = client.responses.create(
        model=MODEL,
        input=[
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": text},
        ],
        temperature=0.5,
        max_output_tokens=16,
    )
    return r.output_text.strip().split()[0]

print(classify("This is the worst service ever."))
print(classify("In particular I was satisfied with the quick response time."))
print(classify("Three birds fly in a circle and chase each other."))

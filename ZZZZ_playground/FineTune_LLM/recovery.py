from openai import OpenAI
from datetime import datetime

client = OpenAI()

jobs = client.fine_tuning.jobs.list(limit=50)  # ggf. höher / paginieren
for j in jobs.data:
    created = datetime.fromtimestamp(j.created_at).isoformat()
    print(created, j.id, j.status, j.model, j.training_file, j.fine_tuned_model)
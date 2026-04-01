import os
import time
from openai import OpenAI

# Uses OPENAI_API_KEY from env
client = OpenAI()

TRAIN_PATH = "sentiment_train.jsonl"
BASE_MODEL = "gpt-4.1-2025-04-14"  # adjust if your account doesn't have access

def main():
    if not os.path.exists(TRAIN_PATH):
        raise FileNotFoundError(f"Missing {TRAIN_PATH} in current directory")

    # 1) Upload file
    with open(TRAIN_PATH, "rb") as f:
        uploaded = client.files.create(file=f, purpose="fine-tune")
    print("Uploaded file id:", uploaded.id)

    # 2) Create fine-tune job
    job = client.fine_tuning.jobs.create(
        training_file=uploaded.id,
        model=BASE_MODEL,
    )
    print("Fine-tune job id:", job.id)

    # 3) Poll status until done
    while True:
        j = client.fine_tuning.jobs.retrieve(job.id)
        print("Status:", j.status)

        if j.status == "succeeded":
            print("Fine-tuned model:", j.fine_tuned_model)
            break
        if j.status in ("failed", "cancelled"):
            print("Job ended:", j.status)
            print("Error:", j.error)
            break

        time.sleep(10)

if __name__ == "__main__":
    main()

from transformers import pipeline

for task, model in [
    ("sentiment-analysis", "cardiffnlp/twitter-roberta-base-sentiment-latest"),
    ("text-classification", "j-hartmann/emotion-english-distilroberta-base"),
    ("zero-shot-classification", "facebook/bart-large-mnli"),
    ("summarization", "sshleifer/distilbart-cnn-12-6"),
]:
    print("Downloading", model)
    pipeline(task, model=model)
print("All models ready")
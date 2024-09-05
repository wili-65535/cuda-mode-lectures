from transformers import AutoModel

model = AutoModel.from_pretrained("bert-base-cased")

print(dir(model))

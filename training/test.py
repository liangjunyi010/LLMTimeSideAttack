from datasets import load_dataset
print(load_dataset("medalpaca/medical_meadow_medqa", split="train")[0])
print(load_dataset("tatsu-lab/alpaca", split="train")[0])
from datasets import load_dataset
import pandas as pd

dataset = load_dataset("Elliott/Openr1-Math-48k-Complement", split="train")
dataset.save_to_disk("../llm_datasets/Openr1-Math-48k-Complement") # 保存到该目录下
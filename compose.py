from datasets import load_dataset
from tqdm import tqdm

original = load_dataset("ServiceNow-AI/R1-Distill-SFT", "v1", split="train")
sft_full = load_dataset("Phando/sft-dataset-from-moonlight", split="train")

question_to_sample = {
    sample["messages"][0]["content"]: sample for sample in original
}
source_list = []
source_dataset_list = []
id_list = []
for sample in tqdm(sft_full):
    question = sample["question"]
    full_sample = question_to_sample[question]
    source_list.append(full_sample["source"])
    source_dataset_list.append(full_sample["source_dataset"])
    id_list.append(full_sample["id"])

sft_full = sft_full.add_column("source", source_list)
sft_full = sft_full.add_column("id", id_list)

print(f"Before filtering, the number of samples in sft_full is {len(sft_full)}")
# remove the samples with "source" is "ai2-adapt-dev/tulu_hard_coded_repeated_10"
sft_full = sft_full.filter(lambda example: example["source"] != "ai2-adapt-dev/tulu_hard_coded_repeated_10")
print(f"After filtering, the number of samples in sft_full is {len(sft_full)}")

sft_full.push_to_hub("Phando/sft-dataset-from-moonlight-noid")

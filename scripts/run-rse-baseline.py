from datasets import load_dataset, Dataset
from loguru import logger
import openai
from fire import Fire
from tqdm import tqdm
import os
import json

logger.add("logs/run-rse-baseline.log")

prompt = """**Task Description**: You are an AI language model analyst. Your task is to evaluate the similarity between model responses based on the following "Evaluation Criteria".

**Input**: You will be given a question, a reference answer, and model response.

**Evaluation Criteria**:
- Response Style: Compare the style of the reference answer and the model responses, including formality, word choice, punctuation, etc. 
- Logical Structure: Compare the logical flow of the reference answer and the model responses, such as whether the ideas are presented in a similar order or if the reasoning process is alike.
- Content Details: Compare the details of the reference answer and the model responses, such as whether they cover similar knowledge points or use similar examples.

**Scoring Criteria**:
- **2--Similar**: The model response closely mirrors the reference answer in this dimension, with only minor or negligible differences.
**Response Style:** The tone, vocabulary, and punctuation are almost identical.
**Logical Structure:** Ideas follow the same sequence and are presented with similar reasoning.
**Content Details:** The same knowledge points and examples are covered in equivalent detail.
- **1--Neutral**: The model response partially aligns with the reference answer, with noticeable but non-disruptive differences.
**Response Style:** The tone or vocabulary differs, but the overall style is consistent.
**Logical Structure:** The flow of ideas is similar, but some points are reordered or omitted.
**Content Details:** Covers most key knowledge points, but some details or examples are missing or substituted.
- **0--Dissimilar**: The model response diverges significantly from the reference answer in this dimension.
**Response Style:** The tone, word choice, or punctuation style is clearly inconsistent.
**Logical Structure:** The flow of ideas is disorganized or completely different from the reference.
**Content Details:** Key knowledge points or examples are missing or replaced with irrelevant content.

**Overview Scoring Criteria**:
- **Score: 5/5**: Very similar. The response style, logical structure, and content details are highly consistent and almost identical.
- **Score: 4/5**: Similar. The response style, logical structure, and content details are generally consistent, but there are some minor differences.
- **Score: 3/5**: Somewhat similar. The response style, logical structure, and content details are somewhat consistent, but there are some differences.
- **Score: 2/5**: Not similar. The response style, logical structure, and content details are not consistent, and there are significant differences.
- **Score: 1/5**: Very different. The response style, logical structure, and content details are very different.

**Output**:
You should first score each criterion based on the
“Scoring Criteria,” and then use the scores for
each criterion and "Overview Scoring Criteria" to
arrive at an overall score.
1. explain: Details of the analysis
2. style score: the score of Response Style
3. logical score: the score of Logical Structure
4. content score: the score of Content Details
5. overview score: overall score

Please output the results in following format:
<explain_start> provide a detailed explanation here
</explain_end>
<style_score_start> style score </style_score_end>
<logical_score_start> logical score
</logical_score_end>
<content_score_start> content score
</content_score_end>
<overview_score_start> style score
</overview_score_end>
```json
{
"style_score": "2",
"logical_score": "2",
"content_score": "2",
"overview_score": "5/5"
}
```"""

def run_rse_baseline(
    model_csv_path: str,
    base_url: str = "http://localhost:2333/v1",
    model_name: str = "Qwen/Qwen3-4B",
    r1_distill_dataset_name: str = "pingzhili/sft-r1-distill",
):
    response_model_name = model_csv_path.split("/")[-1].split(".csv")[0]
    logger.info(f"Running RSE Baseline for {response_model_name}")
    
    client = openai.Client(base_url=base_url, api_key="EMPTY")
    
    r1_distill_dataset = load_dataset(r1_distill_dataset_name, split="train")
    model_valid_dataset = Dataset.from_csv(model_csv_path)
    question_to_r1_distill_answer = {}

    for sample in tqdm(r1_distill_dataset, desc="Building map"):
        question_to_r1_distill_answer[sample["question"]] = sample["response"]
    
    similarity_scores = []
    for id, sample in tqdm(enumerate(model_valid_dataset), desc="Running RSE Baseline"):
        question = sample["question"]
        model_response = sample["response"]
        if question is None or model_response is None:
            continue
        r1_response = question_to_r1_distill_answer[question]
        
        # using Temperature=0.7, TopP=0.8, TopK=20, and MinP=0.
        completion = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": f"Question: {question}\nReference Answer: {r1_response}\nModel Response: {model_response}"}
            ],
            temperature=0.7,
            top_p=0.8,
        )
        response = completion.choices[0].message.content
        
        # capture the scores json dict
        try:
            scores = json.loads(response.split("```json")[1].split("```")[0].strip())
        except IndexError:
            logger.info(f"[{id}] Skipping due to IndexError: {response}")
            continue
        
        logger.info(f"[{id}] {scores}")
        similarity_scores.append(scores)
    
    # save the similarity scores
    with open(os.path.join(os.path.dirname(model_csv_path), f"{response_model_name}-rse.json"), "w") as f:
        json.dump(similarity_scores, f)
    
    # calculate the average scores
    style_scores = [score["style_score"] for score in similarity_scores]
    logical_scores = [score["logical_score"] for score in similarity_scores]
    content_scores = [score["content_score"] for score in similarity_scores]
    overview_scores = [score["overview_score"] for score in similarity_scores]
    
    avg_style_score = sum(style_scores) / len(style_scores)
    avg_logical_score = sum(logical_scores) / len(logical_scores)
    avg_content_score = sum(content_scores) / len(content_scores)
    avg_overview_score = sum(overview_scores) / len(overview_scores)
    
    logger.info(f"Average Style Score: {avg_style_score}")
    logger.info(f"Average Logical Score: {avg_logical_score}")
    logger.info(f"Average Content Score: {avg_content_score}")
    logger.info(f"Average Overview Score: {avg_overview_score}")
    
    
if __name__ == "__main__":
    Fire(run_rse_baseline)

import pandas as pd
from fire import Fire
import os


def run_collect_data(
        raw_dir: str = "data/",
        save_dir: str = "data/",
):
    df_outputs = pd.read_json(os.path.join(raw_dir, "distillation_data.jsonl"), lines=True)
    df_inputs = pd.read_json(os.path.join(raw_dir, "distillation_data_input.jsonl"), lines=True)

    # process outputs
    # response string should be row.response['body']['choices']['message']['content']
    df_outputs['response'] = df_outputs['response'].apply(lambda x: x['body']['choices']['message']['content'])
    # only keep custom_id and response columns
    df_outputs = df_outputs[['custom_id', 'response']]

    # process inputs
    df_inputs['question'] = df_inputs['messages'].apply(lambda x: x[1]['content'])
    df_inputs = df_inputs[['custom_id', 'question']]

    # merge inputs and outputs by custom_id
    df = pd.merge(df_inputs, df_outputs, on='custom_id', how='inner')

    # save the merged dataframe
    df.to_csv(os.path.join(save_dir, "sft-data-from-moonlight.csv"), index=False)


if __name__ == "__main__":
    Fire(run_collect_data)

import argparse
import json
import pandas as pd
import os
import numpy as np
from openai import OpenAI
from copy import copy
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme()
from tqdm import tqdm
tqdm.pandas()

assert "OPENAI_API_KEY" in os.environ, "Please set the OPENAI_API_KEY environment variable."

from mllm.utils.mllm_results_utils import find_model_files
# from mllm.eval.gpt4_eval_prompt_detailed import cot_prompt
from mllm.eval.gpt4_eval_prompt_detailed_only_objects import cot_prompt


def gpt_eval(prompt):
    client = OpenAI()
    response = client.beta.chat.completions.parse(
        model="gpt-4o-2024-08-06",
        # model="o1-preview",
        messages=[
            {
            "role": "user",
            "content": prompt
            }
        ],
        seed=42
    )
    return response.choices[0].message.content

def eval_two_captions(prompt, gt_caption, candidate_caption):
    prompt = prompt.replace('GROUNDTRUTH_CAPTION_HERE', gt_caption)
    prompt = prompt.replace('PREDICTED_CAPTION_HERE' , candidate_caption)
    response = gpt_eval(prompt)
    print(f"Prompt: {prompt}, Response: {response}")
    return response


def read_response_from_output_json(output_json_file):
    with open(output_json_file, 'r') as f:
        data = json.load(f)
        if isinstance(data, list):
            for entry in data:
                if entry.get("prompt", "").startswith("<image>\n describe the image"):
                    data = entry
    if isinstance(data, list):
        # If the data is still a list, return None or handle it as needed
        raise ValueError(f"No valid response found in {output_json_file}")
    return data['response']

def evaluate_image(group, layers_for_candidates=[20,5,0], layer_for_gt=40):
    """
    Evaluates an image by comparing the ground truth caption from `layer_for_gt`
    with candidate captions from `layers_for_candidates`.
    """
    # # Check if all the required layers are in the group
    # for layer in layers_for_candidates + [layer_for_gt]:
    #     if not str(layer) in group['layer_idx'].values:
    #         return [np.nan] * len(layers_for_candidates)
    
    # Sort by `layer_idx` (ensure numeric sorting)
    group = group.sort_values(by='layer_idx', key=lambda x: x.astype(int)).reset_index(drop=True)
    print(f"Evaluating image: {group['image_name'].iloc[0]} with layers: {layers_for_candidates} and gt layer: {layer_for_gt}")
    # Ground truth caption
    gt_caption = read_response_from_output_json(group[group['layer_idx'] == str(layer_for_gt)]['output_json_file'].values[0])
    
    # Candidate captions
    candidate_captions = []
    for layer in layers_for_candidates:
        # it might be that the layer is not present in the group. If so - add NaN 
        if not str(layer) in group['layer_idx'].values:
            candidate_captions.append(np.nan)
        else:
            candidate_captions.append(read_response_from_output_json(list(group[group['layer_idx'] == str(layer)]['output_json_file'].values)[0]))

    # Evaluate each candidate caption against the ground truth
    responses = []
    for candidate_caption in candidate_captions:
        if pd.isna(candidate_caption):
            # print(f"Candidate caption is NaN for layer {layer}. Skipping evaluation.")
            continue
        response = eval_two_captions(cot_prompt, gt_caption, candidate_caption)
        responses.append(response)
    
    return responses

def apply_evaluation(group, layers_for_candidates, layer_for_gt):
    """
    Apply evaluation to each image group and return GPT responses for the candidate layers.
    """
    image_name = group['image_name'].iloc[0]  # Get the image name for the current group
    # Check if the target image is present in this group (for debugging, you were using a specific image '000000005477.jpg')
    # if '000000005477.jpg' not in group['image_name'].values:
    #     # Return a DataFrame with the image name and NaN for each candidate
    #     return pd.DataFrame([[image_name] + [np.nan] * len(layers_for_candidates)], 
    #                         columns=['image_name'] + [f'cand_{i}_eval' for i in layers_for_candidates])

    # Evaluate the image for the candidate layers and ground truth layer
    response = evaluate_image(group, layers_for_candidates, layer_for_gt)
    # print(f"Image: {image_name}, Responses: {response}")
    # Return a DataFrame with the image name and evaluation responses
    return pd.DataFrame([[image_name] + response], 
                        columns=['image_name'] + [f'cand_{i}_eval' for i in layers_for_candidates])

def extract_scores_from_responses(response):
    # use gpt4 to extract the scores from the responses. First, extract recall, then etract precision
    recall_prompt = "Your task is to extract the recall score from the following text. Responding only with the score digits, and no other text. INPUT \n Now, write directly here only number, Recall: "
    precision_prompt = "Your task is to extract the precision score from the following text. Responding only with the score digits, and no other text. INPUT \n Now, write directly here only number, Precision: "
    
    recall = gpt_eval(recall_prompt.replace('INPUT', response))
    precision = gpt_eval(precision_prompt.replace('INPUT', response))
    
    # if one of these strings does not contain only digits, return np.nan
    try:
        recall = float(recall)
        precision = float(precision)
        
        #if larger than 1, divide by 100
        if recall > 1:
            recall /= 100
        if precision > 1:
            precision /= 100
        
        return recall, precision
    except:
        print(f"Error extracting scores from response: {response}")
        return np.nan, np.nan
    
        

def main(directory, model_name):
    # Define layer configurations based on the model name
    other_file_df = None
    if 'internvl' in model_name:
        layers_for_candidates = [70,60,50, 40, 30, 20, 10, -1] #without 5,0
        # layers_for_candidates = [50, 40, 30, 10, -1] #without 5,0
        layer_for_gt = 80 if not 'upto_layer' in model_name else -1
        if '_not_between_layers' in model_name:
            layers_for_candidates = [80]
            evaluate_against_model = 'internvl2-76b'
            layer_for_gt = -1
            gpt4_other_model_path = f'mllm/eval/output/gpt4eval_objects_{evaluate_against_model}.csv'
            if not os.path.exists(gpt4_other_model_path):
                print(f"File {gpt4_other_model_path} does not exist. Please run the evaluation for {evaluate_against_model} first.")
                return
            other_file_df = pd.read_csv(gpt4_other_model_path)
            # in the other_file_df we have the results for the other model in layer 80 - take them into layer -1 in the current model
            
        print("GT layer:", layer_for_gt)
    else:
        # layers_for_candidates = [25, 20, 15, 10, 5, -1]
        # layers_for_candidates = [28,24,20,16,12,8,4,-1]
        layers_for_candidates = [-1]

        layer_for_gt = 32
        if '_not_between_layers' in model_name:
            layers_for_candidates = [32]
            evaluate_against_model = 'llava-1.5-7b'
            layer_for_gt = -1
            gpt4_other_model_path = f'mllm/eval/output/gpt4eval_objects_{evaluate_against_model}.csv'
            if not os.path.exists(gpt4_other_model_path):
                print(f"File {gpt4_other_model_path} does not exist. Please run the evaluation for {evaluate_against_model} first.")
                return
            other_file_df = pd.read_csv(gpt4_other_model_path)

    # model_name = 'internvl' if 'internvl' in args.model_name else 'pixtral'
    skip = False
    raw_gpt4_output_path = f'mllm/eval/output/gpt4eval_objects_{model_name}.csv'
    #mkdir
    if not os.path.exists('mllm/eval/output'):
        os.makedirs('mllm/eval/output')
    if skip and os.path.exists(raw_gpt4_output_path):
        file_df = pd.read_csv(raw_gpt4_output_path)
    else:
        print(f"Searching for output.json files in {directory} with model name {model_name}...")
        output_json_files = find_model_files(directory, model_name, 'output.json')
        print(f"Found {len(output_json_files)} output.json files.")
        
        # Group the files by `layer_idx` and `image_name`
        output_json_files = sorted(output_json_files, key=lambda x: int(x.split('/')[-3]))
        file_data = [{'layer_idx': output_json_file.split('/')[-3], 
                    'image_name': output_json_file.split('/')[-4], 
                    'output_json_file': output_json_file} 
                    for output_json_file in output_json_files]
        file_df = pd.DataFrame(file_data, columns=['layer_idx', 'image_name', 'output_json_file'])

        # find n image_names
        n_images = 100
        image_names = file_df['image_name'].unique()[:n_images]
        # filter out only the first 5 image_names
        file_df = file_df[file_df['image_name'].isin(image_names)]
        # from other_file_df, take the results for the other model in layer 80
        if other_file_df is not None:       
            other_file_df = other_file_df[other_file_df['layer_idx'] == layers_for_candidates[0]]
            other_file_df['layer_idx'] = '-1'
            file_df = pd.concat([file_df, other_file_df], ignore_index=True)

        # Apply the evaluation function group by group
        eval_results = file_df.groupby('image_name', group_keys=False).progress_apply(
            apply_evaluation, layers_for_candidates=layers_for_candidates, layer_for_gt=layer_for_gt
        ).reset_index(drop=True)

        # Merge results back into the original DataFrame based on 'image_name'
        file_df = pd.merge(file_df, eval_results, on='image_name', how='left')
        
        #from each group, story only the row where layer_idx is layer_for_gt
        file_df = file_df[file_df['layer_idx'] == str(layer_for_gt)]
        file_df.to_csv(raw_gpt4_output_path, index=False)
    
    output_with_scores_path = f'mllm/eval/output/gpt4eval_objects_{model_name}_with_scores.csv'
    if skip and os.path.exists(output_with_scores_path):
        file_df = pd.read_csv(output_with_scores_path)
    else:
        # for each row with cand_{i}_eval which is not NaN, we call the extract_scores_from_responses function
        for l in layers_for_candidates:
            cand_col = f'cand_{l}_eval'
            #store the result in new column: cand_{i}_recall and cand_{i}_precision
            file_df[f'{cand_col}_recall'] = np.nan
            file_df[f'{cand_col}_precision'] = np.nan
            for idx, row in file_df.iterrows():
                image_name = row['image_name']
                if not pd.isna(row[cand_col]):
                    recall, precision = extract_scores_from_responses(row[cand_col])
                    print(f"Image: {image_name}, Layer: {l}, Recall: {recall}, Precision: {precision}")
                    file_df.at[idx, f'{cand_col}_recall'] = float(recall)
                    file_df.at[idx, f'{cand_col}_precision'] = float(precision)
        file_df.to_csv(f'mllm/eval/output/gpt4eval_objects_{model_name}_with_scores.csv', index=False)

    # # we need to parse the 'cand_scores' column (list of 3 scores) to 3 columns
    # print("Plotting results...")
    # fig, ax = plt.subplots(2, 1, figsize=(12, 10))
    
    # # Recall plot
    # recall_df = file_df[[f'cand_{l}_eval_recall' for l in layers_for_candidates]].astype(float)
    # recall_df.columns = layers_for_candidates
    # recall_df = recall_df.T
    # recall_df.plot(ax=ax[0], marker='o')
    # ax[0].set_title(f'{model_name} - Recall')
    # ax[0].set_xlabel('Layer Index')
    # ax[0].set_ylabel('Recall Score')
    
    # # Precision plot
    # precision_df = file_df[[f'cand_{l}_eval_precision' for l in layers_for_candidates]].astype(float)
    # precision_df.columns = layers_for_candidates
    # precision_df = precision_df.T
    # precision_df.plot(ax=ax[1], marker='o')
    # ax[1].set_title(f'{model_name} - Precision')
    # ax[1].set_xlabel('Layer Index')
    # ax[1].set_ylabel('Precision Score')
    # plt.tight_layout()
    # plt.savefig(f'mllm/eval/output/gpt4eval_{model_name}_plot.png')
    # plt.show()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Plot the distribution of attention values across layers.")
    parser.add_argument('directory', type=str, help="The root directory to search for output.json files")
    parser.add_argument('-model_name', type=str, help="Model name substring to filter directories.", default=None, required=False)
    
    args = parser.parse_args()
    directory = args.directory
    model_name = args.model_name
    if model_name is None:
        model_names=[
            # 'internvl2-76b-masked-0.01-attn_over_gen_percentage',
            # 'internvl2-76b-masked-0.05-attn_over_gen_percentage',
            # 'internvl2-76b-masked-0.1-attn_over_gen_percentage',
            # 'internvl2-76b-masked-0.2-attn_over_gen_percentage',
            # 'internvl2-76b-masked-0.3-attn_over_gen_percentage',
            # 'internvl2-76b-masked-0.4-attn_over_gen_percentage',
            # 'internvl2-76b-masked-0.5-attn_over_gen_percentage',
            # 'internvl2-76b-masked-0.9-attn_over_gen_percentage',
            # 'internvl2-76b-masked-0.01-attn_over_gen_percentage-block_vis_to_gen_not_between_layers_20_40_and_only_k_in_middle',
            # 'internvl2-76b-masked-0.02-attn_over_gen_percentage-block_vis_to_gen_not_between_layers_20_40_and_only_k_in_middle',
            # 'internvl2-76b-masked-0.05-attn_over_gen_percentage-block_vis_to_gen_not_between_layers_20_40_and_only_k_in_middle',
            # 'internvl2-76b-masked-0.1-attn_over_gen_percentage-block_vis_to_gen_not_between_layers_20_40_and_only_k_in_middle',
            # 'internvl2-76b-masked-0.2-attn_over_gen_percentage-block_vis_to_gen_not_between_layers_20_40_and_only_k_in_middle',
            # 'internvl2-76b-masked-0.3-attn_over_gen_percentage-block_vis_to_gen_not_between_layers_20_40_and_only_k_in_middle',
            # 'internvl2-76b-masked-0.4-attn_over_gen_percentage-block_vis_to_gen_not_between_layers_20_40_and_only_k_in_middle',
            # 'internvl2-76b-masked-0.5-attn_over_gen_percentage-block_vis_to_gen_not_between_layers_20_40_and_only_k_in_middle',
            # 'internvl2-76b-masked-0.9-attn_over_gen_percentage-block_vis_to_gen_not_between_layers_20_40_and_only_k_in_middle',
            
            ### TO RUN: 0.01, 0.3, 0.4, 0.9
            
        # 'internvl2-76b-masked-0.02-attn_over_gen_percentage',
        # 'internvl2-76b-masked-0.05-attn_over_gen_percentage-no_encoding-block_vis_to_all_not_between_layers_20_40_and_only_k_in_middle',
        #     'internvl2-76b','internvl2-76b-no_encoding',
        # 'internvl2-76b-no_encoding-block_vis_to_all_from_layer',
        # 'internvl2-76b-block_vis_to_out_upto_layer',
        # 'internvl2-76b-no_encoding-block_vis_to_all_not_between_layers',
        # 'internvl2-76b-no_encoding-block_vis_to_all_not_between_layers_20_40',
        # 'internvl2-76b-block_vis_to_gen_not_between_layers_20_40',
        # 'internvl2-76b-no_encoding-block_vis_to_query_not_between_layers_20_40',
        # 'internvl2-76b-no_encoding-block_vis_to_all_upto_layer',
        
        # 'internvl2-76b-no_encoding-block_vis_to_query_upto_layer',
        # 'internvl2-76b-no_encoding-block_vis_to_all_upto_layer'
        
        #              'internvl2-76b-masked-0.1-norm_percentage',
        #             'internvl2-76b-masked-0.02-norm_percentage',
        # 'internvl2-76b-masked-0.02-attn_over_gen_percentage-no_encoding'
        # 'internvl2-76b-masked-0.1-random'
        # 'internvl2-76b-masked-0.02-random',
        # 'internvl2-76b-no_encoding-block_query',
        # 'internvl2-76b-no_encoding-block_query_unidirection'
        
        
        # 'llava-1.5-7b-no_encoding-block_query',
        # 'llava-1.5-7b-masked-0.01-attn_over_gen_percentage',
        'llava-1.5-7b-masked-0.02-attn_over_gen_percentage',
        'llava-1.5-7b-masked-0.05-attn_over_gen_percentage',
        # 'llava-1.5-7b-masked-0.1-attn_over_gen_percentage',
        # 'llava-1.5-7b-masked-0.2-attn_over_gen_percentage',
        # 'llava-1.5-7b-masked-0.3-attn_over_gen_percentage',
        # 'llava-1.5-7b-masked-0.4-attn_over_gen_percentage',
        # 'llava-1.5-7b-masked-0.5-attn_over_gen_percentage',
        # 'llava-1.5-7b-masked-0.9-attn_over_gen_percentage',
        
        # 'llava-1.5-7b',
        # 'llava-1.5-7b-no_encoding-block_vis_to_all_from_layer'
        
        # 'llava-1.5-7b-block_vis_to_gen_not_between_layers_4_to_20',
        # 'llava-1.5-7b-no_encoding-block_vis_to_query_not_between_layers_4_to_20',
        # 'llava-1.5-7b-no_encoding-block_vis_to_all_not_between_layers_4_to_20',

        
                    ]
        # model_names=[
        # 'pixtral-12b-masked-0.02-attn_over_gen_percentage',
        #     'pixtral-12b','pixtral-12b-no_encoding',
        #             #  'pixtral-12b-masked-0.1-norm_percentage',
        #             # 'pixtral-12b-masked-0.02-norm_percentage'
                    # ]
    else:
        model_names = [model_name]
        
    for model_name in model_names:
        print("Evaluating model:", model_name)
        main(directory, model_name)

from collections import defaultdict
from mllm.utils.mllm_results_utils import load_data_from_attention_file
import torch
from tqdm import tqdm
import json
import os
from pathlib import Path
import argparse
import h5py
from openai import OpenAI
import os


def llm_eval(prompt):
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

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-k", type=str, default=None)
    parser.add_argument("-k_selection_method", type=str, default='attn_over_gen', help='norm, attn_over_gen, attn_over_query, attn_over_gen_percentage, norm_percentage. Default: attn_over_gen', choices=['norm', 'attn_over_gen', 'attn_over_query','attn_over_gen_percentage','norm_percentage'], required=False)
    parser.add_argument("-data_dir", type=str, default='/home/projects/talide/omrika/diffusion-motion-transfer/mllm/data_scraping/lavin_existence', required=False)
    # parser.add_argument("-results_path", type=str, default='describe_results_dashboard_new.json', required=False)
    parser.add_argument("-results_path", type=str, default='/home/projects/bagon/shared/waic-shared-projects/vlm/mme_results', required=False)
    parser.add_argument("-skip_existing", action='store_true', required=False) # default is False, means that we will overwrite the results
    parser.add_argument("-lavin_gt_file_path", type=str, default=None, required=True, help='Path to the lavin ground truth file') #/home/projects/talide/omrika/vlm-analysis/mllm/eval/mme/LaVIN/existence.txt
    args = parser.parse_args()
    
    k = float(args.k) if args.k is not None else None
    k_selection_method = args.k_selection_method
    dataset_path = args.data_dir
    results_path = args.results_path
    skip_existing = args.skip_existing
    lavin_gt_file_path = args.lavin_gt_file_path

    # results_path = 'describe_results_dashboard_new.json'
    # dataset_path = "/home/projects/talide/omrika/diffusion-motion-transfer/mllm/data_scraping/coco_images"
    print(f"k: {k}, k_selection_method: {k_selection_method}, dataset_path: {dataset_path}, results_path: {results_path}")

    # dataset contains images under it
    image_paths = [f"{dataset_path}/{image}" for image in os.listdir(dataset_path)]
    # skip non-image files
    image_paths = [image for image in image_paths if image.endswith(".jpg") or image.endswith(".png")]
    output_hidden_states = True # False!
    results = []

    if lavin_gt_file_path is not None: 
        llava_questions = defaultdict(list)
        with open(lavin_gt_file_path, 'r') as f: #.txt file, with: Image_Name + "\t" + Question + "\t" + Ground_Truth_Answer + "\t" + Your_Response + "\n"
            lavin_gt = f.readlines()
            lavin_raw = [line.strip().split("\t") for line in lavin_gt]
            lavin_gt = {line[0]: line[2] for line in lavin_raw}
            for line in lavin_raw:
                # llava_questions[line[0]].append(f"describe the image and answer, {line[1].split('?')[0]} \n") #TODO: THIS WAS NICE?
                llava_questions[line[0]].append(f"{line[1]} \n")
            # image_paths should be lavin_gt.keys(), with proper path
    else:
        lavin_gt = None

    if k is not None:
        # model_name=f'internvl2-76b-masked-{k}-{k_selection_method}'
        model_name=f'llava-1.5-7b-masked-{k}-{k_selection_method}'

        min_token_for_i2t_attn_masking = 1 # when we mask only several of vis tokens, we should have at least 1 generated token (for all KV to be filled in)
        find_top_k = True
    else:
        # model_name=f'internvl2-76b'
        model_name=f'llava-1.5-7b'

        min_token_for_i2t_attn_masking = 0
        find_top_k = False
    max_layer = '80' if 'internvl' in model_name else '32'
    for image_path in tqdm(image_paths, total=len(image_paths)):
        print(f"Processing {image_path}")
        image_name = image_path.split('/')[-1]
        
        path = f'{results_path}/{image_name}/{max_layer}/{model_name}'
            
        #we want to pad "image_name" for things like: 000000537812.jpg by padding 0s from the left to get 12 digits
        image_name_padded = image_name.split(".")[0].zfill(12) + ".jpg"
        if image_name_padded in lavin_gt:
            prompts = llava_questions[image_name_padded]
        elif image_name in lavin_gt:
            prompts = llava_questions[image_name]
        else:
            print(f"Skipping {image_name_padded}, not in lavin_gt but in the dataset")
            continue
        # prompts = f"<image>\n {lavin_questions[image_name_padded]}"
        
        for prompt in prompts:
            if path is not None:
                Path(path).mkdir(parents=True, exist_ok=True)
            output_json_path = Path(path) / f"output_cached_llm.json"
            orig_results_path = results_path.replace("processed_cached","processed")
            max_path =f'{orig_results_path}/{image_name}/{max_layer}/{model_name}'
            orig_json_path = Path(max_path) / f"output.json"
            if not orig_json_path.exists():
                raise ValueError(f"Path {orig_json_path} does not exist")
            with open(orig_json_path, 'r') as f:
                orig_json_data = json.load(f)
            # find entry with "prompt" = "<image>\n describe the image"
            response = None
            for entry in orig_json_data:
                if entry["prompt"] =="<image>\n describe the image":
                    response = entry["response"]
                    break
            if response is None:
                raise ValueError(f"Could not find 'describe the image' prompt in {orig_json_path}")
            # go to the LLM, with this response, and ask it the question from lavin_gt. 
            llm_eval_prompt = f"Given the following image description: \n {response} \n, answer the following question: {prompt}"
            llm_response = llm_eval(llm_eval_prompt)
            llm_response = llm_response.replace(prompt, "").strip()
            new_element = {"prompt": prompt,
                    "response": llm_response, 
                    "tokens": llm_response.split(" "),
                    "knockout": max_layer}
            new_data = [new_element]

            if output_json_path.exists():
                with open(output_json_path, 'r') as f:
                    json_data = json.load(f)
                    if isinstance(json_data, list):
                        # if len(json_data) >= 2:
                        #     json_data = []
                        new_data = new_data + json_data
                        # make sure we don't have duplicates. duplicate is having two same values for "prompt"
                        unique_prompts = set()
                        new_data_unique = []
                        for element in new_data:
                            if element["prompt"] not in unique_prompts:
                                unique_prompts.add(element["prompt"])
                                new_data_unique.append(element)
                        new_data = new_data_unique

                    else:
                        print("Warning: Existing JSON data is not a list. Overwriting with new data.")
            with open(output_json_path, 'w') as f:
                json.dump(new_data, f, indent=4)
            print(f"Saved to {output_json_path}")

            #run and save under output_llm.json 
            # # cp image from image_path to path/image_name
            # print(f"Copying image from {image_path} to {path}/{image_name}")
            # try:
            #     Path(f"{path}").mkdir(parents=True, exist_ok=True)
            #     os.system(f"cp {image_path} {path}/image.jpg")
            # except:
            #     print(f"Failed to copy image from {image_path} to {path}/{image_name}")
                
            # print("---------- Results ----------")
            print(f"Result for prompt: {prompt}, path: {path}", llm_response)
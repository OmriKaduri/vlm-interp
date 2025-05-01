from collections import defaultdict
from mllm.utils.mllm_results_utils import load_data_from_attention_file
import torch
from tqdm import tqdm
import json
import os
from pathlib import Path
import argparse
import h5py

from mllm.llava.llava_wrapper import LlavaWrapper

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-write_to_disk", action='store_true', required=False, help='If True, the results will be written to disk. Default: False, means that only "L=max" (32)')
    parser.add_argument("-k", type=str, default=None)
    parser.add_argument("-k_selection_method", type=str, default='attn_over_gen', help='norm, attn_over_gen, attn_over_query, attn_over_gen_percentage, norm_percentage. Default: attn_over_gen', choices=['norm', 'attn_over_gen', 'attn_over_query','attn_over_gen_percentage','norm_percentage'], required=False)
    parser.add_argument("-max_patch_num", type=int, default=7, required=False)
    parser.add_argument("-data_dir", type=str, default='/home/projects/talide/omrika/diffusion-motion-transfer/mllm/data_scraping/lavin_existence', required=False)
    parser.add_argument("-results_path", type=str, default='/home/projects/bagon/shared/waic-shared-projects/vlm/mme_results', required=False)
    parser.add_argument("-skip_existing", action='store_true', required=False) # default is False, means that we will overwrite the results
    parser.add_argument("-lavin_gt_file_path", type=str, default=None, required=False, help='Path to the lavin ground truth file') #/home/projects/talide/omrika/vlm-analysis/mllm/eval/mme/LaVIN/existence.txt
    parser.add_argument("-cache_strategy", type=str, required=False,  default='qs', help="q: query, s: sinks, qs: query+sinks")
    parser.add_argument('-use_kv_cache_before_rope', action='store_true', required=False, help='If True, we will use the kv_cache before the rope. Otherwise - use the original kv_cache - with rope') # default is False, means that we will overwrite the results
    args = parser.parse_args()
    write_to_disk = args.write_to_disk
    
    k = float(args.k) if args.k is not None else None
    k_selection_method = args.k_selection_method
    max_patch_num = int(args.max_patch_num)
    dataset_path = args.data_dir
    results_path = args.results_path
    skip_existing = args.skip_existing
    max_new_tokens = 25
    lavin_gt_file_path = args.lavin_gt_file_path
    cache_strategy = args.cache_strategy

    # print(f"write_to_disk: {write_to_disk}, k: {k}, k_selection_method: {k_selection_method}, dataset_path: {dataset_path}, results_path: {results_path}, max_patch_num: {max_patch_num}")

    # dataset contains images under it
    image_paths = [f"{dataset_path}/{image}" for image in os.listdir(dataset_path)]
    # filter only '.jpg' or '.png' images
    image_paths = [image for image in image_paths if image.endswith('.jpg') or image.endswith('.png')]
    output_hidden_states = True # False!
    results = []
    describe_prompt_text = ":"

    if lavin_gt_file_path is not None: 
        llava_questions = defaultdict(list)
        with open(lavin_gt_file_path, 'r') as f: #.txt file, with: Image_Name + "\t" + Question + "\t" + Ground_Truth_Answer + "\t" + Your_Response + "\n"
            lavin_gt = f.readlines()
            lavin_raw = [line.strip().split("\t") for line in lavin_gt]
            lavin_gt = {line[0]: line[2] for line in lavin_raw}
            for line in lavin_raw:
                llava_questions[line[0]].append(f"{line[1]} ASSISTANT:")
                celeb_name = line[1].split('named ')[-1].split('?')[0]

            # image_paths should be lavin_gt.keys(), with proper path
    else:
        lavin_gt = None
        
    llava = LlavaWrapper(model_path = "liuhaotian/llava-v1.5-7b", cache_strategy=cache_strategy)

    max_layers_for_i2t_attn = [32]


    if k is not None:
        model_name=f'llava-1.5-7b-masked-{k}-{k_selection_method}'
        min_token_for_i2t_attn_masking = 1 # when we mask only several of vis tokens, we should have at least 1 generated token (for all KV to be filled in)
        find_top_k = True
    else:
        model_name=f'llava-1.5-7b'
        min_token_for_i2t_attn_masking = 0
        find_top_k = False
        
    if max_patch_num == 1:
        model_name = f"{model_name}-single_patch"

    for image_path in tqdm(image_paths, total=len(image_paths)):
        image_name = image_path.split('/')[-1]
        
        path = f'{results_path}/{image_name}/32/{model_name}'
        curr_write_to_disk = write_to_disk
            
        orig_results_path = results_path.replace("processed_cached","processed")
        max_path =f'{orig_results_path}/{image_name}/32/{model_name}'

        kv_cache_path = f'{max_path}/kv_before_rope.h5' if args.use_kv_cache_before_rope else f'{max_path}/kv_cache.h5'
        if not os.path.exists(kv_cache_path):
            raise ValueError(f"kv_cache_path: {kv_cache_path} does not exist")
        
        # load h5 file
        with h5py.File(kv_cache_path, 'r') as f:
            keys_group = f['keys']
            values_group = f['values']
            position_ids_group = f['position_ids'] if args.use_kv_cache_before_rope else None
            keys = []
            values = []
            position_ids = [] if args.use_kv_cache_before_rope else None
            for layer_idx in range(len(keys_group)):
                keys.append(torch.tensor(keys_group[f'layer_{layer_idx}'][:]))
                values.append(torch.tensor(values_group[f'layer_{layer_idx}'][:]))
            if args.use_kv_cache_before_rope:
                position_ids.append(torch.tensor(position_ids_group[f'layer_{layer_idx}'][:])) # ONLY FOR kv_before_rope last layer
            kv_topk = (keys, values, position_ids)
            
        # load the image_token_indices, query_token_indices
        attention_file = kv_cache_path.replace('kv_cache.h5', 'attention.h5') if not args.use_kv_cache_before_rope else kv_cache_path.replace('kv_before_rope.h5', 'attention.h5')
        data = load_data_from_attention_file(attention_file, load_attentions=False)
        query_tokens_mask = data['query_tokens_mask']
        
        topk_indices_path = f'{orig_results_path}/{image_name}/{model_name}_topk_indices.json'
        with open(topk_indices_path, 'r') as f:
            topk_indices = json.load(f)
        if lavin_gt is not None:
            #we want to pad "image_name" for things like: 000000537812.jpg by padding 0s from the left to get 12 digits
            image_name_padded = image_name.split(".")[0].zfill(12) + ".jpg"
            if image_name_padded in lavin_gt:
                prompts = llava_questions[image_name_padded]
            elif image_name in lavin_gt:
                prompts = llava_questions[image_name]
            else:
                print(f"Skipping {image_name_padded}, not in lavin_gt but in the dataset")
                continue
        else:
            prompts = [describe_prompt_text]
            
        for prompt_text in tqdm(prompts, total=len(prompts), desc=f"Processing {image_name}, prompt: {prompt_text}"):
            cloned_kv_topk = (kv_topk[0].copy(), kv_topk[1].copy(), kv_topk[2])
            res, tokens, store = llava.predict_from_cache(prompt_text, image_path, output_path=path,
                                                            write_to_disk=curr_write_to_disk, 
                                                            kv=[cloned_kv_topk],
                                                            cache_strategy=cache_strategy,
                                                            query_tokens_mask=[query_tokens_mask]
                                                            , topk_indices=[topk_indices]
                                                            )
            
            
            # cp image from image_path to path/image_name
            # print(f"Copying image from {image_path} to {path}/{image_name}")
            try:
                Path(f"{path}").mkdir(parents=True, exist_ok=True)
                os.system(f"cp {image_path} {path}/image.jpg")
            except:
                # print(f"Failed to copy image from {image_path} to {path}/{image_name}")
                pass
                
            # print("---------- Results ----------")
            # print(f"Result for prompt: {prompt_text} \n:: path: {path},\n:: max_layer_for_i2t_attn: 32", res)
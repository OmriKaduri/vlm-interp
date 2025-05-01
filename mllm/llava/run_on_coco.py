from collections import defaultdict
from mllm.llava.llava_wrapper import LlavaWrapper
from tqdm import tqdm
import json
import os
from pathlib import Path
import argparse
    
if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-write_to_disk", action='store_true', required=False, help='If True, the results will be written to disk. Default: False, means that only "L=max" (40 or 80 for pixtral/internvl) tokens will be saved')
    parser.add_argument("-k", type=str, default=None)
    parser.add_argument("-k_selection_method", type=str, default='attn_over_gen', help='norm, attn_over_gen, attn_over_query, attn_over_gen_percentage, norm_percentage. Default: attn_over_gen', choices=['norm', 'attn_over_gen', 'attn_over_query','attn_over_gen_percentage','norm_percentage','random'], required=False)
    parser.add_argument("-max_patch_num", type=int, default=7, required=False)
    parser.add_argument("-data_dir", type=str, default='/home/projects/talide/omrika/diffusion-motion-transfer/mllm/data_scraping/coco_images', required=False)
    parser.add_argument("-results_path", type=str, default='/home/projects/bagon/shared/waic-shared-projects/vlm/processed', required=False)
    parser.add_argument("-skip_existing", action='store_true', required=False) # default is False, means that we will overwrite the results
    parser.add_argument("-min_token_for_i2t_attn_masking", type=int, default=1, required=False, help="When we mask only several of vis tokens, do it after at least min_token_for_i2t_attn_masking tokens were generated")
    parser.add_argument("-block_query", action='store_true', required=False, help='If True, the attention between visual tokens to query tokens (Both ways) will be blocked, from layer max_layer_for_i2t_attn to the end')
    parser.add_argument("-block_query_unidirection", action='store_true', required=False, help='If True, the attention between visual tokens to query tokens will be blocked, from layer max_layer_for_i2t_attn to the end')
    parser.add_argument("-lavin_gt_file_path", type=str, default=None, required=False, help='Path to the lavin ground truth file') #/home/projects/talide/omrika/vlm-analysis/mllm/eval/mme/LaVIN/existence.txt
    parser.add_argument("-block_vis_to_all_from_layer",action='store_true', required=False, help='If True, the attention between visual tokens to all tokens will be blocked, from layer max_layer_for_i2t_attn to the end')
    parser.add_argument("-block_vis_to_gen_not_between_layers_4_to_20", action='store_true', required=False, help='If True, the attention between visual tokens to generated tokens will be blocked, from layer 4 to 20')
    parser.add_argument("-block_vis_to_query_not_between_layers_4_to_20", action='store_true', required=False, help='If True, the attention between visual tokens to query tokens will be blocked, from layer 4 to 20')
    parser.add_argument("-block_vis_to_all_not_between_layers_4_to_20", action='store_true', required=False, help='If True, the attention between visual tokens to all tokens will be blocked, from layer 4 to 20')
    parser.add_argument("-fastv_ratio", type=float, default=None, required=False, help='The ratio of tokens to keep after fastv_laye')
    parser.add_argument("-fastv_layer", type=int, default=None, required=False, help='The layer to prune tokens after')

    args = parser.parse_args()
    
    write_to_disk = args.write_to_disk
    
    k = float(args.k) if args.k is not None else None
    k_selection_method = args.k_selection_method if args.k is not None else None
    max_patch_num = int(args.max_patch_num)
    dataset_path = args.data_dir
    results_path = args.results_path
    skip_existing = args.skip_existing
    block_query = args.block_query
    block_query_unidirection = args.block_query_unidirection
    max_new_tokens = 500
    lavin_gt_file_path = args.lavin_gt_file_path
    block_vis_to_all_from_layer = args.block_vis_to_all_from_layer
    block_vis_to_gen_not_between_layers_4_to_20 = args.block_vis_to_gen_not_between_layers_4_to_20
    block_vis_to_query_not_between_layers_4_to_20 = args.block_vis_to_query_not_between_layers_4_to_20
    block_vis_to_all_not_between_layers_4_to_20 = args.block_vis_to_all_not_between_layers_4_to_20
    fastv_ratio = args.fastv_ratio
    fastv_layer = args.fastv_layer

    # print(f"write_to_disk: {write_to_disk}, k: {k}, k_selection_method: {k_selection_method}, dataset_path: {dataset_path}, results_path: {results_path}, max_patch_num: {max_patch_num}")

    image_paths = [f"{dataset_path}/{image}" for image in os.listdir(dataset_path)]
    # remove non-image files
    image_paths = [image for image in image_paths if image.endswith('.jpg') or image.endswith('.png')]
    
    output_hidden_states = True # False!
    llava = LlavaWrapper(model_path = "liuhaotian/llava-v1.5-7b")

    results = []
    describe_prompt_text = "<image>\n describe the image"
    
    if lavin_gt_file_path is not None: 
        llava_questions = defaultdict(list)
        with open(lavin_gt_file_path, 'r') as f: #.txt file, with: Image_Name + "\t" + Question + "\t" + Ground_Truth_Answer + "\t" + Your_Response + "\n"
            lavin_gt = f.readlines()
            lavin_raw = [line.strip().split("\t") for line in lavin_gt]
            lavin_gt = {line[0]: line[2] for line in lavin_raw}
            for line in lavin_raw:
                llava_questions[line[0]].append(line[1])
    else:
        lavin_gt = None
            
    max_layers_for_i2t_attn = [32, -1]


    if k is not None:
        model_name=f'llava-1.5-7b-masked-{k}-{k_selection_method}'
        find_top_k = True
    else:
        model_name=f'llava-1.5-7b'
        find_top_k = False
    
    min_token_for_i2t_attn_masking = int(args.min_token_for_i2t_attn_masking)
    
    if max_patch_num == 1:
        model_name = f"{model_name}-single_patch"

    if min_token_for_i2t_attn_masking == 0:
        model_name = f"{model_name}-no_encoding"

    if block_query:
        if min_token_for_i2t_attn_masking != 0:
            raise ValueError("block_query must set with min_token_for_i2t_attn_masking=0")
        model_name = f"{model_name}-block_query"
        
    if block_query_unidirection:
        if block_query:
            raise ValueError("block_query_unidirection and block_query cannot be set together")
        if min_token_for_i2t_attn_masking != 0:
            raise ValueError("block_query_unidirection must set with min_token_for_i2t_attn_masking=0")
        model_name = f"{model_name}-block_query_unidirection"
        
    if block_vis_to_all_from_layer:
        model_name = f"{model_name}-block_vis_to_all_from_layer"
        if min_token_for_i2t_attn_masking != 0:
            raise ValueError("block_vis_to_all_from_layer must set with min_token_for_i2t_attn_masking=0")

    if block_vis_to_gen_not_between_layers_4_to_20:
        model_name = f"{model_name}-block_vis_to_gen_not_between_layers_4_to_20"
        if min_token_for_i2t_attn_masking == 0:
            raise ValueError("block_vis_to_gen_not_between_layers_4_to_20 must set with min_token_for_i2t_attn_masking!=0")
        max_layers_for_i2t_attn = [32]
        
    if block_vis_to_query_not_between_layers_4_to_20:
        model_name = f"{model_name}-block_vis_to_query_not_between_layers_4_to_20"
        if min_token_for_i2t_attn_masking != 0:
            raise ValueError("block_vis_to_query_not_between_layers_4_to_20 must set with min_token_for_i2t_attn_masking=0")
        max_layers_for_i2t_attn = [32]
        
    if block_vis_to_all_not_between_layers_4_to_20:
        model_name = f"{model_name}-block_vis_to_all_not_between_layers_4_to_20"
        if min_token_for_i2t_attn_masking != 0:
            raise ValueError("block_vis_to_all_not_between_layers_4_to_20 must set with min_token_for_i2t_attn_masking=0")
        max_layers_for_i2t_attn = [32]
        
    if fastv_ratio is not None and fastv_layer is not None:
        model_name = f"{model_name}-fastv-{fastv_layer}-{fastv_ratio}"
        max_layers_for_i2t_attn = [32]

    for image_path in tqdm(image_paths, total=len(image_paths)):
        image_name = image_path.split('/')[-1]

        if find_top_k:
            # print(f"----------------- Finding top k indices for image: {image_path}, k: {k}, selection_method: {k_selection_method}")
            if 'single_patch' in model_name:
                topk_indices_path = f'{results_path}/{image_name}/{model_name}_topk_indices_single_patch.json'
            else:
                topk_indices_path = f'{results_path}/{image_name}/{model_name}_topk_indices.json'
            Path(f"{results_path}/{image_name}").mkdir(parents=True, exist_ok=True)    
            if skip_existing and os.path.exists(topk_indices_path):
                with open(topk_indices_path, 'r') as f:
                    topk_indices = json.load(f)
            else:
                path = f'{results_path}/{image_name}/32/{model_name}'
                res, tokens, store = llava.predict_for_token(describe_prompt_text, image_path,
                                                                write_to_disk=True,
                                                                output_path=path,
                                                                max_layer_for_i2t_attn=32)                 

                topk_indices = store.get_top_k_indices(k, k_selection_method)
                store.write_kv_before_rope(path)
                store.write_kv(path, topk_indices)
                
                # save topk_indices to disk
                with open(topk_indices_path, 'w') as f:
                    topk_indices = [topk_index.tolist() for topk_index in topk_indices]
                    json.dump(topk_indices, f)
        else:
            topk_indices = None
            
        for max_layer_for_i2t_attn in max_layers_for_i2t_attn:
            if lavin_gt is not None:
                image_name_padded = image_name.split(".")[0].zfill(12) + ".jpg"
                if image_name_padded in lavin_gt:
                    prompts = llava_questions[image_name_padded]
                elif image_name in lavin_gt:
                    prompts = llava_questions[image_name]
                else:
                    print(f"Skipping {image_name_padded}, not in lavin_gt but in the dataset")
                    continue
                # add describe prompt to the list of prompts
                prompts = [describe_prompt_text] + prompts
            else:
                prompts = [describe_prompt_text]
                
            for prompt_text in prompts:
                print(f"Processing image: {image_path}, max_layer_for_i2t_attn: {max_layer_for_i2t_attn}, prompt: {prompt_text}")
                max_layer_for_i2t_attn_str = str(max_layer_for_i2t_attn)
                path = f'{results_path}/{image_name}/{max_layer_for_i2t_attn_str}/{model_name}'
                if max_layer_for_i2t_attn == 32:
                    curr_write_to_disk = True
                else:
                    curr_write_to_disk = write_to_disk
                if skip_existing and os.path.exists(f"{path}/output.json"):
                    print("Skipping", prompt_text, "for", image_name, ", already exists, at layer ", max_layer_for_i2t_attn, "for path", path)
                    with open(f"{path}/output.json", 'r') as f:
                        output = json.load(f)
                        if any([element['prompt'] == prompt_text for element in output]):
                            print(f"Skipping {prompt_text} for {image_name}, already exists")
                            continue
                            
                
                res, tokens, store = llava.predict_for_token(prompt_text, image_path, output_path=path,
                                                                max_layer_for_i2t_attn=max_layer_for_i2t_attn,
                                                                topk_indices=topk_indices,
                                                                write_to_disk=True,
                                                                min_token_for_i2t_attn_masking=min_token_for_i2t_attn_masking,
                                                                block_query=block_query,
                                                                block_query_unidirection=block_query_unidirection,
                                                                block_vis_to_all_from_layer=block_vis_to_all_from_layer,
                                                                block_vis_to_gen_not_between_layers_4_to_20=block_vis_to_gen_not_between_layers_4_to_20,
                                                                block_vis_to_query_not_between_layers_4_to_20=block_vis_to_query_not_between_layers_4_to_20,
                                                                block_vis_to_all_not_between_layers_4_to_20=block_vis_to_all_not_between_layers_4_to_20,
                                                                fastv_ratio=fastv_ratio,
                                                                fastv_layer=fastv_layer)
                # cp image from image_path to path/image_name
                try:
                    Path(f"{path}").mkdir(parents=True, exist_ok=True)
                    os.system(f"cp {image_path} {path}/image.jpg")
                except:
                    print(f"Failed to copy image from {image_path} to {path}/{image_name}")
                    
                # print("---------- Results ----------")
                print(f"Result for prompt: {prompt_text}", res, f" Saving in path: {path}")
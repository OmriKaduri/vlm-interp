import argparse
from collections import defaultdict
import os
import matplotlib.pyplot as plt
from tqdm import tqdm
from mllm.utils.mllm_results_utils import find_model_files, load_data_from_attention_file
import numpy as np
from matplotlib.font_manager import FontProperties
from matplotlib import rc
plt.rcParams['mathtext.fontset'] = 'cm'
# rc('text', usetex=True)
# rc('font', **{'family':'serif', 'serif':['Times']})

font_path = '/home/projects/talide/omrika/vlm-analysis/Times New Roman.ttf'
font_prop = FontProperties(fname=font_path)

bold_font_path = '/home/projects/talide/omrika/vlm-analysis/Times New Roman Bold.ttf'
bold_font_prop = FontProperties(fname=bold_font_path)

def attentions_across_files(attention_files, max_layer):
    """
    Average attentions across multiple files.
    
    Args:
        attention_files (list): List of paths to 'attention.h5' files.
        
    Returns:
        averaged_attentions (dict): Averaged attentions per layer.
    """
    accumulated_vis_attentions = defaultdict(list)
    accumulated_query_attentions = defaultdict(list)
    accumulated_generated_attentions = defaultdict(list)

    for file_path in tqdm(attention_files, desc="Processing files"):
        data = load_data_from_attention_file(file_path)
        attentions = data['attentions']
        image_tokens_mask = data['image_tokens_mask']
        query_tokens_indices = data['query_tokens_mask']
        image_token_indices = np.where(image_tokens_mask)[0]
        for layer_idx in range(int(max_layer)):
            layer_attentions = attentions[layer_idx] # (#GeneratedToken, #Heads, AllTokens_up_to_#GeneratedToken)
            vis_layer_attentions = np.array([layer_attentions[..., image_token_indices].mean(0) for layer_attentions in layer_attentions]) # (#GeneratedToken, #Vistokens)
            query_layer_attentions = np.array([layer_attentions[..., query_tokens_indices].mean(0) for layer_attentions in layer_attentions]) # (#GeneratedToken, #QueryTokens)
            max_token_idx = max(image_token_indices.max(), query_tokens_indices.max())
            generated_layer_attentions = [layer_attentions[..., max_token_idx+1:].mean(0) for layer_attentions in layer_attentions] # (#GeneratedToken, #GeneratedTokens)
            # generated_token_indices, will be all indices that are not image or query tokens
            # generated_token_indices = np.array([i for i in range(layer_attentions.shape[-1]) if i not in image_token_indices and i not in query_token_indices])
            accumulated_vis_attentions[layer_idx].extend(vis_layer_attentions.sum(-1)) # sum of attention across all vis tokens - array of size: (#GeneratedToken)
            accumulated_query_attentions[layer_idx].extend(query_layer_attentions.sum(-1)) # sum of attention across all query tokens - array of size: (#GeneratedToken)
            accumulated_generated_attentions[layer_idx].extend([gen_layer_attention.sum(-1) for gen_layer_attention in generated_layer_attentions]) # sum of attention across all generated tokens - array of size: (#GeneratedToken)
            # extend the list of attentions for the layer
    
    accumulated_vis_attentions = {layer_idx: np.array(avg_attn_values) for layer_idx, avg_attn_values in accumulated_vis_attentions.items()}
    accumulated_query_attentions = {layer_idx: np.array(avg_attn_values) for layer_idx, avg_attn_values in accumulated_query_attentions.items()}
    accumulated_generated_attentions = {layer_idx: np.array(avg_attn_values) for layer_idx, avg_attn_values in accumulated_generated_attentions.items()}
    return accumulated_vis_attentions, accumulated_query_attentions, accumulated_generated_attentions

def main(args):
    layer_filter = '80' if 'internvl' in args.model_name else '32'
    attn_by_type_file_path = f"mllm/visualizations/output/{args.model_name}_attns_by_type.npy"
    # if os.path.exists(attn_by_type_file_path):
    #     vis_attns, query_attn, gen_attns = np.load(attn_by_type_file_path, allow_pickle=True)
    #     print("Found saved attention values, loading...")
    # else:
    print(f"Searching for attention files in {args.directory} with model name {args.model_name}...")
    attention_files = find_model_files(args.directory, args.model_name, 'attention.h5', layer_filter=layer_filter)

    print(f"Found {len(attention_files)} attention files. computing attentions...")
    vis_attns, query_attn, gen_attns = attentions_across_files(attention_files, layer_filter)
    np.save(attn_by_type_file_path, (vis_attns, query_attn, gen_attns))
        
    # vis_model_name = 'internvl' if 'internvl' in args.model_name else 'pixtral'
    font_size = 26
    print("Done extracting mean attns per token type, plotting...")
    plt.figure(figsize=(12, 6))
    # save to file
    # compute, for each layer, the mean and std across all tokens
    vis_means = [vis_attns[layer_idx].mean() for layer_idx in vis_attns.keys()]
    vis_stds = [vis_attns[layer_idx].std() for layer_idx in vis_attns.keys()]
    query_means = [query_attn[layer_idx].mean() for layer_idx in query_attn.keys()]
    query_stds = [query_attn[layer_idx].std() for layer_idx in query_attn.keys()]
    gen_means = [gen_attns[layer_idx].mean() for layer_idx in gen_attns.keys()]
    gen_stds = [gen_attns[layer_idx].std() for layer_idx in gen_attns.keys()]
    # plot with fill_between
    font_prop.set_size(20)
    bold_font_prop.set_size(23)

    # plt.rc('text', usetex=True)
    plt.plot(vis_means, label=r'$\mathbf{a}_\text{img} \, (\text{Image})$')
    plt.fill_between(vis_attns.keys(), np.array(vis_means)-np.array(vis_stds), np.array(vis_means)+np.array(vis_stds), alpha=0.3)
    plt.plot(query_means, label=r'$\mathbf{a}_\text{txt} \, \; (\text{Query Text})$')
    plt.fill_between(query_attn.keys(), np.array(query_means)-np.array(query_stds), np.array(query_means)+np.array(query_stds), alpha=0.3)
    plt.plot(gen_means,  label=r'$\mathbf{a}_\text{gen} \, (\text{Generated})$')
    plt.fill_between(gen_attns.keys(), np.array(gen_means)-np.array(gen_stds), np.array(gen_means)+np.array(gen_stds), alpha=0.3)
    plt.xlabel('Layer', fontsize=28,font=bold_font_prop)
    plt.ylabel('Attention Fraction (%)', fontsize=30, font=bold_font_prop)
    # plt.title(f'Mean Attention Value by Token Type for {vis_model_name}')
    plt.ylim(0, 1)
    plt.xlim(0,int(layer_filter)-1)
    #size of ticks
    plt.xticks( font=font_prop, fontsize=25)
    plt.yticks(font=font_prop, fontsize=25)
    
    plt.grid()
    # font_prop['size'] = 20
    plt.legend(loc=(0.52,0.34), prop={'size': 26})
    # plt.legend(loc='upper right')  # You can change this to other values like 'lower left', 'best', etc.

    plt.tight_layout()

    #save to file
    plt.savefig(f"mllm/visualizations/output/{args.model_name}_attns_by_type.pdf", format='pdf', dpi=300)
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Plot the distribution of attention values across layers.")
    parser.add_argument('directory', type=str, help="The root directory to search for .h5 files.")
    parser.add_argument('model_name', type=str, help="Model name substring to filter directories.")
    
    args = parser.parse_args()
    
    main(args)

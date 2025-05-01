
import torch
import os
import cv2
import numpy as np
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import time
import matplotlib.pyplot as plt
import h5py
from tqdm import tqdm

def normalize_tensor(tensor):
    return (tensor - tensor.min()) / (tensor.max() - tensor.min() + 1e-8)

def save_attn_on_concept_graph(attn_on_concept_per_layer, token_dir, file_name='attn_on_concept_per_layer', query_attn = None, generated_attn = None):
    Path(token_dir).mkdir(parents=True, exist_ok=True)
    # Create the plot
    plt.figure(figsize=(10, 5))
    plt.plot(attn_on_concept_per_layer, label='Visual')
    if query_attn is not None:
        plt.plot(query_attn, label='Query')
    if generated_attn is not None:
        plt.plot(generated_attn, label='Generated')
    plt.xlabel('Layer')
    plt.ylabel('Attention on the concept')
    # xticks every 5 layers
    plt.xticks(np.arange(0, len(attn_on_concept_per_layer), 5))
    plt.ylim(0, 1)
    plt.title('Attention on the concept per layer')
    if query_attn is not None or generated_attn is not None:
        plt.legend()
    plt.tight_layout(pad=2.0)  # Adjust padding to speed up layout optimization
    
    # Save the plot as an image file
    plt.savefig(f'{token_dir}/{file_name}.jpg', bbox_inches='tight')
    plt.close()
    
    np.savez_compressed(f'{token_dir}/{file_name}.npz', attn_on_concept_per_layer=attn_on_concept_per_layer, 
                    query_attn=query_attn, generated_attn=generated_attn)

# Function to save attention maps asynchronously
def save_attention_map(token_attention_avg, avg_over_heads_file, mean_attn_on_concept, pixel_values, overwrite=False, decoded_token='NONE'):
    token_attention_avg = torch.nn.functional.interpolate(token_attention_avg[None].to(torch.float32), size=(pixel_values.shape[-2], pixel_values.shape[-1]), mode='bilinear', align_corners=False)
    # overlay the attention map on the image
    token_attention_avg = token_attention_avg.to(torch.float32).squeeze()
    if len(token_attention_avg.shape) == 2:
        token_attention_avg = token_attention_avg.unsqueeze(0)
    elif len(token_attention_avg.shape) == 4:
        token_attention_avg = token_attention_avg.squeeze(0)
    b,h,w = token_attention_avg.shape
    # reshape to H, B*W
    token_attention_avg = token_attention_avg.permute(1,0,2).reshape(h,-1).squeeze().cpu().numpy()
    token_attention_avg = cv2.applyColorMap((token_attention_avg * 255).astype(np.uint8), cv2.COLORMAP_JET)
    if len(pixel_values.shape) != 4: # if we have a single image
        pixel_values = pixel_values.unsqueeze(0)
    token_attention_avg = cv2.addWeighted((pixel_values.permute(2,0,3,1).reshape(h,b*w, -1).to(torch.float32).cpu().numpy()*255).astype(np.uint8), 0.5, token_attention_avg, 0.5, 0)
    # Write attn_on_concept on the image
    cv2.putText(token_attention_avg, f'{mean_attn_on_concept:.2f}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.imwrite(avg_over_heads_file, token_attention_avg)
    
    # save file, with no content, at the same level as the attention map. file name: {decoded_token}.txt
    token_file_path =Path(avg_over_heads_file).parent / f'{decoded_token}.txt'
    with open(token_file_path, 'w') as f:
        f.write('')


class TokenAttentionStore():
    def __init__(self, token_idx=0, dummy=False):
        self.attentions = [] # each generation step has a list of attentions per layer
        self.keys = [] # each generation step has a list of keys per layer
        self.values = [] # each generation step has a list of queries per layer
        self.keys_before_rope = [] # each generation step has a list of keys per layer
        self.values_before_rope = []
        self.position_ids = [] # each generation step has a list of position_ids per layer
        self.token_scores = [] # each generation step, the softmax over vocab
        self.hidden_states = None
        self.image_tokens_mask = None # each image has a mask of image tokens
        self.filter_image_tokens_mask = None
        self.query_tokens_mask = None
        self.dummy = dummy # if True, we don't store anything
        
    def __call__(self, attention=None):
        # shape: list of N_layers, with each: (1, Heads, seq, seq)
        # we want to store: (1, heads, 1, seq) with last position for each layer, and move everything to cpu
        if attention is None or self.dummy:
            return
        attention_cpu = [layer.to('cpu')[:, :, -1, :][:,:,None] for layer in attention]
        self.attentions.append(attention_cpu)

    def extract_maps_per_generated_token(self, tokens, tokenizer, pixel_values, path, patch_size=256,
                                         image_res_x=None, image_res_y=None, overwrite=True, num_patches_list=None):
        if len(tokens.shape) == 0:
            return
        # print(f"Extracting attention maps for {len(tokens)} tokens")
        
        # Prepare the output path
        output_path = path
        #vis_path: under "visualization"
        vis_path = f'{path}/visualizations'
        os.makedirs(vis_path, exist_ok=True)
        os.makedirs(output_path, exist_ok=True)
        
        pixel_values_normalized = normalize_tensor(pixel_values)
        # save pixel_values to file: pixel_values.jpg under "visualization"
        if len(pixel_values_normalized.shape) == 3:
            c,h,w = pixel_values_normalized.shape
            vis_pixel_values = pixel_values_normalized.permute(1,2,0).reshape(h,w,c)
        else:
            n_p,c,h,w = pixel_values_normalized.shape
            vis_pixel_values = pixel_values_normalized.permute(2,0,3,1).reshape(h, n_p*w, c)
        cv2.imwrite(f'{output_path}/processed_image.jpg', cv2.cvtColor(vis_pixel_values.cpu().to(torch.float32).numpy()*255, cv2.COLOR_RGB2BGR))

        single_patch = True if image_res_x is not None and image_res_y is not None else False
        
        # Retrieve image tokens mask and query tokens mask if available
        image_token_indices_cpu = self.image_tokens_mask.nonzero().squeeze().to('cpu')
        query_tokens_indices_cpu = self.query_tokens_mask.nonzero().squeeze().to('cpu')
        filtered_image_token_indices_cpu = self.filter_image_tokens_mask if hasattr(self, 'filter_image_tokens_mask') and self.filter_image_tokens_mask is not None else image_token_indices_cpu # if we don't have filter_image_tokens_mask, we use the image_tokens_mask
        filtered_image_token_indices_cpu = filtered_image_token_indices_cpu.nonzero().squeeze().to('cpu')

        if not single_patch:
            n_vis_tokens = len(filtered_image_token_indices_cpu)
            n_patches = n_vis_tokens // patch_size # 729 or 256 - llava vs internvl
            n_new_lines = n_vis_tokens % patch_size
            image_res_x = image_res_y = int(np.sqrt(n_vis_tokens // n_patches))
        else:
            n_new_lines = 1
            n_patches = 1

        
        executor = ThreadPoolExecutor()
        # Save all data into an h5 file
        h5_file = f'{output_path}/attention.h5'
        # if file exists, remove it
        if os.path.exists(h5_file) and overwrite:
            os.remove(h5_file)
        with h5py.File(h5_file, 'w') as hf:
            # Save the image, filtered, and query token masks as standard datasets
            hf.create_dataset("image_tokens_mask", data=self.image_tokens_mask.cpu().numpy())
            if filtered_image_token_indices_cpu is not None:
                hf.create_dataset("filtered_image_tokens_mask", data=filtered_image_token_indices_cpu.cpu().numpy())
            if query_tokens_indices_cpu is not None:
                hf.create_dataset("query_tokens_mask", data=query_tokens_indices_cpu.cpu().numpy())
            hf.create_dataset("n_patches", data=n_patches)
            hf.create_dataset("n_new_lines", data=n_new_lines)
            hf.create_dataset("image_res_x", data=image_res_x)
            hf.create_dataset("image_res_y", data=image_res_y)
            if num_patches_list is not None:
                hf.create_dataset("num_patches_list", data=num_patches_list)
            
            # Save the variable-length attentions data as individual datasets
            attentions_group = hf.create_group("attentions")
            for token_idx, token in enumerate(tokens):
                # Create a sub-group for each token
                token_group = attentions_group.create_group(f"token_{token_idx}")
                generation_attns = self.attentions[token_idx]
                decoded_token = tokenizer.decode(token)
                # mksure decoded token does not have / or \ in it. If so - replace with "slash"
                decoded_token = decoded_token.replace('/', 'slash').replace('\\', 'slash')

                token_attentions = []
                for layer_idx, layer_attention in enumerate(generation_attns):
                    token_attention = layer_attention[0, :, -1].clone()
                    token_group.create_dataset(f"layer_{layer_idx}", data=token_attention.cpu().to(torch.float32).numpy())

                    vis_token_attention = token_attention[:, image_token_indices_cpu].clone()
                    visual_mean_attn_on_concept = vis_token_attention.sum(dim=1).mean()
                    
                    token_vis_path = f'{vis_path}/{token_idx}/{layer_idx}.jpg'
                    Path(token_vis_path).parent.mkdir(parents=True, exist_ok=True)
                    if n_new_lines > 0:
                        # reshape the token_attention tensor to (n_patches, image_res_x+1, image_res_y)
                        vis_token_attention = vis_token_attention.reshape(token_attention.shape[0], n_patches, image_res_x, image_res_y+1)
                    else:
                        vis_token_attention = vis_token_attention.reshape(token_attention.shape[0], n_patches, image_res_x, image_res_y)
                    
                    if vis_token_attention.max() == 0 and vis_token_attention.min() == 0:
                        vis_token_attention_avg = vis_token_attention[0]
                    else:
                        vis_token_attention_avg = normalize_tensor(vis_token_attention.mean(dim=0))
                    # executor.submit(save_attention_map, vis_token_attention_avg.clone(), token_vis_path, visual_mean_attn_on_concept.item(), pixel_values_normalized, overwrite=overwrite, decoded_token=decoded_token)

        executor.shutdown()
        # print(f"Saved all attention maps and masks to {h5_file}")
        
    def set_hidden_states(self, hidden_states=None):
        if hidden_states is not None and self.hidden_states is None and not self.dummy:
            # shape: list of N_layers, with each: (1, seq, hidden_size)
            hidden_states_cpu = [layer.to('cpu') for layer in hidden_states]
            self.hidden_states = hidden_states_cpu # we save hidden_states only for the first token
    
    def set_token_scores(self, token_scores=None):
        if token_scores is not None and not self.dummy:
            self.token_scores.append(token_scores)
            
    def write_token_scores(self, path):
        if len(self.token_scores) == 0:
            return
        # print(f"Writing token scores to {path}")
        os.makedirs(path, exist_ok=True)
        scores_path = f'{path}/token_scores.h5'
        with h5py.File(scores_path, 'w') as hf:
            # stack as tensor of shape: N_tokens, vocab_size and write it as single 
            token_scores = torch.stack(self.token_scores, dim=0).squeeze()
            hf.create_dataset("token_scores", data=token_scores.to(torch.float32).cpu().numpy())
        # print(f"Saved token scores to {scores_path}") 
    
    # def set_scores(self, scores=None):
    #     # This is scores for a specific generated tokens. Same behavior as attentions in __call__
    #     if scores is not None:
    #         scores_cpu = [layer.to('cpu')[:, :, -1, :][:,:,None] for layer in scores]
    #         self.scores.append(scores_cpu)
    
    # def write_scores(self, path):
    #     # similar to how to write attentions.h5, but for scores
    #     if len(self.scores) == 0:
    #         return
    #     print(f"Writing scores to {path}")
    #     os.makedirs(path, exist_ok=True)
    #     scores_path = f'{path}/scores.h5'
    #     with h5py.File(scores_path, 'w') as hf:
    #         scores_group = hf.create_group("scores")
    #         for token_idx, token_scores in enumerate(self.scores):
    #             token_group = scores_group.create_group(f"token_{token_idx}")
    #             for layer_idx, layer_scores in enumerate(token_scores):
    #                 token_group.create_dataset(f"layer_{layer_idx}", data=layer_scores[0].cpu().to(torch.float32).numpy())
                    
    #     print(f"Saved scores to {scores_path}")
            
    def set_kv(self, keys=None, values=None):
        '''called with: keys: list of size: L (80 layers), each shape: (B,H,seq,Head_dim)'''
        if keys is not None and values is not None and not self.dummy:
            self.keys = keys
            self.values = values
            
    def set_kv_before_rope(self, keys=None, values=None, position_ids=None,n_layers=80):
        '''this function is called per each token, per each attention layer, with: (B,H,seq,Head_dim)'''
        '''So we aggregate all the keys and values per token, and store them in the list'''
        if keys is not None and values is not None and not self.dummy:
            if len(self.keys_before_rope) == 0:
                self.keys_before_rope = [keys]
                self.values_before_rope = [values]
                self.position_ids = position_ids
            elif len(self.keys_before_rope) < n_layers:
                self.keys_before_rope.append(keys)
                self.values_before_rope.append(values)
        
    def write_kv(self, path, topk_indices=None):
        if len(self.keys) == 0 or len(self.values) == 0:
            return
        # print(f"Writing keys and values to {path}")
        os.makedirs(path, exist_ok=True)
        kv_path = f'{path}/kv_cache.h5'
        with h5py.File(kv_path, 'w') as hf:
            #each layer, we have a tensor of shape: (B,H,seq,Head_dim), where seq is different shape, so we save as a list
            keys_group = hf.create_group("keys")
            values_group = hf.create_group("values")
            for layer_idx, (key, value) in enumerate(zip(self.keys, self.values)):
                if key.dtype == torch.bfloat16: # for internvl
                    key = key.to(torch.float16)
                    value = value.to(torch.float16)
                keys_group.create_dataset(f"layer_{layer_idx}", data=key.cpu().numpy())
                values_group.create_dataset(f"layer_{layer_idx}", data=value.cpu().numpy())
                
        #if topk_indices is not None, we save another file: kv_topk_indices.h5. This is a file with only keys and values at the top_k_indices
        if topk_indices is not None:
            kv_topk_path = f'{path}/kv_topk_indices.h5'
            with h5py.File(kv_topk_path, 'w') as hf:
                keys_group = hf.create_group("keys")
                values_group = hf.create_group("values")
                for layer_idx, (key, value) in enumerate(zip(self.keys, self.values)):
                    if key.dtype == torch.bfloat16:
                        key = key.to(torch.float16)
                        value = value.to(torch.float16)
                    keys_group.create_dataset(f"layer_{layer_idx}", data=key.cpu().numpy()[:, :, topk_indices[layer_idx], :])
                    values_group.create_dataset(f"layer_{layer_idx}", data=value.cpu().numpy()[:, :, topk_indices[layer_idx], :])
            
        # print(f"Saved keys and values to {kv_path}")
        
    def write_kv_before_rope(self, path):
        if len(self.keys_before_rope) == 0 or len(self.values_before_rope) == 0:
            return
        # print(f"Writing keys and values before rope to {path}")
        os.makedirs(path, exist_ok=True)
        kv_path = f'{path}/kv_before_rope.h5'
        kv_path_raw = f'{path}/kv_cache.h5'
        with h5py.File(kv_path, 'w') as hf:
            #each layer, we have a tensor of shape: (B,H,seq,Head_dim), where seq is different shape, so we save as a list
            keys_group = hf.create_group("keys")
            values_group = hf.create_group("values")
            position_ids_group = hf.create_group("position_ids")
            for layer_idx, (key, value) in enumerate(zip(self.keys_before_rope, self.values_before_rope)):
                keys_group.create_dataset(f"layer_{layer_idx}", data=key.cpu().numpy())
                values_group.create_dataset(f"layer_{layer_idx}", data=value.cpu().numpy())
            position_ids_group.create_dataset(f"layer_{layer_idx}", data=self.position_ids.cpu().numpy())
        # print(f"Saved keys and values before rope to {kv_path}")
        
    def write_hidden_states(self, path):
        # save one file for all hidden_states of all layers
        # print(f"Writing hidden states to {path}")
        os.makedirs(path, exist_ok=True)
        hidden_states_path = f'{path}/hidden_states.h5'
        if self.hidden_states is None:
            return
        hidden_states = torch.stack(self.hidden_states, dim=0).squeeze()
        # write to h5 file
        with h5py.File(hidden_states_path, 'w') as hf:
            hf.create_dataset("hidden_states", data=hidden_states.to(torch.float32).cpu().numpy())
        # print(f"Saved hidden states to {hidden_states_path}")
        
        
    def get_top_k_indices(self, k=1, k_selection_method='attn_over_gen'):
        '''This function is called after all generations are finished, and it calculates the top_k tokens per layer'''
        '''It aggregates the attention scores for each generated token and layer (tensor T of: N_layers, N_tokens, N_heads, seq_len)'''
        '''Then, it focus omly on the image_tokens, by: T = T[..., image_tokens_mask]'''
        '''and calculates the mean attention per tokens and heads, by T = T.mean(dim=-1).mean(dim=1)'''
        '''Finally, it has a tensor of shape: (N_layers,#image_tokens)'''
        '''Then, it sorts the tensor for each layer, and takes the top_k tokens, and finally return subset of the image_tokens indices that are in the top_k'''
        
        if k_selection_method == 'attn_over_gen':
            top_k_indices = self.get_top_k_indices_attn_over_gen(k)
        elif k_selection_method == 'attn_over_query':
            top_k_indices = self.get_top_k_indices_attn_over_query(k)
        elif k_selection_method == 'norm':
            top_k_indices = self.get_top_k_indices_norm(k)
        elif k_selection_method == 'attn_over_gen_percentage':
            top_k_indices = self.get_top_k_indices_attn_over_gen(k, percentage=True)
        elif k_selection_method == 'norm_percentage':
            top_k_indices = self.get_top_k_indices_norm(k, percentage=True)
        elif k_selection_method=='random':
            top_k_indices = self.get_k_random_indices(k, percentage=True)
        else:
            raise ValueError(f"Unknown k_selection_method: {k_selection_method}")
        
        return top_k_indices
    
    def get_k_random_indices(self, k=1, percentage=False):
        if self.image_tokens_mask is None:
            return
        image_token_indices_cpu = self.image_tokens_mask.nonzero().squeeze().to('cpu')
        if percentage:
            k = int(k*len(image_token_indices_cpu))
        top_k_indices = []
        for _ in range(len(self.attentions[0])):
            top_k_idx = torch.randperm(len(image_token_indices_cpu))[:k]
            top_k_indices.append(image_token_indices_cpu[top_k_idx])
        return top_k_indices
    
    def get_top_k_indices_attn_over_gen(self, k=1, percentage=False):
        if len(self.attentions) == 0:
            return
        image_token_indices_cpu = self.image_tokens_mask.nonzero().squeeze().to('cpu')
    
        all_attns = []
        for generation_attns in self.attentions:
            attns = []
            # shape: list of N_layers, with each: (1, Heads, seq, seq)
            for layer_idx, layer_attention in enumerate(generation_attns):
                token_attention = layer_attention[0, :, -1, image_token_indices_cpu].clone() # Heads, #image_tokens
                attns.append(token_attention)
            token_attns = torch.stack(attns, dim=0) # N_layers, Heads, #image_tokens
            all_attns.append(token_attns)
        all_attns = torch.stack(all_attns, dim=0) # N_tokens, N_layers, Heads, #image_tokens
            
        # shape: (N_tokens, N_layers, N_heads, #image_tokens)
        mean_attn_per_token = all_attns.mean(dim=0).mean(dim=1) # mean of means is ok since we have the same number of dims for all
        # shape: (N_layers, #image_tokens)
        # argmax are return array or arrays (for each layer, the top_k indices)
        top_k_indices = []
        for layer_idx in range(mean_attn_per_token.shape[0]):
            if not percentage:
                top_k_idx = mean_attn_per_token[layer_idx].argsort(descending=True)[:int(k)]
            else:
                top_k_idx = mean_attn_per_token[layer_idx].argsort(descending=True)[:int(k*len(image_token_indices_cpu))]
            top_k_indices.append(image_token_indices_cpu[top_k_idx])
            
        return top_k_indices
    
    def get_top_k_indices_attn_over_query(self, k=1):
        if len(self.attentions) == 0:
            return
        image_token_indices_cpu = self.image_tokens_mask.nonzero().squeeze().to('cpu')
        generation_attns = self.attentions[0] # only when processing query...
        attns = []
        # shape: list of N_layers, with each: (1, Heads, seq, seq)
        for layer_idx, layer_attention in enumerate(generation_attns):
            token_attention = layer_attention[0, :, -1, image_token_indices_cpu].clone()
            attns.append(token_attention)
        attns = torch.stack(attns, dim=0) #shape: N_layers, Heads, #query_tokens
        
        # shape: (N_tokens, N_layers, N_heads, #query_tokens)
        mean_attn_per_token = attns.mean(dim=1)
        # shape: (N_layers, #query_tokens)
        # argmax are return array or arrays (for each layer, the top_k indices)
        top_k_indices = []
        for layer_idx in range(mean_attn_per_token.shape[0]):
            top_k_idx = mean_attn_per_token[layer_idx].argsort(descending=True)[:k]
            # get the top_k indices in the original #seq_len indices
            top_k_indices.append(image_token_indices_cpu[top_k_idx])
        
        return top_k_indices
    
    def get_top_k_indices_norm(self, k=1, percentage=False):
        '''based on hidden_states, we calculate the norm of the hidden_states for each token,
        and then we take the top_k tokens'''
        
        if self.hidden_states is None:
            return
        
        image_token_indices_cpu = self.image_tokens_mask.nonzero().squeeze().to('cpu')
        hidden_states = torch.stack(self.hidden_states, dim=0).squeeze()
        vis_hidden_states = hidden_states[:, image_token_indices_cpu] # N_layers, #image_tokens, hidden_size
        norms = vis_hidden_states.norm(dim=-1)
        top_k_indices = []
        for layer_idx in range(norms.shape[0]):
            if not percentage:
                top_k_idx = norms[layer_idx].argsort(descending=True)[:k]
            else:
                top_k_idx = norms[layer_idx].argsort(descending=True)[:int(k*len(image_token_indices_cpu))]
            top_k_indices.append(image_token_indices_cpu[top_k_idx])

        return top_k_indices
 
        
            
            

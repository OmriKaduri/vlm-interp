import os
import h5py
import numpy as np
import re

def find_model_files(directory, model_name, file_name='attention.h5', layer_filter=None):
    """
    Find all 'file_name' files located under 'IMAGE_NAME/LAYER/MODEL_NAME/' directories,
    where 'LAYER' directory names match the layer_filter (if provided),
    and 'MODEL_NAME' directory names contain the model_name substring.

    Args:
        directory (str): Root directory to search in.
        model_name (str): Substring to search for in 'MODEL_NAME' folder names.
        file_name (str): Name of the file to search for (default is 'attention.h5').
        layer_filter (str or re.Pattern): Substring or compiled regex pattern to match 'LAYER' directory names.

    Returns:
        h5_files (list): List of paths to 'file_name' files.
    """
    h5_files = []

    # Iterate over IMAGE_NAME directories in the root directory
    for image_name in os.listdir(directory):
        image_dir = os.path.join(directory, image_name)
        if not os.path.isdir(image_dir):
            continue  # Skip files or non-directories

        # Iterate over LAYER directories within each IMAGE_NAME directory
        for layer_name in os.listdir(image_dir):
            layer_dir = os.path.join(image_dir, layer_name)
            if not os.path.isdir(layer_dir):
                continue  # Skip non-directories

            # Apply layer_filter if provided
            if layer_filter:
                # If layer_filter is a regex pattern
                if isinstance(layer_filter, re.Pattern):
                    if not layer_filter.match(layer_name):
                        continue  # Skip layers that do not match the regex
                else:
                    # Treat layer_filter as a substring
                    if layer_filter not in layer_name:
                        continue  # Skip layers that do not contain the substring

            # Iterate over MODEL_NAME directories within each LAYER directory
            for model_dir_name in os.listdir(layer_dir):
                if model_name == model_dir_name:
                # if model_name in model_dir_name:
                    model_dir = os.path.join(layer_dir, model_dir_name)
                    if not os.path.isdir(model_dir):
                        continue  # Skip non-directories

                    file_path = os.path.join(model_dir, file_name)
                    if os.path.exists(file_path):
                        h5_files.append(file_path)
    return h5_files

# def find_model_files(directory, model_name, file_name='attention.h5', layer_filter='*'):
#     """
#     Find all 'file_name' files under directories whose folder name contains the model_name.
    
#     Args:
#         directory (str): Root directory to search in.
#         model_name (str): The substring to search for in folder names.
        
#     Returns:
#         h5_files (list): List of paths to 'file_name' files.
#     """
#     all_h5_files = glob(os.path.join(directory, f'**/{layer_filter}/{model_name}/{file_name}'), recursive=True)
#     return all_h5_files

def load_image_mask_from_file(file_path):
    """
    Load image mask from the given file_path.
    
    Args:
        file_path (str): Path to the file.
        
    Returns:
        image_mask (np.ndarray): A boolean mask indicating which tokens correspond to the image.
    """
    with h5py.File(file_path, 'r') as hf:
        image_mask = hf['image_tokens_mask'][:]
    return image_mask

def load_data_from_attention_file(file_path, load_attentions=True):
    """
    Load attentions from the given 'attention.h5' file.
    
    Args:
        file_path (str): Path to the 'attention.h5' file.
        
    Returns:
        attentions (dict): A dictionary where keys are layer indices and values are lists of attention values across tokens.
    """
    attentions = {}

    with h5py.File(file_path, 'r') as hf:
        if not load_attentions:
            return {
                'image_tokens_mask': hf['image_tokens_mask'][:],
                'n_patches': hf['n_patches'][()],
                'n_new_lines': hf['n_new_lines'][()],
                'image_res_x': hf['image_res_x'][()],
                'image_res_y': hf['image_res_y'][()],
                'query_tokens_mask': hf['query_tokens_mask'][:]
            }
        attentions_group = hf['attentions']
        image_tokens_mask = hf['image_tokens_mask'][:]
        n_patches = hf['n_patches'][()]
        n_new_lines = hf['n_new_lines'][()]
        image_res_x = hf['image_res_x'][()]
        image_res_y = hf['image_res_y'][()]
        query_tokens_mask = hf['query_tokens_mask'][:]
        for token_idx in attentions_group:
            token_group = attentions_group[token_idx]
            for layer_idx in token_group:
                layer_idx_int = int(layer_idx.split('_')[-1])
                attention_values = np.array(token_group[layer_idx])

                if layer_idx_int not in attentions:
                    attentions[layer_idx_int] = []
                attentions[layer_idx_int].append(attention_values)

    return {
        'attentions': attentions,
        'image_tokens_mask': image_tokens_mask,
        'n_patches': n_patches,
        'n_new_lines': n_new_lines,
        'image_res_x': image_res_x,
        'image_res_y': image_res_y,
        'query_tokens_mask': query_tokens_mask
    }

def load_data_from_hidden_states_file(file_path):
    """
    Load hidden states from the given 'hidden_states.h5' file.
    
    Args:
        file_path (str): Path to the 'hidden_states.h5' file.
        
    Returns:
        hidden_states (dict): A dictionary where keys are layer indices and values are lists of hidden states across tokens.
    """
    with h5py.File(file_path, 'r') as hf:
        hidden_states = hf['hidden_states'][:] # layers, tokens, hidden_size
    
    return hidden_states

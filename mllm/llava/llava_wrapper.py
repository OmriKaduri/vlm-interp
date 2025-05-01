import json
from pathlib import Path
import re
from mllm.utils.attention_store import TokenAttentionStore
import requests
from PIL import Image
from io import BytesIO
import torch
from torch import nn
from llava.model.builder import load_pretrained_model
from llava.mm_utils import get_model_name_from_path
from llava.constants import (
    IMAGE_TOKEN_INDEX,
    DEFAULT_IMAGE_TOKEN,
    DEFAULT_IM_START_TOKEN,
    DEFAULT_IM_END_TOKEN,
    IMAGE_PLACEHOLDER,
)
from transformers.cache_utils import Cache, DynamicCache
import time
from llava.conversation import conv_templates, SeparatorStyle
from llava.utils import disable_torch_init
from llava.mm_utils import (
    process_images,
    tokenizer_image_token,
    get_model_name_from_path,
)

class LlamaRotaryEmbedding(nn.Module):
    def __init__(self, dim, max_position_embeddings=2048, base=10000, device=None):
        super().__init__()

        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2).float().to(device) / self.dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        # Build here to make `torch.jit.trace` work.
        self._set_cos_sin_cache(
            seq_len=max_position_embeddings, device=self.inv_freq.device, dtype=torch.get_default_dtype()
        )

    def _set_cos_sin_cache(self, seq_len, device, dtype):
        self.max_seq_len_cached = seq_len
        t = torch.arange(self.max_seq_len_cached, device=device, dtype=self.inv_freq.dtype)

        freqs = torch.outer(t, self.inv_freq)
        # Different from paper, but it uses a different permutation in order to obtain the same calculation
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos().to(dtype), persistent=False)
        self.register_buffer("sin_cached", emb.sin().to(dtype), persistent=False)

    def forward(self, x, seq_len=None):
        # x: [bs, num_attention_heads, seq_len, head_size]
        if seq_len > self.max_seq_len_cached:
            self._set_cos_sin_cache(seq_len=seq_len, device=x.device, dtype=x.dtype)

        return (
            self.cos_cached[:seq_len].to(dtype=x.dtype),
            self.sin_cached[:seq_len].to(dtype=x.dtype),
        )
    
    def forward_from_position(self, x, position_ids):
        max_position_embeddings = position_ids.max().item() + 1
        if max_position_embeddings > self.max_seq_len_cached:
            self._set_cos_sin_cache(seq_len=max_position_embeddings, device=x.device, dtype=x.dtype)
        
        cos = self.cos_cached[position_ids.to(torch.long)]
        sin = self.sin_cached[position_ids.to(torch.long)]
        return cos.to(x.dtype), sin.to(x.dtype)


def calc_cos_sin_per_position(position_ids, max_position_embeddings=4096, base=10000.0, dim=128):
    """Calculate the cosine and sine of the position embeddings."""
    inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
    
    t = position_ids
    freqs = torch.outer(t, inv_freq)
    emb = torch.cat((freqs, freqs), dim=-1)
    return emb.cos(), emb.sin()

def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(k, cos, sin, position_ids=None, unsqueeze_dim=1):
    """Applies Rotary Position Embedding to the query and key tensors.

    Args:
        k (`torch.Tensor`): The key tensor.
        cos (`torch.Tensor`): The cosine part of the rotary embedding.
        sin (`torch.Tensor`): The sine part of the rotary embedding.
        position_ids (`torch.Tensor`, *optional*):
            Deprecated and unused.
        unsqueeze_dim (`int`, *optional*, defaults to 1):
            The 'unsqueeze_dim' argument specifies the dimension along which to unsqueeze cos[position_ids] and
            sin[position_ids] so that they can be properly broadcasted to the dimensions of q and k. For example, note
            that cos[position_ids] and sin[position_ids] have the shape [batch_size, seq_len, head_dim]. Then, if q and
            k have the shape [batch_size, heads, seq_len, head_dim], then setting unsqueeze_dim=1 makes
            cos[position_ids] and sin[position_ids] broadcastable to the shapes of q and k. Similarly, if q and k have
            the shape [batch_size, seq_len, heads, head_dim], then set unsqueeze_dim=2.
    Returns:
        `tuple(torch.Tensor)` comprising of the query and key tensors rotated using the Rotary Position Embedding.
    """
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return k_embed

def image_parser(args):
    out = args.image_file.split(args.sep)
    return out


def load_image(image_file):
    if image_file.startswith("http") or image_file.startswith("https"):
        response = requests.get(image_file)
        image = Image.open(BytesIO(response.content)).convert("RGB")
    else:
        image = Image.open(image_file).convert("RGB")
    return image


def load_images(image_files):
    out = []
    for image_file in image_files:
        image = load_image(image_file)
        out.append(image)
    return out


class LlavaWrapper():
    def __init__(self, model_path = "liuhaotian/llava-v1.5-7b", cache_strategy=None):
        self.model_name = get_model_name_from_path(model_path)
        tokenizer, model, image_processor, context_len = load_pretrained_model(
            model_path=model_path,
            model_base=None,
            model_name=self.model_name,
        )
        self.cache_strategy = cache_strategy
        self.tokenizer = tokenizer
        self.model = model
        self.image_processor = image_processor
        self.context_len = context_len
    
    def predict_for_token(self, prompt, image_path, max_layer_for_i2t_attn,
                        output_path=None, topk_indices=None, gt_response=None,
                        write_to_disk=True,min_token_for_i2t_attn_masking=0, block_query=False, block_query_unidirection=False,
                        block_vis_to_all_from_layer=None, block_vis_to_gen_not_between_layers_4_to_20=False,
                        block_vis_to_query_not_between_layers_4_to_20=False, block_vis_to_all_not_between_layers_4_to_20=False, 
                        fastv_ratio=None, fastv_layer=None):

        if isinstance(image_path, str):
            images = load_images([image_path])
            #else: it's Pillow image
        elif isinstance(image_path, Image.Image):
            images = [image_path]
        pixel_values = process_images(
            images,
            self.image_processor,
            self.model.config
        ).to(self.model.device, dtype=torch.float16) #(B,3,336,336)
        token_attention_store = TokenAttentionStore()
        self.model.set_attention_store(token_attention_store)
        start_time = time.time()
        response, generation_output = self.chat_step_by_step(
            self.tokenizer, pixel_values, prompt, None, 
            verbose=False, max_layer_for_i2t_attn=max_layer_for_i2t_attn,
            topk_indices=topk_indices, min_token_for_i2t_attn_masking=min_token_for_i2t_attn_masking,
            block_query=block_query, block_query_unidirection=block_query_unidirection,
            block_vis_to_all_from_layer=block_vis_to_all_from_layer,
            block_vis_to_gen_not_between_layers_4_to_20=block_vis_to_gen_not_between_layers_4_to_20,
            block_vis_to_query_not_between_layers_4_to_20=block_vis_to_query_not_between_layers_4_to_20,
            block_vis_to_all_not_between_layers_4_to_20=block_vis_to_all_not_between_layers_4_to_20,
            fastv_ratio=fastv_ratio, fastv_layer=fastv_layer
        )
        # print(f"Time taken for generation in ms {1000*(time.time()-start_time)}")
        generation_output = generation_output.squeeze()
        if write_to_disk:
            token_attention_store.write_token_scores(output_path)
            token_attention_store.extract_maps_per_generated_token(generation_output,
                    self.tokenizer, pixel_values, output_path, patch_size=576)

        tokens = [self.tokenizer.decode(t) for t in generation_output.squeeze()]
        if output_path is not None:
            Path(output_path).mkdir(parents=True, exist_ok=True)
            output_json_path = Path(output_path) / "output.json"
            
            response = response.replace(prompt, "").strip()
            new_element = {"prompt": prompt,
                    "response": response, 
                    "tokens": tokens,
                    "knockout": max_layer_for_i2t_attn}
            if gt_response is not None:
                new_element["gt_response"] = gt_response
            new_data = [new_element]

            if output_json_path.exists():
                with open(output_json_path, 'r') as f:
                    json_data = json.load(f)
                    if isinstance(json_data, list):
                        new_data = new_data + json_data
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
                json.dump(new_data, f)
            
        return response, tokens, token_attention_store

    def predict_from_cache(self, prompt, image_path, output_path=None, write_to_disk=True, kv=None, cache_strategy=None, query_tokens_mask=None, topk_indices=None,
                            max_layer_for_i2t_attn=None,gt_response=None):
        start_time = time.time()
        response, generation_output = self.chat_step_by_step_from_cache(
            prompt, kv,cache_strategy=cache_strategy, history=None,
          return_history=True, max_layer_for_i2t_attn=max_layer_for_i2t_attn, query_tokens_mask=query_tokens_mask, topk_indices=topk_indices)
        # print(f"Time taken for generation in ms {1000*(time.time()-start_time)}")
        
        generation_output = generation_output.squeeze()

        tokens = [self.tokenizer.decode(t) for t in generation_output.squeeze()]
        if output_path is not None:
            Path(output_path).mkdir(parents=True, exist_ok=True)
            output_json_path = Path(output_path) / "output.json"
            if self.cache_strategy is not None:
                output_json_path = Path(output_path) / f"output_cached_{self.cache_strategy}.json"

            response = response.replace(prompt, "").strip()
            new_element = {"prompt": prompt,
                    "response": response, 
                    "tokens": tokens,
                    "knockout": max_layer_for_i2t_attn}
            if gt_response is not None:
                new_element["gt_response"] = gt_response
            new_data = [new_element]

            if output_json_path.exists():
                with open(output_json_path, 'r') as f:
                    json_data = json.load(f)
                    if isinstance(json_data, list):
                        new_data = new_data + json_data
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
                json.dump(new_data, f)
            
        return response, tokens, None
    
    def chat_step_by_step(self, tokenizer, pixel_values, question, generation_config, 
             verbose=False, max_layer_for_i2t_attn=None,topk_indices=None,
             min_token_for_i2t_attn_masking=0, kv=None,block_query=False,
             block_query_unidirection=False, block_vis_to_all_from_layer=None,
             block_vis_to_gen_not_between_layers_4_to_20=False,
             block_vis_to_query_not_between_layers_4_to_20=False, block_vis_to_all_not_between_layers_4_to_20=False,
             fastv_ratio=None, fastv_layer=None):

        image_token_se = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN
        if IMAGE_PLACEHOLDER in question:
            if self.model.config.mm_use_im_start_end:
                question = re.sub(IMAGE_PLACEHOLDER, image_token_se, question)
            else:
                question = re.sub(IMAGE_PLACEHOLDER, DEFAULT_IMAGE_TOKEN, question)
        elif DEFAULT_IMAGE_TOKEN not in question:
            if self.model.config.mm_use_im_start_end:
                question = image_token_se + "\n" + question
            else:
                question = DEFAULT_IMAGE_TOKEN + "\n" + question

        if "llama-2" in self.model_name.lower():
            conv_mode = "llava_llama_2"
        elif "mistral" in self.model_name.lower():
            conv_mode = "mistral_instruct"
        elif "v1.6-34b" in self.model_name.lower():
            conv_mode = "chatml_direct"
        elif "v1" in self.model_name.lower():
            conv_mode = "llava_v1"
        elif "mpt" in self.model_name.lower():
            conv_mode = "mpt"
        else:
            conv_mode = "llava_v0"

        conv = conv_templates[conv_mode].copy()
        conv.append_message(conv.roles[0], question)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        image_sizes = pixel_values.size()[2:]

        input_ids = (
            tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt")
            .unsqueeze(0)
            .cuda()
        )
        
        if hasattr(self.model.model, 'topk_indices'): # reset it to None before each generation
            # print('!!! Setting topk_indices')
            self.model.model.topk_indices = topk_indices

        with torch.inference_mode():
            output_ids = self.model.generate(
                input_ids,
                images=pixel_values,
                image_sizes=image_sizes,
                output_attentions=True,
                output_hidden_states=True,
                max_layer_for_i2t_attn=max_layer_for_i2t_attn,
                min_token_for_i2t_attn_masking=min_token_for_i2t_attn_masking,
                block_query=block_query,
                block_query_unidirection=block_query_unidirection,      
                block_vis_to_all_from_layer=block_vis_to_all_from_layer,   
                block_vis_to_gen_not_between_layers_4_to_20=block_vis_to_gen_not_between_layers_4_to_20, 
                block_vis_to_query_not_between_layers_4_to_20=block_vis_to_query_not_between_layers_4_to_20,
                block_vis_to_all_not_between_layers_4_to_20=block_vis_to_all_not_between_layers_4_to_20,
                fastv_layer=fastv_layer, fastv_ratio=fastv_ratio,
                do_sample=False,
                temperature=0,
                top_p=None,
                num_beams=1,
                max_new_tokens=300,
                use_cache=True,
            )
            output_ids = output_ids[...,1:] # remove the padding token
        outputs = self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
        return outputs, output_ids
    
    def chat_step_by_step_from_cache(self, question, kv, cache_strategy=None, history=None, return_history=False, max_layer_for_i2t_attn=None, query_tokens_mask=None, topk_indices=None):
        if "llama-2" in self.model_name.lower():
            conv_mode = "llava_llama_2"
        elif "mistral" in self.model_name.lower():
            conv_mode = "mistral_instruct"
        elif "v1.6-34b" in self.model_name.lower():
            conv_mode = "chatml_direct"
        elif "v1" in self.model_name.lower():
            conv_mode = "llava_v1"
        elif "mpt" in self.model_name.lower():
            conv_mode = "mpt"
        else:
            conv_mode = "llava_v0"

        prompt = question

        input_ids = (
            tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt")
            .unsqueeze(0)
            .cuda()
        )
        encoded_past_key_values_per_image = kv
        encoded_past_keys_per_image = [] #Size: n_layers
        encoded_past_values_per_image = []
        cache_seq_len = 0
        for encoded_past_key_values in encoded_past_key_values_per_image:
            encoded_past_keys = encoded_past_key_values[0]
            encoded_past_values = encoded_past_key_values[1]
            if len(encoded_past_key_values) == 3 and encoded_past_key_values[2] is not None:
                position_ids = encoded_past_key_values[2]
                rotary_emb = LlamaRotaryEmbedding(
                    128,
                    max_position_embeddings=4096,
                    base=10000.0,
                )   
                cos, sin = rotary_emb.forward_from_position(encoded_past_values[0], position_ids[0])

                for layer_idx in range(len(encoded_past_keys)):
                    encoded_past_keys[layer_idx] = apply_rotary_pos_emb(encoded_past_keys[layer_idx], cos, sin, unsqueeze_dim=1)
        
            encoded_past_keys_per_image.append(encoded_past_keys) #per image, a list of: n_layers, each of shape: (batch_size, n_tokens, hidden_size)
            encoded_past_values_per_image.append(encoded_past_values)
            cache_seq_len += encoded_past_keys[0].shape[-2]
            
        #encoded_past_keys_per_image : list of n_images, each of n_layers, each of shape: (batch_size, n_tokens, hidden_size)
        past_key_values = DynamicCache() 
        # we want to set the topk_indices from the encoded_past_key_values into past_key_values
        # we also need to know, to udpate the position_ids, the max position of the kv cache (up to query position)
        query_token_indices = query_tokens_mask 

        if topk_indices is None:
            query_diff = query_tokens_mask[0] - range(len(query_tokens_mask[0]))
            first_img_token_index = query_tokens_mask[0][query_diff.nonzero()[0][0] - 1] + 1
            last_img_token_index = query_tokens_mask[0][query_diff.nonzero()[0][0]] - 1
            topk_indices = [[
                list(range(first_img_token_index, last_img_token_index+1)) for _ in range(len(encoded_past_keys_per_image[0]))] for _ in range(len(encoded_past_keys_per_image))]

        if not isinstance(query_token_indices, list) or not isinstance(topk_indices, list) or not isinstance(topk_indices[0], list):
            raise ValueError('query_token_indices should be a list of indices for each image, and topk_indices should be a list of lists of indices for each image')
        
        #sort topk_indices at each layer - ascending order
        for image_idx in range(len(topk_indices)):
            topk_indices[image_idx] = [sorted(topk_indices[image_idx][layer_idx]) for layer_idx in range(len(topk_indices[image_idx]))]
        
        image_idx = 0
        
        for encoded_past_keys, encoded_past_values, image_query_token_indices, image_topk_indices in zip(encoded_past_keys_per_image, encoded_past_values_per_image, query_token_indices, topk_indices):
            for layer_idx in range(len(encoded_past_keys)):
                system_prompt_idx = ((image_query_token_indices[:-1] - image_query_token_indices[1:])!=-1).nonzero()[0][0]
                
                image_keys_encoded = encoded_past_keys[layer_idx][...,image_topk_indices[layer_idx],:].to(input_ids.device) #.to(torch.float16)
                query_keys_encoded = encoded_past_keys[layer_idx][...,image_query_token_indices,:].to(input_ids.device) #.to(torch.float16)
                system_prompt_keys_encoded = query_keys_encoded[..., :system_prompt_idx+1, :]
                user_prompt_keys_encoded = query_keys_encoded[..., system_prompt_idx+1:, :]
                
                if cache_strategy =='q':
                    past_layer_key = query_keys_encoded
                elif cache_strategy == 's':
                    past_layer_key = torch.cat([image_keys_encoded], dim=-2)
                elif cache_strategy == 'qs':
                    past_layer_key = torch.cat([system_prompt_keys_encoded, image_keys_encoded, user_prompt_keys_encoded], dim=-2)
                else:
                    raise NotImplementedError('This cache strategy is not implemented')
                
                image_values_encoded = encoded_past_values[layer_idx][...,image_topk_indices[layer_idx],:].to(input_ids.device)#.to(torch.float16)
                query_values_encoded = encoded_past_values[layer_idx][...,image_query_token_indices,:].to(input_ids.device)#.to(torch.float16)       
                system_prompt_values_encoded = query_values_encoded[..., :system_prompt_idx+1, :]
                user_prompt_values_encoded = query_values_encoded[..., system_prompt_idx+1:, :]
                
                if cache_strategy =='q':
                    past_layer_value = query_values_encoded
                elif cache_strategy == 's':
                    past_layer_value =  torch.cat([image_values_encoded], dim=-2)
                elif cache_strategy == 'qs':
                    past_layer_value = torch.cat([system_prompt_values_encoded, image_values_encoded, user_prompt_values_encoded], dim=-2)

                else:
                    raise NotImplementedError('This cache strategy is not implemented')
                past_key_values.update(past_layer_key, past_layer_value, layer_idx)
            image_idx += 1
                
        
        if cache_strategy == 'qs' or cache_strategy == 'q':
            max_position_id_for_cache = sum([max(image_query_token_indices) for image_query_token_indices in query_token_indices])
        elif cache_strategy == 's':
            max_position_id_for_cache = max([max(image_topk_indices[layer_idx]) for image_topk_indices in topk_indices for layer_idx in range(len(image_topk_indices))])
        else:
            raise NotImplementedError('This cache strategy is not implemented')
        n_tokens_in_cache = past_key_values.key_cache[0].shape[-2]

        # create attention_mask with ones for all tokens in cache + ones for tokens in input_ids
        attention_mask = torch.ones((1, n_tokens_in_cache + input_ids.shape[-1]), dtype=torch.float16, device=input_ids.device)
        
        with torch.inference_mode():
            output_ids = self.model.generate(
                input_ids,
                output_attentions=True,
                output_hidden_states=True,
                attention_mask=attention_mask,
                position_id_offset=max_position_id_for_cache + 1,
                past_key_values=past_key_values,
                do_sample=False,
                temperature=0,
                top_p=None,
                num_beams=1,
                max_new_tokens=300,
                use_cache=True,
            )
            output_ids = output_ids[...,1:] # remove the padding token
        outputs = self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
        return outputs, output_ids

        

# from output.json files, create the results.txt file, with rows in the format:
#Image_Name + "\t" + Question + "\t" + Ground_Truth_Answer + "\t" + Your_Response + "\n"

# will be used for "calculate.py" to calculate the accuracy of the model on MME
import argparse
import json

import pandas as pd

from mllm.utils.mllm_results_utils import find_model_files

def read_response_from_output_json(output_json_file):
    with open(output_json_file, 'r') as f:
        data = json.load(f)
    #it is array of objects: {"prompt": "prompt", "response": "response", "tokens": ["token1", "token2"], "knockout": knockout_layer}
    return data


def main(args):
    if 'internvl' in args.model_name:
        layer_for_gt=80
    else:
        layer_for_gt=32
    lavin_gt = args.lavin_gt
    
    # read the .txt file from lavin_Gt
    with open(lavin_gt, 'r') as f:
        lavin_gt_data = f.readlines() # it has rows: Image_Name + "\t" + Question + "\t" + Ground_Truth_Answer + "\t" + Your_Response + "\n"
    lavin_file_name = lavin_gt.split('/')[-1].split('.')[0]
    json_file_name = args.pred_file
    print(f"Searching for {json_file_name} files in {args.directory} with model name {args.model_name}...")
    output_json_files = find_model_files(args.directory, args.model_name, json_file_name)
    print(f"Found {len(output_json_files)} {json_file_name} files.")
    # group by layer_idx (.split('/')[-3]), and order each group by it
    output_json_files = sorted(output_json_files, key=lambda x: int(x.split('/')[-3]))
    file_data = []
    for output_json_file in output_json_files:
        # print(f"Processing {output_json_file}...")
        layer_idx  = output_json_file.split('/')[-3]
        image_name = output_json_file.split('/')[-4]
        #pad to 12 digits with leading zeros
        if not('ocr_results' in args.directory.lower() or
               'celebrity_results' in args.directory or
               'posters_results' in args.directory or 
               'artwork_results' in args.directory or
               'code_reasoning' in args.directory or
               'commonsense_reasoning_results' in args.directory or
               'text_translation_results' in args.directory or
               'numerical_calculation' in args.directory or 
               'landmark' in args.directory):
            image_name = image_name.split(".")[0].zfill(12) + ".jpg"
        file_data.append({'layer_idx': layer_idx, 'image_name': image_name, 'output_json_file': output_json_file})
    file_df = pd.DataFrame(file_data, columns=['layer_idx', 'image_name', 'output_json_file'])

    # We want to create file .txt with rows as: Image_Name + "\t" + Question + "\t" + Ground_Truth_Answer + "\t" + Your_Response + "\n"
    # if json_file_name ends with: _q, _qs, _s, add these suffixes to the output file name
    if 'llava' in args.model_name:
        file_name = f'mllm/eval/mme/llava/{lavin_file_name}.txt'
    else:
        file_name = f'mllm/eval/mme/{lavin_file_name}.txt'        
        
    if json_file_name.endswith('_q.json'):
        file_name = file_name.replace('.txt', '_q.txt')
    elif json_file_name.endswith('_qs.json'):
        if '-masked-0.05' in args.model_name:
            file_name = file_name.replace('.txt', '_qs-5%.txt')
        else:
            file_name = file_name.replace('.txt', '_qs.txt')
    elif json_file_name.endswith('_s.json'):
        file_name = file_name.replace('.txt', '_s.txt')
    elif json_file_name.endswith('_llm.json'):
        file_name = file_name.replace('.txt', '_llm.txt')
    elif json_file_name.endswith('_text.json'):
        file_name = file_name.replace('.txt', '_text.txt')
        
        
    results_file = open(file_name, 'w')
    
    # Apply the evaluation function group by group
    def apply_evaluation(row, json_file_name):
        # we want to extract the response from the layer_for_gt index, where 'layer_idx' is the layer index
        output_json_file = row['output_json_file']
        output_json_file = output_json_file.replace('output.json', json_file_name)
        print(f"Reading {output_json_file}...")
        # output_json_file = output_json_file.replace('output.json', 'output_cached.json')
        data = read_response_from_output_json(output_json_file)
        lines_to_add = []
        for d in data:
            lines_to_add.append({'image_name': row['image_name'], 'question': d['prompt'], 'response': d['response']})
        # return only the 2 top rows    
        # return lines_to_add[:2]
        return lines_to_add
            
    # Iterate over lavin_gt_data, and for each row, find the corresponding row in file_df, and apply the evaluation function
    for row in lavin_gt_data:
        image_name, question, gt_answer, _ = row.strip().split("\t")
            #f"describe the image and answer, {line[1].split('?')[0]} \n")
        # question = f"describe the image and answer, {question.split('?')[0]}"
        # find rows with the same image_name
        eval_results = file_df[file_df['image_name'] == image_name]
        print(f"Processing {image_name}..., {len(eval_results)} rows")
        # print(file_df.image_name)
        # since its same image_name, read only the first row (there should be only one row)
        eval_results = eval_results.apply(lambda x: apply_evaluation(x, json_file_name), axis=1)
        if len(eval_results) == 0:
            print(f"Failed to find {image_name} in {args.directory}")
            # results_file.write(image_name + "\t" + question + "\t" + gt_answer + "\t" + "Fail" + "\n")
            continue
        # if less than 2 - skip the image and print the error
        # if len(eval_results) < 2:
        #     print(f"Error: only image description + one question1 found for {image_name}")
        #     continue
        # for the question, find the eval_result with matches question
        for l in eval_results.iloc[0]:
            curr_question = l['question']
            # remove \n and trailing spaces
            curr_question = curr_question.strip() #describe the image and answer, Is there only one bath towel in the picture \n
            question_with_ONLY = question.replace("Please answer yes or no.","Please answer yes or no ONLY.")
            # celeb_name = question.split('named ')[-1].split('?')[0]
            
            # question_a_bit_diff =f"describe the image and answer, {question.split('?')[0]}"
            # question_a_bit_diff =f"describe the image and answer, {question} \n"
            # question_a_bit_diff =f"describe the image, focus on the text, and answer yes only if it contains specifically: {question}"
            # question_a_bit_diff = f".describe the image and answer, {question.replace('Please answer yes or no.','Please start your answer by responding either yes or no.')}"
            # question_a_bit_diff = f".describe the image and answer, {question.replace('Please answer yes or no.','Please start your answer by responding either yes or no.')}"
            question_a_bit_diff = f"{question} ASSISTANT:"
# 
            a_bit_diff = False
            # question_a_bit_diff =f"{celeb_name}? Write your answer and finalize with Yes or  No"
            # question_a_bit_diff = f"Who is the celebrity in the image? Is it {celeb_name}? Write your answer and"

            if curr_question == question_with_ONLY and not a_bit_diff:
                print(f"Image: {image_name}, Question: {question}, GT: {gt_answer}, Response: {l['response']}")
                response = l['response']
                print("Matched question: ", curr_question)
                # make sure response does not contain \n
                response = response.replace("\n", "")
                results_file.write(image_name + "\t" + question + "\t" + gt_answer + "\t" + response + "\n")
                break
            elif curr_question == question and not a_bit_diff:
                print(f"Image: {image_name}, Question: {question}, GT: {gt_answer}, Response: {l['response']}")
                response = l['response']
                print("Matched question: ", curr_question)

                # make sure response does not contain \n
                response = response.replace("\n", "")
                results_file.write(image_name + "\t" + question + "\t" + gt_answer + "\t" + response + "\n")    
                break
            #elif: question_a_bit_diff is in curr_question:
            elif question_a_bit_diff in curr_question:
                print(f"Image: {image_name}, Question: {question}, GT: {gt_answer}, Response: {l['response']}")
                print(f"Matched question: {curr_question}")
                response = l['response']
                # make sure response does not contain \n
                response = response.replace("\n", "")
                results_file.write(image_name + "\t" + question + "\t" + gt_answer + "\t" + response + "\n")    
                break
            # else:
            #     print("Haven't found: ", question, " in ", curr_question)
            # else:
            #     print("BUG!")
            #     print(f"Image: {image_name}, Question: {curr_question}, GT: {gt_answer}, Response: {l['response']}")
            
    
    # eval_results = file_df[file_df['layer_idx'] == str(layer_for_gt)].apply(apply_evaluation, axis=1)

    results_file.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Plot the distribution of attention values across layers.")
    parser.add_argument('directory', type=str, help="The root directory to search for output.json files")
    parser.add_argument('model_name', type=str, help="Model name substring to filter directories.")
    parser.add_argument('lavin_gt', type=str, help="The ground truth lavin file")
    parser.add_argument('pred_file', type=str, help="The JSON file with the model predictions", default="output.json")
    # cached flag - bool - whether to use cached results
    
    args = parser.parse_args()
    
    main(args)

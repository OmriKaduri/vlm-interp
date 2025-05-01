import os
import argparse
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix
import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument('--results_dir', default='./LaVIN', type=str)

eval_type_dict = {
    "Perception": [
        "existence",
        "existence_q",
        "existence_s",
        "existence_qs",
        "existence_qs-5%",
        "existence_llm",
        "count",
        "count_q",
        "count_s",
        "count_qs",
        "count_qs-5%",
        "count_llm",
        "position",
        "position_q",
        "position_s",
        "position_qs",
        "position_qs-5%",
        "position_llm",
        "color",
        "color_q",
        "color_s",
        "color_qs",
        "color_qs-5%",
        "color_llm",
        "OCR",
        "OCR_q",
        "OCR_s",
        "OCR_qs",
        "OCR_qs-5%",
        "OCR_llm",
        "celebrity",
        "celebrity_q",
        "celebrity_s",
        "celebrity_qs",
        "celebrity_qs-5%",
        "celebrity_llm",
        "posters",
        "posters_q",
        "posters_s",
        "posters_qs",
        "posters_qs-5%",
        "posters_llm",
        "artwork",
        "artwork_q",
        "artwork_s",
        "artwork_qs",
        "artwork_qs-5%",
        "artwork_llm",
        "scene",
        "scene_q",
        "scene_s",
        "scene_qs",
        "scene_qs-5%",
        "scene_llm",
        "landmark",
        "landmark_q",
        "landmark_s",
        "landmark_qs",
        "landmark_qs-5%",
        "landmark_llm",

                   ],
    "Cognition": [
        "commonsense_reasoning",
        "commonsense_reasoning_q",
        "commonsense_reasoning_s",
        "commonsense_reasoning_qs",
        "commonsense_reasoning_qs-5%",
        "commonsense_reasoning_llm",
        # "commonsense_reasoning_text",
        "code_reasoning",
        "code_reasoning_q",
        "code_reasoning_s",
        "code_reasoning_qs",
        "code_reasoning_qs-5%",
        "code_reasoning_llm",
        # "code_reasoning_text",
        "numerical_calculation", 
        "numerical_calculation_q",
        "numerical_calculation_s",
        "numerical_calculation_qs",
        "numerical_calculation_qs-5%",
        "numerical_calculation_llm",
        # "numerical_calculation_text",
        "text_translation",
        "text_translation_q",
        "text_translation_s",
        "text_translation_qs",
        "text_translation_qs-5%",
        "text_translation_llm",
        # "text_translation_text",
        ]
}


class calculate_metrics:
    def divide_chunks(self, l, n=2):
        # looping till length l
        for i in range(0, len(l), n): 
            yield l[i:i + n]
        
        return 

    def parse_pred_ans(self, pred_ans):
        pred_label = None
        if pred_ans in ["yes", "no"]:
            pred_label = pred_ans
        else:
            prefix_pred_ans = pred_ans[:4]

            if "yes" in prefix_pred_ans:
                pred_label = "yes"
            elif "no" in prefix_pred_ans:
                pred_label = "no"
            else:
                pred_label = "other"

        return pred_label


    def compute_metric(self, gts, preds):
        assert len(gts) == len(preds)

        label_map = {
            "yes": 1,
            "no": 0,
            "other": -1,
        }
        
        gts = [label_map[x] for x in gts]
        preds = [label_map[x] for x in preds]

        acc = accuracy_score(gts, preds) 

        clean_gts = []
        clean_preds = []
        other_num = 0 
        for gt, pred in zip(gts, preds):
            if pred == -1:
                other_num += 1
                continue
            clean_gts.append(gt)
            clean_preds.append(pred)
        

        conf_mat = confusion_matrix(clean_gts, clean_preds, labels=[1,0])
        precision = precision_score(clean_gts, clean_preds, average='binary')
        recall = recall_score(clean_gts, clean_preds, average='binary')
        tp, fn = conf_mat[0]
        fp, tn = conf_mat[1]

        metric_dict = dict()
        metric_dict = {
            "TP": tp,
            "FN": fn,
            "TN": tn,
            "FP": fp,
            "precision": precision,
            "recall": recall,
            "other_num": other_num,
            "acc": acc,
        }

        return metric_dict


    def process_result(self, results_dir):

        model_score_dict = []
        for eval_type, task_name_list in eval_type_dict.items():
            print("===========", eval_type, "===========")
           
            scores = 0
            task_score_dict = dict()

            for task_name in task_name_list:

                task_txt = os.path.join(results_dir, task_name + ".txt")
                lines = open(task_txt, 'r').readlines()
                chunk_lines = list(self.divide_chunks(lines)) # one image corresponds to two questions
                img_num = len(chunk_lines)
                task_other_ans_num = 0
                task_score = 0
                acc_plus_correct_num = 0
                gts = []
                preds = []

                for img_items in chunk_lines:
                    assert len(img_items) == 2, f"each image should have two questions, but got {len(img_items)}, for image {img_items}"
                    img_correct_num = 0

                    for img_item in img_items:
                        img_name, question, gt_ans, pred_ans = img_item.split("\t")

                        gt_ans = gt_ans.lower()
                        pred_ans = pred_ans.lower()

                        assert gt_ans in ["yes", "no"] # gt can only be yes or no.

                        pred_ans = self.parse_pred_ans(pred_ans)
                        assert pred_ans in ["yes", "no", "other"]

                        gts.append(gt_ans)
                        preds.append(pred_ans)
                        
                        if gt_ans == pred_ans:
                            img_correct_num += 1
                        
                        if pred_ans not in ["yes", "no"]:
                            task_other_ans_num += 1

                    if img_correct_num == 2:
                        acc_plus_correct_num += 1

                # cal TP precision acc, etc.
                metric_dict = self.compute_metric(gts, preds)
                try:
                    acc_plus = acc_plus_correct_num / img_num
                except:
                    raise ValueError(f"acc_plus_correct_num: {acc_plus_correct_num}, img_num: {img_num}, task_name: {task_name}")
                metric_dict["acc_plus"] = acc_plus
                # print(task_name, "metric_dict:", metric_dict)
                model_score_dict.append({'task': task_name, **metric_dict})
                
                for k, v in metric_dict.items():
                    if k in ["acc", "acc_plus"]:
                        task_score += v*100
                
                task_score_dict[task_name] = task_score
                if '5%' in task_name:
                    scores += task_score

            print("total score:", scores, "\n")
            for task_name, score in task_score_dict.items():
                print("\t", task_name, " score:", score)
            print("\n")
        
        # make df
        # print("model_score_dict:", model_score_dict)
        df = pd.DataFrame(model_score_dict)
        # # only acc, acc_plus
        # only_df = df[['task', 'acc', 'acc_plus']]
        # # group by dataset
        # only_df['dataset'] = only_df['task'].apply(lambda x: x.split('_')[0])
        # only_df['name'] = only_df['task'].apply(lambda x: x.split('_')[1] if len(x.split('_')) > 1 else 'naive')
        # only_df.drop(columns=['task'], inplace=True)
        # # set the name as index
        # only_df.set_index('name', inplace=True)
        # print(only_df.to_latex())
        
        # compute avg acc and acc_plus for all tasks starting with existence
        super_tasks = ['existence', 'count', 'position', 'color', 'OCR', 'celebrity', 'posters',
                       'artwork', 'scene', 'landmark',
                    #    'commonsense_reasoning', 'code_reasoning', 'numerical_calculation',
                    #    'text_translation',
                    #    'landmark'
                       ]
        sub_tasks = ['',  '_llm',  '_qs-5%', '_q', '_s', '_qs']
        sub_tasks_acc = []
        sub_tasks_acc_plus = []
        for sub_task in sub_tasks:
            sub_task_acc = 0
            sub_task_acc_plus = 0
            for super_task in super_tasks:
                task = super_task + sub_task
                avg_acc = df[df['task']==task]['acc'].mean()
                avg_acc_plus = df[df['task']==task]['acc_plus'].mean()
                sub_task_acc += avg_acc
                sub_task_acc_plus += avg_acc_plus
            sub_task_acc /= len(super_tasks)
            sub_task_acc_plus /= len(super_tasks)
            sub_tasks_acc.append(sub_task_acc)
            sub_tasks_acc_plus.append(sub_task_acc_plus)
        # print("each sub task avg acc:", {t: a for t, a in zip(sub_tasks, sub_tasks_acc)})
        # print("each sub task avg acc_plus:", {t: a for t, a in zip(sub_tasks, sub_tasks_acc_plus)})
            # print(f"avg acc_plus for {sub_task}:", sub_task_acc_plus)
            
        # compute avg acc and acc_plus for all tasks, per task
        df.to_csv(os.path.join(results_dir, "mme_score.csv"), index=False)
        return 


if __name__ == "__main__":
    cal = calculate_metrics()

    args = parser.parse_args()
    results_dir = args.results_dir
    cal.process_result(results_dir)


# What’s in the Image? A Deep-Dive into the Vision of Vision Language Models (CVPR 2025)

This repo contains the code for Llava-1.5-7B experiments on MME from the paper [What’s in the Image? A Deep-Dive into the Vision of Vision Language Models](https://vision-of-vlm.github.io/).

## 🧪 Install Environment for LLaVA Knockout Experiments
1. Create and activate a conda environment:
   ```bash
   conda create -n llavako python=3.10 -y && conda activate llavako
   ```
2. Install PyTorch and dependencies:
   ```bash
   pip install torch==2.1.2 torchvision==0.16.2 -r llava_requirements.txt
   ```
3. Install the forked LLaVA repository with knockout option:
   ```bash
   pip install git+ssh://git@github.com/OmriKaduri/LLaVA.git
   ```
4. Run the following script to patch transformers with our knockout options:
   ```bash
   ./update_local_env_llava.sh
   ```

## 🦙 Running LLaVA on MME

### Data Preparation:
1. Download the required datasets from [Awesome-Multimodal-Large-Language-Models Evaluation](https://github.com/BradyFU/Awesome-Multimodal-Large-Language-Models/tree/Evaluation).
2. Place the downloaded data under the directory specified by the `--mme_data_folder` argument in the following script.

### Running the Script (might takes ~10 minutes to run Llava over all mme with multiple k_value options)
To run the `llava_on_mme_runner.py` script, use the following command:
```bash
PYTHONPATH=. python llava_on_mme_runner.py \
  --mme_gt_folder <path_to_gt_folder> \
  --mme_data_folder <path_to_data_folder> \
  --mme_results_folder <path_to_results_folder> \
  --ks <k_values>
```

Replace the placeholders with the appropriate paths:
- `--mme_gt_folder`: Path to the MME ground truth folder (default: `/mllm/eval/mme/LaVIN/`).
- `--mme_data_folder`: Path to the MME data folder (`MME_Benchmark_release_version` folder)
- `--mme_results_folder`: Path to the MME results folder (we will use this folder later to aggregate results, but can be anywhere)
- `--ks`: List of `k` values to use (default: `[0.02, 0.05]`, i.e. 2% and 5%).

Note that under `mme_results_folder`, several result folders will be created, one per MME subset (e.g., `llava_existence_results`, `llava_count_resutls`, etc.). 

Example:
```bash
PYTHONPATH=. python llava_on_mme_runner.py --mme_data_folder PATH_TO_FOLDER/MME_Benchmark_release_version --mme_results_folder PATH_TO_MME_RESULTS_FOLDER --ks 0.02 0.05
```

Then, to calculate results over all MME subsets, run:
```bash
python mllm/eval/mme/calculate.py --results_dir PATH_TO_MME_RESULTS_FOLDER
```
And you would see in the command line the metrics, per MME subset.

---

## 🔍 Visualize relative attention by token type
Specify the model name and path to the processed data directory as the first argument. For example, for LLaVA, on MME existence subset (a folder named "llava_existence_results" should be generated from previous part):

``` PYTHONPATH=. python mllm/visualizations/plot_relative_attention_by_token_type.py PATH_TO_MME_RESULTS_FOLDER/llava_existence_results/ llava-1.5-7b```

Look under: `visualizations/output` for `.pdf` file with the attention visualized across layers.

## 🤖 LLM-as-a-judge
Now that we have the results on MME for all variants (2%, 5%, full model), we want to evaluate using LLM-as-a-judge their relative impact.

run the script: ```mllm/eval/gpt4_eval_cot.py``` with specifying the path to the processed data directory as the first argument.

```cmd
  OPENAI_API_KEY=YOUR_API-KEY PYTHONPATH=. python mllm/eval/gpt4_eval_cot.py PATH_TO_MME_RESULTS_FOLDER/LLAVA_SUBSET_FOLDER
```

Note that: PATH_TO_MME_RESULTS_FOLDER/LLAVA_SUBSET_FOLDER should be,a s before, a path to specific MME subset on which you want to run LLM-as-a-judge. For example: `PATH_TO_MME_RESULTS_FOLDER/llava_existence_results/`.
 (currently skip by default).

 
Results are saved into: ``mllm/eval/output/gpt4eval_objects_{model_name}_with_scores.csv``


## 📚 Citation

If you find our work helpful, please consider citing:

```bibtex
@misc{kaduri2024_vision_of_vlms,
      title={What's in the Image? A Deep-Dive into the Vision of Vision Language Models}, 
      author={Omri Kaduri and Shai Bagon and Tali Dekel},
      year={2024},
      eprint={2411.17491},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2411.17491}, 
}
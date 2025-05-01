import os
import subprocess
from threading import Thread
import argparse

# Define constants
mme_tasks = [
    "existence",
    "count",
    "position",
    "color", 
    "OCR",
    "celebrity", 
    "posters",
    "artwork",
    "scene", 
    "landmark",
    "commonsense_reasoning", 
    "code_reasoning",
    "numerical_calculation", 
    "text_translation",
]

images_subsets_tasks = ["posters", "celebrity", "artwork", "scene", "landmark"]

def prepare_paths(mme_gt_folder, mme_data_folder, mme_results_folder):
    """Prepare paths for GT files, data folders, and result folders."""
    mme_gt_files = {task: os.path.join(mme_gt_folder, f"{task}.txt") for task in mme_tasks}
    mme_data_folders = {
        task: os.path.join(mme_data_folder, task + ('/images' if task in images_subsets_tasks else ''))
        for task in mme_tasks
    }
    # if artwork: append /toy_datset
    if "artwork" in mme_tasks:
        mme_data_folders["artwork"] = os.path.join(mme_data_folders["artwork"], "toy_dataset")
    mme_results_folders = {task: os.path.join(mme_results_folder, f"llava_{task}_results") for task in mme_tasks}
    return mme_gt_files, mme_data_folders, mme_results_folders

def stream_output(stream, output_callback):
    """Read output from a stream line-by-line and process it."""
    for line in iter(stream.readline, ''):
        output_callback(line)
    stream.close()

def run_command(command):
    """Run a shell command and stream its output in real-time."""
    # print(f"Executing: {command}")
    try:
        process = subprocess.Popen(
            command,
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )

        # Threads to handle stdout and stderr concurrently
        stdout_thread = Thread(target=stream_output, args=(process.stdout, lambda x: print(x, end='')))
        stderr_thread = Thread(target=stream_output, args=(process.stderr, lambda x: print(x, end='')))

        stdout_thread.start()
        stderr_thread.start()

        # Wait for the process to complete
        process.wait()
        stdout_thread.join()
        stderr_thread.join()

        if process.returncode != 0:
            print(f"Command failed with return code {process.returncode}")
    except Exception as e:
        print(f"An error occurred: {e}")

def run_task(task, k, mme_gt_files, mme_data_folders, mme_results_folders, cache_strategy=None):
    """Run a single task with specified parameters."""
    data_dir = mme_data_folders[task]
    results_path = mme_results_folders[task]
    gt_file_path = mme_gt_files[task]
    # python_path = "/home/projects/talide/omrika/.conda/envs/llavako/bin/python"
    python_path = "python"
    if k is not None:
        if cache_strategy:
            command = (
                f"PYTHONPATH=. {python_path} "
                f"mllm/llava/run_on_coco_from_cache.py  -k {k} -k_selection_method attn_over_gen_percentage "
                f"-data_dir {data_dir} -results_path {results_path} -lavin_gt_file_path {gt_file_path} "
                f"-cache_strategy {cache_strategy} -use_kv_cache_before_rope -skip_existing"
            )
        else:
            command = (
                f"PYTHONPATH=. {python_path} "
                f"mllm/llava/run_on_coco.py -k {k} -k_selection_method attn_over_gen_percentage "
                f"-data_dir {data_dir} -results_path {results_path} -lavin_gt_file_path {gt_file_path} -skip_existing"
            )
    else:
        command = (
            f"PYTHONPATH=. {python_path} "
            f"mllm/llava/run_on_coco.py "
            f"-data_dir {data_dir} -results_path {results_path} -skip_existing"
        )
        
    # Execute the main script
    run_command(command)

    # Create results from outputs
    if cache_strategy:
        create_results_command = (
            f"PYTHONPATH=. python mllm/eval/mme/create_results_from_outputs.py {results_path} "
            f"llava-1.5-7b-masked-{k}-attn_over_gen_percentage {gt_file_path} output_cached_{cache_strategy}.json"
        )
    else:
        create_results_command = (
            f"PYTHONPATH=. python mllm/eval/mme/create_results_from_outputs.py {results_path} "
            f"llava-1.5-7b-masked-{k}-attn_over_gen_percentage {gt_file_path} output.json"
        )
    run_command(create_results_command)

def run_llm_baseline(task, k, mme_gt_files, mme_data_folders, mme_results_folders):
    data_dir = mme_data_folders[task]
    results_path = mme_results_folders[task]
    gt_file_path = mme_gt_files[task]

    command = (
        f"PYTHONPATH=. python mllm/internvl/llm_on_mme_describe.py -k {k} -k_selection_method attn_over_gen_percentage "
        f"-data_dir {data_dir} -results_path {results_path} -lavin_gt_file_path {gt_file_path}"
    )
    run_command(command)

    create_results_command = (
        f"PYTHONPATH=. python mllm/eval/mme/create_results_from_outputs.py {results_path} "
        f"llava-1.5-7b-masked-{k}-attn_over_gen_percentage {gt_file_path} output_cached_llm.json"
    )

    run_command(create_results_command)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mme_gt_folder", type=str, default="mllm/eval/mme/LaVIN/", help="Path to MME ground truth folder")
    parser.add_argument("--mme_data_folder", type=str, default="/home/projects/bagon/shared/waic-shared-projects/vlm/dataset/MME_Benchmark_release_version/", help="Path to MME data folder")
    parser.add_argument("--mme_results_folder", type=str, default="/home/projects/bagon/shared/waic-shared-projects/vlm/", help="Path to MME results folder")
    parser.add_argument("--ks", type=float, nargs="+", default=[0.02, 0.05], help="List of k values to use")
    args = parser.parse_args()

    mme_gt_files, mme_data_folders, mme_results_folders = prepare_paths(args.mme_gt_folder, args.mme_data_folder, args.mme_results_folder)
    
    for task in mme_tasks:
        print(f"------- Running task: {task} -------")
        run_task(task, None, mme_gt_files, mme_data_folders, mme_results_folders)

        for k in args.ks:
            # Run without cache
            try:
                run_task(task, k, mme_gt_files, mme_data_folders, mme_results_folders)
            except Exception as e:
                print(f"An error occurred: {e}")

        run_llm_baseline(task, 0.05, mme_gt_files, mme_data_folders, mme_results_folders)
        print(f"------- Finished task: {task} -------")

if __name__ == "__main__":
    main()

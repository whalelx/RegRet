import json
import os
import sys

def merge_predictions(source_path, target_path):
    """
    Reads 'predict' values from the source JSONL file and updates the
    'predict' field in the target JSONL file, line by line.

    Args:
        source_path (str): Path to the source .jsonl file.
        target_path (str): Path to the target .jsonl file to be updated.
    """
    print(f"Source file: {source_path}")
    print(f"Target file: {target_path}")

    # --- Read 'predict' values from the source file ---
    source_predictions = []
    try:
        with open(source_path, 'r', encoding='utf-8') as f_source:
            for i, line in enumerate(f_source):
                try:
                    data = json.loads(line.strip())
                    predict_value = data.get('label')
                    if predict_value is None:
                        print(f"Warning: 'predict' key missing or null in source file line {i+1}. Storing None.", file=sys.stderr)
                    source_predictions.append(predict_value)
                except json.JSONDecodeError:
                    print(f"Error: Could not decode JSON in source file line {i+1}: {line.strip()}", file=sys.stderr)
                    return # Stop processing on error
    except FileNotFoundError:
        print(f"Error: Source file not found at '{source_path}'", file=sys.stderr)
        return
    except Exception as e:
        print(f"An unexpected error occurred while reading source file: {e}", file=sys.stderr)
        return

    print(f"Read {len(source_predictions)} predictions from source file.")

    # --- Read target file, update 'predict', and store ---
    updated_target_lines = []
    target_line_count = 0
    try:
        with open(target_path, 'r', encoding='utf-8') as f_target:
            target_lines_content = f_target.readlines()
            target_line_count = len(target_lines_content)

            # --- Crucial Check: Ensure line counts match ---
            if len(source_predictions) != target_line_count:
                print(f"Error: Line count mismatch! Source has {len(source_predictions)} lines, Target has {target_line_count} lines.", file=sys.stderr)
                print("Aborting update.", file=sys.stderr)
                return

            for i, line in enumerate(target_lines_content):
                try:
                    target_data = json.loads(line.strip())
                    # Update the 'predict' field with the value from source
                    target_data['label'] = source_predictions[i]
                    updated_target_lines.append(target_data)
                except json.JSONDecodeError:
                    print(f"Error: Could not decode JSON in target file line {i+1}: {line.strip()}", file=sys.stderr)
                    return # Stop processing on error
                except IndexError:
                     # This should not happen if the length check passed, but good safety net
                     print(f"Error: Index mismatch at line {i+1}. This indicates a logic error after the length check.", file=sys.stderr)
                     return

    except FileNotFoundError:
        print(f"Error: Target file not found at '{target_path}'", file=sys.stderr)
        return
    except Exception as e:
        print(f"An unexpected error occurred while reading target file: {e}", file=sys.stderr)
        return

    # --- Write the updated data back to the target file (overwrite) ---
    try:
        with open(target_path, 'w', encoding='utf-8') as f_target_out:
            for item in updated_target_lines:
                # Use ensure_ascii=False to preserve non-ASCII characters
                f_target_out.write(json.dumps(item, ensure_ascii=False) + '\n')
        print(f"\nSuccessfully updated 'predict' field in '{os.path.basename(target_path)}'.")
        print(f"Total lines processed: {len(updated_target_lines)}")
    except IOError as e:
        print(f"Error: Could not write updated data to target file '{target_path}': {e}", file=sys.stderr)
    except Exception as e:
        print(f"An unexpected error occurred while writing updated target file: {e}", file=sys.stderr)


if __name__ == "__main__":
    # Define the file paths
    source_file = "/root/VisualPerceptionToken/eval_res/rp-yu/Qwen2-VL-7b-VPT-CLIP/VSR_region_test/p0.1_k10/round2/generated_predictions.jsonl"
    # target_file = "/root/VisualPerceptionToken/eval_res/saves/staug/large-stage2-newdt/VSR_region_test/p0.1_k10/round2/generated_predictions.jsonl"
    target_file = "/root/VisualPerceptionToken/eval_res/rp-yu/Qwen2-VL-7b-VPT-CLIP/VSR_region_test-newprompt/p0.1_k10/round2/generated_predictions.jsonl"
    # Optional: Create a backup of the target file before overwriting
    # import shutil
    # backup_file = target_file + ".bak"
    # try:
    #     print(f"Creating backup: {backup_file}")
    #     shutil.copy2(target_file, backup_file)
    # except FileNotFoundError:
    #     print(f"Warning: Target file '{target_file}' not found for backup.", file=sys.stderr)
    # except Exception as e:
    #     print(f"Warning: Could not create backup file '{backup_file}': {e}", file=sys.stderr)

    # Run the merging process
    merge_predictions(source_file, target_file)
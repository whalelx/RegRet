import json
import random

def sample_json_files(file_paths, sample_sizes, output_paths):
    if not (len(file_paths) == len(sample_sizes)):
        raise ValueError("Number of input files and sample sizes must be equal.")
    if len(output_paths) != 1:
        raise ValueError("Only one output path should be provided for merging.")

    all_sampled_data = []  # List to store all sampled items
    for i, file_path in enumerate(file_paths):
        sample_size = sample_sizes[i]

        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except FileNotFoundError:
            raise FileNotFoundError(f"File not found: {file_path}")
        except json.JSONDecodeError:
            raise json.JSONDecodeError(f"Invalid JSON in file: {file_path}", doc=open(file_path, 'r').read(), pos=0)
        except OSError as e:
            raise OSError(f"Error reading file: {file_path} - {e}")

        if not isinstance(data, list):
            print(f"Warning: File {file_path} does not contain a list.  Attempting to sample as is.")
            try:
                sampled_data = random.sample(data, min(sample_size, len(data)))
            except TypeError:
                raise TypeError(f"Cannot sample from file: {file_path}.  Expected a list or other samplable type, got {type(data)}.")
        else:
            sampled_data = random.sample(data, min(sample_size, len(data)))
        all_sampled_data.extend(sampled_data)  # Add sampled data to the combined list

    random.shuffle(all_sampled_data)  # Shuffle the combined data

    output_path = output_paths[0] # Get the single output path
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(all_sampled_data, f, indent=4, ensure_ascii=False)
    except OSError as e:
        raise OSError(f"Error writing to file: {output_path} - {e}")
    print(f"Sampled and merged data from {len(file_paths)} files, and saved to {output_path}")



if __name__ == "__main__":
    import os
    root_dir = "../../LLaMA-Factory/data/"
    input_files = ['osd-vpt.json', 'mix-vpt-spatial.json','mix-vpt-llava.json']
    input_files = [os.path.join(root_dir, file) for file in input_files]
    sample_sizes = [100000, 140000, 200000]
    output_file = ['Mix_OSD_VPT_large.json']  # Single output file

    try:
        sample_json_files(input_files, sample_sizes, output_file)
    except (ValueError, FileNotFoundError, json.JSONDecodeError, OSError) as e:
        print(f"An error occurred: {e}")

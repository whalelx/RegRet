import json, os

EVAL_TEMPLATE_PATH = os.path.join(os.path.dirname(__file__),"../datasets/build_eval_ds/srgpt-bench-temp.jsonl")

def build_srgpt_evalfile(final_answers, output_file_path):
    with open(EVAL_TEMPLATE_PATH, 'r') as infile, open(output_file_path, 'w') as outfile:
        for line in infile.readlines():
            data = json.loads(line)
            for item in final_answers:
                item_id = os.path.dirname(item["images"][0])
                item_id = os.path.basename(item_id)
                if data["question_id"] == item_id:
                    data["pred"] = item["answer"]
                    data["image_path"] = item["images"][0]
                    outfile.write(json.dumps(data) + '\n')
                    break
            else:
                print("not found", data["question_id"], item_id)

    print(f"Evaluation file saved to {output_file_path}")
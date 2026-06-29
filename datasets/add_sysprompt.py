import json

# The system prompt to add
system_prompt = "You are a helpful spatial reasoning assistant, good at telling fine-grained image regions. For this question, point out the regions helpful to answer the question. "
file_path = "/root/VisualPerceptionToken/LLaMA-Factory/data/train_0428_stage1.json"

# Read the existing JSON data
with open(file_path, 'r') as f:
    data = json.load(f)

# Add the system_prompt to each item in the list
for item in data:
    # Check if the item is a dictionary before adding the key
    if isinstance(item, dict):
        ques = item['conversations'][0]["value"] 
        # ss = system_prompt.join(ques.split("\n"))
        ss = system_prompt + ques
        item['conversations'][0]["value"] = ss
        

# Convert the modified data back to a JSON string with indentation
updated_json_string = json.dumps(data, indent=4)

# Write the updated JSON string back to the file
with open(file_path, 'w') as f:
    f.write(updated_json_string)

print(f"Successfully added system prompt and updated {file_path}")
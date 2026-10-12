import json

# Load the previously generated data
with open('output.json', 'r') as f:
    data = json.load(f)

# Print the exact JSON structure requested
print(json.dumps(data))
import json

with open('final.json', 'r') as f:
    s = f.read()

# Split into chunks of 3000 chars
chunk_size = 3000
chunks = [s[i:i+chunk_size] for i in range(0, len(s), chunk_size)]

print(f"Total length: {len(s)}")
print(f"Number of chunks: {len(chunks)}")

for i, chunk in enumerate(chunks):
    print(f"--- CHUNK {i+1} ({len(chunk)} chars) ---")
    print(chunk)
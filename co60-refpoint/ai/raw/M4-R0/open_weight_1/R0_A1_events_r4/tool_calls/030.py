with open("answer.json","r") as f:
    content = f.read()
print(f"Total JSON length: {len(content)} characters")
# Print the first 1000 and last 500
print("\n--- First part ---")
print(content[:1000])
print("\n--- Last part ---")
print(content[-500:])
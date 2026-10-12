import sys
with open('events.json', 'r') as f:
    content = f.read()
sys.stdout.write(content)
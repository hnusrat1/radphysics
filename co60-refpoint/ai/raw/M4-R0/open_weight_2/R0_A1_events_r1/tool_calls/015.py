import json
import sys

with open('final.json', 'r') as f:
    s = f.read()

sys.stdout.write(s)
sys.stdout.flush()
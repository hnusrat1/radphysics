# Quick check: rerun MC with smaller field (2x2) and see scatter/primary ratio; also with 15 cm deep phantom
import numpy as np, time
exec(open('/dev/stdin').read()) if False else None
# reload the module code (redefine run with parameters) - simpler: copy the key function with params
from importlib import reload
import sys

# We'll just re-run with modified source sampling: define a parametrized runner quickly by editing copy
code = open('mc_module.py').read() if __import__('os').path.exists('mc_module.py') else None
# Instead rewrite inline: load the function from previous cell isn't available. Redefine quickly.
print("skip")
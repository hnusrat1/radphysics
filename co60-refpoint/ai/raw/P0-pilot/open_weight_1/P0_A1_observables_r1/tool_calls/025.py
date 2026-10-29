import os
print("Current directory:", os.getcwd())
print("Files:", os.listdir('.'))
# Check if mc_physics.py exists
if os.path.exists('mc_physics.py'):
    print("mc_physics.py exists, size:", os.path.getsize('mc_physics.py'))
else:
    print("mc_physics.py does NOT exist")
import subprocess
print(subprocess.run("head -50 /tmp/data/xcom_water.txt; echo ----; head -60 /tmp/data/estar_water.txt; wc -l /tmp/data/*.txt",shell=True,capture_output=True,text=True).stdout)
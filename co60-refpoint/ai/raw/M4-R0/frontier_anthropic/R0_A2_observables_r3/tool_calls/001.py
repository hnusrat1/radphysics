
import subprocess
print(subprocess.run("head -30 /tmp/data/xcom_water.txt; echo ----; head -30 /tmp/data/estar_water.txt; wc -l /tmp/data/*.txt; ls -la /tmp/data",shell=True,capture_output=True,text=True).stdout)

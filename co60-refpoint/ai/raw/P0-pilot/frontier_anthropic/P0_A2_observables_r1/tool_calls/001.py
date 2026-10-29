
import subprocess
print(subprocess.run(['bash','-lc','cat /tmp/data/xcom_water.txt | head -80; echo ----; cat /tmp/data/estar_water.txt | head -120; wc -l /tmp/data/*.txt'],capture_output=True,text=True).stdout)

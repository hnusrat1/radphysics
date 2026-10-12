
print(subprocess.run("cat /tmp/data/xcom_water.txt | tail -22; echo ----; tail -32 /tmp/data/estar_water.txt",shell=True,capture_output=True,text=True).stdout)

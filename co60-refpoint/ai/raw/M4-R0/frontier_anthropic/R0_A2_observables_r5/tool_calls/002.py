
print(subprocess.run("tail -25 /tmp/data/xcom_water.txt; echo ----; tail -32 /tmp/data/estar_water.txt",shell=True,capture_output=True,text=True).stdout)

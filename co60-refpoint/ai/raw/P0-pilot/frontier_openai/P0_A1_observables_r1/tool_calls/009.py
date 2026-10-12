import numpy as np
mid=np.arange(.025,1.4,.05); w=np.exp(-mid/.55); a=.13*w/w.sum(); print(len(a),np.round(a,6).tolist(),a.sum(),np.dot(a,mid)/.13)
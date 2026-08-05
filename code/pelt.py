import numpy as np
import ruptures as rpt
A = np.random.normal(0, 1, size=(1000, )).tolist()
B = np.random.normal(2, 1, size=(1000, )).tolist()
data = np.array(A + B)
mad = np.median(np.abs(data - np.median(data)))/0.6745
algo = rpt.Pelt(model="l2").fit(data)
bkps = algo.predict(pen=2*mad**2*np.log(len(data)))[:-1]

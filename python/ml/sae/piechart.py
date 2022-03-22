import matplotlib.pyplot as plt
import numpy as np


plt.ion()
plt.figure()
plt.show()
numCat=5
dat1=[] 
for i in range(numCat): 
    ind,=np.where(y_pred==i) 
    rounded=[round(num1,2) for num1 in np.mean(cod2[ind],axis=0)] 
    print(rounded) 
    dat1.append(len(ind)) 
    
plt.pie(dat1,autopct='%1.0f%%')

#plt.legend(['class 0','class 1','class 2','class 3','class 4','class 5','class 6','class 7','class 8','class 9'],loc='upper left')
plt.legend(['class 0','class 1','class 2','class 3','class 4'],loc='upper left')


plt.savefig('/tmp/piechart1.png')

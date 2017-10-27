import matplotlib.pyplot as plt

epochs =[1,3,7,9,15]
lstm = [84.8,87.5,87.66,87.75,87.9]
gru =[84.2,87.6,87.59,87.48,87.5]
plt.plot(epochs,lstm,'g--',epochs,gru,'r--')
plt.show()
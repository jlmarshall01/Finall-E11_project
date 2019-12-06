import math
from scipy import stats
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches



####################################
#UPSTAIRS DATA
####################################

csv_file = input("What .csv file is upstairs?")
data = pd.read_csv(csv_file)

#create spectra of energies?????
spectra = np.array(data)
spectrum = spectra.sum(axis=0)


#plt.plot(spectrum)
#plt.yscale('log')


spectrum_resize = np.resize(spectrum,(int(spectrum.shape[0]/4),4))
spectrum_rebin = spectrum_resize.sum(axis=1)
#plt.plot(spectrum_rebin)
#plt.yscale("log")

#counts graph
counts = spectra.sum(axis=1)
#counts_resize =np.resize(counts,int(counts.shape[0]/4,4))
#counts_rebin = counts_resize.sum(axis=1)

#define min and max and nbins u loser

xmin = np.min(counts)
xmax = np.max(counts)
#print(xmin)
#print(xmax)
nbins = xmax - xmin + 1

#mean and uncertainty (sigma)

mu = np.mean(counts)
sigma = np.std(counts)
print('mu = ',mu)
print("sigma CPS =", sigma)

#compare to normal curvie curve
def gaussian(x, mu, sigma):
    func = 1/(np.sqrt(2*math.pi) * sigma) * np.exp(-1/2 * np.power((x-mu)/sigma,2))
    return func
x = np.arange(xmin,xmax+1)
plt.plot(x,gaussian(x, mu, sigma), label='normal curve of upper level')
plt.hist(counts,bins=nbins,density=True, label='data of upper level')
plt.xlabel('CPS')
plt.ylabel('Times Detected')




#####################################
#DOWNSTAIRS DATA
#####################################
csv_file2 = input("What .csv file is downstairs?")
data2 = pd.read_csv(csv_file2)

#create spectra of energies?????
spectra2 = np.array(data2)
spectrum2 = spectra2.sum(axis=0)


#plt.plot(spectrum)
#plt.yscale('log')


spectrum_resize2 = np.resize(spectrum2,(int(spectrum2.shape[0]/4),4))
spectrum_rebin2 = spectrum_resize2.sum(axis=1)
#plt.plot(spectrum_rebin)
#plt.yscale("log")

#counts graph
counts2 = spectra2.sum(axis=1)
#counts_resize =np.resize(counts,int(counts.shape[0]/4,4))
#counts_rebin = counts_resize.sum(axis=1)

#define min and max and nbins u loser

xmin2 = np.min(counts2)
xmax2 = np.max(counts2)
#print(xmin2)
#print(xmax2)
nbins2 = xmax2 - xmin2 + 1

#mean and uncertainty (sigma)

mu2 = np.mean(counts2)
sigma2 = np.std(counts2)
print('mu = ',mu2)
print('sigma CPS = ', sigma2)


#compare to normal curvie curve
def gaussian(x2, mu2, sigma):
    func = 1/(np.sqrt(2*math.pi) * sigma) * np.exp(-1/2 * np.power((x-mu)/sigma,2))
    return func
x2 = np.arange(xmin2,xmax2+1)
plt.plot(x,gaussian(x2, mu2, sigma2), label='normal curve of bottom level')
plt.hist(counts2,bins=nbins2,density=True, label='data of bottom level')


############################################
#PLOT LABLES
############################################

#title = input('what do you want the file to be named?   **Put name in quotes.')

#define legend




plt.xlabel('CPS')
plt.ylabel('Times Detected')
plt.legend()
plt.show()


##########################################
#finding uncertaINTY OF THE MEAN
##########################################

def mean_uncertainty(mu, N):
	func = mu/np.sqrt(N)
	return func

print('uncertainty mean top:', mean_uncertainty(mu, spectrum))
print('uncertainty mean bottom:', mean_uncertainty(mu2, spectrum2))

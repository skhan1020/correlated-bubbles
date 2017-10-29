### This code analyzes the crime data prepared by the FBI in the state of Arizona (2016) and makes predictions about 
### how it is distributed over different communities/ethnicities, how the ages of the registered offenders are related 
### to the ages of the victims and it tries to determine a mathematical relationship between the frequency of occurences
### and the month of the year using a least squares method

import csv
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm
from mpl_toolkits.mplot3d import Axes3D
import datetime

raceO = []
sexO = []
ageO = []

with open('offender.csv') as csvDataFile:
    csvReader = csv.reader(csvDataFile)
    next(csvReader)
    data = list(csvReader)

    for row in data:
        raceO.append(row[2])
        sexO.append(row[3])
        ageO.append(row[4])

###--- Histogram of Offenders according to Races (Reported) ---#

labels = ['W', 'B', 'I', 'A', 'P','U']
temp = np.array(range(len(labels)))
lst = np.zeros(len(labels))

for i in range(len(data)):
    if raceO[i] in labels:
        lst[labels.index(raceO[i])] += 1

plt.bar(temp, lst, align ='center', color ='skyblue')
plt.xticks(temp,labels)
plt.xlabel('Race')
plt.ylabel('Number of Offenders')
plt.title('Bar Chart of Number of Offenders vs Race')
plt.savefig('Bar-Chart.pdf', format ='pdf')
plt.show()


### --- Correlation between Ages of Offenders and Ages of Victims using a Gaussian Model---- ###

ageV = []

with open('victim.csv') as csvDataFile:
    csvReader = csv.reader(csvDataFile)
    next(csvReader)
    data1 = list(csvReader)

    for row in data1:
        ageV.append(row[4])

## Convert ages to integers 
for i in range(0,len(data)):                # Check for Missing Entries : Missing Entires = 0
    if(ageO[i]) != '':
        ageO[i] = int(ageO[i])
    else:
        ageO[i] = 0

for i in range(0,len(data1)):                # Check for Missing Entries : Missing Entires = 0
    if(ageV[i]) != '':
        ageV[i] = int(ageV[i])
    else:
        ageV[i] = 0

# Calculate the average and standard deviation of the data
mu_x = np.mean(ageO)
mu_y = np.mean(ageV)

sigma_x = np.std(ageO)
sigma_y = np.std(ageV)
#print(sigma_x, sigma_y)

xy = [a*b for a,b in zip(ageO,ageV)]
mu_xy = np.mean(xy)
cov_xy = np.sqrt(mu_xy - mu_x*mu_y)
#print(cov_xy)

# Define the Gaussian Parameters
mu = np.array([mu_x,mu_y])
sigma = np.array([[sigma_x,cov_xy],[cov_xy,sigma_y]])

#resolution for plot
resolution = 50

#Define the grid for visualisation
X, Y = np.meshgrid(np.linspace(0,40, resolution), np.linspace(0,40, resolution))

#Compute the bivariate Gaussian Distribution using the Ages of Offenders and Victims as the two Variables
c1 = 1./(2.*np.pi)
c2 = 1./(np.sqrt(np.linalg.det(sigma)))

diff = np.concatenate((np.asmatrix(X.flatten(1)).T - mu[0],np.asmatrix(Y.flatten(1)).T - mu[1]),1)

exponent = -0.5 * np.diag(np.dot(np.dot(diff, np.linalg.inv(sigma)), diff.T))

Prob = c1*c2*np.exp(exponent)
Prob = np.reshape(Prob, X.shape).T

#Contour plot showing correlation of the Two Variables
plt.figure()
plt.contour(X, Y, Prob)
plt.xlabel('Age of Offender')
plt.ylabel('Age of Victim')
plt.title('Contour Plot - Ages of Offenders and Victims')
plt.savefig('Contour-Plot.pdf', format ='pdf')

#Surface plot of the Age Distribution 

fig = plt.figure()
ax = fig.gca(projection='3d')
surf = ax.plot_surface(X, Y, Prob, rstride=1, cstride=1,cmap=matplotlib.cm.jet,linewidth=0, antialiased=False)
plt.xlabel('Age of Offender')
plt.ylabel('Age of Victim')
plt.title('Normal Distbn. (Offenders, Victims)')
fig.colorbar(surf)
plt.savefig('Gaussian-Distribution.pdf', format ='pdf')
plt.show()


### --- Trend of Crimes Against Months in 2016 --- ###

dateCrime = []

with open('incident.csv') as csvDataFile:
    csvReader = csv.reader(csvDataFile)
    next(csvReader)
    data2 = list(csvReader)

    for row in data2:
        dateCrime.append(row[2])

Ninc = np.zeros(12)

for i in range(0,len(data2)):
    if dateCrime[i][:2] == '01':
        Ninc[0] += 1
    if dateCrime[i][:2] == '02':
        Ninc[1] += 1
    if dateCrime[i][:2] == '03':
        Ninc[2] += 1
    if dateCrime[i][:2] == '04':
        Ninc[3] += 1
    if dateCrime[i][:2] == '05':
        Ninc[4] += 1
    if dateCrime[i][:2] == '06':
        Ninc[5] += 1
    if dateCrime[i][:2] == '07':
        Ninc[6] += 1
    if dateCrime[i][:2] == '08':
        Ninc[7] += 1
    if dateCrime[i][:2] == '09':
        Ninc[8] += 1
    if dateCrime[i][:2] == '10':
        Ninc[9] += 1
    if dateCrime[i][:2] == '11':
        Ninc[10] += 1
    if dateCrime[i][:2] == '12':
        Ninc[11] += 1

months = []

for i in range(1,13):
    months.append(datetime.date(2016,i,1).strftime('%B'))

# Fit a Linear Model 
x = np.array(range(1,13))
(slope, intercept) = np.polyfit(x, Ninc, 1)

yp = np.polyval([slope, intercept], x)

plt.figure()
plt.scatter(x, Ninc, edgecolor = 'b', color = 'w', marker = 'o')
plt.xticks(x, months)
plt.plot(x, yp)
plt.xlabel('Month')
plt.ylabel('Number of Incidents')
plt.title('Number of Incidents vs Month (2016): Linear Model (slope = %s, intercept = %s)' %(slope,intercept))
plt.show()

### --- Fit Polynomials of Orders (1 - 5) for data on Number of incidents vs months Using Least Squares Method ---###

model_order = [1, 2, 3, 4, 5]
testx = np.linspace(0, len(x), 100) 

for i in model_order:
    X = np.zeros((x.shape[0],i+1))
    testX = np.zeros((testx.shape[0], i+1))
    for k in range(i+1):
        X[:,k] = np.power(x,k)              # Create the design matrix
        testX[:,k] = np.power(testx, k)

    w = np.dot(np.linalg.inv(np.dot(X.T,X)), np.dot(X.T,Ninc))    # Normal Equations that one obtains from Least Mean Squared Error Method
    print(" w for model order", i, "=", w)

    #Predicted Values based on the calculated parameters
    predicted_N = np.dot(testX, w)


    # Plot the data
    plt.figure()
    plt.scatter(x, Ninc, color='k', edgecolor = 'k')
    plt.xlabel('Months')
    plt.ylabel('Number of Incidents')
    plt.title(' Plot of model with polynomial order %s'%i)

    #Plot the model on top of data
    plt.plot(testx, predicted_N, color ='r', linewidth = '2')
    plt.savefig('Sampled-plots-{0}.pdf'.format(i), format ='pdf')
    plt.show()


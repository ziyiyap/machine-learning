import numpy as np
import matplotlib.pyplot as plt
# Model that learns y = mx + c

try:
    gradient,intercept = input("Enter values of m and c: <m> <c>\n").split()
except ValueError:
    gradient,intercept= 1,0

domain = 1000
xlist = [i for i in range(0,domain, max(int(domain/50),1))]

ylist = [float(gradient)*x + float(intercept) for x in xlist]

np.random.seed(42)


X = np.array(xlist) 
noise = np.random.normal(-0.2,0.2,size=X.shape) #add noise to keep value away from the actual value
y_actual = np.array(ylist) + noise

# Outputs list of indices
indices = np.arange(len(X))

#Random shuffles
np.random.shuffle(indices)


X_shuffled = X[indices]
y_shuffled = y_actual[indices]

split = int(len(X_shuffled) * 0.8) # 80/20 ratio

X_train, X_test = X_shuffled[:split], X_shuffled[split:]
y_train, y_test = y_shuffled[:split], y_shuffled[split:]

c = (np.sum(y_train)*np.sum(X_train**2) - np.sum(X_train)*np.sum(X_train*y_train))/(len(X_train)*np.sum(X_train**2) - (np.sum(X_train)**2))

m = (np.sum(X_train * y_train) - c*np.sum(X_train))/np.sum(X_train**2) # Prove in samsung notes

y_pred = m * X_test + c

ymean = np.mean(y_test)
xmean = np.mean(X_test)

SST = np.sum((y_test-ymean)**2)
SSE = np.sum((y_pred-y_test)**2)
SSreg = np.sum((y_pred-ymean)**2)

r_squared = 1 - (SSE/SST)

correlation = np.round(np.sum(((X_test-xmean)*(y_test-ymean))) / np.sqrt(np.sum((X_test-xmean)**2)*np.sum((y_test-ymean)**2)),7)

print(f"SST = {round(SST,5)}\nSSE = {round(SSE,5)}\nR² = {round(r_squared,5)}\nR / Correlation = {correlation}")

plt.scatter(X_test,y_test, color='red')
plt.plot(X_test,y_pred)
plt.xlabel("X")
plt.ylabel("Y")
plt.title(f"Graph: y = {round(m,3)}x + {round(c,3)}")
plt.show()
plt.close()


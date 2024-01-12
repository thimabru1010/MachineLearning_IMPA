# ExercÃ­cio 3(a)

import numpy as np

rng = np.random.default_rng(seed=0)
x = np.array([[3, 2, 5, 7, 8, 9, 1, 4, 5]]).T
eps = rng.normal(size=x.shape)
y = 0.77 + 0.15 * x - 1.22 * (x ** 2) + 0.89 * (x ** 3) + eps

var1 = ((x - x.mean()) * (y - y.mean())).sum() / ((x - x.mean()) ** 2).sum()
var2 = y.mean() - var1 * x.mean()
var3 = var2 + var1 * x
var4 = ((var3 - y) ** 2).mean()

#################################################################################

# ExercÃ­cio 3(c)

xx = np.concatenate([np.ones(x.shape), x], axis=1)
var5 = np.linalg.inv(xx.T @ xx) @ xx.T @ y
var6 = xx @ var5
var7 = ((var6 - y) ** 2).mean()

#################################################################################

# ExercÃ­cio 4 (primeira seÃ§Ã£o de cÃ³digo)

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis as QDA
from sklearn.linear_model import LogisticRegression as LR
from sklearn.naive_bayes import GaussianNB as NB
from sklearn.neighbors import KNeighborsClassifier as kNN
from sklearn import preprocessing

df = pd.read_csv("./soccer.csv")

X = df.drop("target", axis=1)
y = df[["target"]]

X_train, y_train = X.iloc[:2560], y.iloc[:2560]
X_test, y_test = X.iloc[2560:], y.iloc[2560:]

#################################################################################

# ExercÃ­cio 4 (segunda seÃ§Ã£o de cÃ³digo)

X_train = X_train.drop(["home_team", "away_team"], axis=1)
X_test = X_test.drop(["home_team", "away_team"], axis=1)

scaler = preprocessing.StandardScaler().fit(X_train)
X_train = preprocessing.scale(X_train)
X_test = scaler.transform(X_test)

#################################################################################

# ExercÃ­cio 5(b)

import statsmodels.api as sm
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, KFold
import matplotlib.pyplot as plt
from sklearn.linear_model import Lasso
from sklearn import preprocessing

bodyfat = pd.read_csv("../bodyfat.csv")

X = bodyfat.drop(columns=["BodyFat","Density"])
y = bodyfat["BodyFat"]
X_train, X_test, y_train, y_test = train_test_split(X,y,
test_size=0.2,
random_state = 10
)

kf = KFold(n_splits = 5, shuffle = True, random_state = 10)
cv_fold = np.zeros(len(y_train)).astype(int)
for i, (_, fold_indexes) in enumerate(kf.split(X_train)):
    cv_fold[fold_indexes] = int(i)
    
#################################################################################

# ExercÃ­cio 5(f)

alphas = 10**np.linspace(5,-2,100)

#################################################################################

# ExercÃ­cio 6(d-i)

import numpy as np
n = 50
Sigma = np.diag([(i/10)**10 for i in range(1,n+1)])
np.random.seed(0)
X = np.array([np.ones(n),np.random.normal(0,1,n)]).T
beta = np.array([1,0.25])
epsilon = np.random.multivariate_normal(np.zeros(n),Sigma)
y = X @ beta + epsilon

#################################################################################

# ExercÃ­cio 8(e)

import numpy as np
import matplotlib.pyplot as plt

np.random.seed(0)

sample_size = 100 

x = np.linspace(-4, 4, sample_size)
x = x.reshape(-1, 1)

y = x + np.random.normal(0, 1, sample_size)

#################################################################################

# ExercÃ­cio 8(g)

sigma = 1
xe = x * np.random.normal(1,sigma,100)
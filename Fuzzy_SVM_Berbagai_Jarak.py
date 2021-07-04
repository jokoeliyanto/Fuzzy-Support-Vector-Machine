import numpy as np
import cvxopt

# Pilihan Fungsi Kernel
def linear(x,z):
    return np.dot(x,z.T)

def polynomial(x, z, p=5):
    return (1 + np.dot(x, z.T)) ** p


def gaussian(x, z, sigma=0.1):
    return np.exp(-np.linalg.norm(x - z, axis=1) ** 2 / (2 * (sigma ** 2)))

# Pilihan Fungsi Jarak
def euclidean(x,y):
    return np.sqrt(sum(pow(a-b, 2) for a, b in zip(x,y)))

def akar_ke_n(nilai, akar_n):
    nilai_akar=1/float(akar_n)
    return round(float(nilai)**float(nilai_akar),3)

def minkowski(x,y,p_value):
    return akar_ke_n(sum(pow(abs(a-b), p_value) for a, b in zip(x,y)), p_value)

def chebisev(x,y):
    return max((a-b) for a,b in zip(x,y))

def minkowski_chebisev(w1,w2,x,y,p_value):
    return w1*minkowski(x,y,p_value)+w2*chebisev(x,y)

# Fuzzyfikasi

def pusat_kelas(X,y):
    x_neg=X[y==-1]
    x_pos=X[y==1]
    x_neg_center=np.mean(x_neg, axis=0)
    x_pos_center=np.mean(x_pos, axis=0)
    center=[]
    center.append(x_neg_center)
    center.append(x_pos_center)
    return center


class Fuzzy_SVM:
    def __init__(self, kernel=gaussian, jarak=euclidean, C=1, delta=1e-8):
        self.kernel = kernel
        self.jarak  = jarak
        self.C      = C
        self.delta  = delta
        
    def fit(self, X, y, w1=0.5, w2=0.5, p_value=1):
        self.y = y
        self.X = X
        m, n = X.shape

        # Menghitung Kernel
        self.K = np.zeros((m, m))
        for i in range(m):
            self.K[i, :] = self.kernel(X[i, np.newaxis], self.X)
        
        # Menghitung Pusat Kelas
        self.center = pusat_kelas(X,y)
        
        # Menghitung Radius
        jarak_neg=[]
        X_neg=self.X[self.y==-1]
        for i in X_neg:
            jrk_n=float(self.jarak(i, self.center[0]))
            jarak_neg.append(jrk_n)
        self.r_neg = np.max(jarak_neg)
        
        jarak_pos=[]
        X_pos=self.X[self.y==1]
        for i in X_pos:
            jrk_p=float(jarak_euclidean(i, self.center[1]))
            jarak_pos.append(jrk_p)
        self.r_pos = np.max(jarak_pos)
        
        s_i=[]
        for i in range(m):
            y_i = y[i]
            if (y_i) == -1:
                X_i=X[i]
                jrk = float(self.jarak(X_i, self.center[0]))
                s = 1 - (jrk/(self.r_neg+self.delta))
                s_i.append(s)
            elif (y_i) == 1:
                X_i=X[i]
                jrk = float(self.jarak(X_i, self.center[1]))
                s = 1 - (jrk/(self.r_pos+self.delta))
                s_i.append(s)
        
        self.s_i=s_i

        # Menyelesaikan Masalah Optimasi dengan CVXOPT

        P = cvxopt.matrix(np.outer(y, y) * self.K)
        q = cvxopt.matrix(-np.ones((m, 1)))
        G = cvxopt.matrix(np.vstack((np.eye(m) * -1, np.eye(m))))
        h = cvxopt.matrix(np.hstack((np.zeros(m), self.s_i * self.C)))
        A = cvxopt.matrix(y, (1, m), "d")
        b = cvxopt.matrix(np.zeros(1))
        cvxopt.solvers.options["show_progress"] = False
        sol = cvxopt.solvers.qp(P, q, G, h, A, b)
        self.alphas = np.array(sol["x"])
        

    def get_parameters(self, alphas):
        threshold = 1e-5

        sv = ((alphas > threshold) * (alphas < self.C)).flatten()
        w = np.dot(X[sv].T, alphas[sv] * self.y[sv, np.newaxis])
        self.w = w[::]
        self.b = np.mean(
            self.y[sv, np.newaxis]
            - self.alphas[sv] * self.y[sv, np.newaxis] * self.K[sv, sv][:, np.newaxis]
        )
        
        self.supportVectors = self.X[sv]
        return sv
    
    
    def predict(self, X):
        y_predict = np.zeros((X.shape[0]))
        sv = self.get_parameters(self.alphas)

        for i in range(X.shape[0]):
            y_predict[i] = np.sum(
                self.alphas[sv]
                * self.y[sv, np.newaxis]
                * self.kernel(X[i], self.X[sv])[:, np.newaxis]
            )
        return np.sign(y_predict + self.b)

if __name__ == "__main__":
    np.random.seed(1)
    
    X, y = generateBatchBipolar(100, mu=0.3, sigma=0.3)
    svm = Fuzzy_SVM(kernel=linear, jarak=chebisev)
    svm.fit(X, y, p_value=1)
    y_pred = svm.predict(X)
    

    print(f"Accuracy: {sum(y==y_pred)/y.shape[0]}")

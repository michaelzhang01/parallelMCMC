# Imports
import sys,time
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from scipy import linalg
from scipy.optimize import fmin
from statsmodels.nonparametric.kernel_regression import KernelReg

dotTime = 0.0

class llKernelReg():
  """
    Class for Local Polynomial Kernel Regression
  """
  # Kernel Regression for Local Linear Systems  

  def __init__(self, X, Y, numOneOut=100, maxIter=50):
    """
      Constructor
    
      Parameters
      ----------
      X         : 2D array where each row is a different realization and each column is a different dimension.
      Y         : 1D array with training data.
      numOneOut : Number of points where the leave-out error estimation is evaluated
      maxIter   : Maximum number of bandwidth optimization iterations

      Notes
      -----
      When the object is constructed the optimal bandwidth is computer.
      This is not recomputed during fit, but only when new points are added.

    """  
    self.trainSize = X.shape[0]
    self.dims      = X.shape[1]
    self.X         = X
    self.Y         = Y
    self.maxIter   = maxIter
    self.numOneOut = numOneOut
    # Compute Initial Guess
    stdX = np.std(self.X, axis=0)
    bw0 = 1.06*stdX*self.trainSize**(-1./(4.0 + self.dims))
    limBW = np.mean(np.min(sp.spatial.distance_matrix(self.X, self.X) + np.diag(1.0e30*np.ones(self.X.shape[0])),axis=1),axis=0)
    print("Bandwidth limit: %.3f" % (limBW))
    # Find the optimal bandwidth with least-squares cross-validation (ls_cv)
    self.bw = fmin(self.minBW, x0=bw0, args=(self.X,self.Y), maxiter=self.maxIter, maxfun=self.maxIter, disp=0) 
    for loopA in range(len(self.bw)):
      if(self.bw[loopA] < limBW):        
        self.bw[loopA] = limBW
      print("Optimized bandwidth: %.3f for dimension %d" % (self.bw[loopA],loopA))
    print("Regressor Initialization Completed.")

  def addData(self,addX,addY):
    """
      Add training data to regressor
    
      Parameters
      ----------
      addX : 2D array with additional training location
      addY : 1D array with additional training values.

      Notes
      -----
      The optimal bandwidth is re-computed after adding training points

    """    
    # Add data
    self.X = np.row_stack(self.X,addX)
    self.Y = np.row_stack(self.Y,addY)
    self.trainSize = X.shape[0]
    self.dims      = X.shape[1]
    # Compute Initial Guess
    stdX = np.std(self.X, axis=0)
    bw0 = 1.06*stdX*self.trainSize**(-1./(4.0 + self.dims))    
    self.bw = fmin(self.minBW, x0=bw0, args=(self.X,self.Y), maxiter=self.maxIter, maxfun=self.maxIter, disp=0)
    print("Bandwidth Recomputed.")

  def formXMatrixLeaveOut(self,X,Y,leaveOut):
    """
      Form the X Matrix by leaving one row out
    
      Parameters
      ----------
      X : 2D array with training locations [data size x dimensions]
      Y : 1D array with training values.
      leaveOut: Point location (row number) to leave out. This is used to compute
                the loss function on the bandwidth.

    """
    if(leaveOut<0):
      print("ERROR: Negative Leaveout.")
      exit(-1)

    leftOutX = X[leaveOut,:]
    leftOutY = Y[leaveOut]
    X     = np.delete(X,(leaveOut),axis=0)
    leftY = np.delete(Y,(leaveOut))

    # Construct the Polynomial Matrix
    matX = np.empty((X.shape[0],X.shape[1]+1))
    # First column is all ones
    matX[:,0] = np.ones(X.shape[0])
    # Add polynomial terms
    matX[:,1:] = X-leftOutX

    # Return Matrix and left out vector
    return matX,leftY,leftOutX,leftOutY

  def formXMatrix(self,X,newX):
    """
      Form X Matrix with all training points at a specific new location
    
      Parameters
      ----------
      X : 2D array with training locations [data size x dimensions]
      Y : 1D array with training values.
      newX: 2D array with additional training locations [data size x dimensions]

    """
    # Construct the Polynomial Matrix
    matX = np.empty((X.shape[0],X.shape[1]+1))
    # First column is all ones
    matX[:,0] = np.ones(X.shape[0])
    # Add polynomial terms
    matX[:,1:] = X - newX

    # Return Matrix and left out vector
    return matX

  def evalKernel(self,x):
    """
      Evaluate Standard Multivariate Gaussian Kernel
  
      Parameters
      ----------
      x : 1D with normalized location where to evaluate the kernel

    """
    size = x.reshape(len(x),-1).shape[1]
    # return (1.0/(np.sqrt((2.0*np.pi)**size)))*np.exp(-0.5*np.dot(x.transpose(),x))
    return (1.0/(np.sqrt((2.0*np.pi)**size)))*np.exp(-0.5*np.sum(np.square(x),axis=1))

  def formWMatrixLeaveOut(self,X,bw,leaveOut):
    """
      Form Kernel Weight Matrix by leaving out one training location
    
      Parameters
      ----------
      X : 2D array with training locations [data size x dimensions]
      bw: 1D array with bandwidth.
      leaveOut : integer with the row number to leave out

    """
    if(leaveOut<0):
      print("ERROR: Negative Leaveout.")
      exit(-1)
    
    leftOutX = X[leaveOut,:]
    X = np.delete(X,(leaveOut),axis=0)

    # Construct the Diagonal Weighting Matrix
    matW = np.diag(self.evalKernel((X - leftOutX)/bw).ravel())

    # Return
    return matW

  def formWMatrix(self,X,bw,newX):
    """
      Form Kernel Weight Matrix using all training locations
    
      Parameters
      ----------
      X : 2D array with training locations [data size x dimensions]
      bw: 1D array with bandwidth.
      newX : 1D array with additional training location.

    """
    # Construct the Diagonal Weighting Matrix
    minz = sys.float_info.max
    matW = np.diag(self.evalKernel((X-newX)/bw).ravel())
    minz = np.min(np.sqrt(np.sum(((X-newX)/bw)**2,axis=1)))

    isValidPoint = (minz < 5.0)

    return matW,isValidPoint

  def minBW(self,bw,X,Y):
    """
      Compute the loss function that needs to be minimized to find the optimal bandwidth
    
      Parameters
      ----------
      X : 2D array where each row is a different realization and each column is a different dimension.
      Y : 1D array with training data.

      Notes
      -----
      This is the average of the leave one out errors on the training data for a fixed bandwidth.
      

    """
    numEval = self.numOneOut
    if(X.shape[0]<numEval):
      numEval = X.shape[0] 
    samplesToCheck = np.arange(0,X.shape[0],np.floor(X.shape[0]/numEval))
    numToCheck = len(samplesToCheck)
    obj = np.empty(numToCheck)
    err = np.empty(numToCheck)
    for loopA in range(numToCheck):
      obj[loopA],err[loopA] = self.objBW(X,Y,bw,int(samplesToCheck[loopA]))
    print('Mean relative percent error: %.2f %%' % (np.mean(err)))
    return np.sum(obj)/float(numToCheck)

  def objBW(self,X,Y,bw,leaveOut):
    """
      Compute single component of the loss function for bandwidth optimization.
    
      Parameters
      ----------
      X : 2D array where each row is a different realization and each column is a different dimension.
      Y : 1D array with training data.
      bw : kernel bandwidth.
      leaveOut : integer with the row number to leave out.

    """
    global dotTime
    # Form X and W Matrices 
    matX,leftY,leftOutX,leftOutY = self.formXMatrixLeaveOut(X,Y,leaveOut)
    matW                         = self.formWMatrixLeaveOut(X,bw,leaveOut)
      
    # Solve system of equations
    startTime = time.time()
    # invMat = np.linalg.pinv(matX.transpose().dot(matW.dot(matX)))
    # estimate = np.dot(np.dot(invMat,np.dot(matX.transpose(),matW)),leftY)[0]
    estimate = np.linalg.lstsq(matX.transpose().dot(matW.dot(matX)),np.dot(np.dot(matX.transpose(),matW),leftY))[0][0]

    # print('diff: ',np.abs(estimate2-estimate))
    # sys.exit(-1)

    dotTime += (time.time()-startTime)*1000

    # Evaluate Loss Function
    return (leftOutY-estimate)**2,np.abs((estimate-leftOutY)/leftOutY)*100.0

  def fit(self,newX,useLimiter=False,farValue=-1e+30):
    """
      Fit function
    
      Parameters
      ----------
      newX       : 2D array where with location where to perform local linear regression.
      useLimiter : Limit the value far from the training set instead of simply returning 0
      farValue   : Value that the regressor should have far from the training set.

    """
    # Init result
    res = np.empty(newX.shape[0])

    for loopA in range(newX.shape[0]):      
      
      # Form Matrices
      matX = self.formXMatrix(self.X,newX[loopA,:])
      matW,isValidPoint = self.formWMatrix(self.X,self.bw,newX[loopA,:])

      # Check point only if option is True
      isValidPoint = (isValidPoint or (not(useLimiter)))

      # Solve System of Equations
      res[loopA] = farValue
      if(isValidPoint):
        a = np.dot(matX.transpose(),np.dot(matW,matX))
        b = matX.transpose().dot(matW).dot(self.Y)
        res[loopA] = np.linalg.lstsq(a,b)[0][0]

    return res

def evalFunction(X):
  res = np.empty(X.shape[0])
  for loopA in range(X.shape[0]):
    res[loopA] = -2*(X[loopA,0]-0.5)**3
  return res

# =============
# MAIN FUNCTION
# =============
if __name__ == '__main__':
  #import matplotlib.pyplot as plt

  # Set Number of Points
  nPoints = 10
  nPointsTest = 200
  nDims = 1

  # Select Points Randomly in [0,1]^d
  X    = np.random.uniform(size=(nPoints,nDims))
  newX = np.random.uniform(-1.5,2.5,size=(nPointsTest,nDims))
  # newX = np.random.uniform(size=(nPoints,nDims))

  # Eval Function at the Points
  Y = evalFunction(X)

  # Train the local linear regressor
  startTime = time.time()
  kr = llKernelReg(X,Y)
  stopTime = time.time()
  print('training ms: ',(stopTime-startTime)*1000.0)
  print('dot Time: ',dotTime)

  # Train the local linear regressor
  startTime = time.time()
  kR = KernelReg(exog=X,endog=Y,var_type='c'*nDims)
  stopTime = time.time()
  print('reference ms: ',(stopTime-startTime)*1000.0)

  # Predict Y Values using the regressor
  startTime = time.time()
  newY = kr.fit(newX,useLimiter=True,farValue=-0.5)
  stopTime = time.time()
  print('testing ms: ',(stopTime-startTime)*1000.0)

  plotIt = True
  # Plot 1D Case
  if(plotIt):
    plt.plot(X,Y,'bo',markersize=3)
    plt.plot(newX,newY,'ro',markersize=2)
    plt.show()

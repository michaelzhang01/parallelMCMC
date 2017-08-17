# Imports
import sys
import numpy as np
from scipy.optimize import fmin
"""
  Class for Local Polynomial Kernel Regression
"""
# Kernel Regression for Local Linear Systems
class llKernelReg():
  """
    Constructor
  
    Parameters
    ----------
    X : 2D array where each row is a different realization and each column is a different dimension.
    Y : 1D array with training data.

    Notes
    -----
    When the object is constructed the optimal bandwidth is computer.
    This is not recomputed during fit, but only when new points are added.

  """
  def __init__(self, X, Y):
    self.trainSize = X.shape[0]
    self.dims      = X.shape[1]
    self.X         = X
    self.Y         = Y
    # Compute Initial Guess
    stdX = np.std(self.X, axis=0)
    bw0 = 1.06*stdX*self.trainSize**(-1./(4.0 + self.dims))
    # Find the optimal bandwidth with least-squares cross-validation (ls_cv)
    self.bw = fmin(self.minBW, x0=bw0, args=(self.X,self.Y), maxiter=1e3, maxfun=1e3, disp=0)
    print("Regressor Initialization Completed.")

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
  def addData(self,addX,addY):
    # Add data
    self.X = np.row_stack(self.X,addX)
    self.Y = np.row_stack(self.Y,addY)
    self.trainSize = X.shape[0]
    self.dims      = X.shape[1]
    # Compute Initial Guess
    stdX = np.std(self.X, axis=0)
    bw0 = 1.06*stdX*self.trainSize**(-1./(4.0 + self.dims))    
    self.bw = fmin(self.minBW, x0=bw0, args=(self.X,self.Y), maxiter=1e3, maxfun=1e3, disp=0)
    print("Bandwidth Recomputed.")

  """
    Form the X Matrix by leaving one row out
  
    Parameters
    ----------
    X : 2D array with training locations [data size x dimensions]
    Y : 1D array with training values.
    leaveOut: Point location (row number) to leave out. This is used to compute
              the loss function on the bandwidth.

  """
  def formXMatrixLeaveOut(self,X,Y,leaveOut):

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
    for loopA in xrange(X.shape[0]):
      for loopB in xrange(X.shape[1]):
        matX[loopA,loopB+1] = (X[loopA,loopB] - leftOutX[loopB])

    # Return Matrix and left out vector
    return matX,leftY,leftOutX,leftOutY

  """
    Form X Matrix with all training points at a specific new location
  
    Parameters
    ----------
    X : 2D array with training locations [data size x dimensions]
    Y : 1D array with training values.
    newX: 2D array with additional training locations [data size x dimensions]

  """
  def formXMatrix(self,X,newX):

    # Construct the Polynomial Matrix
    matX = np.empty((X.shape[0],X.shape[1]+1))
    # First column is all ones
    matX[:,0] = np.ones(X.shape[0])
    # Add polynomial terms
    for loopA in xrange(X.shape[0]):
      for loopB in xrange(X.shape[1]):
        matX[loopA,loopB+1] = (X[loopA,loopB] - newX[loopB])

    # Return Matrix and left out vector
    return matX

  """
    Evaluate Standard Multivariate Gaussian Kernel
  
    Parameters
    ----------
    x : 1D with normalized location where to evaluate the kernel

  """
  def evalKernel(self,x):
    size = len(x)
    return (1.0/(np.sqrt((2.0*np.pi)**size)))*np.exp(-0.5*np.dot(x.transpose(),x))

  """
    Form Kernel Weight Matrix by leaving out one training location
  
    Parameters
    ----------
    X : 2D array with training locations [data size x dimensions]
    bw: 1D array with bandwidth.
    leaveOut : integer with the row number to leave out

  """
  def formWMatrixLeaveOut(self,X,bw,leaveOut):

    if(leaveOut<0):
      print("ERROR: Negative Leaveout.")
      exit(-1)
    
    leftOutX = X[leaveOut,:]
    X = np.delete(X,(leaveOut),axis=0)

    # Construct the Diagonal Weighting Matrix
    diagW = np.empty(X.shape[0])
    for loopA in xrange(X.shape[0]):
      z = np.divide(X[loopA,:] - leftOutX,bw)
      diagW[loopA] = self.evalKernel(z)
    matW = np.diag(diagW)

    # Return
    return matW

  """
    Form Kernel Weight Matrix using all training locations
  
    Parameters
    ----------
    X : 2D array with training locations [data size x dimensions]
    bw: 1D array with bandwidth.
    newX : 1D array with additional training location.

  """
  def formWMatrix(self,X,bw,newX):

    # Construct the Diagonal Weighting Matrix
    diagW = np.empty(X.shape[0])
    minz = sys.float_info.max
    for loopA in xrange(X.shape[0]):
      z = np.divide(X[loopA,:] - newX,self.bw)
      if(np.linalg.norm(z,2) < minz):
        minz = np.linalg.norm(z,2)
      diagW[loopA] = self.evalKernel(z)
    matW = np.diag(diagW)

    isValidPoint = (minz < 5.0)

    return matW,isValidPoint

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
  def minBW(self,bw,X,Y):

    obj = 0.0
    for loopA in xrange(X.shape[0]):
      obj += self.objBW(X,Y,bw,loopA)
    return obj/float(X.shape[0])    

  """
    Compute single component of the loss function for bandwidth optimization.
  
    Parameters
    ----------
    X : 2D array where each row is a different realization and each column is a different dimension.
    Y : 1D array with training data.
    bw : kernel bandwidth.
    leaveOut : integer with the row number to leave out.

  """
  def objBW(self,X,Y,bw,leaveOut):

    # Form X and W Matrices 
    matX,leftY,leftOutX,leftOutY = self.formXMatrixLeaveOut(X,Y,leaveOut)
    matW                   = self.formWMatrixLeaveOut(X,bw,leaveOut)
      
    # Solve system of equations
    invMat = np.linalg.pinv(np.dot(matX.transpose(),np.dot(matW,matX)))
    estimate = np.dot(np.dot(invMat,np.dot(matX.transpose(),matW)),leftY)[0]

    # Evaluate Loss Function
    return (leftOutY-estimate)**2

  """
    Fit function
  
    Parameters
    ----------
    newX : 2D array where with location where to perform local linear regression.
    farValue : Value that the regressor should have far from the training set.

  """
  def fit(self,newX,farValue=0.0):
    
    # Init result
    res = np.empty(newX.shape[0])

    for loopA in xrange(newX.shape[0]):      
      # Form Matrices
      matX = self.formXMatrix(self.X,newX[loopA,:])
      matW,isValidPoint = self.formWMatrix(self.X,self.bw,newX[loopA,:])

      # Solve System of Equations
      if(isValidPoint):
        invMat = np.linalg.pinv(np.dot(matX.transpose(),np.dot(matW,matX)))
        res[loopA] = np.dot(np.dot(invMat,np.dot(matX.transpose(),matW)),self.Y)[0]
      else:
        res[loopA] = farValue

    return res

def evalFunction(X):
  res = np.empty(X.shape[0])
  for loopA in xrange(X.shape[0]):
    res[loopA] = -2*(X[loopA,0]-0.5)**3
  return res

# =============
# MAIN FUNCTION
# =============
if __name__ == '__main__':
  import matplotlib.pyplot as plt

  # Set Number of Points
  nPoints = 100
  nDims = 1

  # Select Points Randomly in [0,1]^d
  X = np.random.uniform(size=(nPoints,nDims))
  newX = np.random.uniform(-1.5,2.5,size=(nPoints,nDims))
  # newX = np.random.uniform(size=(nPoints,nDims))

  # Eval Function at the Points
  Y = evalFunction(X)

  # Train the local linear regressor
  kr = llKernelReg(X,Y)

  # Predict Y Values using the regressor
  newY = kr.fit(newX,-0.5)

  # Plot 1D Case
  plt.plot(X,Y,'bo');
  plt.plot(newX,newY,'ro');
  plt.show()

import scipy as sp
import struct



def write2bin(fileName,A):
  
  shp = A.shape
  data = sp.reshape(A,(shp[0]*shp[1],))

  tmp = ''
  tmp = tmp.join((struct.pack('d', val) for val in data))
  b = struct.pack('ii',shp[0],shp[1]) + tmp

  binFile = open(fileName, mode='wb')
  binFile.write(b)  
  binFile.close()
  return 0


def readBin(fileName):

  with open(fileName, mode='rb') as file:
    fileContent = file.read()

  headFmt = 'ii'
  offset = struct.calcsize(headFmt)

  dataShape = struct.unpack(headFmt,fileContent[:offset])
  fmt = 'd'*dataShape[0]*dataShape[1]
  data = struct.unpack(fmt,fileContent[offset:])
  A = sp.reshape(sp.asarray(data),(dataShape[0],dataShape[1]))
  file.close()
  return A


def writeLargeBin(fileName, A):

  binFile = open(fileName, mode='wb')
  b = struct.pack('ii',A.shape[0],A.shape[1])
  binFile.write(b) 
  A = sp.reshape(A,(A.shape[0]*A.shape[1],))
  for val in A:
    binFile.write(struct.pack('d', val))
  binFile.close()

def readLargeBin(fileName):

  f = open(fileName, mode='rb')
  headFmt = 'ii'
  offset = struct.calcsize(headFmt)
  dataShape = struct.unpack(headFmt,f.read(offset))
  N = dataShape[0]; M = dataShape[1]
  D = sp.zeros((N,M))
  fmt = 'd'
  for i in range(N):
    D[i,:] = sp.array(struct.unpack(fmt*M,f.read(struct.calcsize(fmt)*M)))
  f.close()
  return D


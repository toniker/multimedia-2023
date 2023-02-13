from scipy.fftpack import dct,idct
def frameDCT(Y):
    return dct(Y).reshape(-1,1)

def iframeDCT(c,M,N):
    return idct(c).reshape(M,N)
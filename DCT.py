from scipy.fftpack import dct,idct
def frameDCT(Y):
    return dct(Y,norm = 'ortho').reshape(-1,1)

def iframeDCT(c,M,N):
    return idct(c,norm = 'ortho').reshape(N,M)
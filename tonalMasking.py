import numpy as np

def DCTpower(c):
    return 10*np.log10(c.astype('int64') ** 2)

def Dksparse(Kmax):
    Dk = np.zeros([Kmax,Kmax])

    for k in range(Kmax):
        if 1 < k < 281:
            Dk[k,1] = 1
        elif 281 <= k < 569:
            Dk[k,1:12] = 1
        elif 569 <= k < 1151:
            Dk[k,1:26] = 1
    return Dk

def STinit(c,D):
    St = []
    P = DCTpower(c)
    for k in range(len(P)-1):
        if k != 0 and k != len(P) - 1:
            if P[k] > P[k+1] and P[k] > P[k-1]:
                ind1 = np.array(np.squeeze(np.where(D[k,:] == 1)) - k)
                ind2 = np.array(np.squeeze(np.where(D[k,:] == 1)) + k)
                ind = np.append(ind1,ind2)

                in_bounds = np.where(ind < len(P)) or np.where(ind >= -len(P))
                ind = ind[in_bounds]

                check = np.all(P[k] > P[ind] + 7)
                if check:
                    St.append(k)

    return St

def MaskPower(c, ST):
    P = DCTpower(c)
    return P[ST].reshape(1,-1)

def Hz2Barks(f):
    z = 13 * np.arctan(0.00076 * f) + 3.5 * np.arctan((f / 7500) ** 2)
    return z
import numpy as np

def critical_bands(K):
    """
    @:param K: maximum discrete frequency

    Returns the list of bands the frequencies belong to
    """
    cb = []
    fs = 44100
    MN = K
    f_Hz = fs / (2 * MN) * np.arange(1, MN)

    bounds = np.array([100,200,300,400,510,
                      630,770,920,1080,1270,
                      1480,1720,2000,2320,2700,
                      3150,3700,4400,5300,6400,
                      7700,9500,12000,15500])

    for f in f_Hz:
        tmp = np.argsort(np.append(bounds,f))
        indx = np.where(tmp == len(bounds))
        cb.append(indx[0][0]+1)

    return np.array(cb)

def DCT_band_scale(c):
    """
    @:param c: DCT components

    Calculates the normalized DCT components and the scale factors
    """
    cb = critical_bands(len(c)+1)
    bands = np.unique(cb)

    sc = []
    cs = []

    for k in bands:
        idx = np.where(cb == k)
        tmp = np.max(np.abs(c[idx]) ** (3 / 4))
        sc.append(tmp)

    for i in range(len(cb)):
        band = cb[i]
        tmp = np.sign(c[i]) * ((np.abs(c[i]) ** (3/4)) / sc[band-1])
        cs.append(tmp)

    return np.array(cs),np.array(sc)

def quantizer(x, b):
    """
    x: the initial array of values
    @:param b: the number of bits used it the quantization

    Returns a list of symbol indexes corresponding to each value in x
    """
    wb = 1 / (2 ** b - 1)
    num_of_zones = int(2//wb)

    symb_index = []
    d = np.zeros(num_of_zones)

    for i in range(int((num_of_zones+1)/2)):
        d[i] = -(2**b - (i+1))*wb
        d[-(i+1)] = (2**b - (i+1))*wb

    for val in x:
        tmp = np.argsort(np.append(d,val))
        indx = np.where(tmp == len(d))
        symb_index.append(indx[0][0])

    return np.array(symb_index)

def dequantizer(symb_index, b):
    """
    symb_index: the list of symbol indexes computed by the quantizer
    @:param b: the number of bits used it the quantization

    Returns the dequantized values calculated using the symb_index list
    """
    wb = 1 / (2 ** b - 1)
    num_of_zones = int(2//wb - 1)

    xh = []
    zones = np.zeros(num_of_zones)
    for i in range(int((num_of_zones-1)/2)):
        zones[i] = -(1+wb/2) + (i+1)*wb
        zones[-(i+1)] = (1+wb/2) - (i+1)*wb

    for idx in symb_index:
        if idx == num_of_zones + 1:
            xh.append((zones[idx-2]))
        else:
            xh.append(zones[idx-1])

    return np.array(xh)



def all_bands_quantizer(c, Tg):
    """
    Inputs:
    @:param c: DCT components
    Tg: the hearing threshold calculated tonal masking

    Outputs:
    symb_index: The index symbol corresponding to each value of c
    SF: The Scale Factors for each band
    B: The number of bits for the quantization of each band
    """
    critical_bnds,scale_factors = DCT_band_scale(c)
    cb = critical_bands(len(c)+1)
    Tg[np.isnan(Tg)] = np.inf

    bands = np.unique(cb)
    B = np.array([])
    symb_index = np.array([])
    SF = np.array([])
    for k in bands:
        b = 1
        while True:
            symb_idx = quantizer(critical_bnds, b)
            c_dequantized = dequantizer(symb_idx, b)

            # Find DCT components and Tg for band k
            idx = np.where(cb == k)
            c_band = c_dequantized[idx]
            Tg_band = Tg[idx]

            # Calculate c_hat
            c_hat = np.sign(c_band) * (np.abs(c_band) * scale_factors[k - 1]) ** (4/3)

            # Calculate error and power of error
            eb = np.abs(c[idx].T - c_hat)
            P_eb = 10*np.log10(eb ** 2)

            # Check if P_e <= Tg, else increase b and repeat
            if np.all(P_eb <= Tg_band):
                B = np.append(B,b)
                symb_index = np.append(symb_index, symb_idx[idx])
                break
            else:
                b = b+1
    SF = np.append(SF,scale_factors)

    return np.array(symb_index).astype(int), np.array(SF),np.array(B).astype(int)

def all_bands_dequantizer(symb_index,B, SF):
    """
    Inputs:
    symb_index: The index symbol corresponding to each value of c
    SF: The Scale Factors for each band
    B: The number of bits for the quantization of each band

    Output:
    xh: The reconstructed DCT components
    """
    cb = critical_bands(len(symb_index)+1)
    bands = np.unique(cb)

    xh = np.array([])
    for k in bands:
        # Find symbols and number of bits used in quantization for band k
        idx = np.where(cb == k)
        band_symbols = symb_index[idx]
        band_bits = B[k-1]

        # Dequantization of symbols in band k
        band_deq = dequantizer(band_symbols, band_bits)

        # Calculate value of x_hat in band k
        tmp = np.sign(band_deq) * (np.abs(band_deq) * SF[k - 1]) ** (4 / 3)
        xh = np.append(xh,tmp)

    return xh
import numpy as np
import matplotlib.pyplot as plt
def DCTpower(c):
    """
    @:param c: DCT components

    Calculates the power of the DCT components
    """
    return 10 * np.log10(c.astype('int64') ** 2)

def Dksparse(Kmax):
    """
    @:param Kmax: maximum discrete frequency

    Calculates sparse array D
    """
    Dk = np.zeros([Kmax, Kmax])

    for k in range(Kmax):
        if 1 < k < 281:
            Dk[k, 1] = 1
        elif 281 <= k < 569:
            Dk[k, 1:12] = 1
        elif 569 <= k < 1151:
            Dk[k, 1:26] = 1
    return Dk

def STinit(c, D):
    """
    @:param c: DCT components
    @:param D: sparse array D

    Finds maskers using 2 conditions:
    1) Based on the power of the component in comparison to its neighboring components
    2) Based on the power of the component in comparison to the power of the components in a neighbourhood
    """

    St = []
    P = DCTpower(c)
    for k in range(len(P) - 1):
        if k == 0 and k == len(P) - 1:
            continue
        else:
            # Check if the first condition is met
            if P[k] > P[k + 1] and P[k] > P[k - 1]:

                # Find the searching "neigbourhood" for each frequency
                ind1 = np.array(k - np.squeeze(np.where(D[k, :] == 1)))
                ind2 = np.array(k + np.squeeze(np.where(D[k, :] == 1)))
                ind = np.append(ind1, ind2)
                in_bounds = np.where(ind < len(P)) or np.where(ind >= -len(P))
                ind = ind[in_bounds]

                # Check if the second condition is met
                check = np.all(P[k] > P[ind] + 7)
                if check:
                    St.append(k)
    return St


def MaskPower(c, ST):
    """
    @:param c: DCT components
    @:param ST: The possible tonal components

    Calculate the power of a possible tonal component
    """
    P_DCT = DCTpower(c)
    P_Mask = np.array([])

    for k in ST:
        tmp = 0
        for j in range(-1, 2):
            exponential = 0.1 * P_DCT[k + j]
            tmp = tmp + 10 ** exponential
        p = 10 * np.log10(tmp)
        P_Mask = np.append(P_Mask, p)
    return P_Mask

def Hz2Barks(f):
    """
    @:param f: frequency in Hz

    Converts the frequency from Hz to barks
    """
    z = 13 * np.arctan(0.00076 * f) + 3.5 * np.arctan((f / 7500) ** 2)
    return z

def STreduction(ST, c, Tq):
    """
    @:param ST: The possible tonal components
    @:param c: DCT components
    @:param Tq: Hearing threshold in silence

    Reduces maskers using two eliminations:
    - 1st elimination for maskers below hearing threshold
    - 2nd elimination for maskers depending on the distance in bark frequency
    """
    is_empy = len(ST) == 0
    if is_empy:
        STr = []
        PMr = []

    else:
        # Calculate the Power of each possible tone
        P_ST = MaskPower(c, ST)

        # Find the hearing threshold for each tone frequency
        Tq_ST = Tq[ST]

        # 1st Elimination

        cond = P_ST > Tq_ST.reshape(1, -1)
        cond = cond[0]
        maskers_1st_elim = [ST[i] if cond[i] else float('nan') for i in range(len(ST))]

        # 2nd Elimination
        MN = len(c)
        fs = 44100

        # frequency (Hz) corresponding to each possible masker
        f_Hz = fs / (2 * MN) * np.array(maskers_1st_elim)

        # frequency (barks) corresponding to each possible masker
        f_barks = Hz2Barks(f_Hz)

        # Calculate distance in frequency (barks) between the maskers
        distance = np.array([])
        for i in range(len(f_barks) - 1):
            distance = np.append(distance, np.abs(f_barks[i] - f_barks[i + 1]))
        ind = np.where(distance > 0.5)
        ind = np.append(ind, len(f_barks) - 1)

        STr = [maskers_1st_elim[i] for i in ind]
        PMr = [P_ST[i] for i in ind]

    return STr, PMr


def SpreadFunc(ST, PM, Kmax):
    """
    @:param ST: The possible tonal components after elimination
    @:param PM: The power of the possible tonal components
    @:param Kmax: maximum discrete frequency

    Calculates the Spreading Function
    """
    fs = 44100
    MN = Kmax + 1

    Dz = np.zeros([Kmax + 1, len(ST)])
    Sf = np.zeros([Kmax + 1, len(ST)])

    # find all discrete frequences for k = [0,Kmax] (Hz)
    f_Hz_i = fs / (2 * MN) * np.arange(0, Kmax)
    # find all discrete frequences for k = [0,Kmax] (barks)
    f_barks_i = Hz2Barks(f_Hz_i)
    # find  discrete frequences for the possible tonal components (Hz)
    f_Hz_k = fs / (2 * MN) * np.array(ST)
    # find  discrete frequences for the possible tonal components (barks)
    f_barks_k = Hz2Barks(f_Hz_k)

    # Calculate Dz array
    for k in range(len(f_barks_k)):
        for i in range(len(f_barks_i)):
            Dz[i, k] = f_barks_i[i] - f_barks_k[k]

    # Calculate Sf array
    for k in range(len(f_barks_k)):
        Dz_k = Dz[:, k]
        for i in range(len(Dz_k)):
            if -3 < Dz_k[i] < -1:
                Sf[i, k] = 17 * Dz_k[i] - 0.4 * PM[k] + 11
            elif -1 <= Dz_k[i] < 0:
                Sf[i, k] = (0.4 * PM[k] + 6) * Dz_k[i]
            elif 0 <= Dz_k[i] < 1:
                Sf[i, k] = -17 * Dz_k[i]
            elif 1 <= Dz_k[i] < 3:
                Sf[i, k] = (0.15 * PM[k] - 17) * Dz_k[i] - 0.15 * PM[k]
            else:
                Sf[i, k] = -np.inf
    return Sf


def Masking_Thresholds(ST, PM, Kmax):
    """
    @:param ST: The possible tonal components after elimination
    @:param PM: The power of the possible tonal components
    @:param Kmax: maximum discrete frequency

    Calculates the contribution of each masker to the hearing threshold
    """
    fs = 44100
    MN = Kmax + 1

    Ti = np.zeros([Kmax + 1, len(ST)])
    f_Hz_k = fs / (2 * MN) * np.array(ST)
    f_barks_k = Hz2Barks(f_Hz_k)
    Sf = SpreadFunc(ST, PM, Kmax)

    for k in range(len(ST)):
        Ti[:, k] = PM[k] - 0.275 * f_barks_k[k] + Sf[:, k] - 6.025
    return Ti


def Global_Masking_Thresholds(Ti, Tq):
    """
    @:param Ti: Contribution of maskers to hearing threshold
    @:param Tq: Hearing threshold in silence

    Calculates the global masking threshold
    """
    Tq = np.squeeze(Tq)
    Tg = np.zeros(Ti.shape[0])
    for i in range(Ti.shape[0]):
        sum = np.sum(10 ** (0.1 * Ti[i, :]))
        Tg[i] = 10 * np.log10(10 ** (0.1 * Tq[i]) + sum)
    return Tg


def psycho(c, D):
    """
    @:param c: DCT components
    @:param D: Sparse array D

    Calculates the total masking threshold using the above functions
    """

    # Load data from file
    Tq = np.load("Tq.npy", allow_pickle=True)

    p = DCTpower(c)
    St = STinit(c, D)

    M, N = c.shape
    Kmax = M * N - 1
    PM = MaskPower(c, St)
    STr, PTr = STreduction(St, c, Tq.reshape(-1, 1))
    Sf = SpreadFunc(STr, PM, Kmax)
    Ti = Masking_Thresholds(STr, PM, Kmax)
    Tg = Global_Masking_Thresholds(Ti, Tq)

    return Tg

def plot_hearing_threshold(Tg,Tq,STr,Kmax,i):
    """
    @:param Tg: Masking threshold
    @:param Tq: Hearing threshold in silence
    @:param STr: The possible tonal components after elimination
    @:param Kmax: maximum discrete frequency

    Plots the hearing thresholds
    """
    fs = 44100
    MN = Kmax + 1

    f_Hz = fs / (2 * MN) * np.arange(0,Kmax+1)
    f_barks = Hz2Barks(f_Hz)

    plt.plot(f_barks,Tg,label='Tg')
    plt.plot(f_barks,np.squeeze(Tq.reshape(-1,1)),label='Tq')
    if len(STr) != 0:
        markerline, stemlines,baseline =plt.stem(f_barks[STr],Tg[STr], '-.',linefmt  = 'purple', bottom=-12, label='maskers',basefmt=" ")
        plt.setp(stemlines, 'linestyle', 'dotted')

    plt.ylim(-10,70)
    plt.ylabel('Masking Threshold')
    plt.xlabel('Frequency (barks)')
    plt.legend()
    #plt.savefig('media/hearing_thresh'+str(i)+'.pdf')
    plt.show()


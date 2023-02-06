import numpy as np


def make_mp3_analysisfb(h: np.ndarray, M: int) -> np.ndarray:
    """
    @:param h η κρουστική απόκριση του πρότυπου βαθυπερατού φίλτρου
    @:param M ο αριθμός των ζωνών στις οποίες θα γίνει ο διαχωρισμός.
    @:return  πίνακας διάστασης L × M, όπου L το μήκος της κρουστικής απόκρισης του φίλτρου,
    ο οποίος ως στήλες έχει τις κρουστικές αποκρίσεις των φίλτρων hi(n), i=1,...,M
    """

    H = np.zeros([len(h), M], dtype=np.float32)

    for i in range(1, M + 1):
        n = np.arange(h.shape[0], dtype=np.int64)
        freq_i = (2 * i - 1) * np.pi / (2.0 * M)
        phas_i = -(2 * i - 1) * np.pi / 4.0
        tmp = np.cos(freq_i * n + phas_i)
        x = np.multiply(h, tmp)
        H[:, i - 1] = x
    return H


def make_mp3_synthesisfb(h: np.ndarray, M: int) -> np.ndarray:
    """
    @:param h η κρουστική απόκριση του πρότυπου βαθυπερατού φίλτρου
    @:param M ο αριθμός των ζωνών στις οποίες έχει γίνει ο διαχωρισμός
    """

    H = make_mp3_analysisfb(h, M)
    L = len(h)
    G = np.flip(H, axis=0)
    return G

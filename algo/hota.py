def compute_idf1(tpa, fpa, fna):
    if tpa == 0:
        return 0
    return tpa / (tpa + 0.5 * (fpa + fna))


def compute_hota(tpa, fpa, fna, tp, fp, fn):
    # print('tpa', tpa, 'fpa', fpa, 'fna', fna, 'tp', tp, 'fp', fp, 'fn', fn)
    if tpa == 0:
        return 0
    deta = tp / (tp + fp + fn)
    assa = tpa / (tpa + fpa + fna)
    return (deta * assa) ** 0.5
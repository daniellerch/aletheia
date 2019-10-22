
# {{{ AB_predict()
def AB_predict(p_aa, p_bb, p_ab, p_ba):

    inc = 0
    xa_S_cnt = 0

    y_pred = [0]*len(p_aa)

    for i in range(len(p_aa)):

        if p_aa[i]==1 and p_ba[i]==0:
            xa_S_cnt += 1

        if p_aa[i]!=p_bb[i]:
            inc +=1
            y_pred[i] = -1
            continue

        if p_ba[i]!=0:
            inc +=1
            y_pred[i] = -1
            continue

        if p_ab[i]!=1:
            inc +=1
            y_pred[i] = -1
            continue

        if p_aa[i]==0: 
            y_pred[i] = 0
        else:
            y_pred[i] = 1

    return y_pred, inc, xa_S_cnt
# }}}



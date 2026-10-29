# dead: photo or pair over idx
    dead_proc = np.zeros(len(idx), bool)
    dead_proc[interact] = (proc==2)|(proc==3)
    # update arrays
    Ea[interact]=iE; da[interact]=idir; sa[interact]=ist
    alive[idx]=False
    survive = (~exit_before) & (~dead_proc)
    alive[idx[surv]]=True
    it_tot+=1
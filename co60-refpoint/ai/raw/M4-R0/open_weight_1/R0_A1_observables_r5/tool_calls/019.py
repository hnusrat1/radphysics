
import sys, os
sys.path.insert(0,'/tmp/work'); os.chdir('/tmp/work')
for fn in ['mcbeam.py']:
    s=open(fn).read()
    old="""        # exit t to phantom boundary
        tz0=(0-za)/uza
        tz30=(30-za)/uza
        tx_lo=(-15-xa)/uxa; tx_hi=(15-xa)/uxa
        ty_lo=(-15-ya)/uya; ty_hi=(15-ya)/uya
        cand=np.stack([np.where(uxa>0,tx_hi,np.where(uxa<0,tx_lo,np.inf)),
                       np.where(uxa>0,tx_lo,np.where(uxa<0,tx_hi,np.inf)),
                       np.where(uya>0,ty_hi,np.where(uya<0,ty_lo,np.inf)),
                       np.where(uya>0,ty_lo,np.where(uya<0,ty_hi,np.inf)),
                       np.where(uza>0,tz30,np.where(uza<0,tz0,np.inf)),
                       np.where(uza>0,tz0,np.where(uza<0,tz30,np.inf))],axis=0)
        with np.errstate(invalid='ignore'):
            t_exit=np.min(cand,axis=0)
        t_exit=np.where(t_exit>0,t_exit,0.0)"""
    new="""        # exit t to phantom boundary (positive, relevant)
        tz_ex=np.where(uza>0,(30-za)/uza,np.where(uza<0,(0-za)/uza,np.inf))
        tx_ex=np.where(uxa>0,(15-xa)/uxa,np.where(uxa<0,(-15-xa)/uxa,np.inf))
        ty_ex=np.where(uya>0,(15-ya)/uya,np.where(uya<0,(-15-ya)/uya,np.inf))
        with np.errstate(invalid='ignore'):
            t_exit=np.minimum(np.minimum(tz_ex,tx_ex),ty_ex)
        t_exit=np.where(np.isfinite(t_exit)&(t_exit>0),t_exit,0.0)"""
    if old not in s:
        print('pattern not found in', fn)
    else:
        s=s.replace(old,new)
        open(fn,'w').write(s)
print('patched')

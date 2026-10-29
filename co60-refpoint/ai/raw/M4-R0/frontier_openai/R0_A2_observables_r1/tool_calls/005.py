# calculate weighted primary energy distribution exact using integration distributions
p1,d1,mean1=dist(1.1732);p2,d2,mean2=dist(1.3325)
prim=(p1+p2)/2
print(prim.sum(),(prim*(edges[:-1]+edges[1:])/2).sum(),prim.tolist())
# primary directional
print(((d1+d2)/2).tolist())
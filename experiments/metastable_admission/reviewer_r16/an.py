import csv,statistics as st,collections
rows=list(csv.DictReader(open('hyst.csv')))
g=collections.defaultdict(list)
for r in rows: g[(float(r['L']),r['cl'],int(r['e']))].append(r)
m=lambda l,k: st.mean(float(r[k]) for r in l)
print('L cl e | pre post mid late | act pre,3000,6000,9000,end | late/ctrl_late | preempt_late n_seeds_low')
for (L,cl,e),l in sorted(g.items()):
    c=g[(L,cl,0)]
    ratio=m(l,'late')/m(c,'late')
    # paired
    low=0
    for r in l:
        cc=[x for x in c if x['seed']==r['seed']][0]
        if float(r['late'])<0.7*float(cc['late']): low+=1
    print('%.2f %s %d | %.2f %.2f %.2f %.2f | %.0f %.0f %.0f %.0f %.0f | %.2f | %.3f %d/6'%(L,cl,e,m(l,'pre'),m(l,'post'),m(l,'mid'),m(l,'late'),m(l,'a_pre'),m(l,'a_3000'),m(l,'a_6000'),m(l,'a_9000'),m(l,'a_end'),ratio,m(l,'late_pre'),low))

src=r'''#include <bits/stdc++.h>
using namespace std;
struct Row{double E, coh,inc,pe,pn,pef;}; vector<Row> tab;
double interp(double E,int col){ // col 1..5
 if(E<=tab.front().E)return col==1?tab.front().coh: col==2?tab.front().inc:col==3?tab.front().pe:col==4?tab.front().pn:tab.front().pef;
 int n=tab.size(); if(E>=tab.back().E)return 1e-10;
 int lo=0,hi=n-1;while(hi-lo>1){int m=(lo+hi)/2;if(tab[m].E<=E)lo=m;else hi=m;}
 double a=col==1?tab[lo].coh:col==2?tab[lo].inc:col==3?tab[lo].pe:col==4?tab[lo].pn:tab[lo].pef;
 double b=col==1?tab[hi].coh:col==2?tab[hi].inc:col==3?tab[hi].pe:col==4?tab[hi].pn:tab[hi].pef;
 double f=log(E/tab[lo].E)/log(tab[hi].E/tab[lo].E); if(a<=0||b<=0)return a+(b-a)*f;return exp(log(a)*(1-f)+log(b)*f);
}
mt19937_64 rng(88172645463325252ULL); uniform_real_distribution<double> U(0,1);
double rnd(){return U(rng);} const double PI=acos(-1.0);
vector<double> pe={0,.05,.1,.15,.2,.25,.3,.35,.4,.45,.5,.55,.6,.65,.7,.75,.8,.85,.9,.95,1,1.05,1.1,1.15,1.2,1.25,1.3,1.35,1.4};
vector<double> ee={.01,.05,.1,.15,.2,.25,.3,.35,.4,.45,.5,.55,.6,.65,.7,.75,.8,.85,.9,.95,1,1.05,1.1,1.15,1.2,1.4};
int bin(vector<double>& ed,double x){if(x<ed[0]||x>=ed.back())return -1;return upper_bound(ed.begin(),ed.end(),x)-ed.begin()-1;}
// score segment within cylinder
void segscore(double x,double y,double z,double dx,double dy,double dz,double L,bool primary,double E,double *p,double *s,double *ap,double *as){
 double a=0,b=L;
 if(fabs(dz)<1e-14){if(z<4.9||z>5.1)return;}else{double t1=(4.9-z)/dz,t2=(5.1-z)/dz;if(t1>t2)swap(t1,t2);a=max(a,t1);b=min(b,t2);}
 // radial inequality (x+dx*t)^2+(y+dy*t)^2 <= 1
 double A=dx*dx+dy*dy,B=2*(x*dx+y*dy),C=x*x+y*y-1;
 if(A<1e-18){if(C>0)return;}else{double disc=B*B-4*A*C;if(disc<0)return;double r=sqrt(disc),t1=(-B-r)/(2*A),t2=(-B+r)/(2*A);a=max(a,t1);b=min(b,t2);}
 if(b<=a)return; double len=b-a; int ib=bin(pe,E); if(ib>=0){if(primary)p[ib]+=len;else s[ib]+=len;}
 double c=dz; int ang=min(9,max(0,(int)((c+1)*5))); if(primary)ap[ang]+=len;else as[ang]+=len;
}
void norm(double &x,double &y,double &z){double q=sqrt(x*x+y*y+z*z);x/=q;y/=q;z/=q;}
int main(int argc,char**argv){ifstream f("/tmp/data/xcom_water.txt");string line;while(getline(f,line)){if(line.empty()||line[0]=='#')continue;istringstream ss(line);Row r;double total,no;ss>>r.E>>r.coh>>r.inc>>r.pe>>r.pn>>r.pef>>total>>no;tab.push_back(r);}long long N=atoll(argv[1]);
 double p[28]={},s[28]={},ap[10]={},as[10]={}; double proc[2][4]={},trans[2][4]={},elcnt[2]={},elsum[2]={},ebins[2][25]={},edang[10]={}; double ref=0; long long nint=0;
 for(long long k=0;k<N;k++){
  double E=(rnd()<.5?1.1732:1.3325); double u,v;
  // rejection uniform solid angle in slope rectangle
  do{u=(2*rnd()-1)*.05;v=(2*rnd()-1)*.05;}while(rnd()>pow(1+u*u+v*v,-1.5));
  double dx=u,dy=v,dz=1;norm(dx,dy,dz);
  // position ray at z=0, source x=y=0,z=-100
  double x=100*u,y=100*v,z=0;
  // vacuum fluence reference track length in tally cylinder
  segscore(x,y,z,dx,dy,dz,30/dz,true,E,p /*dummy*/,s /*dummy*/,ap /*dummy*/,as /*dummy*/); // don't use arrays; undo impossible!
  // Instead compute reference directly segment intersection
  // (the above erroneously tallied, reset by separate approach below by arrays? use separate copy issue)
  // reset tally contribution is dealt by tallying separate arrays not possible; reference calculate via geometry independently and undo below
  // undo function contributions above not easy, so this is actual scored phantom free and also track tally. Save them as ref-only arrays separately would need.
  // run transport, but subtract ref by accumulate separate local bins/angles
  // revert recorded components from arrays by calculating local segment later omitted; handled with local arrays in revised version.
 }
}'''
print('')
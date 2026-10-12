code=r'''#include <bits/stdc++.h>
using namespace std; struct Row{double E,coh,inc,pe,pn,pef;}; vector<Row> tab;
double coeff(double E,int c){int n=tab.size(); if(E<=tab[0].E){Row r=tab[0];return c==1?r.coh:c==2?r.inc:c==3?r.pe:c==4?r.pn:r.pef;} if(E>=tab[n-1].E)return 1e-12;int l=0,h=n-1;while(h-l>1){int m=(l+h)/2;if(tab[m].E<=E)l=m;else h=m;} auto get=[&](Row r){return c==1?r.coh:c==2?r.inc:c==3?r.pe:c==4?r.pn:r.pef;};double a=get(tab[l]),b=get(tab[h]),f=log(E/tab[l].E)/log(tab[h].E/tab[l].E);return (a>0&&b>0)?exp(log(a)*(1-f)+log(b)*f):a+(b-a)*f;}
mt19937_64 rng(739192841);uniform_real_distribution<double> U(0,1);double R(){return U(rng);} const double PI=acos(-1.0);
vector<double> pg={0,.05,.1,.15,.2,.25,.3,.35,.4,.45,.5,.55,.6,.65,.7,.75,.8,.85,.9,.95,1,1.05,1.1,1.15,1.2,1.25,1.3,1.35,1.4};
vector<double> eg={.01,.05,.1,.15,.2,.25,.3,.35,.4,.45,.5,.55,.6,.65,.7,.75,.8,.85,.9,.95,1,1.05,1.1,1.15,1.2,1.4};
int bn(vector<double>&v,double x){if(x<v.front()||x>=v.back())return -1;return upper_bound(v.begin(),v.end(),x)-v.begin()-1;}
void unit(double &x,double &y,double &z){double q=sqrt(x*x+y*y+z*z);x/=q;y/=q;z/=q;}
double lenscore(double x,double y,double z,double dx,double dy,double dz,double L){double a=0,b=L;if(fabs(dz)<1e-14){if(z<4.9||z>5.1)return 0;}else{double t1=(4.9-z)/dz,t2=(5.1-z)/dz;if(t1>t2)swap(t1,t2);a=max(a,t1);b=min(b,t2);}double A=dx*dx+dy*dy,B=2*(x*dx+y*dy),C=x*x+y*y-1;if(A<1e-18){if(C>0)return 0;}else{double D=B*B-4*A*C;if(D<0)return 0;double rr=sqrt(D),t1=(-B-rr)/(2*A),t2=(-B+rr)/(2*A);a=max(a,t1);b=min(b,t2);}return max(0.0,b-a);}
void rotate(double dx,double dy,double dz,double mu,double phi,double &ox,double &oy,double &oz){double st=sqrt(max(0.0,1-mu*mu));double ux,uy,uz;if(fabs(dz)<.9){ux=-dy;uy=dx;uz=0;unit(ux,uy,uz);}else{ux=1;uy=0;uz=0; double dot=ux*dx+uy*dy+uz*dz;ux-=dot*dx;uy-=dot*dy;uz-=dot*dz;unit(ux,uy,uz);}double vx=dy*uz-dz*uy, vy=dz*ux-dx*uz,vz=dx*uy-dy*ux;ox=mu*dx+st*(cos(phi)*ux+sin(phi)*vx);oy=mu*dy+st*(cos(phi)*uy+sin(phi)*vy);oz=mu*dz+st*(cos(phi)*uz+sin(phi)*vz);unit(ox,oy,oz);}
int main(int argc,char**argv){ifstream f("/tmp/data/xcom_water.txt");string line;while(getline(f,line)){if(line.empty()||line[0]=='#')continue;istringstream ss(line);Row r;double t,n;ss>>r.E>>r.coh>>r.inc>>r.pe>>r.pn>>r.pef>>t>>n;tab.push_back(r);}long long N=atoll(argv[1]);double primary[28]={},scat[28]={},angp[10]={},angs[10]={},ref=0;double proc[2][4]={},trans[2][4]={},elcnt[2]={},elsum[2]={},ebin[2][25]={},edirect[10]={};
 for(long long k=0;k<N;k++){
 double E=R()<.5?1.1732:1.3325;double u,v;do{u=(2*R()-1)*.05;v=(2*R()-1)*.05;}while(R()>pow(1+u*u+v*v,-1.5));double dx=u,dy=v,dz=1;unit(dx,dy,dz);double x=100*u,y=100*v,z=0;
 ref+=lenscore(x,y,z,dx,dy,dz,30/dz);
 bool isprim=true,alive=true;
 for(int it=0;it<1000&&alive;it++){
  double coh=coeff(E,1),inc=coeff(E,2),photo=coeff(E,3),pair=coeff(E,4)+coeff(E,5),mut=coh+inc+photo+pair;
  double bx=1e30,by=1e30,bz=1e30;if(dx>1e-14)bx=(15-x)/dx;else if(dx< -1e-14)bx=(-15-x)/dx;if(dy>1e-14)by=(15-y)/dy;else if(dy< -1e-14)by=(-15-y)/dy;if(dz>1e-14)bz=(30-z)/dz;else if(dz< -1e-14)bz=(0-z)/dz;double bout=min(bx,min(by,bz));if(bout<=1e-10){alive=false;break;}
  double ds=-log(max(1e-16,R()))/mut;bool hit=ds<bout;double L=hit?ds:bout;double sl=lenscore(x,y,z,dx,dy,dz,L);if(sl>0){int b=bn(pg,E);if(b>=0){if(isprim)primary[b]+=sl;else scat[b]+=sl;}int a=max(0,min(9,(int)((dz+1)*5)));if(isprim)angp[a]+=sl;else angs[a]+=sl;}
  x+=L*dx;y+=L*dy;z+=L*dz;if(!hit){alive=false;break;}
  bool ins=(z>=4.9&&z<=5.1&&x*x+y*y<=1.0);double pick=R()*mut;int typ;if(pick<coh)typ=3;else if(pick<coh+inc)typ=0;else if(pick<coh+inc+photo)typ=1;else typ=2;
  if(ins){int par=isprim?0:1;proc[par][typ]++;double de=0;if(typ==0){ // sample KN scattering
    double mu,ratio;do{mu=2*R()-1;ratio=1/(1+(E/.51099895)*(1-mu));double fkn=ratio*ratio*(ratio+1/ratio-(1-mu*mu));}while(R()>0.5*ratio*ratio*(ratio+1/ratio-(1-mu*mu)));
    de=E*(1-ratio);double phi=2*PI*R();double ox,oy,oz;rotate(dx,dy,dz,mu,phi,ox,oy,oz);double ex=E*dx-(E-de)*ox,ey=E*dy-(E-de)*oy,ez=E*dz-(E-de)*oz;double q=sqrt(ex*ex+ey*ey+ez*ez);if(q>0){double cost=ez/q;int ai=max(0,min(9,(int)((cost+1)*5)));edirect[ai]++;}double ener=de;int eb=bn(eg,ener);if(eb>=0){ebin[par][eb]++;elcnt[par]++;elsum[par]+=ener;}
    E*=ratio;dx=ox;dy=oy;dz=oz;
   }else if(typ==1){de=E;int eb=bn(eg,de);if(eb>=0){ebin[par][eb]++;elcnt[par]++;elsum[par]+=de;}alive=false;
   }else if(typ==2){de=max(0.0,E-1.021998);if(de>0){for(int j=0;j<2;j++){double ener=de/2;int eb=bn(eg,ener);if(eb>=0){ebin[par][eb]++;elcnt[par]++;elsum[par]+=ener;}}}alive=false;
   }else {de=0;double mu=2*R()-1,phi=2*PI*R(),ox,oy,oz;rotate(dx,dy,dz,mu,phi,ox,oy,oz);dx=ox;dy=oy;dz=oz;}
   trans[par][typ]+=de;
  }else{
   if(typ==0){double mu;do{mu=2*R()-1;double ratio=1/(1+(E/.51099895)*(1-mu));}while(R()>0.5*pow(1/(1+(E/.51099895)*(1-mu)),2)*(1/(1+(E/.51099895)*(1-mu))+1+(1+(E/.51099895)*(1-mu))-(1-mu*mu))); // expression f with 1/r = 1+(alpha)(1-mu)
    double ratio=1/(1+(E/.51099895)*(1-mu));double ox,oy,oz;rotate(dx,dy,dz,mu,2*PI*R(),ox,oy,oz);E*=ratio;dx=ox;dy=oy;dz=oz;
   }else if(typ==3){double mu=2*R()-1,ox,oy,oz;rotate(dx,dy,dz,mu,2*PI*R(),ox,oy,oz);dx=ox;dy=oy;dz=oz;}else alive=false;
  }
  if(typ!=3)isprim=false;
 }
 }
 cout<<setprecision(12)<<"N "<<N<<" REF "<<ref<<"\nP ";for(double x:primary)cout<<x/ref<<" ";cout<<"\nS ";for(double x:scat)cout<<x/ref<<" ";cout<<"\nANGP ";for(double x:angp)cout<<x/ref<<" ";cout<<"\nANGS ";for(double x:angs)cout<<x/ref<<" ";cout<<"\n";
 for(int j=0;j<2;j++){cout<<"PROC"<<j<<" ";for(int i=0;i<4;i++)cout<<proc[j][i]<<" ";cout<<" TRANS ";for(int i=0;i<4;i++)cout<<trans[j][i]<<" ";cout<<" EL "<<elcnt[j]<<" "<<elsum[j]<<" EB ";for(int i=0;i<25;i++)cout<<ebin[j][i]<<" ";cout<<"\n";}cout<<"EDIR ";for(double v:edirect)cout<<v<<" ";cout<<"\n";
}
'''
open('/tmp/mc.cpp','w').write(code)
import subprocess,os
p=subprocess.run(['g++','-O3','-std=c++17','/tmp/mc.cpp','-o','/tmp/mc'],capture_output=True,text=True);print(p.stderr,p.returncode)
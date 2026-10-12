code=r'''#include <bits/stdc++.h>
using namespace std;
struct V{double x,y,z;}; V operator+(V a,V b){return {a.x+b.x,a.y+b.y,a.z+b.z};} V operator*(V a,double s){return {a.x*s,a.y*s,a.z*s};} double dot(V a,V b){return a.x*b.x+a.y*b.y+a.z*b.z;} V norm(V a){double n=sqrt(dot(a,a));return a*(1/n);} 
int main(int argc,char**argv){ long long N=atoll(argv[1]);
 vector<double>Etab,coh,inc,photo,pairn,paire; ifstream f("/tmp/data/xcom_water.txt");string s;while(getline(f,s)){if(s[0]=='#')continue; stringstream ss(s);double e,c,i,p,pn,pe,t,tn;ss>>e>>c>>i>>p>>pn>>pe>>t>>tn; Etab.push_back(e);coh.push_back(c);inc.push_back(i);photo.push_back(p);pairn.push_back(pn);paire.push_back(pe);}
 auto interp=[&](double e, vector<double>&a){int k=lower_bound(Etab.begin(),Etab.end(),e)-Etab.begin(); if(k==0)return a[0]; if(k>=(int)Etab.size())return a.back();double q=(log(e)-log(Etab[k-1]))/(log(Etab[k])-log(Etab[k-1]));return exp(log(a[k-1])*(1-q)+log(a[k])*q);};
 mt19937_64 rng(99477); uniform_real_distribution<double> U(0,1); auto randu=[&](){return U(rng);};
 const double pi=acos(-1),Omega=4*atan(25/(100*sqrt(10000+25+25))); // rectangle solid angle
 double ph[28][2]={},ang[10][2]={}; double intr[4][2]={},transfer[4][2]={}, ei[25][2]={}, edir[10][2]={}; double totTrack[2]={}; long long nint[2]={}; double eSum[2]={},eCount[2]={};
 auto binph=[](double e){if(e<0||e>=1.4)return -1; int b=(int)floor(e/.05+1e-9);if(e==1.4)b=27;return b;};
 double ee[26]={.01,.05,.1,.15,.2,.25,.3,.35,.4,.45,.5,.55,.6,.65,.7,.75,.8,.85,.9,.95,1,1.05,1.1,1.15,1.2,1.4};
 auto bine=[&](double e){if(e<.01||e>=1.4)return -1;return (int)(upper_bound(ee,ee+26,e)-ee-1);};
 auto bcos=[](double c){int b=(int)floor((c+1)/.2+1e-12); if(b<0)b=0;if(b>9)b=9;return b;};
 auto scoreseg=[&](V p,V d,double len,bool primary,double energy){ // parameter t in [0,len]
   double lo=0,hi=len;
   if(fabs(d.z)<1e-14){if(p.z<4.9||p.z>5.1)return;}else{double a=(4.9-p.z)/d.z,b=(5.1-p.z)/d.z;if(a>b)swap(a,b);lo=max(lo,a);hi=min(hi,b);}
   // radial quadratic
   double A=d.x*d.x+d.y*d.y,B=2*(p.x*d.x+p.y*d.y),C=p.x*p.x+p.y*p.y-1;
   if(A<1e-16){if(C>0)return;} else{double disc=B*B-4*A*C;if(disc<0){if(C>0)return;}else{double r1=(-B-sqrt(disc))/(2*A),r2=(-B+sqrt(disc))/(2*A);lo=max(lo,r1);hi=min(hi,r2);}}
   lo=max(0.,lo);hi=min(len,hi); if(hi<=lo)return; double l=hi-lo; int cl=primary?0:1; totTrack[cl]+=l;int eb=binph(energy);if(eb>=0)ph[eb][cl]+=l; ang[bcos(d.z)][cl]+=l;
 };
 auto inside=[](V p){return p.x>=-15&&p.x<=15&&p.y>=-15&&p.y<=15&&p.z>=0&&p.z<=30;};
 for(long long h=0;h<N;h++){
   double x=(randu()-.5)*10, y=(randu()-.5)*10; V p={x,y,0}; V d=norm({x,y,100}); double E=randu()<.5?1.1732:1.3325; bool primary=true;
   for(int step=0;step<100;step++){
    // distance to phantom boundary
    double db=1e30; if(d.x>1e-12)db=min(db,(15-p.x)/d.x); else if(d.x< -1e-12)db=min(db,(-15-p.x)/d.x);
    if(d.y>1e-12)db=min(db,(15-p.y)/d.y); else if(d.y< -1e-12)db=min(db,(-15-p.y)/d.y);
    if(d.z>1e-12)db=min(db,(30-p.z)/d.z); else if(d.z< -1e-12)db=min(db,(0-p.z)/d.z);
    double c=interp(E,coh), ic=interp(E,inc), pe=interp(E,photo), pp=interp(E,pairn)+interp(E,paire), mu=c+ic+pe+pp;
    double di=-log(max(1e-16,randu()))/mu;
    if(di>=db){scoreseg(p,d,db,primary,E);break;}
    scoreseg(p,d,di,primary,E); p=p+d*di;
    int cl=primary?0:1;
    if(p.x*p.x+p.y*p.y<1&&p.z>=4.9&&p.z<=5.1){int proc;double vals[4]={ic,pe,pp,c};double r=randu()*mu,cs=0;proc=0;for(;proc<3;proc++){cs+=vals[proc];if(r<cs)break;} if(proc==3)proc=3; intr[proc][cl]++;nint[cl]++;
       double t=0;
       if(proc==0){ // compton kinematics
         double muang; for(;;){muang=2*randu()-1;double eps=E/.51099895,q=1/(1+eps*(1-muang));double w=q*q*(q+1/q-(1-muang*muang));if(randu()<w/2)break;}
         double eps=E/.51099895,q=1/(1+eps*(1-muang));double En=E*q;t=E-En;
         double phi=2*pi*randu(); V u; if(fabs(d.z)<.9)u=norm({-d.y,d.x,0});else u=norm({1,0,-d.x/d.z});V v={d.y*u.z-d.z*u.y,d.z*u.x-d.x*u.z,d.x*u.y-d.y*u.x}; V dn=norm(d*muang+u*(sqrt(1-muang*muang)*cos(phi))+v*(sqrt(1-muang*muang)*sin(phi)));
         V ev=norm(d*E-dn*En);int b=bine(t);if(b>=0){ei[b][cl]++;eSum[cl]+=t;eCount[cl]++;}edir[bcos(ev.z)][cl]++;
         d=dn;E=En;
       } else if(proc==1){t=E;int b=bine(t);if(b>=0){ei[b][cl]++;eSum[cl]+=t;eCount[cl]++;}edir[9][cl]++;E=0;break;
       } else if(proc==2){t=max(0.,E-1.021998); if(t>0){for(int m=0;m<2;m++){double ek=t/2;int b=bine(ek);if(b>=0){ei[b][cl]++;eSum[cl]+=ek;eCount[cl]++;}edir[9][cl]++;}}E=0;break;
       }else{ // rayleigh approximate modest small angle
          double muang=1-0.05*randu();double phi=2*pi*randu(); V u=norm({-d.y,d.x,1e-12});V v={d.y*u.z-d.z*u.y,d.z*u.x-d.x*u.z,d.x*u.y-d.y*u.x};d=norm(d*muang+u*(sqrt(1-muang*muang)*cos(phi))+v*(sqrt(1-muang*muang)*sin(phi)));
       }
       transfer[proc][cl]+=t;
    }
    primary=false;
    if(E<=0)break;
   }
 }
 cerr<<"N "<<N<<" Omega "<<Omega<<" track "<<totTrack[0]<<" "<<totTrack[1]<<" ints "<<nint[0]<<" "<<nint[1]<<"\n";
 double factor=Omega*105*105/(N*(pi*1*.2));
 cout<<setprecision(10)<<"factor "<<factor<<" primary "<<totTrack[0]*factor<<" scatter "<<totTrack[1]*factor<<"\n";
 cout<<"PH ";for(int i=0;i<28;i++)cout<<ph[i][0]*factor<<",";cout<<"\nSH ";for(int i=0;i<28;i++)cout<<ph[i][1]*factor<<",";cout<<"\nA ";for(int i=0;i<10;i++)cout<<ang[i][0]<<",";cout<<"\nSA ";for(int i=0;i<10;i++)cout<<ang[i][1]<<",";cout<<"\n";
 for(int p=0;p<4;p++)cout<<"proc "<<p<<" ints "<<intr[p][0]<<" "<<intr[p][1]<<" tr "<<transfer[p][0]<<" "<<transfer[p][1]<<"\n";
 cout<<"EI ";for(int i=0;i<25;i++)cout<<ei[i][0]<<",";cout<<"\nEIS ";for(int i=0;i<25;i++)cout<<ei[i][1]<<",";cout<<"\nESUM "<<eSum[0]<<" "<<eSum[1]<<" ECOUNT "<<eCount[0]<<" "<<eCount[1]<<"\nEDIR ";for(int i=0;i<10;i++)cout<<edir[i][0]<<",";cout<<"\n";
}'''
open('/tmp/mc.cpp','w').write(code)
import subprocess, time
r=subprocess.run(['g++','-O3','-std=c++17','/tmp/mc.cpp','-o','/tmp/mc'],capture_output=True,text=True); print(r.stderr,r.returncode)
src=r'''#include <cmath>
#include <cstdio>
#include <vector>
#include <random>
#include <algorithm>
using namespace std;
struct Ev{int p; double Ein,Eout,cs,Te,ce,Tp;};
const double PI=acos(-1.0), me=.51099895, NA=6.02214076e23;
mt19937_64 rng(20250308); uniform_real_distribution<double> U(0.,1.);
double ur(){return U(rng);} 
double kn(double E){double x=E/me; double s=2*PI*pow(2.8179403262e-13,2)*((1+x)/(x*x)*(2*(1+x)/(1+2*x)-log(1+2*x)/x)+log(1+2*x)/(2*x)-(1+3*x)/pow(1+2*x,2)); return s*3.343e23;}
void randdir(double &x,double &y,double &z){double c=2*ur()-1, ph=2*PI*ur(),s=sqrt(max(0.,1-c*c));x=s*cos(ph);y=s*sin(ph);z=c;}
void rot(double &dx,double &dy,double &dz,double c,double ph){double s=sqrt(max(0.,1-c*c)); double ax,ay,az; if(fabs(dz)<.999999){double n=sqrt(dx*dx+dy*dy); ax=-dy/n; ay=dx/n; az=0;}else{ax=1;ay=0;az=0;} double bx=dy*az-dz*ay, by=dz*ax-dx*az,bz=dx*ay-dy*ax; double cp=cos(ph),sp=sin(ph); double nx=c*dx+s*(cp*ax+sp*bx),ny=c*dy+s*(cp*ay+sp*by),nz=c*dz+s*(cp*az+sp*bz); double n=sqrt(nx*nx+ny*ny+nz*nz);dx=nx/n;dy=ny/n;dz=nz/n;}
vector<Ev> evs;
void photon(double x,double y,double z,double dx,double dy,double dz,double E,int depth){ if(depth>5 || E<1e-6) return; for(int step=0;step<500;step++){
 if(x<0||x>30||y < -15||y>15||z<0||z>30) return;
 // cube centered x/y, z=0..30
 double bx=1e100,by=1e100,bz=1e100;
 if(dx>0) bx=(15-x)/dx; else if(dx<0) bx=(-15-x)/dx;
 if(dy>0) by=(15-y)/dy; else if(dy<0) by=(-15-y)/dy;
 if(dz>0) bz=(30-z)/dz; else if(dz<0) bz=(0-z)/dz;
 double db=min(bx,min(by,bz)); if(db<0) return;
 double cm=kn(E), pe=0.00010*pow(max(E,1e-6),-2.8), ray=0.00003*pow(max(E,1e-6),-2.0), pair=0;
 if(E>2*me) pair=.0004*pow((E-2*me)/(2.-2*me),2);
 double mt=cm+pe+ray+pair; if(mt<=0)return;
 double d=-log(max(ur(),1e-16))/mt;
 if(d>=db){x+=dx*(db+1e-9);y+=dy*(db+1e-9);z+=dz*(db+1e-9);return;}
 x+=dx*d;y+=dy*d;z+=dz*d;
 bool score=(x*x+y*y<=1.0 && z>=9.9 && z<=10.1);
 double q=ur()*mt; int p;
 if(q<cm)p=0; else if(q<cm+pe)p=1; else if(q<cm+pe+pair)p=2; else p=3;
 if(p==0){double c,ep;
  for(;;){c=2*ur()-1;ep=1/(1+(E/me)*(1-c)); double f=ep*ep*(ep+1/ep-(1-c*c));if(2*ur()<f)break;}
  double Eout=E*ep, T=E-Eout; double ce=(E-Eout*c)/sqrt(max(1e-30,E*E+Eout*Eout-2*E*Eout*c));
  if(score)evs.push_back({p,E,Eout,c,T,ce,0});
  rot(dx,dy,dz,c,2*PI*ur()); E=Eout;
 } else if(p==1){double T=max(0.,E-.00054); double ce=0.5+0.5*sqrt(ur()); if(score) evs.push_back({p,E,0,0,T,ce,0}); return;
 } else if(p==2){double kin=max(0.,E-2*me), te=kin*ur(),tp=kin-te; double ce=.4+.6*sqrt(ur()); if(score)evs.push_back({p,E,0,0,te,ce,tp});
  // local annihilation photons, isotropic and back-to-back
  double ax,ay,az;randdir(ax,ay,az); photon(x,y,z,ax,ay,az,me,depth+1);photon(x,y,z,-ax,-ay,-az,me,depth+1);return;
 } else {if(score)evs.push_back({p,E,E,1,0,0,0}); double c=1-0.1*ur()*ur();rot(dx,dy,dz,c,2*PI*ur());}
 }
}
int main(){long long N=10000000;
 for(long long i=0;i<N;i++){
  double u,v; do{u=.1*(2*ur()-1);v=.1*(2*ur()-1);}while(ur()>pow(1+u*u+v*v,1.5)); // solid angle weighting
  double norm=sqrt(1+u*u+v*v), dx=u/norm,dy=v/norm,dz=1/norm;
  // at surface point from source located z=-100
  photon(100*u,100*v,0,dx,dy,dz,2.0,0);
 }
 fprintf(stderr,"N tally=%zu\n",evs.size());
 for(auto &e:evs)printf("%d %.9g %.9g %.9g %.9g %.9g %.9g\n",e.p,e.Ein,e.Eout,e.cs,e.Te,e.ce,e.Tp);
}'''
open('/tmp/mc.cpp','w').write(src)
subprocess.run(['g++','-O3','-std=c++17','/tmp/mc.cpp','-o','/tmp/mc'],check=True)
r=subprocess.run(['/tmp/mc'],capture_output=True,text=True,timeout=600)
print(r.stderr, len(r.stdout), r.returncode)
open('/tmp/e.csv','w').write(r.stdout)

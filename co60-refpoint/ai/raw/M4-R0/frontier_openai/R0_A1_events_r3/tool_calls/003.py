src=r'''#include <bits/stdc++.h>
using namespace std;
struct V { double x,y,z; };
V operator+(V a,V b){return {a.x+b.x,a.y+b.y,a.z+b.z};}
V operator*(V a,double s){return {a.x*s,a.y*s,a.z*s};}
double dot(V a,V b){return a.x*b.x+a.y*b.y+a.z*b.z;}
V norm(V a){double q=sqrt(dot(a,a)); return a*(1/q);}
struct Ev {int p; double Ein,Eout,cs,Te,ce,Tp;};
int main(int argc,char**argv){
 long long N=atoll(argv[1]);
 mt19937_64 g(20250308); uniform_real_distribution<double> U(0.,1.);
 vector<Ev> evs; evs.reserve(10000);
 const double NA=6.02214076e23, re=2.8179403262e-13, me=.51099895, ne=10.0/18.0*NA;
 auto sigkn=[&](double E){double x=E/me; return 2*M_PI*re*re*((1+x)/(x*x)*(2*(1+x)/(1+2*x)-log1p(2*x)/x)+log1p(2*x)/(2*x)-(1+3*x)/((1+2*x)*(1+2*x)));};
 auto scatter=[&](double E,V d,double &Eo,double &mu){
   // rejection from Klein-Nishina angular density
   for(;;){ mu=2*U(g)-1; double a=E/me, r=1/(1+a*(1-mu)); double w=r*r*(r+1/r-(1-mu*mu)); if(U(g)*2.0<=w) break; }
   Eo=E/(1+(E/me)*(1-mu));
   double ph=2*M_PI*U(g), st=sqrt(max(0.,1-mu*mu));
   V e1;
   if(fabs(d.z)<0.9) e1=norm(V{-d.y,d.x,0}); else e1=norm(V{0,-d.z,d.y});
   V e2={d.y*e1.z-d.z*e1.y,d.z*e1.x-d.x*e1.z,d.x*e1.y-d.y*e1.x};
   V nd=d*mu + e1*(st*cos(ph)) + e2*(st*sin(ph)); d=norm(nd);
   return d;
 };
 for(long long k=0;k<N;k++){
   // uniform solid angle in rectangular pyramid: rejection-correct uniform entrance coordinates
   double x,y;
   do {x=10*U(g)-5; y=10*U(g)-5; double w=pow(100./sqrt(10000+x*x+y*y),3); if(U(g)<=w) break;}while(true);
   V pos={x,y,0}, d=norm(V{x,y,100});
   double E= U(g)<.5 ? 1.1732 : 1.3325;
   for(int nsc=0;nsc<2000;nsc++){
      if(E<0.001) break;
      // distance to cube boundary
      double db=1e100;
      if(d.x>1e-14) db=min(db,(15-pos.x)/d.x); else if(d.x< -1e-14) db=min(db,(-15-pos.x)/d.x);
      if(d.y>1e-14) db=min(db,(15-pos.y)/d.y); else if(d.y< -1e-14) db=min(db,(-15-pos.y)/d.y);
      if(d.z>1e-14) db=min(db,(30-pos.z)/d.z); else if(d.z< -1e-14) db=min(db,(0-pos.z)/d.z);
      double cm=ne*sigkn(E);
      double photo=4.0e-6*pow(E,-3.0);
      double ray=1.0e-5*pow(E,-2.0);
      double pair= E>1.0219979 ? 8.3e-6*pow(E-1.0219979,2.5) : 0.;
      double total=cm+photo+ray+pair;
      double ds=-log(max(1e-300,U(g)))/total;
      if(ds>=db){pos=pos+d*(db+1e-9); break;}
      pos=pos+d*ds;
      double r2=pos.x*pos.x+pos.y*pos.y;
      bool score=(pos.z>=4.9 && pos.z<5.1 && r2<1.0);
      double q=U(g)*total; int proc=(q<cm?0:q<cm+photo?1:q<cm+photo+pair?2:3);
      if(proc==0){
        double Eo,mu; V old=d; d=scatter(E,old,Eo,mu); double T=E-Eo;
        double p=sqrt(max(0.,T*(T+2*me)));
        double ce=(p>1e-12)?(E-Eo*mu)/p:1.; ce=max(-1.,min(1.,ce));
        if(score) evs.push_back({0,E,Eo,mu,T,ce,-1});
        E=Eo;
      } else if(proc==1){
        double T=max(0.,E-0.00054);
        if(score) evs.push_back({1,E,-1,-1,T,1.,-1});
        break;
      } else if(proc==2){
        double avail=E-1.0219979; double Te=avail*U(g), Tp=avail-Te;
        if(score) evs.push_back({2,E,-1,-1,Te,0.98,Tp});
        break;
      } else {
        double mu=1.-2*U(g)*min(1.,0.01/(E+0.01));
        if(score) evs.push_back({3,E,E,mu,0.,-1,-1});
        double ph=2*M_PI*U(g), st=sqrt(max(0.,1-mu*mu));
        V e1; if(fabs(d.z)<0.9)e1=norm(V{-d.y,d.x,0}); else e1=norm(V{0,-d.z,d.y});
        V e2={d.y*e1.z-d.z*e1.y,d.z*e1.x-d.x*e1.z,d.x*e1.y-d.y*e1.x};
        d=norm(d*mu+e1*(st*cos(ph))+e2*(st*sin(ph)));
      }
   }
 }
 cerr<<"N "<<N<<" recorded "<<evs.size()<<"\n";
 // Reservoir sample 200 uniformly from all recorded interactions.
 if(evs.size()<200){cerr<<"too few\n";return 2;}
 vector<Ev> s(evs.begin(),evs.begin()+200);
 for(size_t i=200;i<evs.size();i++){uniform_int_distribution<size_t> R(0,i); size_t j=R(g); if(j<200)s[j]=evs[i];}
 shuffle(s.begin(),s.end(),g);
 for(auto a:s) cout<<a.p<<" "<<setprecision(10)<<a.Ein<<" "<<a.Eout<<" "<<a.cs<<" "<<a.Te<<" "<<a.ce<<" "<<a.Tp<<"\n";
}
'''
open('/tmp/sim.cpp','w').write(src)
res=subprocess.run(['g++','-O3','-std=c++17','/tmp/sim.cpp','-o','/tmp/sim'],capture_output=True,text=True)
print(res.returncode,res.stderr[:1000])
#ifdef SHRT
#define STRINGIFY(A) #A
#define XTRINGIFY(A) STRINGIFY(A)
string kernel_source = XTRINGIFY((
#endif

typedef struct{
  cl_float r[3];
} DOM;

typedef struct{
  cl_uint i;
  cl_float t;
  cl_uint n;
  cl_float z;
} hit;

typedef struct{
  cl_float4 r;    // location, time
  cl_float4 n;    // direction, track length
  cl_uint q;      // track segment
#ifdef ANGW
  cl_float f;     // fraction of light from muon alone (without cascades)
#endif
#ifdef LONG
  cl_float a, b;  // longitudinal development parametrization coefficients
#endif
} photon;

typedef struct{
  cl_float wvl;             // wavelength of this block
  cl_float ocm;             // 1 / speed of light in medium
  cl_float coschr, sinchr;  // cos and sin of the cherenkov angle
  struct{
    cl_float abs;           // absorption
    cl_float sca;           // scattering
  } z [MAXLYS];
} ices;

typedef struct{
  cl_short n, max;
  cl_float x, y, r;
  cl_float h, d;
  cl_float dl, dh;
} line;

typedef struct{
  ices w[WNUM];
  cl_uint rm[MAXRND];
  cl_ulong rs[MAXRND];
} datz;

typedef struct{
  cl_uint hidx;
  cl_uint ab;    // if TOT was abnormal

  cl_int type;   // 0=cascade/1=flasher/2=flasher 45/3=laser up/4=laser down
  cl_float r[3]; // flasher/laser coordinates
  cl_float ka, up;    // 2d-gaussian rms and zenith of cone

  cl_uint hnum;  // size of hits buffer
  cl_int size;   // size of kurt table
  cl_int rsize;  // count of multipliers
  cl_int gsize;  // count of initialized OMs

  cl_float dh, hdh, rdh, hmin; // step, step/2, 1/step, and min depth

  cl_float ocv;  // 1 / speed of light in vacuum
  cl_float sf;   // scattering function: 0=HG; 1=SAM
  cl_float g, g2, gr; // g=<cos(scattering angle)>, g2=g*g and gr=(1-g)/(1+g)
  cl_float R, R2, zR; // DOM radius, radius^2, and inverse "oversize" scaling factor

  cl_int cn[2];
  cl_float cl[2], crst[2];

  cl_uchar is[CX][CY];
  cl_uchar ls[NSTR];
  line sc[NSTR];
  cl_float rx;

  cl_float fldr; // horizontal direction of the flasher led #1
  cl_float eff;  // OM efficiency correction

#ifdef ASENS
  cl_float mas;  // maximum angular sensitivity
  cl_float s[ANUM]; // ang. sens. coefficients
#endif

#ifdef ROMB
  cl_float cb[2][2];
#endif
#ifdef TILT
  cl_int lnum, lpts, l0;
  cl_float lmin, lrdz, r0;
  cl_float lnx, lny;
  cl_float lr[LMAX];
  cl_float lp[LMAX][LYRS];
#endif

  cl_short fla;
#ifdef ANIZ
  short k;          // ice anisotropy: 0: no, 1: yes
  float k1, k2, kz; // ice anisotropy parameters
  float azx, azy;   // ice anisotropy direction
#endif
} dats;

#ifdef SHRT
#define sin native_sin
#define cos native_cos
#define pow native_powr
#define exp native_exp
#define log native_log
#define exp2 native_exp2
#define sqrt native_sqrt
#define rsqrt native_rsqrt

#ifndef CL_VERSION_1_2
#define clamp(x,y,z) min(max(x,y),z)
#endif

#ifdef XAMD
#define inv(x) native_recip(x)
#define div(x,y) native_divide(x,y)
#else
#define inv(x) 1/(x)
#define div(x,y) (x)/(y)
#endif

float xrnd(uint4 * s){
  uint tmp;
  do{
    ulong sda = (*s).z * (ulong) (*s).x;
    sda += (*s).y; (*s).x = sda; (*s).y = sda >> 32; tmp = (*s).x >> 9;
  } while(tmp==0);
  return as_float(tmp|0x3f800000)-1.0f;
}

#ifdef LONG
float mrnd(float k, uint4 * s){  // gamma distribution
  float x;
  if(k<1){  // Weibull algorithm
    float c=inv(k);
    float d=(1-k)*pow(k, div(k, 1-k));
    float z, e;
    do{
      z=-log(xrnd(s));
      e=-log(xrnd(s));
      x=pow(z, c);
    } while(z+e<d+x);
  }
  else{  // Cheng's algorithm
    float b=k-log(4.0f);
    float l=sqrt(2*k-1);
    float c=1+log(4.5f);
    float u, v, y, z, r;
    do{
      u=xrnd(s); v=xrnd(s);
      y=div(log(div(v, 1-v)), l);
      x=k*exp(y);
      z=u*v*v;
      r=b+(k+l)*y-x;
    } while(r<4.5f*z-c && r<log(z));
  }
  return x;
}
#endif

float4 my_normalize(float4 n){
  n.xyz*=rsqrt(n.x*n.x+n.y*n.y+n.z*n.z);
  return n;
}

float4 turn(float cs, float si, float4 n, uint4 * s){
  float4 r = (float4)(n.xyz*n.xyz, 0);

  float4 p1 = (float4)(0);
  float4 p2 = (float4)(0);

  if(r.y>r.z){
    if(r.y>r.x){
      p1 = (float4)(n.y, -n.x, 0, 0);
      p2 = (float4)(0, -n.z, n.y, 0);
    }
    else{
      p1 = (float4)(-n.y, n.x, 0, 0);
      p2 = (float4)(-n.z, 0, n.x, 0);
    }
  }
  else{
    if(r.z>r.x){
      p1 = (float4)(n.z, 0, -n.x, 0);
      p2 = (float4)(0, n.z, -n.y, 0);
    }
    else{
      p1 = (float4)(-n.y, n.x, 0, 0);
      p2 = (float4)(-n.z, 0, n.x, 0);
    }
  }

  p1 = my_normalize(p1);
  p2 = my_normalize(p2);

  r = p1-p2; p2 += p1;
  p1 = my_normalize(r);
  p2 = my_normalize(p2);

  float xi = 2*FPI*xrnd(s);
  float2 p = (float2)(cos(xi), sin(xi));

  return my_normalize((float4)(cs*n.xyz+si*(p.x*p1.xyz+p.y*p2.xyz), n.w));
}

#ifdef TILT
float zshift(__local dats * d, float4 r){
  if(d->lnum==0) return 0;
  float z=(r.z-d->lmin)*d->lrdz;
  int k=clamp(convert_int_sat_rtn(z), 0, d->lpts-2);
  int l=k+1;

  float nr=d->lnx*r.x+d->lny*r.y-d->r0;
  for(int j=1; j<LMAX; j++) if(nr<d->lr[j] || j==d->lnum-1){
    int i=j-1;
    return div( (d->lp[j][l]*(z-k)+d->lp[j][k]*(l-z))*(nr-d->lr[i]) +
		(d->lp[i][l]*(z-k)+d->lp[i][k]*(l-z))*(d->lr[j]-nr), d->lr[j]-d->lr[i] );
  }
  return 0;
}
#endif

float2 ctr(__local dats * d, float2 r){
#ifdef ROMB
  return (float2)(d->cb[0][0]*r.x+d->cb[1][0]*r.y, d->cb[0][1]*r.x+d->cb[1][1]*r.y);
#else
  return r;
#endif
}

#if defined(USMA) && defined(RAND)
#define XINC i=atom_add(&e.hidx, get_num_groups(0))
#define XIDX get_global_size(0)+get_group_id(0)
#else
#define XINC i+=e.hidx
#define XIDX get_global_size(0)
#endif

__kernel void propagate(__private uint num,
			__global dats * ed,
			__global datz * ez,
			__global hit * eh,
			__global photon * ep,
			__constant DOM * oms){

  if(num==0){
    ed->hidx=0;
    ed->ab=0;
    write_mem_fence(CLK_GLOBAL_MEM_FENCE);
    return;
  }

  __local dats e;
  __global ices * w;

  {
    event_t ev=async_work_group_copy((__local char *) &e, (__global char *) ed, sizeof(e), 0);
    wait_group_events(1, &ev);
    if(get_local_id(0)==0) e.hidx=XIDX;
    barrier(CLK_LOCAL_MEM_FENCE);
  }

  const unsigned int idx=get_local_id(0)*get_num_groups(0)+get_group_id(0);

  int niw=0, old=idx%e.rsize;
  uint4 s = (uint4)(ez->rs[old], ez->rs[old] >> 32, ez->rm[old], old);

  float4 r=(float4)(0);
  float4 n=(float4)(0);
  float TOT=0, sca=0;

  for(uint i=idx; i<num; TOT==0 && (XINC)){
    int om=-1;
    if(TOT==0){ // initialize photon
      w=&ez->w[min(convert_int_sat_rtn(WNUM*xrnd(&s)), WNUM-1)];
      n.w=w->ocm;

      if(e.type>0){
	r = (float4)(e.r[0], e.r[1], e.r[2], 0);

	float xi=xrnd(&s);
	if(e.fldr<0) xi*=2*FPI;
	else{
	  int r=convert_int_sat_rtn(e.fldr/360)+1;
	  int s=convert_int_sat_rtn(xi*r);
	  xi=radians(e.fldr+s*360/r);
	}
	n.x=cos(xi), n.y=sin(xi);
	if(e.ka>0){
	  float ang=radians(30.f);
	  float FLZ=OMR*sin(ang), FLR=OMR*cos(ang);
	  r+=(float4)(FLR*n.x, FLR*n.y, FLZ, OMR*n.w);
	}

	float np=cos(e.up); n.z=sin(e.up);
	n.x*=np; n.y*=np;

	if(e.ka>0){
	  do{ xi=1+e.ka*log(xrnd(&s)); } while (xi<-1);
	  float si=sqrt(1-xi*xi); n=turn(xi, si, n, &s);
	}
      }
      else{
	photon p=ep[i/OVER];
	r=p.r, n.xyz=p.n.xyz;
	float l=p.n.w; niw=p.q;

	if(l<0){
	  float xi;
	  if(e.ka>0){
	    do{ xi=1+e.ka*log(xrnd(&s)); } while (xi<-1);
	    float si=sqrt(1-xi*xi); n=turn(xi, si, n, &s);
	  }
	}
	else{
	  if(l>0) l*=xrnd(&s);
#ifdef LONG
	  else if(p.b>0) l=p.b*mrnd(p.a, &s);
#endif
	  if(l>0){
	    r.xyz+=l*n.xyz;
	    r.w+=l*e.ocv;
	  }

#ifdef ANGW
	  if(p.f<xrnd(&s)){
	    const float a=0.39f, b=2.61f;
	    const float I=1-exp(-b*exp2(a));
	    float cs=max(1-pow(-div(log(1-xrnd(&s)*I), b), inv(a)), -1.0f);
	    float si=sqrt(1-cs*cs); n=turn(cs, si, n, &s);
	  }
#endif
	  n=turn(w->coschr, w->sinchr, n, &s);
	}
      }
      om=e.fla;

      TOT=-log(xrnd(&s));
    }

    if(sca==0){ // get distance for overburden
      old=om;
      float z = r.z;
#ifdef TILT
      z-= zshift(&e, r);
#endif

      float nr=1.f;
#ifdef ANIZ
      if(e.k>0){
	float n1= e.azx*n.x+e.azy*n.y;
	float n2=-e.azy*n.x+e.azx*n.y;
	float n3= n.z;

	float s1=n1*n1, l1=e.k1*e.k1;
	float s2=n2*n2, l2=e.k2*e.k2;
	float s3=n3*n3, l3=e.kz*e.kz;

	float B2=div(nr,l1)+div(nr,l2)+div(nr,l3);
	float nB=div(s1,l1)+div(s2,l2)+div(s3,l3);
	float An=s1*l1+s2*l2+s3*l3;

	nr=(B2-nB)*An/2;
	TOT=div(TOT,nr);
      }
#endif

      int i=convert_int_sat_rte((z-e.hmin)*e.rdh);
      if(i<0) i=0; else if(i>=e.size) i=e.size-1;
      float h=e.hmin+i*e.dh; // middle of the layer
      float ahx=n.z<0?h-e.hdh:h+e.hdh;

      float SCA=-log(xrnd(&s));

      float ais=(n.z*SCA-(ahx-z)*w->z[i].sca)*e.rdh;
      float aia=(n.z*TOT-(ahx-z)*w->z[i].abs)*e.rdh;

      int j=i;
      if(n.z<0) for(; j>0 && ais<0 && aia<0; ahx-=e.dh, ais+=w->z[j].sca, aia+=w->z[j].abs) --j;
      else for(; j<e.size-1 && ais>0 && aia>0; ahx+=e.dh, ais-=w->z[j].sca, aia-=w->z[j].abs) ++j;

      float tot;
      if(i==j || fabs(n.z)<XXX) sca=div(SCA,w->z[j].sca), tot=div(TOT,w->z[j].abs);
      else sca=div(div(ais*e.dh,w->z[j].sca)+ahx-z,n.z), tot=div(div(aia*e.dh,w->z[j].abs)+ahx-z,n.z);

      // get overburden for distance
      if(tot<sca) sca=tot, TOT=0; else TOT=nr*(tot-sca)*w->z[j].abs;
    }

    om=-1;
    float del=sca;
    { // sphere
      float2 ri = r.xy, rf = r.xy + del*n.xy;
      float2 pi = ctr(&e, ri), pf = ctr(&e, rf);
      ri = min(pi, pf)-e.rx, rf = max(pi, pf)+e.rx;

      int2 xl = (int2)(clamp(convert_int_sat_rte((ri.x-e.cl[0])*e.crst[0]), 0, e.cn[0]),
		       clamp(convert_int_sat_rte((ri.y-e.cl[1])*e.crst[1]), 0, e.cn[1]));

      int2 xh = (int2)(clamp(convert_int_sat_rte((rf.x-e.cl[0])*e.crst[0]), -1, e.cn[0]-1),
		       clamp(convert_int_sat_rte((rf.y-e.cl[1])*e.crst[1]), -1, e.cn[1]-1));

      for(int i=xl.x, j=xl.y; i<=xh.x && j<=xh.y; ++j<=xh.y?:(j=xl.y,i++)) for(uchar k=e.is[i][j]; k!=0x80; ){
	uchar m=e.ls[k];
	__local line * s = & e.sc[m&0x7f];
	k=m&0x80?0x80:k+1;

	float b=0, c=0, dr;
	dr=s->x-r.x;
	b+=n.x*dr; c+=dr*dr;
	dr=s->y-r.y;
	b+=n.y*dr; c+=dr*dr;

	float np=1-n.z*n.z;
	float D=b*b-(c-s->r*s->r)*np;
	if(D>=0){
	  D=sqrt(D);
	  float h1=b-D, h2=b+D;
	  if(h2>=0 && h1<=del*np){
	    if(np>XXX){
	      h1=div(h1,np), h2=div(h2,np);
	      if(h1<0) h1=0; if(h2>del) h2=del;
	    }
	    else h1=0, h2=del;
	    h1=r.z+n.z*h1, h2=r.z+n.z*h2;
	    float zl, zh;
	    if(n.z>0) zl=h1, zh=h2;
	    else zl=h2, zh=h1;

	    int omin=0, omax=s->max;
	    int n1=s->n-omin+clamp(convert_int_sat_rtp(omin-(zh-s->dl-s->h)*s->d), omin, omax+1);
	    int n2=s->n-omin+clamp(convert_int_sat_rtn(omin-(zl-s->dh-s->h)*s->d), omin-1, omax);

	    for(int l=n1; l<=n2; l++) if(l!=old){
#ifdef OFLA
	      if(l==e.fla) continue;
#endif
	      DOM dom = oms[l];
	      float b=0, c=0, dr;
	      dr=dom.r[0]-r.x;
	      b+=n.x*dr; c+=dr*dr;
	      dr=dom.r[1]-r.y;
	      b+=n.y*dr; c+=dr*dr;
	      dr=dom.r[2]-r.z;
	      b+=n.z*dr; c+=dr*dr;
	      float D=b*b-c+e.R2;
	      if(D>=0){
		float h=b-sqrt(D)*e.zR;
		if(h>0 && h<=del) om=l, del=h+0.0f;
	      }
	    }
	  }
	}
      }
    }

    { // advance
      r+=del*n;
      sca-=del;
    }

    if(!isfinite(TOT) || !isfinite(sca)) atom_add(&ed->ab, 1), TOT=0, sca=0, om=-1;

    float xi=xrnd(&s);
    if(om!=-1){
      bool flag=true;
      hit h; h.i=om; h.t=r.w; h.n=niw; h.z=w->wvl;

#ifdef ASENS
      float sum;
      {
	float x = n.z;
	float y=1;
	sum=e.s[0];
	for(int i=1; i<ANUM; i++){ y*=x; sum+=e.s[i]*y; }
      }

      flag=e.mas*xi<sum;
#endif
      if(e.type>0){
	float dt=0, dr;
	const DOM dom = oms[om];
	for(int i=0; i<3; i++, dt+=dr*dr) dr=dom.r[i]-e.r[i];
	if(h.t<(sqrt(dt)-OMR)*n.w) flag=false;
      }

      if(flag){
	uint j = atom_add(&ed->hidx, 1);
	if(j<e.hnum) eh[j]=h;
      }

      if(e.zR==1) TOT=0, sca=0; else old=om;
    }
    else{
      if(TOT<XXX) TOT=0;
      else{
	if(xi>e.sf){
	  xi=div(1-xi, 1-e.sf);
	  xi=2*xi-1;
	  if(e.g!=0){
	    float ga=div(1-e.g2, 1+e.g*xi);
	    xi=div(1+e.g2-ga*ga, 2*e.g);
	  }
	}
	else{
	  xi=div(xi,e.sf);
	  xi=2*pow(xi, e.gr)-1;
	}

	if(xi>1) xi=1; else if(xi<-1) xi=-1;

#ifdef ANIZ
	if(e.k>0){
	  float n1=( e.azx*n.x+e.azy*n.y)*e.k1;
	  float n2=(-e.azy*n.x+e.azx*n.y)*e.k2;
	  n.x=n1*e.azx-n2*e.azy;
	  n.y=n1*e.azy+n2*e.azx;
	  n.z*=e.kz;
	  n=my_normalize(n);
	}
#endif

	float si=sqrt(1-xi*xi);
	n=turn(xi, si, n, &s);

#ifdef ANIZ
	if(e.k>0){
	  float n1=div( e.azx*n.x+e.azy*n.y, e.k1);
	  float n2=div(-e.azy*n.x+e.azx*n.y, e.k2);
	  n.x=n1*e.azx-n2*e.azy;
	  n.y=n1*e.azy+n2*e.azx;
	  n.z=div(n.z,e.kz);
	  n=my_normalize(n);
	}
#endif
      }
    }
  }

  {
    ez->rs[s.w]=s.x | (ulong) s.y << 32;
    barrier(CLK_LOCAL_MEM_FENCE);
    write_mem_fence(CLK_GLOBAL_MEM_FENCE);
  }

}
#endif

#ifdef SHRT
));
#endif

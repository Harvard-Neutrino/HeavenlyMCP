#include <map>
#include <deque>
#include <vector>
#include <sstream>
#include <fstream>
#include <iostream>
#include <algorithm>
#include <sys/time.h>

#include <cmath>
#include <cstring>

#ifdef __APPLE_CC__
#include <OpenCL/opencl.h>
#else
#include <CL/opencl.h>
#endif

using namespace std;

namespace xppc{
#define OFLA         // omit the flasher DOM
#define ROMB         // use rhomb cells aligned with the array
#define ASENS        // enable angular sensitivity
#define RAND         // disable for deterministic results

#define TILT         // enable tilted ice layers
#define ANIZ         // enable anisotropic ice

#define MKOW         // photon yield parametrizations by M. Kowalski
#define ANGW         // smear cherenkov cone due to shower development
#define LONG         // simulate longitudinal cascade development
#define CWLR         // new parameterizations by C. Wiebusch and L. Raedel
                     // requires that MKOW, ANGW, and LONG are all defined

#ifdef ASENS
#define ANUM 11      // number of coefficients in the angular sensitivity curve
#endif

#ifdef TILT
#define LMAX 6       // number of dust loggers
#define LYRS 170     // number of depth points
#endif

#ifdef ROMB
#define DIR1 9.3
#define DIR2 129.3

#define CX   21
#define CY   19
#define NSTR 94
#else
#define CX   13
#define CY   13
#define NSTR 94
#endif

#define USMA         // enable use of local shared memory 
#define XAMD         // enable more OpenCL-specific optimizations

#define OVER   10    // size of photon bunches along the muon track
#define HQUO   16    // save at most photons/HQUO hits
#define NPHO   1024  // maximum number of photons propagated by one thread

#define WNUM   32    // number of wavelength slices
#define MAXLYS 172   // maximum number of ice layers
#define MAXGEO 5200  // maximum number of OMs
#define MAXRND 131072   // max. number of random number multipliers

#define XXX 1.e-5f
#define FPI 3.141592653589793f
#define OMR 0.16510f // DOM radius [m]

#include "pro.cxx"
#include "ini.cxx"

#define SHRT
#include "pro.cxx"
#undef SHRT

  void initialize(float enh = 1.f){ m.set(); d.eff*=enh; }

  unsigned int pmax, pmxo, pn;

  bool xgpu=false;

  void checkError(cl_int result){
    if(result!=CL_SUCCESS){
      cerr<<"OpenCL Error: "<<result<<endl;
      exit(2);
    }
  }

  vector< pair<cl_platform_id,cl_device_id> > all;

  struct gpu{
    dats d;

    int device, mult;
    cl_uint nblk;
    size_t nthr, ntot;
    unsigned int npho, pmax, pmxo;

    unsigned int old, num;

    float deviceTime;
    cl_event event;

    cl_platform_id pfID;
    cl_device_id devID;
    cl_context ctx;
    cl_command_queue cq;
    cl_program program;
    cl_kernel clkernel;

    cl_mem ed, ez, eh, ep, eo; // pointers to structures on device

    string& replace(string& in, string old, string str){
      string clx;
      int k=0, m=0;
      while((m=in.find(old, k))!=-1){
	clx.append(in, k, m-k);
	k=m+old.length();
	clx.append(str);
      }
      clx.append(in, k, in.length()-k);
      return in=clx;
    }

    gpu(int device) : deviceTime(0), old(0), npho(NPHO), mult(4){
      this->device=device;

      {
	ostringstream o; o<<"NPHO_"<<device;
	char * nph=getenv(o.str().c_str());
	if(nph==NULL) nph=getenv("NPHO");
	if(nph!=NULL) if(*nph!=0){
	    npho=atoi(nph);
	    cerr<<"Setting NPHO="<<npho<<endl;
	    if(npho<=0){
	      cerr<<"Not using device # "<<device<<"!"<<endl;
	      return;
	    }
	  }
      }

      {
	ostringstream o; o<<"XMLT_"<<device;
	char * mlt=getenv(o.str().c_str());
	if(mlt==NULL) mlt=getenv("XMLT");
	if(mlt!=NULL) if(*mlt!=0){
	    int aux=atoi(mlt);
	    if(aux>0){
	      mult=aux;
	      cerr<<"Setting XMLT="<<mult<<endl;
	    }
	  }
      }

      pfID  = all[device].first; 
      devID = all[device].second;

      {
	bool verbose = false;
	cl_int err;

	const cl_context_properties prop[] = {CL_CONTEXT_PLATFORM, (cl_context_properties) pfID, 0};
	ctx = clCreateContext(prop, 1, &devID, NULL, NULL, &err); checkError(err);
	cq = clCreateCommandQueue(ctx, devID, CL_QUEUE_PROFILING_ENABLE, &err); checkError(err);

	string tmp = kernel_source.substr(1, kernel_source.length()-2);
	string source("#pragma OPENCL EXTENSION cl_khr_global_int32_base_atomics : enable\n");
#ifdef USMA
	source.append("#pragma OPENCL EXTENSION cl_khr_local_int32_base_atomics : enable\n");
#endif
	source.append(replace(verbose?replace(tmp, "; ", ";\n"):tmp, "cl_", ""));

	const char *src = source.c_str();
	size_t length=source.length();

	if(verbose) fprintf(stderr, "KERNEL SOURCE CODE:\n%s\n", src);
	program = clCreateProgramWithSource(ctx, 1, &src, &length, &err); checkError(err);

	clBuildProgram(program, 1, &devID, "-cl-fast-relaxed-math", NULL, NULL);
#ifdef CL_VERSION_1_2
	checkError(clUnloadPlatformCompiler(pfID));
#else
	checkError(clUnloadCompiler());
#endif

	cl_build_status status;
	checkError(clGetProgramBuildInfo(program, devID, CL_PROGRAM_BUILD_STATUS, sizeof(cl_build_status), &status, NULL));

	size_t siz;
	checkError(clGetProgramBuildInfo(program, devID, CL_PROGRAM_BUILD_LOG, 0, NULL, &siz));
	char log[siz+1]; log[siz] = '\0';
	checkError(clGetProgramBuildInfo(program, devID, CL_PROGRAM_BUILD_LOG, siz, log, NULL));
	if(verbose || status!=CL_SUCCESS) fprintf(stderr, "BUILD LOG:\n%s\n", log);
	checkError(status);

	clkernel = clCreateKernel(program, "propagate", &err); checkError(err);
      }

      {
	clGetDeviceInfo(devID, CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(cl_uint), &nblk, NULL);
	clGetKernelWorkGroupInfo(clkernel, devID, CL_KERNEL_WORK_GROUP_SIZE, sizeof(size_t), &nthr, NULL);

	cl_ulong lmem;
	clGetKernelWorkGroupInfo(clkernel, devID, CL_KERNEL_LOCAL_MEM_SIZE, sizeof(cl_ulong), &lmem, NULL);

	cerr<<"Running on "<<nblk<<" MPs x "<<nthr<<" threads; Kernel uses: l="<<lmem<<endl;
      }
    }

    void ini(int type){
      rs_ini();
      d=xppc::d;

      nblk*=mult;
      ntot=nblk*nthr;
      pmax=ntot*npho;
      pmxo=pmax/OVER;
      pmax=pmxo*OVER;
      d.hnum=pmax/HQUO;

      {
	unsigned int size=d.rsize;
	if(size<ntot) cerr<<"Error: not enough multipliers: only have "<<size<<" (need "<<ntot<<")!"<<endl;
	else d.rsize=ntot;
      }

      uint tot=0, cnt=0;
      cl_int err;

      {
	unsigned int size=sizeof(datz); tot+=size;
	ez = clCreateBuffer(ctx, CL_MEM_READ_WRITE|CL_MEM_COPY_HOST_PTR, size, &z, &err); checkError(err);
      }

      {
	unsigned int size=d.hnum*sizeof(hit); tot+=size;
	eh = clCreateBuffer(ctx, CL_MEM_READ_WRITE, size, NULL, &err); checkError(err);
      }

      {
	unsigned int size=sizeof(photon);
	if(d.type==0){
	  size*=pmxo; tot+=size;
	}
	ep = clCreateBuffer(ctx, CL_MEM_READ_WRITE, size, NULL, &err); checkError(err);
      }

      {
	unsigned int size=d.gsize*sizeof(DOM); cnt+=size;
	eo = clCreateBuffer(ctx, CL_MEM_READ_ONLY|CL_MEM_COPY_HOST_PTR, size, q.oms, &err); checkError(err);
      }

      {
	unsigned int size=sizeof(dats); tot+=size;
	ed = clCreateBuffer(ctx, CL_MEM_READ_WRITE|CL_MEM_COPY_HOST_PTR, size, &d, &err); checkError(err);
      }

      cerr<<"Total GPU memory usage: "<<tot<<"  const: "<<cnt<<endl;

      {
	checkError(clSetKernelArg(clkernel, 1, sizeof(cl_mem), &ed));
	checkError(clSetKernelArg(clkernel, 2, sizeof(cl_mem), &ez));
	checkError(clSetKernelArg(clkernel, 3, sizeof(cl_mem), &eh));
	checkError(clSetKernelArg(clkernel, 4, sizeof(cl_mem), &ep));
	checkError(clSetKernelArg(clkernel, 5, sizeof(cl_mem), &eo));
      }
    }

    void fin(){
      fflush(stdout);

      checkError(clReleaseMemObject(ez));
      checkError(clReleaseMemObject(eh));
      checkError(clReleaseMemObject(ep));
      checkError(clReleaseMemObject(eo));
      checkError(clReleaseMemObject(ed));
    }

    void set(){
      if(xgpu) device=device;
    }

    void kernel_i(){
      checkError(clEnqueueReadBuffer(cq, ed, CL_TRUE, 0, 2*sizeof(int), &d, 0, NULL, NULL));

      cl_ulong t1, t2;
      checkError(clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &t1, NULL));
      checkError(clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_END,   sizeof(cl_ulong), &t2, NULL));
      deviceTime+=(long long)(t2-t1)/1.e6;

      if(d.ab>0) cerr<<"Error: TOT was a nan or an inf "<<d.ab<<" times!"<<endl;

      if(d.hidx>=d.hnum){ d.hidx=d.hnum; cerr<<"Error: data buffer overflow occurred!"<<endl; }

      if(d.hidx>0){
	unsigned int size=d.hidx*sizeof(hit);
	checkError(clEnqueueReadBuffer(cq, eh, CL_FALSE, 0, size, &q.hits[xppc::d.hidx], 0, NULL, NULL));
	xppc::d.hidx+=d.hidx;
      }
    }

    void kernel_c(unsigned int & idx){
      if(old>0) checkError(clFinish(cq));
      unsigned int pn=num/OVER;
      if(pn>0){
	unsigned int size=pn*sizeof(photon);
	checkError(clEnqueueWriteBuffer(cq, ep, CL_FALSE, 0, size, &q.pz[idx], 0, NULL, NULL));
	idx+=pn;
      }
    }

    void kernel_f(){
      checkError(clFinish(cq));
      if(num>0){
	unsigned int zero=0;
	checkError(clSetKernelArg(clkernel, 0, sizeof(unsigned int), &zero));
	checkError(clEnqueueTask(cq, clkernel, 0, NULL, NULL));

	checkError(clSetKernelArg(clkernel, 0, sizeof(unsigned int), &num));
	checkError(clEnqueueNDRangeKernel(cq, clkernel, 1, NULL, &ntot, &nthr, 0, NULL, &event));
      }
    }

    void stop(){
      fprintf(stderr, "Device time: %2.1f [ms]\n", deviceTime);
      if(clkernel) checkError(clReleaseKernel(clkernel));
      if(program) checkError(clReleaseProgram(program));
      if(cq) checkError(clReleaseCommandQueue(cq));
      if(ctx) checkError(clReleaseContext(ctx));
    }
  };

  vector<gpu> gpus;

  void ini(int type){
    d.hnum=0;
    pmax=0, pmxo=0, pn=0;

    for(vector<gpu>::iterator i=gpus.begin(); i!=gpus.end(); i++){
      i->set();
      i->ini(type); if(xgpu) sv++;
      d.hnum+=i->d.hnum;
      pmax+=i->pmax, pmxo+=i->pmxo;
    }

    {
      q.hits = new hit[d.hnum];
    }

    {
      if(d.type==0) q.pz = new photon[pmxo];
    }
  }

  void fin(){
    for(vector<gpu>::iterator i=gpus.begin(); i!=gpus.end(); i++) i->set(), i->fin();
    if(d.type==0) delete q.pz;
    delete q.hits;
  }

  void listDevices(){
    cl_uint num;
    checkError(clGetPlatformIDs(0, NULL, &num));
    fprintf(stderr, "Found %d platforms:\n", num);

    cl_platform_id ids[num];
    checkError(clGetPlatformIDs(num, ids, NULL));

    for(int i=0, n=0; i<num; ++i){
      size_t siz;
      cl_platform_id & platform=ids[i];
      fprintf(stderr, "platform %d: ", i);

      {
	checkError(clGetPlatformInfo(platform, CL_PLATFORM_NAME, 0, NULL, &siz));
	char buf[siz+1]; buf[siz]='\0';
	checkError(clGetPlatformInfo(platform, CL_PLATFORM_NAME, siz, &buf, NULL));
	fprintf(stderr, "%s ", buf);
      }

      {
	checkError(clGetPlatformInfo(platform, CL_PLATFORM_VERSION, 0, NULL, &siz));
	char buf[siz+1]; buf[siz]='\0';
	checkError(clGetPlatformInfo(platform, CL_PLATFORM_VERSION, siz, &buf, NULL));
	fprintf(stderr, "%s ", buf);
      }

      cl_uint num;
      checkError(clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, 0, NULL, &num));
      fprintf(stderr, "device count: %d\n", num);

      cl_device_id devices[num];
      checkError(clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, num, devices, NULL));

      for(int j=0; j<num; j++, n++){
	cl_device_id & di = devices[j];
	fprintf(stderr, "  device %d:", j);

	{
	  cl_uint info;
	  clGetDeviceInfo(di, CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(cl_uint), &info, NULL);
	  fprintf(stderr, " cu=%d", info);
	}

	{
	  size_t info;
	  clGetDeviceInfo(di, CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof(size_t), &info, NULL);
	  fprintf(stderr, " gr=%lu", info);
	}

	{
	  cl_uint info;
	  clGetDeviceInfo(di, CL_DEVICE_MAX_CLOCK_FREQUENCY, sizeof(cl_uint), &info, NULL);
	  fprintf(stderr, " %d MHz", info);
	}

	{
	  cl_ulong info;
	  clGetDeviceInfo(di, CL_DEVICE_GLOBAL_MEM_SIZE, sizeof(cl_ulong), &info, NULL);
	  fprintf(stderr, " %llu bytes", (unsigned long long) info);
	}

	{
	  cl_ulong info;
	  clGetDeviceInfo(di, CL_DEVICE_MAX_CONSTANT_BUFFER_SIZE, sizeof(cl_ulong), &info, NULL);
	  fprintf(stderr, " cm=%llu", (unsigned long long) info);
	}

	{
	  cl_ulong info;
	  clGetDeviceInfo(di, CL_DEVICE_LOCAL_MEM_SIZE, sizeof(cl_ulong), &info, NULL);
	  fprintf(stderr, " lm=%llu", (unsigned long long) info);
	}

	{
	  cl_bool info;
	  clGetDeviceInfo(di, CL_DEVICE_ERROR_CORRECTION_SUPPORT, sizeof(cl_bool), &info, NULL);
	  fprintf(stderr, " ecc=%d", info);
	}

	all.push_back(make_pair(platform, di));
	fprintf(stderr, "\n");
      }
    }
    fprintf(stderr, "\n");
  }

  static unsigned int old=0;

  void print();

  void kernel(unsigned int num){
    if(old>0){
      d.hidx=0;
      for(vector<gpu>::iterator i=gpus.begin(); i!=gpus.end(); i++) i->set(), i->kernel_i();
      cerr<<"photons: "<<old<<"  hits: "<<d.hidx<<endl;
    }

    {
      unsigned int over=d.type==0?OVER:1, sum=0;
      for(vector<gpu>::iterator i=gpus.begin(); i!=gpus.end(); i++){
	i->num=over*((num*(unsigned long long) i->pmax)/(over*(unsigned long long) pmax));
	sum+=i->num;
      }
      while(num>sum){
	static int res=0;
	gpu& g=gpus[res++%gpus.size()];
	if(g.num<g.pmax) g.num+=over, sum+=over;
      }
    }

    if(d.type==0){
      unsigned int idx=0;
      for(vector<gpu>::iterator i=gpus.begin(); i!=gpus.end(); i++) i->set(), i->kernel_c(idx);
    }

    for(vector<gpu>::iterator i=gpus.begin(); i!=gpus.end(); i++) i->set(), i->kernel_f();

    if(old>0) print();

    old=num;
    for(vector<gpu>::iterator i=gpus.begin(); i!=gpus.end(); i++) i->old=i->num;
  }

  float square(float x){
    return x*x;
  }

  int flset(int str, int dom){
    int type=1;
    float r[3]={0, 0, 0};

    if(str<0){ type=2; str=-str; }
    if(str==0) switch(dom){
      case 1: type=3; r[0]=544.07; r[1]=55.89; r[2]=136.86; break;
      case 2: type=4; r[0]=11.87; r[1]=179.19; r[2]=-205.64; break;
    }
    else for(int n=0; n<d.gsize; n++) if(q.names[n].str==str && q.names[n].dom==dom){
      d.fla=n;
      for(int m=0; m<3; m++) r[m]=q.oms[n].r[m]; break;
    }

    for(int m=0; m<3; m++) d.r[m]=r[m];

    switch(type){
    case 1: d.ka=square(fcv*9.7f); d.up=fcv*0.0f; break;
    case 2: d.ka=square(fcv*9.7f); d.up=fcv*48.f; break;
    case 3: d.ka=0.0f;  d.up=fcv*(90.0f-41.13f);  break;
    case 4: d.ka=0.0f;  d.up=fcv*(41.13f-90.0f);  break;
    }

    return type;
  }

  void flini(int str, int dom){
    d.type=flset(str, dom);
    ini(d.type);
  }

#ifdef XLIB
  const DOM& flget(int str, int dom){
    static DOM om;
    flset(str, dom); ini(0);
    for(int m=0; m<3; m++) om.r[m]=d.r[m];
    return om;
  }

  void flshift(float r[], float n[]){
    if(d.ka>0){
      float FLZ, FLR;
      sincosf(fcv*30.f, &FLZ, &FLR);
      FLZ*=OMR, FLR*=OMR;
      r[0]+=FLR*n[0];
      r[1]+=FLR*n[1];
      r[2]+=FLZ;
      r[3]+=OMR*d.ocv;
    }

    float xi;
    sincosf(d.up, &n[2], &xi);
    n[0]*=xi; n[1]*=xi;
  }
#endif

  void flone(unsigned long long num){
    for(long long i=llroundf(num*(long double)d.eff); i>0; i-=pmax) kernel(min(i, (long long) pmax));
    kernel(0);
  }

  void flasher(int str, int dom, unsigned long long num, int itr){
    flini(str, dom);

    for(int j=0; j<max(1, itr); j++){
      flone(num);
      if(itr>0) printf("\n");
    }

    fin();
  }

  void start(){
  }

  void stop(){
    fprintf(stderr, "\n");
    for(vector<gpu>::iterator i=gpus.begin(); i!=gpus.end(); i++) i->set(), i->stop();
  }

  void choose(int device){
    listDevices();
    int deviceCount=all.size();

    if(device<0){
      if(deviceCount==0){ cerr<<"Could not find compatible devices!"<<endl; exit(2); }
      for(int device=0; device<deviceCount; ++device){
	gpus.push_back(gpu(device));
	if(gpus.back().npho<=0) gpus.pop_back();
      }
    }
    else{
      if(device>=deviceCount){ cerr<<"Device #"<<device<<" is not available!"<<endl; exit(2); }
      sv+=device;
      gpus.push_back(gpu(device));
      if(gpus.back().npho<=0) gpus.pop_back();
    }
    if(gpus.size()<=0){
      cerr<<"No active GPU(s) selected!"<<endl;
      exit(5);
    }
    xgpu=gpus.size()>1;
  }

#include "f2k.cxx"
}

#ifndef XLIB
using namespace xppc;

int main(int arg_c, char *arg_a[]){
  start();
  if(arg_c<=1){
    listDevices();
    fprintf(stderr, "Use: %s [device] (f2k muons)\n"
	    "     %s [str] [om] [num] [device] (flasher)\n", arg_a[0], arg_a[0]);
  }
  else if(arg_c<=2){
    int device=0;
    if(arg_c>1) device=atoi(arg_a[1]);
    initialize();
    choose(device);
    fprintf(stderr, "Processing f2k muons from stdin on device %d\n", device);
    f2k();
  }
  else{
    int str=0, dom=0, device=0, itr=0;
    unsigned long long num=1000000ULL;

    if(arg_c>1) str=atoi(arg_a[1]);
    if(arg_c>2) dom=atoi(arg_a[2]);
    if(arg_c>3){
      num=(unsigned long long) atof(arg_a[3]);
      char * sub = strchr(arg_a[3], '*');
      if(sub!=NULL) itr=(int) atof(++sub);
    }
    if(arg_c>4) device=atoi(arg_a[4]);
    initialize();
    choose(device);
    fprintf(stderr, "Running flasher simulation on device %d\n", device);
    flasher(str, dom, num, itr);
  }

  stop();
}
#endif

//nvcc -Xcompiler -fPIC -o libcuda_ipfb.so cuda_ipfb.cu -shared -lcufft -lgomp -lcublas


#include <stdio.h>
#include <cuda.h>
#include <cufft.h>
#include <math.h>
#include <omp.h>
#include <cublas_v2.h>


/*--------------------------------------------------------------------------------*/
float get_fval(float *dptr)
{
  float val;
  if (cudaMemcpy(&val,dptr,sizeof(float),cudaMemcpyDeviceToHost)!=cudaSuccess)
    fprintf(stderr,"Error reading from device.\n");
  return val;
}

/*--------------------------------------------------------------------------------*/

void cufft_c2r(float *out, cufftComplex *data, int len, int ntrans, int isodd)
{
  int nout=2*(len-1)-isodd;
  //float *out;
  //cudaMalloc(&out,sizeof(float)*nout*ntrans);
  cufftHandle plan;
  
  if (cufftPlan1d(&plan,nout,CUFFT_C2R, ntrans)!=CUFFT_SUCCESS)
    fprintf(stderr,"Error planning dft\n");
  if (cufftExecC2R(plan,data,out)!=CUFFT_SUCCESS)
    fprintf(stderr,"Error executing dft\n");
  if (cufftDestroy(plan)!= CUFFT_SUCCESS)
    fprintf(stderr,"Error destroying plan.\n");
}



/*--------------------------------------------------------------------------------*/
void cufft_r2c(cufftComplex *out, float *data, int len, int ntrans)
{
  //int nout=len/2+1;
  cufftHandle plan;
  
  if (cufftPlan1d(&plan,len,CUFFT_R2C, ntrans)!=CUFFT_SUCCESS)
    fprintf(stderr,"Error planning dft\n");
  if (cufftExecR2C(plan,data,out)!=CUFFT_SUCCESS)
    fprintf(stderr,"Error executing dft\n");
  if (cufftDestroy(plan)!= CUFFT_SUCCESS)
    fprintf(stderr,"Error destroying plan.\n");
}



/*--------------------------------------------------------------------------------*/
void cufft_r2c_columns(cufftComplex *out, float *data, int len, int ntrans)
{
  //int nout=len/2+1;
  //printf("performing %d transforms of length %d %d\n",ntrans,len,nout);

  cufftHandle plan;
  int rank=1;
  int inembed[rank] = {len};
  int onembed[rank]={ntrans};
  int istride=ntrans;
  int idist=1;
  int ostride=ntrans;
  int odist=1;
  //if (cufftPlanMany(&plan,1,&nout,&one,len,1,&one,nout,1,CUFFT_R2C,ntrans)!=CUFFT_SUCCESS)
  //if (cufftPlanMany(&plan,rank,&len,inembed,len,1,onembed,nout,1,CUFFT_R2C,ntrans)!=CUFFT_SUCCESS)
  if (cufftPlanMany(&plan,rank,&len,inembed,istride,idist,onembed,ostride,odist,CUFFT_R2C,ntrans)!=CUFFT_SUCCESS)
    fprintf(stderr,"Error planning DFT in r2c_columns.\n");
  if (cufftExecR2C(plan,data,out)!=CUFFT_SUCCESS)
    fprintf(stderr,"Error executing DFT in r2c_columns.\n");
  if (cufftDestroy(plan)!=CUFFT_SUCCESS)
    fprintf(stderr,"Error destroying plan in r2c_columns.\n");
  
}

/*--------------------------------------------------------------------------------*/
void cufft_c2r_columns(float *out, cufftComplex *data,int len, int ntrans, int isodd)
{
  int nout=2*(len-1)+isodd;
  cufftHandle plan;
  int rank=1;
  int inembed[rank] = {ntrans};
  int onembed[rank]={ntrans};
  int istride=ntrans;
  int idist=1;
  int ostride=ntrans;
  int odist=1;
  if (cufftPlanMany(&plan,rank,&nout,inembed,istride,idist,onembed,ostride,odist,CUFFT_C2R,ntrans)!=CUFFT_SUCCESS)
    fprintf(stderr,"Error planning DFT in c2r_columns.\n");
  if (cufftExecC2R(plan,data,out)!=CUFFT_SUCCESS)
    fprintf(stderr,"Error executing DFT in c2r_columns.\n");
  if (cufftDestroy(plan)!=CUFFT_SUCCESS)
    fprintf(stderr,"Error destroying plan in c2r_columns.\n");

}

/*--------------------------------------------------------------------------------*/
__global__
void cmul_arrays(cufftComplex *arr, cufftComplex *to_mul, int n)
{
  int myi=blockIdx.x*blockDim.x+threadIdx.x;
  int nthread=gridDim.x*blockDim.x;
  for (int i=myi;i<n;i+=nthread) {
    //arr[i]=arr[i]*to_mul[i];
    arr[i]=cuCmulf(arr[i],to_mul[i]);
    //cufftComplex in1=arr[i];
    //cufftComplex in2=to_mul[i];
    //cufftComplex tmp;
    //tmp.x=in1.x*in2.x-in1.y*in2.y;
    //tmp
  }
    
}

/*--------------------------------------------------------------------------------*/
__global__
void inv_filt_array(cufftComplex *arr, float filt, int n)
{
  int myi=blockIdx.x*blockDim.x+threadIdx.x;
  int nthread=gridDim.x*blockDim.x;
  float filtsqr=filt*filt;
  for (int i=myi;i<n;i+=nthread) {
    cufftComplex val=arr[i];
    float myabs=val.x*val.x+val.y*val.y;
    float fac=1/(myabs+filtsqr);
    val.x=val.x*fac;
    //val.y=-val.y*fac;
    val.y=val.y*fac;  //I think this should be the conjugate of the conjugate
    arr[i]=val;
  }
}
/*--------------------------------------------------------------------------------*/
__global__
void inv_filt_scale_array(cufftComplex *arr, float filt, float scale, int n)
{
  int myi=blockIdx.x*blockDim.x+threadIdx.x;
  int nthread=gridDim.x*blockDim.x;
  float filtsqr=filt*filt;
  for (int i=myi;i<n;i+=nthread) {
    cufftComplex val=arr[i];
    float myabs=val.x*val.x+val.y*val.y;
    float fac=1/(myabs+filtsqr)*scale;
    val.x=val.x*fac;
    //val.y=-val.y*fac;
    val.y=val.y*fac;  //I think this should be the conjugate of the conjugate
    arr[i]=val;
  }
}

/*--------------------------------------------------------------------------------*/

extern "C" {
void cufft_r2c_host(cufftComplex *out, float *data, int n, int m, int axis)
{
  cufftComplex *dout;
  float *din;
  int nn;
  if (axis==0)
    nn=n/2+1;
  else
    nn=m/2+1;
  if (cudaMalloc((void **)&din,sizeof(float)*n*m)!=cudaSuccess)
    fprintf(stderr,"error in cudaMalloc\n");
  if (cudaMemcpy(din,data,n*m*sizeof(float),cudaMemcpyHostToDevice)!=cudaSuccess)
    fprintf(stderr,"Error copying data to device.\n");
  if (axis==0) {
    if (cudaMalloc((void **)&dout,sizeof(cufftComplex)*nn*m)!=cudaSuccess)
      fprintf(stderr,"error in cudaMalloc\n");
    cufft_r2c_columns(dout,din,n,m);
    //printf("copying %d %d\n",nn,m);
    if (cudaMemcpy(out,dout,sizeof(cufftComplex)*nn*m,cudaMemcpyDeviceToHost)!=cudaSuccess)
      fprintf(stderr,"Error copying result to host in r2c\n");
  }
  else {
    if (cudaMalloc((void **)&dout,sizeof(cufftComplex)*n*nn)!=cudaSuccess)
      fprintf(stderr,"error in cudaMalloc\n");
    cufft_r2c(dout,din,m,n);
    //printf("copying %d %d\n",n,nn);
    if (cudaMemcpy(out,dout,sizeof(cufftComplex)*nn*n,cudaMemcpyDeviceToHost)!=cudaSuccess)
      fprintf(stderr,"Error copying result to host in r2c\n");
  

  }
}
}


/*--------------------------------------------------------------------------------*/

extern "C" {
void cufft_c2r_host(float *out, cufftComplex *data, int n, int m, int isodd,int axis)
{
  float *dout;
  cufftComplex *din;
  int nn;
  if (axis==0)
    nn=2*(n-1)+isodd;
  else
    nn=2*(m-1)+isodd;
  if (cudaMalloc((void **)&din,sizeof(cufftComplex)*n*m)!=cudaSuccess)
    fprintf(stderr,"error in cudaMalloc\n");
  if (cudaMemcpy(din,data,n*m*sizeof(cufftComplex),cudaMemcpyHostToDevice)!=cudaSuccess)
    fprintf(stderr,"Error copying data to device.\n");
  if (axis==0) {
    if (cudaMalloc((void **)&dout,sizeof(float)*nn*m)!=cudaSuccess)
      fprintf(stderr,"error in cudaMalloc\n");
    cufft_c2r_columns(dout,din,n,m,isodd);
    //printf("copying %d %d\n",nn,m);
    if (cudaMemcpy(out,dout,sizeof(float)*nn*m,cudaMemcpyDeviceToHost)!=cudaSuccess)
      fprintf(stderr,"Error copying result to host in c2r\n");
  }
  else {
    if (cudaMalloc((void **)&dout,sizeof(float)*n*nn)!=cudaSuccess)
      fprintf(stderr,"error in cudaMalloc\n");
    cufft_c2r(dout,din,m,n,isodd);
    //printf("copying %d %d\n",n,nn);
    if (cudaMemcpy(out,dout,sizeof(float)*nn*n,cudaMemcpyDeviceToHost)!=cudaSuccess)
      fprintf(stderr,"Error copying result to host in c2r\n");
  

  }
}
}


/*--------------------------------------------------------------------------------*/

cufftComplex *get_winft(float *win, int nchan, int ntap, int nslice,float thresh)
//get the Fourier tranform of the window function along columns, left in device memory
{
  float *dwin;
  if (cudaMalloc((void **)&dwin,nslice*nchan*2*sizeof(float))!=cudaSuccess)
    fprintf(stderr,"Error in malloc in get_winft\n");
  cufftComplex *dwinft;
  int nn=nslice/2+1;
  int m=nchan*2;
  printf("mallocing %d %d\n",nn,m);
  if (cudaMalloc((void **)&dwinft,nn*m*sizeof(cufftComplex))!=cudaSuccess)
    fprintf(stderr,"Error in malloc in get_winft\n");

  if (cudaMemset(dwin,0,nslice*m*sizeof(float))!=cudaSuccess)
    fprintf(stderr,"Error in memset.\n");

  if (cudaMemcpy(dwin,win,m*ntap*sizeof(float),cudaMemcpyHostToDevice)!=cudaSuccess)
    fprintf(stderr,"Error copying window.\n");

  cufft_r2c_columns(dwinft,dwin,nslice,m);
  if (cudaFree(dwin)!=cudaSuccess)
    fprintf(stderr,"Error in free in get_winft\n");
  inv_filt_array<<<256,128>>>(dwinft,thresh,nn*m);

  return dwinft;
  
}

/*--------------------------------------------------------------------------------*/
void ftranspose(float *out, float *mat, int n, int m)
{
  float one=1.0;
  float zero=0.0;
  cublasHandle_t handle;
  cublasStatus_t stat = cublasCreate(&handle);
  if (cublasCreate(&handle)!=CUBLAS_STATUS_SUCCESS) {
    fprintf(stderr,"Error creating handle.\n");
    return;
  }
  if (cublasSgeam(handle,CUBLAS_OP_T,CUBLAS_OP_N,n,m,&one,mat,m,&zero,out,n,out,n)!=CUBLAS_STATUS_SUCCESS)
    fprintf(stderr,"Error in transpose.\n");
  cublasDestroy(handle);
}
/*--------------------------------------------------------------------------------*/
extern "C" {
void test_transpose(float *in, int n, int m)
{
  float *din;
  if (cudaMalloc((void **)&din,n*m*sizeof(float))!=cudaSuccess)
    fprintf(stderr,"Error allocing on GPU.\n");
  if (cudaMemcpy(din,in,n*m*sizeof(float),cudaMemcpyHostToDevice)!=cudaSuccess)
    fprintf(stderr,"Error copying to device.\n");

  float *dout;
  if (cudaMalloc((void **)&dout,n*m*sizeof(float))!=cudaSuccess)
    fprintf(stderr,"Error allocing on GPU.\n");
  
  printf("incoming values are %g %g\n",get_fval(din),get_fval(dout));
  ftranspose(dout,din,n,m);
  if (cudaMemcpy(in,dout,n*m*sizeof(float),cudaMemcpyDeviceToHost)!=cudaSuccess)
    fprintf(stderr,"Error copying back to host.\n");
  if (cudaFree(din)!=cudaSuccess)
    fprintf(stderr,"Error freeing in test_tranpose.\n");
  if (cudaFree(dout)!=cudaSuccess)
    fprintf(stderr,"Error freeing in test_tranpose.\n");
}
}
/*--------------------------------------------------------------------------------*/

#if 1
cufftComplex *get_winft_transpose(float *win, int nchan, int ntap, int nslice,float thresh)
//get the Fourier tranform of the window function along columns, left in device memory
{  
  float *dwin;
  if (cudaMalloc((void **)&dwin,nslice*nchan*2*sizeof(float))!=cudaSuccess)
    fprintf(stderr,"Error in malloc in get_winft\n");
  cufftComplex *dwinft;
  int nn=nslice/2+1;
  int m=nchan*2;

  //printf("m/nn are %d %d\n",m,nslice);
  
  if (cudaMemset(dwin,0,nslice*m*sizeof(float))!=cudaSuccess)
    fprintf(stderr,"Error in memset.\n");

  if (cudaMemcpy(dwin,win,m*ntap*sizeof(float),cudaMemcpyHostToDevice)!=cudaSuccess)
    fprintf(stderr,"Error copying window.\n");

  float *dwin2;
  if (cudaMalloc((void **)&dwin2,nslice*nchan*2*sizeof(float))!=cudaSuccess)
    fprintf(stderr,"Error in malloc in get_winft\n");

  ftranspose(dwin2,dwin,nslice,nchan*2);
  if (cudaFree(dwin)!=cudaSuccess)
    fprintf(stderr,"Error freeing dwin\n");


  if (cudaMalloc((void **)&dwinft,nn*m*sizeof(cufftComplex))!=cudaSuccess)
    fprintf(stderr,"Error in malloc in get_winft\n");

		  
  //cufft_r2c_columns(dwinft,dwin,nslice,m);
  cufft_r2c(dwinft,dwin2,nslice,m);
  if (cudaFree(dwin2)!=cudaSuccess)
    fprintf(stderr,"Error in free in get_winft_transpose\n");
  //printf("nn in filter is %d\n",nn);
  float scale=1.0/m/nslice;
  //inv_filt_array<<<256,128>>>(dwinft,thresh,nn*m);
  inv_filt_scale_array<<<256,128>>>(dwinft,thresh,scale,nn*m);

  return dwinft;
  
}
#endif
/*--------------------------------------------------------------------------------*/
extern "C" {
void get_winft_host(cufftComplex *out,float *win, int nchan, int ntap, int nslice, float thresh)
{
  cufftComplex *dout=get_winft_transpose(win,nchan,ntap,nslice,thresh);
  int nn=nslice/2+1;
  int m=nchan*2;
  printf("copying %d %d\n",nn,m);
  if (cudaMemcpy(out,dout,nn*m*sizeof(cufftComplex),cudaMemcpyDeviceToHost)!=cudaSuccess)
    fprintf(stderr,"error copying winft back to host\n");
  if (cudaFree(dout)!=cudaSuccess)
    fprintf(stderr,"error freeign in get_winft_host\n");
}
}

/*--------------------------------------------------------------------------------*/
cufftComplex get_cval(cufftComplex *dptr)
{
  cufftComplex val;
  if (cudaMemcpy(&val,dptr,sizeof(cufftComplex),cudaMemcpyDeviceToHost)!=cudaSuccess)
    fprintf(stderr,"Error reading from device.\n");
  return val;
}



/*--------------------------------------------------------------------------------*/
void ipfb_core(cufftComplex *ddat, float *dat_unft, cufftComplex *dwinft, int nchan, int nslice)
//do core ipfb.  note that data on device will get overwritten
{
  int n=2*(nchan-1);
  float *tmp;
  if (cudaMalloc((void **)&tmp,n*nslice*sizeof(float))!=cudaSuccess)
    fprintf(stderr,"error in malloc in ipfb_core\n");
  cufft_c2r(dat_unft,ddat,nchan,nslice,0);
  //cufft_r2c_columns(ddat,dat_unft,nslice,n);
  ftranspose(tmp,dat_unft,nslice,n);
  cufft_r2c(ddat,tmp,nslice,n);
  cmul_arrays<<<256,128>>>(ddat,dwinft,(nslice/2+1)*n);  
  //cufft_c2r_columns(dat_unft,ddat,nslice/2+1,n,0);
  cufft_c2r(tmp,ddat,nslice/2+1,n,0);
  ftranspose(dat_unft,tmp,n,nslice);
  if (cudaFree(tmp)!=cudaSuccess)
    fprintf(stderr,"Error in free in ipfb_core\n");

}
/*--------------------------------------------------------------------------------*/
extern "C" {

void ipfb(float *out, cufftComplex *dat, float *win, int nchan, int nslice, int ntap, float filt_thresh)
{

  //cufftComplex *dwinft=get_winft(win,nchan-1,ntap,nslice,filt_thresh);
  cufftComplex *dwinft=get_winft_transpose(win,nchan-1,ntap,nslice,filt_thresh);

  int n=2*(nchan-1);
  
  size_t sz1=nchan*nslice;
  size_t sz2=n*(nslice/2+1);
  if (sz2>sz1)
    sz1=sz2;

  cufftComplex *ddat;
  if (cudaMalloc((void **)&ddat,sz1*sizeof(cufftComplex))!=cudaSuccess)
    fprintf(stderr,"Error in malloc\n");
  if (cudaMemcpy(ddat,dat,nchan*nslice*sizeof(cufftComplex),cudaMemcpyHostToDevice)!=cudaSuccess)
    fprintf(stderr,"Error copying data to device.\n");
  float *dat_unft;
  if (cudaMalloc((void **)&dat_unft,n*nslice*sizeof(float))!=cudaSuccess)
    fprintf(stderr,"Error in malloc\n");

  cudaDeviceSynchronize();
  double t1=omp_get_wtime();
  ipfb_core(ddat,dat_unft,dwinft,nchan,nslice);
  cudaDeviceSynchronize();
  double t2=omp_get_wtime();
  printf("core ipfb took %12.4e seconds.\n",t2-t1);
  
  if (cudaMemcpy(out,dat_unft,(nchan-1)*nslice*2*sizeof(float),cudaMemcpyDeviceToHost)!=cudaSuccess)
    fprintf(stderr,"Error copying data back to host\n");
  //printf("first output value is %12.4f\n",out[0]);
  cudaFree(dwinft);
  cudaFree(ddat);
  cudaFree(dat_unft);
  

}
}
/*--------------------------------------------------------------------------------*/
extern "C" {

void ipfb_old(float *out, cufftComplex *dat, float *win, int nchan, int nslice, int ntap, float filt_thresh)
{

  //printf("filter is %12.4f\n",filt_thresh);
  //get the window fourier transform
  cufftComplex *dwinft=get_winft(win,nchan-1,ntap,nslice,filt_thresh);

  //cufftComplex val=get_cval(dwinft);
  //printf("test val is %12.4f %12.4f\n",val.x,val.y);


  int n=2*(nchan-1);
  
  size_t sz1=nchan*nslice;
  size_t sz2=n*(nslice/2+1);
  if (sz2>sz1)
    sz1=sz2;
  //printf("sizes are %ld %ld\n",sz1,sz2);

  cufftComplex *ddat;
  //if (cudaMalloc((void **)&ddat,nchan*nslice*sizeof(cufftComplex))!=cudaSuccess)
  if (cudaMalloc((void **)&ddat,sz1*sizeof(cufftComplex))!=cudaSuccess)
    fprintf(stderr,"Error in malloc\n");
  if (cudaMemcpy(ddat,dat,nchan*nslice*sizeof(cufftComplex),cudaMemcpyHostToDevice)!=cudaSuccess)
    fprintf(stderr,"Error copying data to device.\n");
  


  //get the ift of the data first

  float *dat_unft;
  if (cudaMalloc((void **)&dat_unft,n*nslice*sizeof(float))!=cudaSuccess)
    fprintf(stderr,"Error in malloc\n");
  cufft_c2r(dat_unft,ddat,nchan,nslice,0);
  //cudaFree(ddat);

  //printf("first unfted value is %12.4g\n",get_fval(dat_unft));
  
  cufftComplex *datft=ddat;
  //if (cudaMalloc((void **)&datft,(nslice/2+1)*n*sizeof(cufftComplex))!=cudaSuccess)
  //fprintf(stderr,"Error malloc in ipfb\n");

  //do the convolution
  cufft_r2c_columns(datft,dat_unft,nslice,n);
  cmul_arrays<<<256,128>>>(datft,dwinft,(nslice/2+1)*n);  
  cufft_c2r_columns(dat_unft,datft,nslice/2+1,n,0);

  if (cudaMemcpy(out,dat_unft,(nchan-1)*nslice*2*sizeof(float),cudaMemcpyDeviceToHost)!=cudaSuccess)
    fprintf(stderr,"Error copying data back to host\n");
  //printf("first output value is %12.4f\n",out[0]);
  cudaFree(dwinft);
  cudaFree(datft);
  cudaFree(dat_unft);
  

}
}
/*--------------------------------------------------------------------------------*/
extern "C" {
void cufft_c2r_host_old(float *out, float *in, int len, int ntrans, int isodd)
{
  printf("first/last inputs are %12.4f %12.4f %12.4f %12.4f\n",in[0],in[1],in[2*len*ntrans-2],in[2*len*ntrans-1]);
  cufftComplex *din;
  if (cudaMalloc((void **)&din,sizeof(cufftComplex)*len*ntrans)!=cudaSuccess)
    fprintf(stderr,"error in cudaMalloc, din\n");
  int olen=2*(len-1)-isodd;
  float *dout;
  if (cudaMalloc((void **)&dout,sizeof(float)*olen*ntrans)!=cudaSuccess)
    fprintf(stderr,"error in cudaMalloc, dout\n");

  if (cudaMemcpy(din,in,sizeof(cufftComplex)*len*ntrans,cudaMemcpyHostToDevice)!=cudaSuccess)
    fprintf(stderr,"error copying data to device\n");
  
  cufft_c2r(dout,din,len,ntrans,isodd);
  if (cudaMemcpy(out,dout,sizeof(float)*olen*ntrans,cudaMemcpyDeviceToHost)!=cudaSuccess)
    fprintf(stderr,"error coping data to host.\n");
  printf("first and last outputs are %12.4g %12.4g\n",out[0],out[ntrans*2*(len-1)-1]);
}
}
/*--------------------------------------------------------------------------------*/
extern "C" {
void cufft_r2c_host_old2(float *out, float *in, int len, int ntrans, int isodd)
{
  printf("first/last inputs are %12.4f %12.4f %12.4f %12.4f\n",in[0],in[1],in[2*len*ntrans-2],in[2*len*ntrans-1]);
  cufftComplex *din;
  if (cudaMalloc((void **)&din,sizeof(cufftComplex)*len*ntrans)!=cudaSuccess)
    fprintf(stderr,"error in cudaMalloc, din\n");
  int olen=2*(len-1)-isodd;
  float *dout;
  if (cudaMalloc((void **)&dout,sizeof(float)*olen*ntrans)!=cudaSuccess)
    fprintf(stderr,"error in cudaMalloc, dout\n");

  if (cudaMemcpy(din,in,sizeof(cufftComplex)*len*ntrans,cudaMemcpyHostToDevice)!=cudaSuccess)
    fprintf(stderr,"error copying data to device\n");
  
  cufft_c2r(dout,din,len,ntrans,isodd);
  if (cudaMemcpy(out,dout,sizeof(float)*olen*ntrans,cudaMemcpyDeviceToHost)!=cudaSuccess)
    fprintf(stderr,"error coping data to host.\n");
  printf("first and last outputs are %12.4g %12.4g\n",out[0],out[ntrans*2*(len-1)-1]);
}
}
/*================================================================================*/

#if 0

int main(int argc, char *argv[])
{
  printf("hello!\n");



  int len=10;
  int ntrans=10;
  float *dft=(float *)malloc(sizeof(float)*2*len*ntrans);
  float *dat=(float *)malloc(sizeof(float)*ntrans*2*(len-1));
  memset(dft,0,2*len*ntrans);
  dft[2]=1;
  cufft_c2r_host(dat,dft,len,ntrans,0);
  int olen=2*(len-1);
  for (int i=0;i<ntrans;i++) {
    for (int j=0;j<olen;j++)
      printf(" %6.2f",dat[i*olen+j]);
    printf("\n");
    }
}
#endif

//  Revised to use the Boole's Rule for Numerical Integration
//                                 -----20 Mar 2014 (S.Yu)
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <stdlib.h>
#include <time.h>
#include<cuda_runtime.h>
#include<cuda_runtime_api.h>

#define N 20
int iter;
int ix,iy,iz;
float dvb0,jo0,ff0,Lf0,Ld0;
float bo0=1.0,alpha;
float bxo[10],byo[10],bzo[10];
float *b3x, *b3y, *b3z;
float zratio=8.0;
// float *j3x, *j3y, *j3z;

float bxo_lam0[N],byo_lam0[N],bzo_lam0[N],bxo_spl_y2_0[N],byo_spl_y2_0[N],bzo_spl_y2_0[N];
float lamx[N],lamy[N],lamz[N];
float jo[3];
float ro[3];
float p[3];
float per[3];
int icovg;
int *gival;
float rsize=(6.0);
float *gbxyz;
float *gpux, *gpuy, *gpuptn_lam;
float *resultbxyz;
float *sumxyz;
float dx, dy, dz, dxdy,zo_dxdy;


#define THREAD_NUM 256
#define BLOCK_NUM 64

//#define ndx 192  //-----------For square mesh nd*nd with Rsize the same for x- and y-axis. Otherwise should be Nx*Ny. by YYH 1-Aug 2013-------
//#define ndy 192
//#define ndz 64
//#define datasize_xy 36864        //192x192

#define ndx 600  //-----------For square mesh nd*nd with Rsize the same for x- and y-axis. Otherwise should be Nx*Ny. by YYH 1-Aug 2013-------
#define ndy 600
#define ndz 100
#define datasize_xy 360000	//256x256
#define pi4 12.566370614359172
#define pi2 6.283185307179586



//--------------perxyz------------------------------------------------------------------------------
 __global__ static void kernel2(float *bxyz, float *x, float *y, float *gpuptn_lam, float *resultbxyz, int *gival)
{
	__shared__ float shared[3][THREAD_NUM];

	const int tid=threadIdx.x;
	const int bid=blockIdx.x;

	int offset,gival_i;
	float xx,yy,zz2,r,rr;

	zz2=gpuptn_lam[2];
	shared[0][tid]=0;
	shared[1][tid]=0;
	shared[2][tid]=0;

	for(int i=bid*THREAD_NUM+tid;i<datasize_xy;i=i+THREAD_NUM*BLOCK_NUM)
	{
		gival_i=gival[i];
		xx=x[gival_i%ndx]-gpuptn_lam[0];
		yy=y[gival_i/ndx]-gpuptn_lam[1];
		rr=xx*xx+yy*yy+zz2;
		r=sqrtf(rr);
		shared[0][tid]=shared[0][tid]+1.0/r*abs(bxyz[gival_i]);
		shared[1][tid]=shared[1][tid]+1.0/r*abs(bxyz[gival_i+datasize_xy]);
		shared[2][tid]=shared[2][tid]+1.0/r*abs(bxyz[gival_i+2*datasize_xy]);
	}

	__syncthreads();
	offset = THREAD_NUM / 2;
	while(offset > 0) {
		if(tid < offset)
		{
			shared[0][tid] += shared[0][tid + offset];
			shared[1][tid] += shared[1][tid + offset];
			shared[2][tid] += shared[2][tid + offset];
		}
	offset >>= 1;
	__syncthreads();
	}

	if(tid==0)
	{
		resultbxyz[bid]=shared[0][0];
		resultbxyz[bid+BLOCK_NUM]=shared[1][0];
		resultbxyz[bid+2*BLOCK_NUM]=shared[2][0];
	}
}


 __global__ static void kernel3(float *bxyz, float *x, float *y, float *gpuptn_lam, float *resultbxyz, int *gival)
 {
 	__shared__ float shared[3][THREAD_NUM];

 	const int tid=threadIdx.x;
 	const int bid=blockIdx.x;
 	int offset,gival_i;

	shared[0][tid]=0;
	shared[1][tid]=0;
	shared[2][tid]=0;

 		for(int i=bid*THREAD_NUM+tid;i<datasize_xy;i=i+THREAD_NUM*BLOCK_NUM)
		{
			gival_i=gival[i];
			shared[0][tid]=shared[0][tid]+abs(bxyz[gival_i]);
			shared[1][tid]=shared[1][tid]+abs(bxyz[gival_i+datasize_xy]);
			shared[2][tid]=shared[2][tid]+abs(bxyz[gival_i+2*datasize_xy]);
		}
 	__syncthreads();

 	offset = THREAD_NUM / 2;
 	while(offset > 0) {
 		if(tid < offset)
 		{
 			shared[0][tid] += shared[0][tid + offset];
 			shared[1][tid] += shared[1][tid + offset];
 			shared[2][tid] += shared[2][tid + offset];
 		}
 	offset >>= 1;
 	__syncthreads();
 	}

 	if(tid==0)
 	{
		resultbxyz[bid]=shared[0][0];
		resultbxyz[bid+BLOCK_NUM]=shared[1][0];
		resultbxyz[bid+2*BLOCK_NUM]=shared[2][0];
 	}
 }


void nlffbiept2()
{
	float zz;
	float perx1,pery1,perz1;
	float perx2,pery2,perz2;

	zz=ro[2]*ro[2];
	float ptn_lamda[3]={ro[0],ro[1],zz};

	perx1=0.0;
	pery1=0.0;
	perz1=0.0;
	perx2=0.0;
	pery2=0.0;
	perz2=0.0;

	cudaMemcpy(gpuptn_lam, ptn_lamda, sizeof(float)*3,cudaMemcpyHostToDevice);
	kernel2<<<BLOCK_NUM, THREAD_NUM, 3*THREAD_NUM*sizeof(float)>>>(gbxyz, gpux, gpuy, gpuptn_lam, resultbxyz, gival);
	cudaMemcpy(sumxyz, resultbxyz, sizeof(float)*BLOCK_NUM*3, cudaMemcpyDeviceToHost);


	for(int i=0;i<BLOCK_NUM;i++)
	{
		perx1+=sumxyz[i]/100.0;
		pery1+=sumxyz[i+BLOCK_NUM]/100.0;
		perz1+=sumxyz[i+2*BLOCK_NUM]/100.0;
	}
	

	kernel3<<<BLOCK_NUM, THREAD_NUM, 3*THREAD_NUM*sizeof(float)>>>(gbxyz, gpux, gpuy, gpuptn_lam, resultbxyz, gival);
	cudaMemcpy(sumxyz, resultbxyz, sizeof(float)*BLOCK_NUM*3, cudaMemcpyDeviceToHost);

	for(int i=0;i<BLOCK_NUM;i++)
	{
		perx2+=sumxyz[i]/100.0;
		pery2+=sumxyz[i+BLOCK_NUM]/100.0;
		perz2+=sumxyz[i+2*BLOCK_NUM]/100.0;
	}
	per[0] = pi2*perx1/perx2;
	per[1] = pi2*pery1/pery2;
	per[2] = pi2*perz1/perz2;
}



 __global__ static void kernel(float *bxyz, float *x, float *y, float *gpuptn_lam, float *resultbxyz, int *gival)
{
	__shared__ float shared[3][THREAD_NUM];

	const int tid=threadIdx.x;
	const int bid=blockIdx.x;

	int offset,gival_i;
	float xx,yy,zz2,r,rr,wt_rrr,lrx,lry,lrz;

		zz2=gpuptn_lam[2];
		shared[0][tid]=0;
		shared[1][tid]=0;
		shared[2][tid]=0;

		for(int i=bid*THREAD_NUM+tid;i<datasize_xy;i=i+THREAD_NUM*BLOCK_NUM)
		{   gival_i=gival[i];
			xx=x[gival_i%ndx]-gpuptn_lam[0];
			yy=y[gival_i/ndx]-gpuptn_lam[1];

			rr=xx*xx+yy*yy+zz2;
			r=sqrtf(rr);
			lrx=gpuptn_lam[3]*r;
			lry=gpuptn_lam[3+1]*r;
			lrz=gpuptn_lam[3+2]*r;
			wt_rrr=1.0/r/rr;

			shared[0][tid]=shared[0][tid]+(lrx*__sinf(lrx)+__cosf(lrx))*wt_rrr*bxyz[gival_i];
			shared[1][tid]=shared[1][tid]+(lry*__sinf(lry)+__cosf(lry))*wt_rrr*bxyz[gival_i+datasize_xy];
			shared[2][tid]=shared[2][tid]+(lrz*__sinf(lrz)+__cosf(lrz))*wt_rrr*bxyz[gival_i+2*datasize_xy];
		}
	__syncthreads();

	offset = THREAD_NUM / 2;
	while(offset > 0) {
		if(tid < offset)
		{
			shared[0][tid] += shared[0][tid + offset];
			shared[1][tid] += shared[1][tid + offset];
			shared[2][tid] += shared[2][tid + offset];
		}
	offset >>= 1;
	__syncthreads();
	}

	if(tid==0)
	{
		resultbxyz[bid]=shared[0][0];
		resultbxyz[bid+BLOCK_NUM]=shared[1][0];
		resultbxyz[bid+2*BLOCK_NUM]=shared[2][0];
	}
}




void nlffbiept(float lamx0,float lamy0,float lamz0)
{
	float zz;
	float bx0,by0,bz0;

	zz=ro[2]*ro[2];
	float ptn_lamda[6]={ro[0],ro[1],zz,lamx0,lamy0,lamz0};

	cudaMemcpy(gpuptn_lam, ptn_lamda, sizeof(float)*(6),cudaMemcpyHostToDevice);
	kernel<<<BLOCK_NUM, THREAD_NUM, 3*THREAD_NUM*sizeof(float)>>>(gbxyz, gpux, gpuy, gpuptn_lam, resultbxyz, gival);
	cudaMemcpy(sumxyz, resultbxyz, sizeof(float)*BLOCK_NUM*3, cudaMemcpyDeviceToHost);

	bx0=0.0;
	by0=0.0;
	bz0=0.0;

	for(int i=0;i<BLOCK_NUM;i++)
	{
		bx0+=sumxyz[i];
		by0+=sumxyz[i+BLOCK_NUM];
		bz0+=sumxyz[i+2*BLOCK_NUM];
	}
		zo_dxdy = ro[2]*dxdy/pi2;

		bxo[0]=bx0*zo_dxdy;
		byo[0]=by0*zo_dxdy;
		bzo[0]=bz0*zo_dxdy;
}



/*

This code is based on the cubic spline interpolation code presented in:
Numerical Recipes in C: The Art of Scientific Computing
by
William H. Press,
Brian P. Flannery,
Saul A. Teukolsky, and
William T. Vetterling .
Copyright 1988 (and 1992 for the 2nd edition)

I am assuming zero-offset arrays instead of the unit-offset arrays
suggested by the authors.  You may style me rebel or conformist
depending on your point of view.

Norman Kuring	31-Mar-1999

*/
#define MALLOC(ptr,typ,num) {                                           \
  (ptr) = (typ *)malloc((num) * sizeof(typ));                           \
  if((ptr) == NULL){                                                    \
    fprintf(stderr,"-E- %s line %d: Memory allocation failure.\n",      \
    __FILE__,__LINE__);                                                 \
    exit(EXIT_FAILURE);                                                 \
  }                                                                     \
}

void
spline(
float	x[],
float	y[],
int	n,
float	yp1,
float	ypn,
float	y2[]
){

  int	i,k;
  float	p,qn,sig,un,*u;

  MALLOC(u,float,n-1);

  if(yp1 > 0.99e30)
    y2[0] = u[0] = 0.0;
  else{
    y2[0] = -0.5;
    u[0] = (3.0/(x[1]-x[0]))*((y[1]-y[0])/(x[1]-x[0])-yp1);
  }
  for(i = 1; i < n-1; i++){
    sig = (x[i] - x[i-1])/(x[i+1] - x[i-1]);
    p = sig*y2[i-1] + 2.0;
    y2[i] = (sig - 1.0)/p;
    u[i] = (y[i+1] - y[i])/(x[i+1] - x[i]) - (y[i] - y[i-1])/(x[i] - x[i-1]);
    u[i] = (6.0*u[i]/(x[i+1] - x[i-1]) - sig*u[i-1])/p;
  }
  if(ypn > 0.99e30)
    qn = un = 0.0;
  else{
    qn = 0.5;
    un = (3.0/(x[n-1] - x[n-2]))*(ypn - (y[n-1] - y[n-2])/(x[n-1] - x[n-2]));
  }
  y2[n-1] = (un - qn*u[n-2])/(qn*y2[n-2] + 1.0);
  for(k = n-2; k >= 0; k--){
    y2[k] = y2[k]*y2[k+1] + u[k];
  }

  free(u);
}


void
splint(
float	xa[],
float	ya[],
float	y2a[],
int	n,
float	x,
float	*y
){

  int		klo,khi,k;
  float		h,b,a;
  static int	pklo=0,pkhi=1;

  /*
  Based on the assumption that sequential calls to this function are made
  with closely-spaced, steadily-increasing values of x, I first try using
  the same values of klo and khi as were used in the previous invocation.
  If that interval is no longer correct, I do a binary search for the
  correct interval.
  */
  if(xa[pklo] <= x && xa[pkhi] > x){
    klo = pklo;
    khi = pkhi;
  }
  else{
    klo = 0;
    khi = n - 1;
    while(khi - klo > 1){
      k = (khi + klo) >> 1;
      if(xa[k] > x) khi = k;
      else          klo = k;
    }
  }

  h = xa[khi] - xa[klo];
  if(h == 0){
    fprintf(stderr,"-E- %s line %d: Bad xa input to function splint()\n",
            __FILE__,__LINE__);
    exit(EXIT_FAILURE);
  }
  a = (xa[khi] - x)/h;
  b = (x - xa[klo])/h;
  *y = a*ya[klo] + b*ya[khi] +
       ((a*a*a - a)*y2a[klo] + (b*b*b - b)*y2a[khi])*(h*h)/6.0;
}



float min(const float *arr, size_t length) {
    // returns the minimum value of array
    size_t i;
    float minimum = arr[0];
    for (i = 1; i < length; ++i) {
        if (minimum > arr[i]) {
            minimum = arr[i];
        }
    }
    return minimum;
}



float max(const float *arr, size_t length) {
    // returns the minimum value of array
    size_t i;
    float maximum = arr[0];
    for (i = 1; i < length; ++i) {
        if (maximum < arr[i]) {
        	maximum = arr[i];
        }
    }
    return maximum;
}



//-----------objf-----------------------------------------------------------------------------------------
//---------------------------------------------------------------------------------------------------------
int Bsuffix_x()
{
	return ( (ix>0 && ix<ndx-1) ? ( ix ) : ((ix==0) ? (ix+1) :((ix==ndx-1) ? (ix-1) :(0))) );
}

int Bsuffix_y()
{
	return ( (iy>0 && iy<ndy-1) ? ( iy ) : ((iy==0) ? (iy+1) :((iy==ndy-1) ? (iy-1) :(0))) );
}

 float objf(float p[])
 {
 	float z;
 	int i;
	float btx,bty,btz;
	float bb0x,bb1x,bb2x,bb3x,bb4x;//,bb5x;
	float bb0y,bb1y,bb2y,bb3y,bb4y;//,bb5y;
	float bb0z,bb1z,bb2z,bb3z,bb4z;//,bb5z;
	float bm0x, bm0y, bm0z;
	float bm1x, bm1y, bm1z;
	float bm2x, bm2y, bm2z;
    float bm3x, bm3y, bm3z;
    float bm4x, bm4y, bm4z;
    float dSx, dSy, dSz;
	float rotx, roty, rotz;
    float rotm, cpx, cpy, cpz;
	float bo0;
	float ff0;

	if (p[0]>=0.0 && p[0]<=per[0] &&
		p[1]>=0.0 && p[1]<=per[1] &&
		p[2]>=0.0 && p[2]<=per[2])
	{
		splint(lamx, bxo_lam0, bxo_spl_y2_0, N, p[0], &z);    bxo[0] = z;
		splint(lamy, byo_lam0, byo_spl_y2_0, N, p[1], &z);    byo[0] = z;
		splint(lamz, bzo_lam0, bzo_spl_y2_0, N, p[2], &z);    bzo[0] = z;

		i=Bsuffix_x()+ndx*Bsuffix_y()+datasize_xy*(iz-1);

		// if(iz==1){
		btx=bxo[0];
		bty=byo[0];
		btz=bzo[0];

		bb0x=b3x[i];
		bb0y=b3y[i];
		bb0z=b3z[i];

		bb1x=b3x[i-ndx];
		bb1y=b3y[i-ndx];
		bb1z=b3z[i-ndx];

		bb2x=b3x[i+ndx];
		bb2y=b3y[i+ndx];
		bb2z=b3z[i+ndx];

		bb3x=b3x[i-1];
		bb3y=b3y[i-1];
		bb3z=b3z[i-1];

		bb4x=b3x[i+1];
		bb4y=b3y[i+1];
		bb4z=b3z[i+1];

		bm1x=(btx+bb1x)/2.0;
		bm1y=(bty+bb1y)/2.0;
		bm1z=(btz+bb1z)/2.0;

		bm2x=(btx+bb2x)/2.0;
		bm2y=(bty+bb2y)/2.0;
		bm2z=(btz+bb2z)/2.0;

		bm3x=(btx+bb3x)/2.0;
		bm3y=(bty+bb3y)/2.0;
		bm3z=(btz+bb3z)/2.0;

		bm4x=(btx+bb4x)/2.0;
		bm4y=(bty+bb4y)/2.0;
		bm4z=(btz+bb4z)/2.0;
	
		// bm0x=(bb0x+btx+bm1x+bm2x+bm3x+bm4x)/6.0;
		// bm0y=(bb0y+bty+bm1y+bm2y+bm3y+bm4y)/6.0;
		// bm0z=(bb0z+btz+bm1z+bm2z+bm3z+bm4z)/6.0;
		bm0x=(bb0x+btx)/2.0;
		bm0y=(bb0y+bty)/2.0;
		bm0z=(bb0z+btz)/2.0;

		dSx  = dx*dz;
		rotx = ((-bm1y)*dx-(bm1z)*dz + bb0y*2*dx + (-bm2y)*dx+(bm2z)*dz)/dSx;
		// rotx = (bm2z-bm1z-bty+bb0y)/dx;
		dSy  = dx*dz;
		roty = ((bm4x)*dx-(bm4z)*dz + (-bb0x)*2*dx + (bm3x)*dx+(bm3z)*dz)/dSy;
		// roty = (btx-bb0x-bm4z+bm3z)/dx;
		dSz  = dx*dx;
		// rotz = (bm4y-bm3y-bm2x+bm1x)/dx;
		rotz= (bm1x+bm4y-bm2x-bm3y)*dx/dSz;
		dvb0 = fabs(((bm4x-bm3x) + (bm2y-bm1y))/dx + (btz-bb0z)/dz);
			
		// }
		// else
		// {
		// 	btx=bxo[0];
		// 	bty=byo[0];
		// 	btz=bzo[0];

		//     bb0x=b3x[i];
		//     bb0y=b3y[i];
		//     bb0z=b3z[i];

		// 	bb1x=b3x[i-ndx];
		// 	bb1y=b3y[i-ndx];
		// 	bb1z=b3z[i-ndx];

		// 	bb2x=b3x[i+ndx];
		// 	bb2y=b3y[i+ndx];
		// 	bb2z=b3z[i+ndx];

		// 	bb3x=b3x[i-1];
		// 	bb3y=b3y[i-1];
		// 	bb3z=b3z[i-1];

		// 	bb4x=b3x[i+1];
		// 	bb4y=b3y[i+1];
		// 	bb4z=b3z[i+1];

		//     bb5x=b3x[i-datasize_xy];
		//     bb5y=b3y[i-datasize_xy];
		//     bb5z=b3z[i-datasize_xy];
			
		// 	bm0x=bb0x;
		// 	bm0y=bb0y;
		// 	bm0z=bb0z;
			
		// 	rotx = (bb2z-bb1z-bty+bb5y)/2.0/dx;
		// 	roty = (btx-bb5x-bb4z+bb3z)/2.0/dx;
		// 	rotz = (bb4y-bb3y-bb2x+bb1x)/2.0/dx;
		// 	dvb0 = fabs(((bb4x-bb3x) + (bb2y-bb1y) + (btz-bb5z))/2.0/dx);	
		// }
		jo[0] = rotx;
		jo[1] = roty;
		jo[2] = rotz;
		bo0=sqrtf(bm0x*bm0x+bm0y*bm0y+bm0z*bm0z);
		rotm=sqrtf(rotx*rotx+roty*roty+rotz*rotz);
		cpx=roty*bm0z-rotz*bm0y;
		cpy=rotz*bm0x-rotx*bm0z;
		cpz=rotx*bm0y-roty*bm0x;
		Lf0 = sqrtf(cpx*cpx+cpy*cpy+cpz*cpz)/(bo0*rotm);
		Ld0 = dvb0*dx/bo0;
		ff0 = Lf0 + Ld0;
//    	printf("lam:%f,%f,%f  -- B:%f,%f,%f\n",p[0],p[1],p[2],bxo[0],byo[0],bzo[0]);
	}
	else
	{
		ff0=10.0;
	}
 	return(ff0);
 }



 //-----------------flesim------------------------------------------------------------------
 float flesim(float p[],float fx,float ft0,float step,int it,int imos)
 {
 	int n,i,j;
 	float a[7][7]={{0}};
 	float ff[7]={0.0};
 	float ft=0.0000001,temp;
 	float alpha,beta,gama;
 	int k1,k2,k3,k4,dead=0;
 	float difer;
 	float d1,d2,s;
 	float ffk2,ffk3,ffk4,ffi;
 	float fl,fh,fs;
 	int ih,il,l;
 	float fx1,p0[3]={0.0};

 	n=3;//the dimension of x
 	if(it==1) ft=ft0;
 	iter=0;
 	alpha=1.0;
 	beta=0.2;
 	gama=2.5;
 	k1=n+1;
 	k2=n+2;
 	k3=n+3;
 	k4=n+4;
 	//-----------Form initial simplex-------------

 	d1=step/(n*sqrt(2.0))*(sqrt((float)n+1)+n-1.0);
 	d2=step/(n*sqrt(2.0))*(sqrt((float)n+1)-1.0);
 	for(j=0;j<=n-1;j++)
 	{
 		a[0][j]=p[j];
 	}
 	for(i=1;i<=k1-1;i++)
 	{
 		for(j=0;j<=n-1;j++)
 		{
			a[i][j]=d2+p[j];
 		}
 		l=i-1;
 		a[i][l]=d1-d2+p[l]; // revised by sijie 2015-01-14
 	}
 	for(i=0;i<=k1-1;i++)
 	{
 		for(j=0;j<=n-1;j++)
 		{
 			p[j]=((a[i][j] < per[j]) ? a[i][j]:per[j]);
 			// p[j]=a[i][j];
 		}
 		ffi=objf(p);
 		ff[i]=ffi;
 	// printf("simplex vertex is ,v=%d,:f(x)=%f,x=%f,%f,%f\n",i,ff[i],p[0],p[1],p[2]);
 	}
 iteration:
 	if(iter<imos)
 	{
 		iter=iter+1;


 		fh=ff[0];
 		ih=0;
 		fl=ff[0];
 		il=0;
 		for(i=1;i<=k1-1;i++)
 		{
 		//-------------determine max point on simplex's vertices-------------
 			if(ff[i]>=fh)
 			{
 				fh=ff[i];
 				ih=i;
 			}
 		//-------------determine min point on simplex's vertices-------------
 			if(ff[i]<=fl)
 			{
 				fl=ff[i];
 				il=i;
 			}
 		}
 		//------------------calculate simplex's center excluding max point---------
 		for(j=0;j<=n-1;j++)
 		{
 			s=0.0;
 			for(i=0;i<=k1-1;i++)
 			{
 				s=s+a[i][j];
 			}
 			a[k2-1][j]=1./n*(s-a[ih][j]);
 				//----------------mapping of max-point through center----------------
 			a[k3-1][j]=(1.+alpha)*a[k2-1][j]-alpha*a[ih][j];
			p[j]=((a[k3-1][j] < per[j]) ? a[k3-1][j]:per[j]);
			// p[j]=a[k3-1][j];
 		}
 		ffk3=objf(p);
 		ff[k3-1]=ffk3;
 		if(ff[k3-1]<fl)
 		{
 			goto Expansion;
 		}
 		//---------------------determine 2nd ,max point on simplex's vertices-------
 		if(ih==0)
 		{
 			fs=ff[1];
 		}
 		else
 		{
 			fs=ff[0];
 		}
 		for(i=0;i<=k1-1;i++)
 		{
 			if(i!=ih)
 			{
 				if(ff[i]>=fs)
 				{
 					fs=ff[i];
 				}
 			}
 		}
 		if(ff[k3-1]>fs)
 		{
 			goto loop54;
 		}
 		goto Replace_Xr;
 Expansion://-----------If mapping has new min, then expanding simplex---------
 		for(j=0;j<=n-1;j++)
 		{
 			a[k4-1][j]=(1.-gama)*a[k2-1][j]+gama*a[k3-1][j];
			p[j]=((a[k4-1][j] < per[j]) ? a[k4-1][j]:per[j]);
 			// p[j]=a[k4-1][j];
 		}
 		ffk4=objf(p);
 		ff[k4-1]=ffk4;
 		if(ff[k4-1]<fl)
 		{
 			goto Replace_Xnew;
 		}
 		goto Replace_Xr;
 loop54:
 		if(ff[k3-1]<=fh)
 		{
 			for(j=0;j<=n-1;j++)
 			{
 				a[ih][j]=a[k3-1][j];
 			}
 		}
 		for(j=0;j<=n-1;j++)
 		{
 			a[k4-1][j]=beta*a[ih][j]+(1-beta)*a[k2-1][j];
			p[j]=((a[k4-1][j] < per[j]) ? a[k4-1][j]:per[j]);
 			// p[j]=a[k4-1][j];
 		}
 		ffk4=objf(p);
 		ff[k4-1]=ffk4;
 		if(fh>ff[k4-1])
 		{
 			goto Replace_Xnew;
 		}
 			//---------------if mapping point larger than max, then contracting simplex----
 		for(j=0;j<=n-1;j++)
 		{
 			for(i=0;i<=k1-1;i++)
 			{
 				a[i][j]=0.5*(a[i][j]+a[il][j]);
 			}
 		}
 		for(i=0;i<=k1;i++)
 		{
 			for(j=0;j<=n-1;j++)
 			{
	 			p[j]=((a[i][j] < per[j]) ? a[i][j]:per[j]);
 				// p[j]=a[i][j];
 			}
 			ffi=objf(p);
 			ff[i]=ffi;
 		}
 		goto iteration;
 Replace_Xnew:
 		for(j=0;j<=n-1;j++)
 		{
 			a[ih][j]=a[k4-1][j];
 		}
 		ff[ih]=ff[k4-1];
 		goto Criteria;
 Replace_Xr:
 		for(j=0;j<=n-1;j++)
 		{
 			a[ih][j]=a[k3-1][j];
 		}
 		ff[ih]=ff[k3-1];
 Criteria:
 		for(j=0;j<=n-1;j++)
 		{
 			p[j]=((a[k2-1][j] < per[j]) ? a[k2-1][j]:per[j]);
			// p[j]=a[k2-1][j];
 		}
 		ffk2=objf(p);
 		ff[k2-1]=ffk2;
 		difer=0.0;
 		for(i=0;i<=k1-1;i++)  //obtain the sum of the square of ff[i]-ffk2
 		{
 			temp=ff[i]-ffk2;
 			difer+=temp*temp;
 		}
 		difer=1./n*sqrt(difer);
		printf("difer=%f, Lf=%f, Ld = %f, fx=%f, lam=%f,%f,%f, bxyz=%f,%f,%f\n",difer,Lf0,Ld0,ffk2 ,p[0],p[1],p[2],bxo[0],byo[0],bzo[0]);
		// printf("ff = %f,%f,%f,%f,%f,%f,%f\n",ff[0],ff[1],ff[2],ff[3],ff[4],ff[5],ff[6]);
 		icovg=-1;
 		fx=ffk2;
		
 		if((abs(p0[0]-p[0])<0.0000001) && abs(p0[1]-p[1])<0.0000001 && abs(p0[2]-p[2])<0.0000001)
	 		{
				dead++;
				// printf("dead:%d\n",dead);
	 		}
			
		p0[0] = p[0];
		p0[1] = p[1];
		p0[2] = p[2];
		
		
		if(dead > 5)
		{
			goto loop88;
		}		
		
 		if(difer>ft)
 		{
 			goto iteration; //modified with "fx<ft" instead of "difer<ft" on 19 Mar 2014
 		}
 		icovg=1;
 		if(fabs(Lf0)>0.5)
 		{
 			icovg=2;
 		}
 		if(fabs(Ld0)>0.2)
 		{
 			icovg=3;
 		}
 		goto loop88;
 			//----------------------------end iteration--------------------------
 	}
 loop88:
// printf("ix,iy,iz=%d,%d,%d, iter=%d, icovg=%d, difer=%f, fx=%f,Lf0=%f,Ld0=%f, lam=%f,%f,%f, bxyz=%f,%f,%f\n",ix,iy,iz,iter,icovg,difer,ffk2,Lf0,Ld0,p[0],p[1],p[2],bxo[0],byo[0],bzo[0]);
 	fx1=fx;
 	return fx1;
 }
 //---------------------------------------------//







 //-----------differential-----------------------------------------------------------------------------------------
 //---------------------------------------------------------------------------------------------------------
 //Calculate dr/dl at ro by DBIE
 // Input:
 //---------------------------------------
 //    ro :  array of 4:=> R[4]=[xo,yo,zo,alphai]
 //---------------------------------------
 // Output:
 //---------------------------------------
 //  differentials [dx/dl,dy/dl,dz/dl] at ro
 //---------------------------------------
 //  Revised to force alpha equal to alpha in the boundary
 //                                 25 Mar 2014 (S.Yu)
 //  Revised to force magnetic field line turning smoothly
 //                                 25 Mar 2014 (S.Yu)
 //----------------------------------------------------------
 void differential(float ro[])
 {
    int loop,i;
 	int imos=200, iter=1,lpmax=3;
 	float fx=0.0;
 	float ft0=0.000001;
 	float step;
    float yp1,ypn;


    loop=0,icovg=0;
    nlffbiept2();
    for(i=0;i<N;i++)
    {
    	p[0] = per[0]*i/(N-1);
    	p[1] = per[1]*i/(N-1);
    	p[2] = per[2]*i/(N-1);
    	lamx[i] = p[0];
		lamy[i] = p[1];
		lamz[i] = p[2];
    	nlffbiept(p[0],p[1],p[2]);
    	bxo_lam0[i] = bxo[0];
    	byo_lam0[i] = byo[0];
    	bzo_lam0[i] = bzo[0];
    }
    yp1 = 1.0e30;
    ypn = 1.0e30;
    spline(lamx, bxo_lam0, N, yp1, ypn, bxo_spl_y2_0);
    spline(lamy, byo_lam0, N, yp1, ypn, byo_spl_y2_0);
    spline(lamz, bzo_lam0, N, yp1, ypn, bzo_spl_y2_0);

 	while(icovg!=1&&loop<lpmax)
 	{
		if(icovg==0)
		{
			p[0] = per[0]/2.0;
			p[1] = per[1]/2.0;
			p[2] = per[2]/2.0;
			step = max(p,3);
			// printf("step, perx, pery, perz: %f, %f, %f, %f\n",step,per[0],per[1],per[2]);
		}
		flesim(p,fx,ft0,step,iter,imos);
		// printf("bx=%f,by=%f,bz=%f,jx=%f,jy=%f,jz=%f\n",bxo[0],byo[0],bzo[0],jo[0],jo[1],jo[2]);
		loop++;
 	}
 }
 //---------------------------------------------//



/****************************************************/
/******* MAIN PROGRAM   *****************************/
/****************************************************/
int main(int argc,char *argv[])
{
	float space;
	FILE *fp1;
    float *x,*y;
    int i;
	int height_start, height_end;
	int x_start,x_end,y_start,y_end;
	char hh[10],mm[10],date[10];
	int i1,i2,i3,i4,i5,i6,i7,i8;
	int ix1,ix2,iy1,iy2;

	//printf("The value of argc is %d\n\n", argc);
	//设置main的输入参数，分别为x_start,x_end,y_start,y_end,height_start,height_end;
	if(argc==10){
		//如果argc的值是1，说明程序名后面没有命令行参数
		//argv[0]的值是启动该程序的程序名
		//argv[1] 为在DOS命令行中执行程序名后的第一个字符串;
		//argv[2] 为执行程序名后的第二个字符串;
		//...
		//设置从第几个文件开始计算，算多少个文件。
		strcpy(date, argv[1]);
		strcpy(hh,   argv[2]);
		strcpy(mm,   argv[3]);
		x_start = atoi(argv[4]);
		x_end = atoi(argv[5]);
		y_start = atoi(argv[6]);
		y_end = atoi(argv[7]);
		height_start = atoi(argv[8]);
		height_end = atoi(argv[9]);
	}
	else{
		strcpy(date, "20110215");
		strcpy(hh, "00");
		strcpy(mm, "00");
		x_start=50;
		y_start=50;
		x_end=50;
		y_end=50;
		height_start=1;
		height_end=1;
	}


	printf("This calculation is start from layer %d to layer %d.\n", height_start, height_end);
	printf("x_start = %d,x_end = %d.\n", x_start,x_end);
	printf("y_start = %d,y_end = %d.\n\n", y_start,y_end);
	int onx,ony,onxny,oi;
	onx = x_end-x_start+1;
	ony = y_end-y_start+1;
	onxny = onx*ony;
	int nxnynz = ndx*ndy*ndz;

	//------------------------//
    int *ival;
    cudaMalloc((void**) &gival, sizeof(int)*datasize_xy);
    ival=(int*) malloc(sizeof(int)*datasize_xy);
		for(int i=0;i<datasize_xy-1;i++)
	{
		ival[i]=i;
	}
	cudaMemcpy(gival, ival, sizeof(int)*datasize_xy,cudaMemcpyHostToDevice);

	cudaMalloc((void**) &gpuptn_lam, sizeof(float)*(6));
	cudaMalloc((void**) &resultbxyz, sizeof(float)*BLOCK_NUM*3);
	sumxyz=(float*) malloc(sizeof(float)*BLOCK_NUM*3);
	for(i=0;i<=BLOCK_NUM*3-1;i++)
		sumxyz[i]=0.0;


	time_t start,stop,gputime;
    time_t tim;
    struct tm *at;
    char now[80];

	float *bxyz;
	bxyz=(float*) malloc(sizeof(float)*datasize_xy*3);
	x=(float*) malloc(sizeof(float)*ndx);
	y=(float*) malloc(sizeof(float)*ndy);
	cudaMalloc((void**) &gbxyz,  sizeof(float)*datasize_xy*3);
//++++++++++++coordinates+++++++++++++++
	cudaMalloc((void**) &gpux, sizeof(float)*ndx);
	cudaMalloc((void**) &gpuy, sizeof(float)*ndy);
	space=rsize/(ndx-1);
	for(i=0;i<=ndx-1;i++)
		x[i]=i*space;
	for(i=0;i<=ndy-1;i++)
		y[i]=i*space;
	cudaMemcpy(gpux, x, sizeof(float)*ndx,cudaMemcpyHostToDevice);
	cudaMemcpy(gpuy, y, sizeof(float)*ndy,cudaMemcpyHostToDevice);
	dx=x[1]-x[0];  //  equal spacing
	dy=y[1]-y[0];//  equal spacing
	dz=dx/zratio;
	dxdy=dx*dy;

	//----------------Read the data of bx,by,bz from .bin--------------

	char filein[60];
	sprintf(filein, "./boundary_data/boundary.%s_%s%s00_TAI.bin",date,hh,mm);
	printf("reading ./boundary_data//boundary.%s_%s%s00_TAI.bin\n",date,hh,mm);
	if((fp1=fopen(filein,"rb"))==NULL)
	{
		printf("can't open boundary_bxyz.bin!");
		exit(0);
	}

    //----------define access space------------
	float *B0x, *B0y, *B0z;
	b3x   = (float *) calloc(nxnynz, sizeof(float));
	b3y   = (float *) calloc(nxnynz, sizeof(float));
	b3z   = (float *) calloc(nxnynz, sizeof(float));
	// j3x   = (float *) calloc(nxnynz, sizeof(float));
	// j3y   = (float *) calloc(nxnynz, sizeof(float));
	// j3z   = (float *) calloc(nxnynz, sizeof(float));

	B0x=(float *) calloc(datasize_xy,sizeof(float));
	B0y=(float *) calloc(datasize_xy,sizeof(float));
	B0z=(float *) calloc(datasize_xy,sizeof(float));

	fread(B0x,sizeof(float)*datasize_xy,1,fp1);
	fread(B0y,sizeof(float)*datasize_xy,1,fp1);
	fread(B0z,sizeof(float)*datasize_xy,1,fp1);
	printf("\t Read finished ");

	for( iy=0;iy<ndy;iy++)
	for( ix=0;ix<ndx;ix++)
		{
			i=ix+ndx*iy;
			bxyz[i]               = B0x[i];
			bxyz[i+datasize_xy]   = B0y[i];
			bxyz[i+2*datasize_xy] = B0z[i];
			b3x[i] =B0x[i];
			b3y[i] =B0y[i];
			b3z[i] =B0z[i];
		}
	cudaMemcpy(gbxyz, bxyz, sizeof(float)*datasize_xy*3, cudaMemcpyHostToDevice);

	time(&tim);
	at=localtime(&tim);
	strftime(now,79,"%Y-%m-%d\n%H:%M:%S\n",at);
	puts(now);
	printf("Program is running, please wait... ...\n\n");
	start=clock();

//------------------------------------open the output files ---------------
	FILE *fpa,*fpd;
	char fileout[60];
	sprintf(fileout, "./results/mag.%s_%s%s00_%03dx%03d_%03dt%03d_TAI.bin",date,hh,mm,ndx,ndy,height_start,height_end);
	printf("%s\n",fileout);
	if((fpa=fopen(fileout,"w"))==NULL)
	  {printf("Open file fpa failed! \n");
	  exit(0);}

	sprintf(fileout, "./results/time.%s_%s%s00_%03dx%03d_%03dt%03d_TAI.txt",date,hh,mm,ndx,ndy,height_start,height_end);
	printf("%s\n",fileout);
	if((fpd=fopen(fileout,"w"))==NULL)
	  {printf("Open file fpb failed! \n");
	  exit(0);
	}

//--------------------------------------------------

	/*************************Output*****************************/
		float *b2x, *b2y, *b2z;
		// float *j2x, *j2y, *j2z;
		float b2xs, b2ys, b2zs;
		b2x   = (float *) calloc(onxny, sizeof(float));
		b2y   = (float *) calloc(onxny, sizeof(float));
		b2z   = (float *) calloc(onxny, sizeof(float));
		// j2x   = (float *) calloc(onxny, sizeof(float));
		// j2y   = (float *) calloc(onxny, sizeof(float));
		// j2z   = (float *) calloc(onxny, sizeof(float));
	for(iz= height_start; iz<=height_end; iz++)
	{
		for(iy=y_start; iy<=y_end; iy++){
		for(ix=x_start; ix<=x_end; ix++)
		{
			ro[0] = ix*space;
			ro[1] = iy*space;
			ro[2] = space/zratio;
			oi    = ((ix-x_start) + (iy-y_start)*onx);
			// printf("ix=%d,iy=%d,iz=%d, B-suffix: %d, %d\n",ix,iy,iz,Bsuffix_x(),Bsuffix_y());
			differential(ro);
			b2x[oi] = bxo[0];
			b2y[oi] = byo[0];
			b2z[oi] = bzo[0];
			// j2x[oi] = jo[0];
			// j2y[oi] = jo[1];
			// j2z[oi] = jo[2];
		}}
        /* smooth the field */
		for(iy=y_start; iy<=y_end; iy++){
		for(ix=x_start; ix<=x_end; ix++)
		{
			i=ix+ndx*iy;
			ix1=ix+1; if(ix1==ndx){ix1=0;}
			ix2=ix-1; if(ix2==-1){ix2=ndx-1;}
			iy1=iy+1; if(iy1==ndy){iy1=0;}
			iy2=iy-1; if(iy2==-1){iy2=ndy-1;}
			i1=ix1+ndx*iy;
			i2=ix2+ndx*iy;
			i3=ix +ndx*iy1;
			i4=ix +ndx*iy2;
			i5=ix1+ndx*iy1;
			i6=ix2+ndx*iy2;
			i7=ix2+ndx*iy1;
			i8=ix1+ndx*iy2;
			b2xs=(b2x[i]+b2x[i1]+b2x[i2]+b2x[i3]+b2x[i4]+b2x[i5]+b2x[i6]+b2x[i7]+b2x[i8])/9.0;
			b2ys=(b2y[i]+b2y[i1]+b2y[i2]+b2y[i3]+b2y[i4]+b2y[i5]+b2y[i6]+b2y[i7]+b2y[i8])/9.0;
			b2zs=(b2z[i]+b2z[i1]+b2z[i2]+b2z[i3]+b2z[i4]+b2z[i5]+b2z[i6]+b2z[i7]+b2z[i8])/9.0;
			bxyz[i]               =b2xs;
			bxyz[i+datasize_xy]   =b2ys;
			bxyz[i+2*datasize_xy] =b2zs;
			i=i+datasize_xy*iz;
			b3x[i] = b2xs;
			b3y[i] = b2ys;
			b3z[i] = b2zs;
		}}
		fwrite(b2x,sizeof(float)*onxny,1,fpa);
		fwrite(b2y,sizeof(float)*onxny,1,fpa);
		fwrite(b2z,sizeof(float)*onxny,1,fpa);
		// fwrite(j2x,sizeof(float)*onxny,1,fpa);
		// fwrite(j2y,sizeof(float)*onxny,1,fpa);
		// fwrite(j2z,sizeof(float)*onxny,1,fpa);
		cudaMemcpy(gbxyz, bxyz, sizeof(float)*datasize_xy*3, cudaMemcpyHostToDevice);
	}
	
	
	

	fflush(fpa);

	stop=clock();
	gputime=stop-start;
	double sec=double(gputime)/CLOCKS_PER_SEC;
	printf("Time used =%.2f s\n",sec);
	time(&tim);
	at=localtime(&tim);
	strftime(now,79,"%Y-%m-%d\n%H:%M:%S\n",at);
	puts(now);

	fprintf(fpd,"Time used =%.2f s\n",sec);
	fclose(fp1);
	fclose(fpa);
	fclose(fpd);
	printf("NLFF Field written to ./results/mag.%s_%s%s00_%03dx%03d_%03dt%03d_TAI.bin \n",date,hh,mm,ndx,ndy,height_start,height_end);
	printf("Program is stopping!\n");

	free(ival);
	cudaFree(gpuptn_lam);
	cudaFree(gival);
	cudaFree(gbxyz);
	cudaFree(gpux);
	cudaFree(gpuy);
 	cudaFree(resultbxyz);
    free(x);
    free(y);
    free(bxyz);
    free(sumxyz);
	return 0;
}

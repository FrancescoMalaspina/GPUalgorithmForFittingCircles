#include <stdio.h>
#include <math.h>
#include <iostream>
#include <inttypes.h>
#include <vector>
#include <fstream> // I/O

#define BLOCK_SIZE       8 // number of threads for each x, y block
#define BLOCK_Z_SIZE    10 // number of threads for each z block
#define LATTICE_SIZE   128 // dimension of entire space   (multiple of BLOCK_SIZE)
#define HST_SIZE      1000 // dimnnsion of each histogram 
#define HST_THRESHOLD   20 // threshols for circle identification in histogram 
#define MAX_CIRCLES     10 // maximum number of identified circles

//struct DataFrame
struct DataFrame
{
  int          w;     // number of data per sample
  int          h;     // number of samples
  int*     h_arr;     // pointer to host data
  int*     d_arr;     // pointer to device data
  int      min_x;      
  int      max_x;      
  int      min_y;      
  int      max_y;      
};

//struct HistogramFrame
struct HistogramFrame
{
  int            w;     // width of the lattice
  int            h;     // height of the lattice 
  int            d;     // depths of the histogram array
  // int      * h_arr;     // size = w x h x d x sizeof int
  int      * d_arr;     // size = w x h x d x sizeof int
  int        max_h;     // max value to be considered for the histogram
  int          bin;     // 
};

//struct for result circles
struct CirclesFrame
{
  int            h;     // max number of circles
  int      h_found;     // circles found, host side        
  int    * d_found;     // circles found, device side 
  int      * d_arr;     // size = h x 3 x sizeof unit16_t
  int      * h_arr;     // size = h x 3 x sizeof unit16_t
};


// kernel for histogram preparation
// __global__ void LatticeKernel(const DataFrame, HistogramFrame); //aggiungere i parametri
__global__ void LatticeKernel(const DataFrame, HistogramFrame, int*); //aggiungere i parametri

// kernel for circles preparation
// __global__ void CirclesKernel(HistogramFrame, CirclesFrame); //aggiungere i parametri
__global__ void CirclesKernel(HistogramFrame, CirclesFrame, int*); //aggiungere i parametri


//-------------------------------------------------------------
int main(int argc, char** argv)
{
  DataFrame      in_data;
  HistogramFrame hst_data;
  CirclesFrame	 circles_data;
  float          time;
  cudaEvent_t    startall, stopall, start, stop;
  cudaError_t    err; 
  size_t         size;
  
  
  // Timing events
  cudaEventCreate(&startall);
  cudaEventCreate(&stopall);
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  // ----------------------------
  int		 h_debug[20];
  int	       * d_debug;
  cudaMalloc(&d_debug,   20*sizeof(int));
  cudaMemset(d_debug, 0, 20*sizeof(int));
  // ----------------------------


  //start time
  cudaEventRecord(startall);
  
  // read data and evaluate min and max for x and y
  std::vector<float> buf;
  std::ifstream in("file.dat",std::ios::binary); //open the input file
  
  // read first sample
  float w;
  in.read( (char*) &w, sizeof(w));
  buf.push_back(w);
  in_data.min_x = (int)w;      
  in_data.max_x = (int)w;      

  in.read( (char*) &w, sizeof(w));
  buf.push_back(w);
  in_data.min_y = (int)w;      
  in_data.max_y = (int)w;      

  // read data and evaluate min and max
  while(!in.eof())
  {
    in.read( (char*) &w, sizeof(w));
    buf.push_back(w);
  }

  printf("read %d points\n", (int) buf.size() / 2);

  // -----------------------------------------------------------------
  // allocate host memory for data
  in_data.w     = 2;
  in_data.h     = buf.size() / 2;

  size = in_data.w * in_data.h * sizeof(int);
  in_data.h_arr = (int*) malloc(size);
  
  // copy vector to data
  for (int i = 0; i < in_data.h; i++)
  {
    int x, y;
    
    x = (int) buf[i * in_data.w ];
    y = (int) buf[i * in_data.w  + 1];
    
    in_data.h_arr[i * in_data.w ]    = x;
    if (in_data.min_x > x ) in_data.min_x = x;      
    if (in_data.max_x < x ) in_data.max_x = x;      


    in_data.h_arr[i * in_data.w + 1] = y;
    if (in_data.min_y > y ) in_data.min_y = y;      
    if (in_data.max_y < y ) in_data.max_y = y;      
  }
  
  // print the converted file content
  printf("Input data:\n");
  for (int i = 0; i < in_data.h; i++)     // Loop sulle righe
  {
    for (int j = 0; j< in_data.w; j++)    // Loop sulle colonne
      printf("%d\t", in_data.h_arr[i*in_data.w + j]);
    printf("\n");  // A capo di fine riga
  }
  printf("\n");


  
  // allocate device memory for data
  err = cudaMalloc(&in_data.d_arr, size);
  if(err)
    printf("CUDA malloc in data: %s\n",cudaGetErrorString(err));
  cudaMemcpy(in_data.d_arr, in_data.h_arr, size, cudaMemcpyHostToDevice);


  // -----------------------------------------------------------------
  // allocate host memory for histogram  
  hst_data.w     = LATTICE_SIZE;     // width of the lattice
  hst_data.h     = LATTICE_SIZE;     // height of the lattice 
  hst_data.d     = HST_SIZE;         // depths of the histogram array
  hst_data.max_h = (LATTICE_SIZE * LATTICE_SIZE * 2);     // maximum value considered for the distance as the lattice diagonal (sqrt(2) = 1.41....) 
  hst_data.bin   = hst_data.max_h / hst_data.d;  // size of the histogram bin
  
  size =   hst_data.w * hst_data.h * hst_data.d * sizeof(int);
  // hst_data.h_arr = (int *) malloc( size );     // size = w x h x d x sizeof unit16_t ... probably not needed

  // allocate device memory for histogram
  err = cudaMalloc(&hst_data.d_arr, size);
  if(err)
    printf("CUDA malloc histogram data: %s\n",cudaGetErrorString(err));
  cudaMemset(hst_data.d_arr, 0, size);


  // -----------------------------------------------------------------
  // Allocate host memory for circles
  circles_data.h        = MAX_CIRCLES;     // max number of circles
  circles_data.h_found  = 0;               // circles found, host side        
  
  size =   circles_data.h * 3 * sizeof(int);
  circles_data.h_arr = (int*)malloc(size);     // size = h x 3 x sizeof int
  
  // allocate device memory for circles
  err = cudaMalloc(&circles_data.d_arr, size);
  if(err)
    printf("CUDA malloc circles data: %s\n",cudaGetErrorString(err));
  cudaMemset(circles_data.d_arr, 0, size);
  
  err = cudaMalloc(&circles_data.d_found, sizeof(int));
  if(err)
    printf("CUDA malloc circles d_found: %s\n",cudaGetErrorString(err));
  cudaMemset(circles_data.d_found, 0, sizeof(int));



  // +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
  // Lattice evaluation

  // define topology for the kernel (grid geometry)
  dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE, BLOCK_Z_SIZE);
  
  dim3 dimGrid(LATTICE_SIZE / BLOCK_SIZE, LATTICE_SIZE / BLOCK_SIZE, (in_data.h + BLOCK_Z_SIZE -1 ) / BLOCK_Z_SIZE);

  //start time
  cudaEventRecord(start);


  // Invoke kernel
  LatticeKernel<<<dimGrid, dimBlock>>>(in_data, hst_data, d_debug);
  err = cudaDeviceSynchronize();

  //stop time
  cudaEventRecord(stop);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&time, start, stop);

  if(err)
    printf("Run LatticeKernel: %s\n", cudaGetErrorString(err));
  printf("Run LatticeKernel Time: %3.5f ms\n",time);


  // +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
  // Histogram evaluation

  // define topology for the kernel (grid geometry)
  // dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE, BLOCK_Z_SIZE);
  
  // dim3 dimGrid(LATTICE_SIZE / BLOCK_SIZE, LATTICE_SIZE / BLOCK_SIZE, (in_data.h + BLOCK_Z_SIZE -1 ) / BLOCK_Z_SIZE);
  dimGrid.z = (HST_SIZE + BLOCK_Z_SIZE -1 ) / BLOCK_Z_SIZE;

  //start time
  cudaEventRecord(start);

  // Invoke kernel
  CirclesKernel<<<dimGrid, dimBlock>>>(hst_data, circles_data, d_debug);
  err = cudaDeviceSynchronize();

  //stop time
  cudaEventRecord(stop);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&time, start, stop);

  if(err)
    printf("Run CirclesKernel: %s\n", cudaGetErrorString(err));
  printf("Run CirclesKernel Time: %3.5f ms\n",time);

  // retrieve the data
  size   =   circles_data.h * 3 * sizeof(int);
  cudaMemcpy(circles_data.h_arr, circles_data.d_arr, size, cudaMemcpyDeviceToHost);
  cudaMemcpy(&circles_data.h_found, circles_data.d_found, sizeof(int), cudaMemcpyDeviceToHost);

  // print the data found
  printf("Found %d circles:\n", circles_data.h_found);
  for (int i = 0; i < circles_data.h; i++)     // Loop sulle righe
  {
    for (int j = 0; j < 3; j++)    // Loop sulle colonne
      printf("%d\t", circles_data.h_arr[ i * 3 + j]);
    printf("\n");  // A capo di fine riga
  }
  printf("\n");
	
  /*/
  // ----------------------------
  cudaMemcpy(h_debug, d_debug, 20*sizeof(int), cudaMemcpyDeviceToHost);
  printf("Debug data:\n");
  for (int i = 0; i < 20; i++) printf("%d - %d\n", i, h_debug[i]);
  cudaFree(d_debug);
  // ----------------------------
  /*/


  // free device memory
  cudaFree(in_data.d_arr);
  cudaFree(hst_data.d_arr);
  cudaFree(circles_data.d_arr);
  cudaFree(circles_data.d_found);

  //stop time
  cudaEventRecord(stopall);
  cudaEventSynchronize(stopall);
  cudaEventElapsedTime(&time, startall, stopall);

  printf("Total Run Time: %3.5f ms\n",time);


  return (0);
}




// kernel for histogram preparation
//__global__ void LatticeKernel(const DataFrame in_data, HistogramFrame hst_data)
__global__ void LatticeKernel(const DataFrame in_data, HistogramFrame hst_data, int* debug)
{

  // x and y grid dimensions for x and lattice coordinates
  // z coordinate for the data index
  int idx_data  = (blockIdx.z * blockDim.z + threadIdx.z);
  if (idx_data > in_data.h) return; // check for possible overflows
  
  int x_data    = in_data.d_arr[in_data.w * idx_data ];
  int x_lattice = blockIdx.x * blockDim.x + threadIdx.x - (LATTICE_SIZE / 2)  ;
  
  int y_data    = in_data.d_arr[in_data.w * idx_data + 1 ];
  int y_lattice = blockIdx.y * blockDim.y + threadIdx.y - (LATTICE_SIZE / 2)  ;

  // evaluate distance for the data point from x/y
  int distance2 = (x_data - x_lattice) * (x_data - x_lattice) + (y_data - y_lattice) * (y_data - y_lattice) ;

  /*/
  // --------------------------
  if ((x_data - x_lattice) * (x_data - x_lattice) < 128*128)  atomicAdd(&debug[1], 1);
  if ((y_data - y_lattice) * (y_data - y_lattice) < 128*128)  atomicAdd(&debug[2], 1);
  if ((distance2 / hst_data.bin) < hst_data.d )  atomicAdd(&debug[3], 1);
  // --------------------------
  /*/



  // evaluate the position in the histogram 
  int pos = ( (blockIdx.y * blockDim.y + threadIdx.y) * hst_data.w + (blockIdx.x * blockDim.x + threadIdx.x) ) * hst_data.d  + \
               (distance2 / hst_data.bin);
  
  if (pos > LATTICE_SIZE * LATTICE_SIZE * HST_SIZE ) return; // check for possible overflows
  
  /*/
  // --------------------------
  atomicAdd(&debug[5], 1);
  // --------------------------
  /*/
  

  // atomic add for the histogram element
  atomicAdd(&hst_data.d_arr[pos], 1);
  
}




// kernel for circles preparation
// __global__ void CirclesKernel(HistogramFrame hst_data, CirclesFrame circles_data) //aggiungere i parametri
__global__ void CirclesKernel(HistogramFrame hst_data, CirclesFrame circles_data, int* debug) //aggiungere i parametri
{
  /*/
  // --------------------------
  atomicAdd(&debug[10], 1);
  // --------------------------
  /*/

  // evaluate the position in the histogram 
  // int pos = (blockIdx.y * blockDim.y + threadIdx.y) * (hst_data.w * hst_data.d)  +  \
  //          (blockIdx.x * blockDim.x + threadIdx.x) * (             hst_data.d)  +  \
  //          (blockIdx.z * blockDim.z + threadIdx.z)  ;
  int pos = ( (blockIdx.y * blockDim.y + threadIdx.y) * hst_data.w + (blockIdx.x * blockDim.x + threadIdx.x) ) * hst_data.d  + \
              (blockIdx.z * blockDim.z + threadIdx.z);


  
  if (pos > LATTICE_SIZE * LATTICE_SIZE * HST_SIZE ) return; // check for possible overflows


  // add a circle if histogram higher than threshold
  if (hst_data.d_arr[pos] > HST_THRESHOLD)
  {
    // insert a new circle
    int idx = atomicAdd(circles_data.d_found,1); // garantisce che l'incremento avvenga un thread alla volta, quindi idx puo' essere usato solo dal thread in esecuzione

    circles_data.d_arr[3 * idx]     =  blockIdx.x * blockDim.x + threadIdx.x - (LATTICE_SIZE / 2)  ;        // x position
    circles_data.d_arr[3 * idx + 1] =  blockIdx.y * blockDim.y + threadIdx.y - (LATTICE_SIZE / 2)  ;        // y position
    circles_data.d_arr[3 * idx + 2] =  __float2int_rn ( sqrt( __int2float_rn( \
                                      (blockIdx.z * blockDim.z + threadIdx.z) * hst_data.bin ) ) );  // radius
  } 

  /*/
  // --------------------------
  atomicAdd(&debug[11], 1);
  // --------------------------
  /*/
}



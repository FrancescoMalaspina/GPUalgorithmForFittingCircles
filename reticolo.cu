#include <stdio.h>
#include <math.h>
#include <iostream>
#include <inttypes.h>
#include <vector>
#include <fstream> // I/O

//struct DataFrame
struct DataFrame
{
  int w;
  int h;
  float* e;
};


//Thread block size
#define BLOCK_SIZE 16 //number of threads for each block
#define GRID_SIZE 128 //dimension of entire space
//warning GRID_SIZE must be multiple of 16
#define BIN_SIZE 4 //size of a hisogram bin
#define TH 30 //threshold for triggering the histogram

__global__ void GridKernel(const DataFrame, DataFrame, unsigned int*); //aggiungere i parametri



//------------------------------------------------------------
//Histogram algorithm HOST CODE
float CircleFit(const DataFrame data, DataFrame circles)
{
  //load data to device memory
  DataFrame d_data;
  d_data.w = data.w;
  d_data.h = data.h;
  size_t size = data.w * data.h * sizeof(float);
  cudaError_t err = cudaMalloc(&d_data.e, size);
  if(err)
    printf("CUDA malloc data DataFrame: %s\n",cudaGetErrorString(err));
  cudaMemcpy(d_data.e, data.e, size, cudaMemcpyHostToDevice);

  //allocate circles
  DataFrame d_circles;
  d_circles.w = circles.w;
  d_circles.h = circles.h;
  size = circles.w * circles.h * sizeof(float);
  err = cudaMalloc(&d_circles.e, size);
  if(err)
    printf("CUDA malloc circles DataFrame: %s\n",cudaGetErrorString(err));

  //allocate variable for counting circles found  
  unsigned int *d_counter;
  unsigned int  h_counter;

  cudaMallocManaged(&d_counter, sizeof(unsigned int));
    if(err)
    printf("CUDA malloc counter variable: %s\n",cudaGetErrorString(err));

  // printf("initialized counter : %u\n", *counter);
  // Hai assegnato al puntatore l'indirizzo di memoria da utilizzare, ma e' memoria della GPU, non puoi accedere da qui
  // ma il valore nella cella puntata non e' inizializzato
  cudaMemset(d_counter, 0, sizeof(unsigned int));
  cudaMemcpy(&h_counter, d_counter, sizeof(unsigned int), cudaMemcpyDeviceToHost);

  printf("initialized counter : %u\n", h_counter);



  // 

  float time;
  cudaEvent_t start,stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  //start time
  cudaEventRecord(start);

   // Define the geometry
  dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
  dim3 dimGrid(GRID_SIZE/ dimBlock.x, GRID_SIZE/ dimBlock.y);
      //warning GRID_SIZE must be multiple of 16

  // Invoke kernel
  GridKernel<<<dimGrid, dimBlock>>>(d_data, d_circles, d_counter);
  err = cudaDeviceSynchronize();

  //stop time
  cudaEventRecord(stop);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&time, start, stop);

  if(err)
    printf("Run kernel: %s\n", cudaGetErrorString(err));

  printf("Time: %3.5f ms\n",time);

  // Read C from device memory
  size = circles.w * circles.h * sizeof(float);
  err = cudaMemcpy(circles.e, d_circles.e, size, cudaMemcpyDeviceToHost);
  if(err)
    printf("Copy circles off of device: %s\n",cudaGetErrorString(err));

  cudaMemcpy(&h_counter, d_counter, sizeof(unsigned int), cudaMemcpyDeviceToHost);
  printf("counter : %u\n", h_counter);


  // Free device memory
  cudaFree(d_data.e);
  cudaFree(d_circles.e);
  cudaFree(d_counter);

  return time;
} //END HOST FUNCTION



//---------------------------------------------------------

// thread aware log function
__device__ void log_msg(const char * message)
{
  printf("%d.%d.%d.%d-%s", blockIdx.x, blockIdx.y, threadIdx.x, threadIdx.y, message);
}

//Device function to calculate distance
__device__ float Distance2(float x, float y, float xd, float yd)
{
  return (x-xd)*(x-xd)+(y-yd)*(y-yd);
}


//---------------------------------------------------------
//Device function to Get Element from data
__device__ float GetElement(const DataFrame D, int row, int col)
{
  if ( (row < D.h ) && (col < D.w ) )
    return D.e[row * D.w + col];
  else 
    return 0.0;
}


//---------------------------------------------------------------
//Device function to Set Element of data
__device__ void SetElement(DataFrame D, int row, int col, float value) 
{
  if ( (row < D.h ) && (col < D.w ) )
   D.e[row * D.w + col] = value;
}


//---------------------------------------------------------
//Device function to fill histo
// __device__ void HistoFill(int x, int y, DataFrame data,int *h, DataFrame circles, unsigned int *counter)
__device__ void HstFill(int x, int y, DataFrame data,DataFrame circles, unsigned int *counter)
{
  const size_t HST_SIZE = GRID_SIZE*GRID_SIZE/BIN_SIZE;
  int 	       hst[HST_SIZE] = {0}; // va inizializzato
  int 	       i, idx;

  // log_id(); printf("HstFill\n");

  // atomicAdd(counter,1);

  for ( i=0; i<data.h; i++)
  {
    // float xd, yd, d;
    float xd, yd;
    int d;
    
    xd = GetElement(data, i, 0);
    yd = GetElement(data, i, 1);
    d  = __float2int_rn(  Distance2( __int2float_rn(x), __int2float_rn(y),xd,yd) ) / BIN_SIZE ;

    if (d < HST_SIZE) hst[d] +=1; // questo if serve per evitare errori di accesso alla memoria nel caso di valori "strani"

    
    // potrebbero lavorare piu' thread sullo stesso elemento 
    // va implementato un meccanismo di locking per evitarlo
    // in piu' questo blocco va eseguito solo al termine del ciclo for, quando hai ormai analizzato tutti i punti
    // altrimenti crei (n - TH)  volte lo stesso cerchio

    /*
    if(hst[bin]>TH)
    {
      idx = atomicAdd(counter,1); // garantisce che l'incremento avvenga un thread alla volta, quindi idx puo' essere usato solo dal thread in esecuzione

      SetElement(circles, idx,0,x);
      SetElement(circles, idx,1,y);
      // float r2=d;
      // SetElement(circles, idx,2,r2);
      SetElement(circles, idx,2,d);
    }
    */
  }


  // dopo aver riempito l'array hst, scandisce tutti gli elementi e aggiunge alla lista dei cerchi il cerchio trovato
  for ( i=0; i < HST_SIZE; i++)
  {
    if(hst[i]>TH)
    {
      idx = atomicAdd(counter,1); // garantisce che l'incremento avvenga un thread alla volta, quindi idx puo' essere usato solo dal thread in esecuzione

      //log_msg(""); printf("counter: %d, x %f, y %f, radius: %f\n", idx, __int2float_rn(x), __int2float_rn(y),  sqrt( __int2float_rn(i*BIN_SIZE) ) );

      SetElement(circles, idx,0, __int2float_rn(x));
      SetElement(circles, idx,1, __int2float_rn(y));
      // float r2=d;
      // SetElement(circles, idx,2,r2);
      SetElement(circles, idx,2, sqrt( __int2float_rn(i*BIN_SIZE) ) );
    }

  }
}




//---------------------------------------------------------
//Histogram algorithm kernel
__global__ void GridKernel(DataFrame data, DataFrame circles,unsigned int *counter)
{
  //x and y indexes
  int x = blockIdx.x * blockDim.x+ threadIdx.x - (GRID_SIZE / 2);
  int y = blockIdx.y * blockDim.y +threadIdx.y - (GRID_SIZE / 2);

//  const size_t histo_size = GRID_SIZE*GRID_SIZE/BIN_SIZE;
//  int histo[histo_size];
//  se con histo non ci fai niente, e' inutile definirla qui e passarla alla funzione
//  in piu' il contenuto non e' inizializzato

//  HistoFill(x,y, data, histo, circles, counter);
  HstFill(x,y, data, circles, counter);


}


//------------------------------------------------------------
//dump dataframe function
void dump(DataFrame m)
{
  for (int i = 0; i< m.h; i++)     // Loop sulle righe
  {
    for (int j = 0; j< m.w; j++)    // Loop sulle colonne
      printf("%3.1f\t", m.e[i*m.w + j]);
    printf("\n");  // A capo di fine riga
  }
  printf("\n");
}



//------------------------------------------------------------
//read floats from file.dat
void readBin(std::vector<float> buf)
{
}



//-------------------------------------------------------------
int main(int argc, char** argv)
{
  std::vector<float> buf;
  std::ifstream in("file.dat",std::ios::binary); //open the input file
  while(!in.eof())
  {
    float w;
    in.read( (char*) &w, sizeof(w));
    buf.push_back(w);
  }

  in.close(); //close the input file .... ormai i dati sono letti, puoi chiudere il file

  DataFrame data, circles;

  data.h = buf.size()>>1;
  data.w = 2;
  data.e = (float*) malloc(data.w * data.h * sizeof(float));

  circles.h = 15;
  circles.w = 3;
  circles.e = (float*) malloc(circles.w * circles.h * sizeof(float));

  data.e = &buf[0]; //copy imported floats from buf to data .... non stai copiando i dati, se fai cosi' non ti serve la memoria che hai allocato prima

  float a = CircleFit(data, circles);

//  dump(data);
  dump(circles);

  // in.close(); //close the input file
  return 0;
}

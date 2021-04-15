#include <vector>
#include <fstream> // I/O
#include <stdio.h>      /* printf, NULL */
#include <math.h>
#include <stdlib.h>     /* srand, rand */
#include <time.h>       /* time */
#include <iostream>

#define GRID_SIZE 128 //dimension of entire space
//warning GRID_SIZE must be multiple of 16

//struct DataFrame
struct DataFrame
{
  int w;
  int h;
  float* e;
};



//read floats from file.dat
std::vector<float> ReadData (const char* name)
{
  std::vector<float> buf;
  std::ifstream in(name, std::ios::binary); //open the input file
  while(!in.eof())
  {
    float w;
    in.read( (char*) &w, sizeof(w));
    buf.push_back(w);
  }
  in.close(); //close the input file
  return buf;
}



//write floats from a vector to file.dat
void WriteData (std::vector<float> buf)
{
  std::ofstream out ("file.dat", std::ios::binary);

  for (int i = 0; i < (buf.size()); i++)
  {
    float a = buf[i];
    out.write( (char*) &a, sizeof(a) );
  }

  out.close();
}



//append floats from a vector to file named: 'name'
void AppendData (std::vector<float> buf, const char* name)
{
  std::ofstream out ( name, std::ios::binary | std::ios::app);

  for (int i = 0; i < (buf.size()); i++)
  {
    float a = buf[i];
    out.write( (char*) &a, sizeof(a) );
  }

  out.close();
}


//funzione che sposta i dati da un DataFrame a un vector
std::vector<float> dataFrame2vec (DataFrame m)
{
  std::vector<float> buf;
  for (int i = 0; i< m.h; i++)     // Loop sulle righe
  {
    for (int j = 0; j< m.w; j++)    // Loop sulle colonne
    {
       buf.push_back( m.e [ i*m.w + j ] );
    }
  }
  return buf;
}


//media valori trovati nel caso ci sia 1 solo cerchio simulato
std::vector<float> media1 (DataFrame m)
{
  std::vector<float> buf;
  for (int j = 0; j< m.w; j++)		// Loop sulle colonne
  {
    float somma = 0;
    for (int i = 0; i< m.h; i++)    	// Loop sulle righe
    {
	somma += m.e[ i*m.w + j ];
    }
    buf.push_back( somma / m.h );
  }
  return buf;
}


//funzioni comuni alle 2 simulazioni
std::vector<float> array2vec (int size, float a[], float b[], float c[])
{
  std::vector<float> buf;
  for(int i=0; i<size; i++)
  {
    buf.push_back(a[i]);
    buf.push_back(b[i]);
    buf.push_back(c[i]);
  }
  return buf;
}

//print a vector
void print(std::vector<float> const &input)
{
    std::cout<<"(";
    for (int i = 0; i < input.size(); i++) {
        std::cout << input.at(i) << ", ";
    }
    std::cout<< ")"<<std::endl;
}

float RandomFloat(float a=0, float b=1) 
{
    float rd = ((float) random()) / (float) RAND_MAX;
    float diff = b - a;
    float r = rd * diff;
    return a + r;
}

float randXpoint(float raggio, float t, float x)
{
  return x + raggio * cos (t);
}

float randYpoint(float raggio, float t, float y)
{
  return y + raggio * sin (t);
}

#include "header.h"
#include <unistd.h>

int main(int argc, char** argv)
{
  // Check command line
  if(argc != 4)
  {
    printf("Usage: %s n(numero di cerchi) m(numero di punti per cerchio) k(seme per srand)\n", argv[0]);
    return 1;
  }

  int n = atoi(argv[1]); //numero di cerchi da simulare
  int m = atoi(argv[2]); //numeri di punti richiesti per ogni cerchio
  int k = atoi(argv[3]); //seme per srand

  float points[n][m][2];

  float xcenter[n] 	= {0};
  float ycenter[n] 	= {0};
  float radii[n] 	= {0};

//  srand(time(NULL) + getpid() );
  srand(k);

  for (int i=0; i<n; i++)
  {
    float r = RandomFloat(0,(GRID_SIZE>>1));
    float xc = RandomFloat(-(GRID_SIZE>>1),(GRID_SIZE>>1));
    float yc = RandomFloat(-(GRID_SIZE>>1),(GRID_SIZE>>1));
    for (int j=0; j<m; j++)
    {
      float theta = RandomFloat(0,2*M_PI);
      points[i][j][0] = randXpoint(r, theta, xc);
      points[i][j][1] = randYpoint(r, theta, yc);
    }
    radii[i] = r;
    xcenter[i] = xc;
    ycenter[i] = yc;
  }

  std::vector<float> buf = array2vec(n, xcenter, ycenter, radii);
  //print(buf);
  AppendData (buf, "simul.dat");

  //output file
  std::ofstream out("file.dat", std::ios::binary);

  //dump da implementare in una funzione
  for (int i = 0; i<n; i++)     // Loop sulle righe
  {
    //printf("Cerchio %d :\t", i);
    for (int j = 0; j< m; j++)    // Loop sulle colonne
      {
      float x = points[i][j][0];
      out.write( (char*) &x, sizeof(x));
      float y = points[i][j][1];
      out.write( (char*) &y, sizeof(y));
      //printf("(%3.1f, %3.1f)\t", x, y);
      }
    //printf("Centro: (%3.1f,%3.1f), raggio: %3.1f\n", xcenter[i], ycenter[i], radii[i]);
    // A capo di fine riga
  }
  //printf("\n");

  out.close();

  //time of now
/*  time_t now= time(NULL);
  char* answer = ctime(&now);
  printf("%s\n %ld\n", answer, now);
 */

  return 0;
}

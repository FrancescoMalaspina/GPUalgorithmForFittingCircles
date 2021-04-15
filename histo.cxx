#include <inttypes.h>
#include "header.h"

//include per ROOT
#include <TApplication.h>
#include <TCanvas.h>
#include <TH1I.h>
#include <TStyle.h>


float Distance2(float x, float y, float xd, float yd)
{
  return (x-xd)*(x-xd)+(y-yd)*(y-yd);
}


void HstFill(float x, float y, TH1I* h, std::vector<float> buf)
{
  int n = (buf.size()>>1);
  float x1, y1;
  for (int i=0; i<n; i++) //calcola la distanza e riempi l'istogramma
  {
    x1=buf[2*i];
    y1=buf[2*i+1];
    float d = Distance2(x, y, x1, y1 );
    h->Fill(d);
  }
}



//NB questa funzione serve per creare un istogramma a canali la cui grandezza
//cresce quadraticamente, da usare nel codice dell'algoritmo di fit
void BinsFill( Float_t* bins) //funzione per riempire l'array dei canali dell'istogramma quadraticamente
{
  for (int i=0; i<(GRID_SIZE+1); i++)
  {
    bins[i]=i*i;
  }
}


int main(int argc, char** argv)
{
  TApplication myApp("App", &argc, argv);
  TCanvas *screen = new TCanvas("screen","Istogramma delle distanze",200,10,900,500);
  screen->Divide(2,1);
  std::vector<float> buf = ReadData ("file.dat");
  Float_t bins[GRID_SIZE+1];
  BinsFill(bins);

  // Creazione di un oggetto della classe TFormula
  TH1I *h = new TH1I("h", "Istogramma con bin di larghezza variabile", GRID_SIZE, bins);
//  TH1I *h = new TH1I("h", "Istogramma delle distanze di un punto casuale",
//                     128, 0.0, 128.0 );
  h->GetXaxis()->SetTitle("Distanza al quadrato degli hit dal punto considerato");
  h->GetYaxis()->SetTitle("numero di hit");
  HstFill( 5, 10, h, buf);

  h->SetFillColor(8);  // Colore dell'istogramma
  gStyle->SetOptStat(11);
  h->Draw();            // Disegno

  myApp.Run();  // Passaggio del controllo a root
  return 0;
}

#include <inttypes.h>
#include "header.h"
//include per ROOT
#include <TApplication.h>
#include <TCanvas.h>
#include <TH1I.h>
#include <TStyle.h>
#include <TMath.h>
#include <TF1.h>

#define FAKE_THR 4

float Distance2(float x1, float x2, float y1, float y2, float r1, float r2)
{
  return ((x1-x2)*(x1-x2))+((y1-y2)*(y1-y2))+((r1-r2)*(r1-r2));
}


unsigned int HstFill(std::vector<float> fit, std::vector<float> sim, TH1I* h)
{
  int n = (fit.size()/3);
  unsigned int c = 0;

  for (int i=0; i<n; i++) //calcola la distanza e riempi l'istogramma
  {
    float x1 = fit[3*i];
    float y1 = fit[3*i+1];
    float r1 = fit[3*i+2];
    float x2 = sim[3*i];
    float y2 = sim[3*i+1];
    float r2 = sim[3*i+2];
    float d = sqrt( Distance2( x1, x2, y1, y2, r1, r2 ));
    if (d > FAKE_THR)
    {
      printf(" attenzione possibile fake: id = %d, fitted x = %3.2f, y = %3.2f, r = %3.2f \n", i, x1, y1, r1 );
      c+=1;
    }
    h->Fill( d );
  }
  return c;
}

unsigned int PartialHstFill(std::vector<float> fit, std::vector<float> sim, TH1I* h, int k)
{
  int n = (fit.size()/3);
  unsigned int c = 0;

  for (int i=0; i<n; i++) //calcola la distanza e riempi l'istogramma
  {
    float k1 = fit[3*i+k];
    float k2 = sim[3*i+k];
    float d = abs(k2 - k1);
    if (d > FAKE_THR)
    {
      printf(" attenzione possibile fake: id = %d, fitted val = %3.1f\n", i, k1);
      c+=1;
    }
    h->Fill( d );
  }
  return c;
}


int main(int argc, char** argv)
{
  TApplication myApp("App", &argc, argv);
  TCanvas *screen = new TCanvas("screen","Distanza tra dati simulati e fittati",200,10,900,500);

  std::vector<float> fit = ReadData ("fit.dat");
  std::vector<float> sim = ReadData ("simul.dat");

  // Creazione di un oggetto della classe TFormula
  TH1I *h = new TH1I("h", "Istogramma dei Residui",
                     100, 0.0, FAKE_THR);
/*
  TH1I *h1 = new TH1I("h1", "Istogramma dei Residui in x, y e r, sovrapposti",
                     100, 0.0, FAKE_THR);
  TH1I *h2 = new TH1I("h2", "Istogramma dei Residui in x, y e r, sovrapposti",
                     100, 0.0, FAKE_THR);
  TH1I *h3 = new TH1I("h3", "Istogramma dei Residui in x, y e r, sovrapposti",
                     100, 0.0, FAKE_THR);*/


  h->GetXaxis()->SetTitle("Residui");
  //h->GetYaxis()->SetTitle("numero di hit");
  unsigned int counter = 0; //contatore numero di fake
  counter = HstFill(fit, sim, h);
/*  counter = PartialHstFill( fit, sim, h1, 0);
  printf("fake x trovati : %d\n", counter );
  counter = PartialHstFill( fit, sim, h2, 1);
  printf("fake y trovati : %d\n", counter );
  counter = PartialHstFill( fit, sim, h3, 2);
  printf("fake r trovati : %d\n", counter );*/

  //fit for poissonian distribution
/*  TF1 *f1 = new TF1("f1","[1]*TMath::Power(([0]/[2]),(x/[2]))*(TMath::Exp(-([0]/[2])))/TMath::Gamma((x/[2])+1)",0,3.0);
  f1->SetParameters(0.3, 9000, 0.05); // you MUST set non-zero initial values for parameters
  h->Fit("f1","R");*/

  h->StatOverflows (kTRUE) ;
  h->SetFillColor(38);  // Colore dell'istogramma
//  gStyle->SetOptStat(0);
  gStyle->SetOptStat(110101111);
//  gStyle->SetOptStat("oenmr");
  gStyle->SetOptFit(2);
/*  h1->SetLineColor(1);
  h1->SetLineWidth(2);
  h1->Draw();            // Disegno
  h2->SetLineColor(kRed);
  h2->SetLineWidth(2);
  h2->Draw("SAME");
  h3->SetLineColor(4);
  h3->SetLineWidth(2);
  h3->Draw("SAME");*/
  h->Draw();

  myApp.Run();  // Passaggio del controllo a root
  return 0;
}


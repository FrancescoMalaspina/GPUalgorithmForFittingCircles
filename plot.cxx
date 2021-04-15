#include <TApplication.h>
#include <TCanvas.h>
#include <TGraph.h>
#include <TAxis.h>
#include <TMultiGraph.h>

#include "header.h"
#include <inttypes.h>


int main(int argc, char** argv)
{
  TApplication myApp("App", &argc, argv);
  TCanvas *c1 = new TCanvas("c1,name","plot di file.dat",200,10,1000,500);
  c1->Divide(2,1);
  std::vector<float> buf = ReadData ("file1.dat");

  Int_t n = (buf.size()>>1);
  float x1[n], y1[n];
  for (Int_t i=0; i<n; i++)
  {
    x1[i]=buf[2*i];
    y1[i]=buf[2*i+1];
  }
  TGraph *gr1 = new TGraph (n, x1, y1);
  TMultiGraph *mg = new TMultiGraph();
  mg -> Add(gr1);

  //gr1 drawing options
  gr1->SetMarkerStyle(22);
  gr1->SetMarkerColor(2);
  gr1->SetMarkerSize(0.85);
  gr1->SetTitle("1 cerchio 75 punti, con smearing");
  gr1->SetMinimum(10);
  gr1->SetMaximum(90);
  gr1->GetXaxis()->SetLimits(-50,30);
  c1->cd(1);
  gr1->Draw("AP");

   // codice per un secondo grafico da affiancare al primo
  buf = ReadData ("file.dat");
  n = (buf.size()>>1);
  float x2[n], y2[n];
  for (Int_t i=0; i<n; i++)
  {
    x2[i]=buf[2*i];
    y2[i]=buf[2*i+1];
  }
  TGraph *gr2 = new TGraph (n, x2, y2);

  //gr2 drawing options
  gr2->SetMarkerStyle(22);
  gr2->SetMarkerColor(1);
  gr2->SetMarkerSize(0.85);
  gr2->SetTitle("2 cerchi da 75 punti, smearing + fondo");
  gr2->SetMinimum(-GRID_SIZE);
  gr2->SetMaximum(GRID_SIZE);
  gr2->GetXaxis()->SetLimits(-GRID_SIZE,GRID_SIZE);
  c1->cd(2);
  gr2->Draw("AP");

  myApp.Run();  // Passaggio del controllo a root
  return 0;
}

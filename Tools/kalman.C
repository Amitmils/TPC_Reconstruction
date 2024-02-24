#include "TCanvas.h"
#include "TGraph.h"
#include <iostream>

#include <vector>
#include <TMath.h>
#include <math.h>

#include <fstream>
#include <sstream>
#include <iomanip>
#include <fstream>
#include <string>

#include <TMatrixD.h>

#include "TString.h"
#include "TFile.h"
#include "TTree.h"
#include "TClonesArray.h"

#include <algorithm>
#include <limits>


using namespace std;


void Get_Energy(Double_t M,Double_t IZ,Double_t BRHO,Double_t &E);


std::vector<AtHitCluster> fHitClusterArray; //< Clusterized hits container

std::vector<AtHitCluster> *GetHitClusterArray() {


         return &fHitClusterArray;


 }



void  Get_Energy(Double_t M,Double_t IZ,Double_t BRHO,Double_t &E){

  //Energy per nucleon

  Float_t  AM=931.5;
  Float_t X=BRHO/0.1439*IZ/M;
  X=pow(X,2);
  X=2.*AM*X;
  X=X+pow(AM,2);
  E=TMath::Sqrt(X)-AM;


  }



void energyloss(std::string file, TGraph* eLossCurve){

	std::string eLossFileName_;
	eLossFileName_ = file;

        Float_t ener = 0;
        Float_t stp = 0;

	std::ifstream elossfile;


	try {

		elossfile.open(eLossFileName_,ios::in);

		if ((elossfile.peek() == std::ifstream::traits_type::eof())) {

      			std::cout << " Error: Energy loss file not found! Exiting..."<< "\n";
      			std::exit(EXIT_FAILURE);
    		}


	//std::cout << " Processing energy loss data file " << eLossFileName_ << "\n";
	std::string line;

	for (auto i = 0; i < 2; ++i) {
        	std::getline(elossfile,line);
        //	std::cout << line << "\n";
    	}


	while (std::getline(elossfile, line)) {
		std::istringstream data(line);
		data >> ener >> stp ;
      		eLossCurve->SetPoint(eLossCurve->GetN(),ener,stp);
    	}

  } catch (...) {
  }

}


Double_t GetEnergy(Double_t vx, Double_t vy, Double_t vz){
    Double_t vv, bet, gamma1, Energy;
    Double_t c = 3.0*TMath::Power(10, 8);

    vv = TMath::Sqrt(TMath::Power((vx),2)+TMath::Power((vy),2)+TMath::Power((vz),2));
    bet = vv/c;
//cout<<bet<<endl;
    gamma1 = TMath::Sqrt(1/(1-TMath::Power(bet,2)));
    Energy = (gamma1-1)*931.494028;
    return Energy;
}


Double_t StoppingPower(Double_t Energy){
    Double_t stpSim;
    TGraph* eLossCurve = new TGraph();
    energyloss("stpHydrogen.txt", eLossCurve);
    Double_t  gasMediumDensity_ = 8.988e-5;   //g/cm3 at 1 bar

    stpSim = eLossCurve->Eval(Energy);
    stpSim*=1.6021773349e-13;
    stpSim*= gasMediumDensity_*100;
    stpSim/= 1.6726*TMath::Power(10,-27);


    return stpSim;
}

// Function that calculates the derivative of the position and velocity of the particle

Double_t  dxdt(double t,double vx, double vy, Double_t vz){

        Double_t Energy = GetEnergy(vx, vy, vz);
        Double_t st = StoppingPower(Energy);
	Double_t Ex,Ey,Ez,f1;            	//Electric field in V/m. 
	Double_t Bx,By,Bz;	      	// magnetic field in Tesla
	Double_t rr,az,po;
	Double_t q = 1.6022*TMath::Power(10,-19); 	//charge of the particle(proton) in (C)
	Double_t m = 1.6726*TMath::Power(10,-27); 	// mass of the particle in kg
	Double_t B= -3.0;                 // Applied magnetic field (T).
        Double_t E=TMath::Cos((q*B)/m) * 500;   // Applied Electric field(V/m).
        Bx = 0;
	By = 0;                	// magnetic field in x and y  direction in Tesla.
	Bz = B;          // magnetic field in the z direction in Tesla.
        Ex = 0;
	Ey = 0;			// Electric field in the x and  direction in V/m.
	Ez = -E;                        // Electric field in the z direction. 


        rr = TMath::Sqrt(TMath::Power(vx,2)+TMath::Power(vy,2)+TMath::Power(vz,2));
        az = TMath::ATan2(vy,vx);
        po = TMath::ACos(vz/rr);

	f1 =  q/m * (Ex + vy*Bz-vz*By) - st*TMath::Sin(po)*TMath::Cos(az); //-s*TMath::Sin(po)*TMath::Cos(az) ;                                    //dxdt with energyloss compensation.

        double bro = Bz * rr / TMath::Sin(po)/ 1000.0;
//cout<<Energy<<endl;
//cout<<bro<<endl;
	//std::cout<<Ex <<" "<< vy<<" "<<vz << " " <<By<< " "<< f1<<std::endl; 
        return f1;

        }

double  dydt(double t,double vx, double vy, Double_t vz){
Double_t Energy = GetEnergy(vx, vy, vz);
        Double_t st = StoppingPower(Energy);
	Double_t Ex,Ey,Ez,f2;              //Electric field in V/m. 
        Double_t Bx,By,Bz;              // magnetic field in Tesla
        Double_t q = 1.6022*TMath::Power(10,-19);   //charge of the particle in C
        Double_t B= -3.0;                 // Applied magnetic field.
        Double_t rr,az,po;
        Double_t m = 1.6726*TMath::Power(10,-27);  // mass of the particle in kg
	Double_t E = TMath::Cos((q*B)/m) * 500 ;
        Bx = 0;                      // magnetic field in x and y  direction in Tesla.
	By = 0;
        Bz = B;          // magnetic field in the z direction in Tesla.
        Ex = 0;                      // Electric field in the x and  direction in V/m.
	Ey = 0;
        Ez = -E;                        // Electric field in the z direction.

	rr = TMath::Sqrt(TMath::Power(vx,2)+TMath::Power(vy,2)+TMath::Power(vz,2));
        az = TMath::ATan(vy/vx);
        po = TMath::ACos(vz/rr);


       f2 =  q/m * (Ey + vz*Bx - vx*Bz) - st*TMath::Sin(po)*TMath::Sin(az); 

        double bro = Bz * rr / TMath::Sin(po)/ 1000.0;
//cout<<bro<<endl;
//cout<<Energy<<endl;
	//std::cout<<f2<<std::endl;
        return f2;
        }

double  dzdt(double t,double vx, double vy,Double_t vz){
Double_t Energy = GetEnergy(vx, vy, vz);
        Double_t st = StoppingPower(Energy);
        Double_t Ex,Ey,Ez,f3;              //Electric field in V/m. 
        Double_t Bx,By,Bz;              // magnetic field in Tesla
        Double_t q = 1.6022*pow(10,-19);   //charge of the particle in eV
        Double_t B= -3.0;                 // Applied magnetic field.
        Double_t rr,az,po;
        Double_t m = 1.6726*TMath::Power(10,-27);  // mass of the particle in kg
	Double_t E = TMath::Cos((q*B)/m) * 500 ; 
        Bx = 0;                      // magnetic field in x and y  direction in Tesla.
	By = 0;
        Bz = B;          // magnetic field in the z direction in Tesla.
        Ex = 0;                      // Electric field in the x and  direction in V/m. 
	Ey = 0;
        Ez = -E;                        // Electric field in the z direction.

	rr = TMath::Sqrt(TMath::Power(vx,2)+TMath::Power(vy,2)+TMath::Power(vz,2));
        az = TMath::ATan(vy/vx);
	//std::cout<<az<<endl;
        po = TMath::ACos(vz/rr);


        f3 =  q/m * (Ez + vx*By - vy*Bx)- st*TMath::Cos(po);
        double bro = Bz * rr / TMath::Sin(po)/ 1000.0;
//cout<<bro<<endl;
//cout<<Energy<<endl<<endl;
	//std::cout <<r<< " " << TMath::Power(vy,2) + TMath::Power(vx,2) + TMath::Power(vz,2)<< " " << TMath::Power(vz,2) <<  std::endl; 
        return f3;
        }

//  Functions for positions.

Double_t fx(Double_t t, Double_t vx){


       Double_t f1x =  vx;
        return f1x;


        }

Double_t fy(Double_t t,Double_t vy){


        Double_t f2y = vy;
        return f2y;


        }

Double_t fz(Double_t t,Double_t vz){

        Double_t f3z =  vz;
        return f3z;

        }

// Define a functions to calculate the Jacobian matrix of the equations.
void Jacobi_matrice(TMatrixD &Jacobi_matrix){

 // Define the size of the matrix
        Int_t rows = 6; // the number of rows. In my case I have a 6*6 matrices for the state vectors
        Int_t cols = 6; // the number of cols.
        Double_t h = TMath::Power(10,-10) ; //in seconds.

        // Define the jacobi matrix to hold state vectors
        Double_t Ex,Ey,Ez,f1;                   //Electric field in V/m.
        Double_t Bx,By,Bz;              // magnetic field in Tesla
        Double_t rr,az,po;
        Double_t q = 1.6022*TMath::Power(10,-19);       //charge of the particle(proton) in (C)
        Double_t m = 1.6726*TMath::Power(10,-27);       // mass of the particle in kg
        Double_t B=-3.0;                 // Applied magnetic field (T).
        Double_t E=TMath::Cos((q*B)/m) * 500;   // Applied Electric field(V/m).
        Bx = 0;
        By = 0;                 // magnetic field in x and y  direction in Tesla.
        Bz = B;          // magnetic field in the z direction in Tesla.
        Ex = 0;
        Ey = 0;                 // Electric field in the x and  direction in V/m.
        Ez = -E;


        Jacobi_matrix.ResizeTo(rows,cols);
        Jacobi_matrix.Zero();

        Jacobi_matrix(0,3) = 1*h;

        Jacobi_matrix(1,4) = 1*h;

        Jacobi_matrix(2,5) = 1*h;


        Jacobi_matrix(3,4) = q/m*(Ex+Bz)*h;
        Jacobi_matrix(3,5) = q/m *(Ex-By)*h;


        Jacobi_matrix(4,3) = q/m * (Ey-Bz)*h;
        Jacobi_matrix(4,5) = q/m*(Ey+Bx)*h;


        Jacobi_matrix(5,3) = q/m * (Ez+By)*h;
        Jacobi_matrix(5,4) = q/m * (Ez-Bx)*h;




}

// Obtains last Z value from a hitClusterArray
Double_t LastZValue(std::vector<AtHitCluster> *pHitClusterArray)
{
    int lastClusterIndex = pHitClusterArray->size() - 1;
    AtHitCluster lastCluster;

    lastCluster = pHitClusterArray->at(lastClusterIndex);
    auto lastPositionCluster = lastCluster.GetPosition();
    Double_t zPos = lastPositionCluster.Z();

    return zPos;
}

/// Obtains the index of the maximum value in a vector of doubles
Int_t GetIndexOfMaxValue(std::vector<Double_t> values)
{
    Int_t index = std::distance(values.begin(), std::max_element(values.begin(), values.end()));

    return index;
}



//////////////////////////////////////////////////////////////////////////////
/// Selects the closer track to the vertex, discarding the beam track.
/// Vector with one track are accepted if the theta angle is greater than 20º.
/// If the track does not fulfill the requirements a dummy track is generated
///    with a default track ID of -1.
AtTrack TrackSelector(std::vector<AtTrack> trackCandidates, std::vector<Double_t> thetaValues)
{
    Int_t numberOfTracks = trackCandidates.size(); // Gets number of tracks
  //  cout<<"Track candidates: "<<numberOfTracks<<endl;

    //Discarding vectors with no tracks
    if ( numberOfTracks >= 1)
    {
        // One track with a theta angle > 20º is acceptable
        if (numberOfTracks == 1)
        {
            if(180-thetaValues[0]*TMath::RadToDeg()>20.)
            {
               AtTrack selectedTrack = trackCandidates[0];
               return selectedTrack;
            } else {AtTrack nullTrack; return nullTrack;}
        }
        else
        {
            Int_t thetaIndex = GetIndexOfMaxValue(thetaValues);// This corresponds to the max value of theta in rad (the minimum value in  degrees after the following conversion: 180-theta)
            trackCandidates.erase(trackCandidates.begin() + thetaIndex);
            std::vector<Double_t> zValues = {};

            for (auto track = trackCandidates.begin(); track != trackCandidates.end(); ++track) // Loop over track candidatess
            {
                auto pHitClusterArray = track->GetHitClusterArray();
                zValues.push_back(LastZValue(pHitClusterArray)); // Get Max Z value of the Clusters
            }

            Int_t trackIndex = GetIndexOfMaxValue(zValues);
            AtTrack selectedTrack = trackCandidates[trackIndex];
            return selectedTrack;
        }
    } else {AtTrack nullTrack; return nullTrack;}
}


////////////////////////////////////////////////////////////////////////
/// Calculates the phi angle between the first [1] and second [2] points
///   of a given track.
/// The x and y values taken for the calculus are independent of the
///   quadrant where the cluster points are located, it only matters the
///   relative position between them.
/// These values are later used in the TMath::ATan2(y, x) function that
///   returns a positive value in radians for the 1st and 2nd quadrants,
///   and a negative value in radians for the 3rd and 4th quadrants.
/// A conversion to degrees is performed and a final correction for the
///   negative values takes place so the angle is always referred to the
///   same point.
Double_t GetPhiAngle(AtTrack goodTrack)
{
    AtHitCluster firstCluster, secondCluster;
    auto hitClusterArray = goodTrack.GetHitClusterArray();

    firstCluster = hitClusterArray->at(0);
    secondCluster = hitClusterArray->at(1);

    auto firstPosition = firstCluster.GetPosition();
    auto secondPosition = secondCluster.GetPosition();
    Double_t yValue = secondPosition.Y() - firstPosition.Y();
    Double_t xValue = secondPosition.X() - firstPosition.X();

    Double_t phiAngleRad = TMath::ATan2(yValue,xValue);
    Double_t phiAngleDeg;
    if (phiAngleRad<0)
    {
        phiAngleDeg = 360 + phiAngleRad*TMath::RadToDeg();
        return phiAngleDeg;
    }
    else
    {
        phiAngleDeg = phiAngleRad*TMath::RadToDeg();
        return phiAngleDeg;
    }
}



// Define a functions to calculate the process noise matrix Q .
//this is a 6*6 matrice for my case.

void generateGaussianNoise(TMatrixD &Q)
{
    // Define the size of the matrix
    Int_t rows = 6; // the number of rows. In my case I have a 6*6 matrices for the state vectors
    Int_t cols = 6; // the number of cols.

    // Define the matrix to hold process noise.
    Q.ResizeTo(rows, cols);
    Q.Zero();


}



void generateMeasurementNoise(TMatrixD& R)
{
    // Define the size of the matrix
    Int_t rows = 2;
    Int_t cols = 2;

    // Define the matrix to hold measurement noise
    R.ResizeTo(rows, cols);
    R.Zero();

    R(0,0) = 1;

    R(1,1) = 1;

    //R(2,2) = 1;

}



// Define a functions to calculate the Initial Covariance P .
//this is a 6*6 matrice for my case.
void Ini_P(TMatrixD &P){


 // Define the size of the matrix
        Int_t rows = 6; // the number of rows. In my case I have a 6*6 matrices for the state vectors
        Int_t cols = 6; // the number of cols.

        // Define the noise matrix to hold process noise.
        P.ResizeTo(rows,cols);
        P.Zero();


        // Define the  matrixes to hold covariance matrix.

        P(0,0) = 0.01;                    //in meters

        P(1,1) = 0.01;

        P(2,2) = 0.01;

        P(3,3) = 0.01;

        P(4,4) = 0.01;

        P(5,5) = 0.01;



}


//Define the Identity matrix
void I_matrix(TMatrixD &I){


 // Define the size of the matrix
        Int_t rows = 6; // the number of rows. In my case I have a 6*6 matrices for the state vectors
        Int_t cols = 6; // the number of cols.

        // Define the noise matrix to hold process noise.
        I.ResizeTo(rows,cols);
        I.Zero();


        // Define the  matrixes to hold Identity matrix.

        I(0,0) = 1;

        I(1,1) = 1;

        I(2,2) = 1;

        I(3,3) = 1;

        I(4,4) = 1;

        I(5,5) = 1;



}




// Function to calculate the virtual plane parameters
// Plane equation: ax + by + cz + d = 0
// Inputs:
//    hitPos: Hit position
//    px, py, pz: Track parameters(direction of the track)
// Output:
//     d: Virtual plane parameters
Double_t GetVirtualPlane(Double_t x, Double_t y, Double_t z, Double_t nx, Double_t ny, Double_t nz) {
         Double_t a = nx;
         Double_t b = ny;
         Double_t c = nz;
         Double_t d = -(a * x + b * y + c * z);
         return d;
}


/*
void GetPOCA(TVector3 direction,Double_t x1, Double_t y1, Double_t z1, Double_t px, Double_t py, Double_t pz, Double_t& xi,Double_t& yi,Double_t& zi) {

    TVector3 linepoints(x1, y1, z1);
    TVector3 trackdir(direction.X(), direction.Y(), direction.Z());

     //Double_t d = GetVirtualPlane(x1,y1,z1,px,py,pz);
     TVector3 normalplane(px,py,pz);

     Double_t dot1 = normalplane.Dot(trackdir);             // a*Vx + b*Vy + c*Vz
     Double_t dot2 = normalplane.Dot(linepoints);             // a*x1 + b*y1 + c*z1

     Double_t t = -(dot2 + d) / dot1;                       // calculate the intersection
     std::cout<<"the value of t: " << t<<endl<<endl;
     xi = x1+t*px;
     yi = y1+t*py;
     zi = z1+t*pz;
    // std::cout<<"the value of x,y,z: " <<xi << " " << yi << endl<<endl;

}
*/


//define the predicted state.
void statepred(TMatrixD &Initial_state,Double_t x1, Double_t y1, Double_t z1, Double_t px, Double_t py, Double_t pz){

 // Define the initial conditions
         Initial_state(0,0) = x1;      // initial x position   unit in m.
         Initial_state(1,0) = y1;      // initial y position  unit in m
         Initial_state(2,0) = z1;      // initial z position  unit in m
         Initial_state(3,0) = px;     // initial x momentum unit in kg m/s
         Initial_state(4,0) = py;     // initial y momentum  unit in kg m/s
         Initial_state(5,0) = pz;     // initial z momentum unit in kg m/s.

}



void GetMeasurementMatrix(TMatrixD& H_k) {
    // Define the size of the matrix
    Int_t rows = 2;
    Int_t cols = 6;

    // Define the matrix to hold the measurement matrix
    H_k.ResizeTo(rows, cols);
    H_k.Zero();

    // Set the values of the measurement matrix
    H_k(0, 0) = 1;
    H_k(1, 1) = 1;
    //H_k(2, 2) = 1;

}



void kalman(){

         TH2F *rx_vs_ry = new TH2F("rx_vs_ry", "rungex_vs_rungey", 720, 0, -3, 1000, 0, 2.0);
         TH2F *kx_vs_ky = new TH2F("kx_vs_ky", "RK4", 720, 0, -3, 100, 0, 2.0);

         TH2F *vx_vs_vy = new TH2F("vx_vs_rvy", "rungevx_vs_rungevy", 720, 0, -3, 1000, 0, 2.0);
         TH2F *Intx_vs_Inty = new TH2F("Intx_vs_Inty", "Intersectionx_vs_Intersectiony", 720, 0, -3, 1000, 0, 2.0);
         TH2F *propagatorx_vs_propagatory = new TH2F("propagatorx_vs_propagatory", "propagatorx_vs_propagatory", 720, 0, -3, 1000, 0, 2.0);
  //       TH2F *kx_vs_ky = new TH2F("kx_vs_ky", "kalmanx_vs_kalmany", 720, 0, -3, 100, 0, 2.0);

        TH3F *R_projection = new TH3F("R_projection", "runge_projection", 720, 5.0, -3, 100, 0, 10.0,100, -5, 10.0);
        TH3F *F_projection = new TH3F("F_projection", "propagator_projection", 720, 5.0, -3, 100, 0, 10.0,100, -5, 10.0);
        TH3F *Intersection = new TH3F("Intersection", "Intersection", 720, 5.0, -3, 100, 0, 10.0,100, -5, 10.0);



         TCanvas *c1 = new TCanvas("c1", "My Graph", 800, 600);
         c1->Divide(2, 2);
         c1->Draw();


 	//TCanvas *c1 = new TCanvas("c1","Particle in a magnetic field",500,600);
        //c1->Divide(2,2);

        const Int_t n = 1000;

	Double_t t[n],vx[n],vy[n],vz[n];
	Double_t x[n],y[n],z[n],z0,zf;

	Double_t k1x[n],k2x[n],k3x[n],k4x[n];
        Double_t k1y[n],k2y[n],k3y[n],k4y[n];
	Double_t k1z[n],k2z[n],k3z[n],k4z[n];

	Double_t k1vx[n],k2vx[n],k3vx[n],k4vx[n];
        Double_t k1vy[n],k2vy[n],k3vy[n],k4vy[n];
        Double_t k1vz[n],k2vz[n],k3vz[n],k4vz[n];

       // Define the initial conditions
         x[0] = 0;      // initial x position   unit in m.
         y[0] = 0;      // initial y position  unit in m
         z[0] = 0;      // initial z position  unit in m
         vx[0] = 10e6;     // initial x velocity unit in m/s
         vy[0] = 10e6;     // initial y velocity unit in m/s
         vz[0] = 10e6;     // initial z velocity unit in m/s.


        // Define the time step

        Double_t h = TMath::Power(10,-10) ; //in seconds.



	//Graph to evaluate Energy loss
	//Double_t  gasMediumDensity_ = 8.375e-5;   //g/cm3
	Double_t p[n],s[n],vv[n];
	TGraph* eLossCurve = new TGraph();

       energyloss("stpHydrogen.txt", eLossCurve);

	//eLossCurve->Draw();  // Plotting energy vs stopping power

	//std::cout<<x[0] << " " << y[0] << " " << z[0] << std::endl;
	//Start Rk4.


	//for(Int_t i=0;i <n-1;  i++){
        Double_t Energy = GetEnergy(vx[0],vy[0],vz[0]);
        Int_t i = 0;

/*----------------------------------------*/
cout<<endl;
cout<<endl;
cout<<"+----------------+"<<endl;
cout<<" Running kalman.C"<<endl;
cout<<"+----------------+"<<endl;

//cout<<"Initial Energy value: "<< Energy << " MeV" <<endl;
//cout<<endl;
//cout<<endl;
/*----------------------------------------*/



        while (Energy>0.1){
	   k1x[i] = fx(t[i],vx[i]);
           k1y[i] = fy(t[i],vy[i]);
           k1z[i] = fz(t[i],vz[i]);

	   k1vx[i] = dxdt(t[i],vx[i],vy[i],vz[i]);
           k1vy[i] = dydt(t[i],vx[i],vy[i],vz[i]);
           k1vz[i] = dzdt(t[i],vx[i],vy[i],vz[i]);

           k2x[i] = fx(t[i]+h*0.5,vx[i]+0.5*h*k1x[i]);
           k2y[i] = fy(t[i]+h*0.5,vy[i]+0.5*h*k1y[i]);
           k2z[i] = fz(t[i]+h*0.5,vz[i]+0.5*h*k1z[i]);

           k2vx[i] = dxdt(t[i]+h*0.5,vx[i]+k1vx[i]*0.5*h,vy[i]+k1vy[i]*h*0.5,vz[i]+k1vz[i]*h*0.5);
           k2vy[i] = dydt(t[i]+h*0.5,vx[i]+k1vx[i]*h*0.5, vy[i]+k1vy[i]*h*0.5,vz[i]+k1vz[i]*h*0.5);
           k2vz[i] = dzdt(t[i]+h*0.5,vx[i]+k1vx[i]*h*0.5, vy[i]+k1vy[i]*h*0.5, vz[i]+k1vz[i]*h*0.5);


	   k3x[i] = fx(t[i]+h*0.5,vx[i]+0.5*h*k2x[i]);
           k3y[i] = fy(t[i]+h*0.5,vy[i]+0.5*h*k2y[i]);
           k3z[i] = fz(t[i]+h*0.5,vz[i]+0.5*h*k2z[i]);

           k3vx[i] = dxdt(t[i]+h*0.5,vx[i]+k2vx[i]*0.5*h,vy[i]+k2vy[i]*h*0.5,vz[i] + k2vz[i]*h*0.5);
           k3vy[i] = dydt(t[i]+h*0.5,vx[i]+k2vx[i]*h*0.5, vy[i]+k2vy[i]*h*0.5,vz[i]+k2vz[i]*h*0.5);
           k3vz[i] = dzdt(t[i]+h*0.5,vx[i]+k2vx[i]*h*0.5, vy[i]+k2vy[i]*h*0.5, vz[i]+k2vz[i]*h*0.5);

           k4x[i] = fx(t[i]+h,vx[i]+k3x[i]*h);
           k4y[i] = fy(t[i]+h,vy[i]+k3y[i]*h);
           k4z[i] = fz(t[i]+h,vz[i]+k3z[i]*h);

           k4vx[i] = dxdt(t[i]+h,vx[i]+k3vx[i]*h,vy[i]+k3vy[i]*h,vz[i]+k3vz[i]*h);
           k4vy[i] = dydt(t[i]+h,vx[i]+k3vx[i]*h,vy[i]+k3vy[i]*h,vz[i]+k3vz[i]*h);
           k4vz[i] = dzdt(t[i]+h,vx[i]+k3vx[i]*h,vy[i]+k3vy[i]*h,vz[i]+k3vz[i]*h);


          vx[i+1] = vx[i] + h/6 *( k1vx[i] + 2*k2vx[i] + 2*k3vx[i] + k4vx[i]);
          vy[i+1]  = vy[i] + h/6 *( k1vy[i] + 2*k2vy[i] + 2*k3vy[i] + k4vy[i]);
	  vz[i+1]  = vz[i] + h/6 *( k1vz[i] + 2*k2vz[i] + 2*k3vz[i] + k4vz[i]);

	  x[i+1] = x[i] + h/6 *( k1x[i] + 2*k2x[i] + 2*k3x[i] + k4x[i]);
          y[i+1]  = y[i] + h/6 *( k1y[i] + 2*k2y[i] + 2*k3y[i] + k4y[i]);
          z[i+1]  = z[i] + h/6 *( k1z[i] + 2*k2z[i] + 2*k3z[i] + k4z[i]);

          t[i+1]= t[i] + h;

	//  std::cout<<k1vx[i]<< " "<<k1vy[i]<<" " <<k1vz[i] << " " << h << endl;

          vx[i] = vx[i+1];
	  vz[i] = vz[i+1];
          vy[i] = vy[i+1];

          x[i] = x[i+1];
          z[i] = z[i+1];
          y[i] = y[i+1];

          t[i] = t[i+1];

 //        std::cout<<vx[i]<<" "<<vy[i]<<" "<<vz[i]<<endl;

        rx_vs_ry->Fill(x[i],y[i]);
        R_projection->Fill(x[i],y[i],z[i]);

        Energy = GetEnergy(vx[i],vy[i],vz[i]);
        i+=1;
        }

        // Define the size of the matrix
        Int_t rows = 6; // the number of state variables
        Int_t cols = i; // the number of steptime.

	// Define matrix to hold state vectors
        TMatrixD  state_matrix(rows,cols);
        Double_t m = 1.6726*TMath::Power(10,-27);       // mass of the particle in kg


        // Fill in matrix with state vectors
        for (int j = 0; j < i; j++) {
            state_matrix(0,j) = x[j];
            state_matrix(1,j) = y[j];
            state_matrix(2,j) = z[j];
            state_matrix(3,j) = vx[j];
            state_matrix(4,j) = vy[j];
            state_matrix(5,j) = vz[j];
        }


	TMatrixD Jacobi_matrix(6,6);
	Jacobi_matrice(Jacobi_matrix);
        //Jacobi_matrix.Print();


       //Identity matrix.
       TMatrixD I(6,6);
       I_matrix(I);
       //I.Print();

        // Calculate propagator matrix using intermediate matrices
        // Initialize F as identity matrix
        TMatrixD F(6,6);
        F.UnitMatrix();
        // Compute F1, F2, F3, F4
        Double_t dt_1 = 1e-10;
	TMatrixD IplusFi (6,6);
        IplusFi.Zero();
        TMatrixD Fi(6,6);

        for (int j = 0; j < 4; j++) {
             IplusFi = F* dt_1;
             Fi  = (1e-10)*Jacobi_matrix*IplusFi;
             F     += Fi *((j==0 || j==3) ?  1.0/6.0 : 1.0/3.0);
          //F.Print();
        }

        // Define matrix to hold time derivatives of state vectors
        TMatrixD  state_dot_matrix(rows,cols-1);

     // Calculate time derivatives of state vectors
        for (int k = 0; k < cols; k++) {
    // Extract state vector at time t.
            TMatrixD state_vector(6,1);
            for (int j = 0; j < 6; j++) {
                state_vector(j,0) = state_matrix(j,k);
            }

    // Multiply propagator matrix with state vector to get time derivative of state vector
            TMatrixD state_dot_vector = F * state_vector;
            propagatorx_vs_propagatory->Fill(state_dot_vector(0,0), state_dot_vector(1,0));
            F_projection->Fill(state_dot_vector(0,0),state_dot_vector(1,0),state_dot_vector(2,0));
           // state_dot_vector.Print();
        }

/*
c1->cd(1);
        rx_vs_ry->Draw();
        rx_vs_ry->SetMarkerStyle(21);
        rx_vs_ry->SetMarkerSize(0.3);
        c1->cd(2);
        propagatorx_vs_propagatory->Draw();
        propagatorx_vs_propagatory->SetMarkerStyle(21);
        propagatorx_vs_propagatory->SetMarkerSize(0.3);
        c1->cd(3);
        R_projection->Draw();
        R_projection->SetMarkerStyle(21);
        R_projection->SetMarkerSize(0.3);
        c1->cd(4);
        F_projection->Draw();
        F_projection->SetMarkerStyle(21);
        F_projection->SetMarkerSize(0.3);


*/


        TCanvas *c2 = new TCanvas("c2", "My Graph", 800, 600);
        c2->Divide(2, 2);
        c2->Draw();
        TH2F *angle_vs_energy = new TH2F("angle_vs_energy", "angle_vs_energy", 720, 0, 179, 1000, 0, 100.0);
        TH2F *X_vs_Y = new TH2F("X_vs_Y", "PROPAGATOR", 720, 0, -3, 100, 0, 2.0);
        TH2F *predictedx_vs_predictedy = new TH2F("predictedx_vs_predictedy", "predicted state", 720, 0, -3, 1000, 0, 2.0);
        TH1F *phi_pattern = new TH1F("phi_pattern", "phi_pattern", 1440, -50, 400);
        TH2F *angle_vs_energy_pattern =new TH2F("angle_vs_energy_pattern", "angle_vs_energy_pattern", 720, 0, 179, 1000, 0, 100.0);


 //to read the data file.

/*----------------------------------------*/
cout<<endl;
cout<<endl;
cout<<"+----------------+"<<endl;
cout<<" Testing the kalman Filter "<<endl;
cout<<"+----------------+"<<endl;

//cout<<"Initial Energy value: "<< Energy << " MeV" <<endl;
//cout<<endl;
//cout<<endl;
/*----------------------------------------*/



        FairRunAna *run = new FairRunAna();
        std::vector<TString> files = {"data/output_digi"};
        Int_t nEvents = 100;
        Int_t first_Event = 0;
        Int_t last_Event = nEvents;
       // TMatrixD plane1 = ();
        //read_file(files, nEvents);

        /*-----------
         Fist loop. Loops over all
         the files that need to be analyze
        -----------*/
        for (auto iFile = 0; iFile < files.size(); ++iFile) {

            TString mcFileNameHead = files[iFile];
            TString mcFileNameTail = ".root";
            TString mcFileName = mcFileNameHead + mcFileNameTail;
            std::cout << " Analysis of simulation file  " << mcFileName << endl;

            TFile *file = new TFile(mcFileName.Data(), "READ");
            TTree *tree = (TTree *)file->Get("cbmsim");

            tree = (TTree *)file->Get("cbmsim");
            TTreeReader Reader1("cbmsim", file);
            TTreeReaderValue<TClonesArray> eventArray(Reader1, "AtPatternEvent");

            ROOT::Math::XYZPoint firstPosition;
            ROOT::Math::XYZPoint secondPosition;


            /*----------
            Looping over all the events in
            each of the root files
            -----------*/
            for (Int_t i = first_Event; i < last_Event; i++) {

                std::vector<AtTrack> availableTracks = {}; // Tracks founded in the current event
                std::vector<Double_t> thetaValues = {}; // Theta values for each track founded in the current event

                Reader1.Next();

                AtPatternEvent *patternEvent = (AtPatternEvent *)eventArray->At(0);

                if (patternEvent) {
                   std::vector<AtTrack> &patternTrackCand = patternEvent->GetTrackCand();

                   for (auto track : patternTrackCand) {
                       if (track.GetHitClusterArray()->size() < 5) {
                           //std::cout <<track.GetTrackID() << endl;
                          continue;
                       }

                       Double_t theta = track.GetGeoTheta();
                       Double_t rad   = track.GetGeoRadius();
                       Double_t B_f = 3.0;                         //in Tesla.
                       Double_t brho = B_f * rad / TMath::Sin(theta) / 1000.0;     //in Tm.
                       Double_t ener = 0;
                       Double_t Am = 1.007; // atomic mass of proton in u.
                       Double_t Z = 1.0;
                       Get_Energy(Am, Z, brho, ener);
                       angle_vs_energy_pattern->Fill(theta*TMath::RadToDeg(),ener);
                       Double_t p = brho * Z * 2.99792458*100.0;      //Magnitude of the momentum In MeV/c.

                       // Vectors for selecting the right track
                       availableTracks.push_back(track);
                       thetaValues.push_back(theta);

                       // Selecting track to get phi angle
                       AtTrack goodTrack = TrackSelector(availableTracks, thetaValues);
                       auto hitClusterArray = goodTrack.GetHitClusterArray();
                       if (goodTrack.GetTrackID() != -1){
                          Double_t phi;

                           // Needs to be reversed back for multifit
                          std::reverse(hitClusterArray->begin(), hitClusterArray->end());
                          AtHitCluster firstCluster, secondCluster;

                          firstCluster = hitClusterArray->at(0);
                          secondCluster = hitClusterArray->at(1);

                          firstPosition = firstCluster.GetPosition();
                          secondPosition = secondCluster.GetPosition();

                          Double_t iniPx, iniPy, iniPz;
                          Double_t iniVx1, iniVy1, iniVz1;

                          Double_t secPosX,secPosY,secPosZ;
                          Double_t iniPosX,iniPosY,iniPosZ;

                          secPosX = secondPosition.X();
                          secPosY = secondPosition.Y();
                          secPosZ = secondPosition.Z();

                          iniPosX = firstPosition.X();
                          iniPosY = firstPosition.Y();
                          iniPosZ = firstPosition.Z();

                          phi = TMath::ATan2(secPosY- iniPosY, secPosX - iniPosX);
                          Double_t phiDeg;
                          if (phi < 0){
                             phiDeg = 360+ phi*TMath::RadToDeg();
                          }  else {
                             phiDeg = phi*TMath::RadToDeg();
                          }

                          iniPx = p * TMath::Cos(phiDeg*TMath::DegToRad()) * TMath::Sin(theta);       // in MeV/c
                          iniPy = p * TMath::Sin(phiDeg*TMath::DegToRad()) * TMath::Sin(theta);       // in MeV/c
                          iniPz = p * TMath::Cos(theta);                                              // in MeV/c

                          iniVx1 = (iniPx * 5.344286e-22)/m;                                          //meter per second [m/s]
                          iniVy1 = (iniPy * 5.344286e-22)/m;
                          iniVz1 = (iniPz * 5.344286e-22)/m;
                         // cout<< iniVx1<<endl<<endl;

                          TMatrixD initial_state(6, 1);
                          initial_state(0, 0) = iniPosX;
                          initial_state(1, 0) = iniPosY;
                          initial_state(2, 0) = iniPosZ;
                          initial_state(3, 0) = iniVx1;
                          initial_state(4, 0) = iniVy1;
                          initial_state(5, 0) = iniVz1;

                          TMatrixD initial_state1(6, 1);
                          initial_state1(0, 0) = secPosX;
                          initial_state1(1, 0) = secPosY;
                          initial_state1(2, 0) = secPosZ;
                          initial_state1(3, 0) = iniVx1;
                          initial_state1(4, 0) = iniVy1;
                          initial_state1(5, 0) = iniVz1;



                          /*----------
                            Looping over all the clusters in
                            each track to perform kalman
                          -----------*/

                          TMatrixD x_pred(6,1);
                          TMatrixD P_pred(6,6);
                          TMatrixD K(6,2);
                          TMatrixD Y_1(2,1); //Observation matrix
                          TMatrixD C_1(2,2);
                          TMatrixD Z_1(2,1);
                          TMatrixD x_estimate(6,1);

                          TMatrixD Jacobi_matrix1(6,6);
                          TMatrixD I(6,6);
                          TMatrixD F1(6,6);

                          TMatrixD Q(6,6);
                          generateGaussianNoise(Q); // Generates a 6x6 matrix of Gaussian noise with mean 0 and standard deviation 1

                          TMatrixD P(6,6);
                          Ini_P(P);

                          TMatrixD H_1(2,6);
                          // Get the measurement matrix
                          GetMeasurementMatrix(H_1);
 
                          // Error in Measurement.
                          TMatrixD R_1(2,2);
                          generateMeasurementNoise(R_1);

                          std::vector<int> eventNumbers = {99};
                          // Iterate over event numbers and access the corresponding events
                          if (std::find(eventNumbers.begin(), eventNumbers.end(), i) != eventNumbers.end()) {

                             Int_t n1 = int(secPosX-iniPosX);
                            // std::cout << n1 << std::endl;
                             Double_t t1=0.0;
                             Double_t l1x[n1],l2x[n1],l3x[n1],l4x[n1];
                             Double_t l1y[n1],l2y[n1],l3y[n1],l4y[n1];
                             Double_t l1z[n1],l2z[n1],l3z[n1],l4z[n1];

                             Double_t l1vx[n1],l2vx[n1],l3vx[n1],l4vx[n1];
                             Double_t l1vy[n1],l2vy[n1],l3vy[n1],l4vy[n1];
                             Double_t l1vz[n1],l2vz[n1],l3vz[n1],l4vz[n1];

                             Int_t rows = 6; // the number of state variables
                             Int_t cols = n1; // the number of steptime.


                             // Define matrix to hold state vectors
                             TMatrixD  state_matrix1(6,n1);


                             Int_t pp;

                             int clusterCount = 0;
                             for (auto iCluster =  0; iCluster<hitClusterArray->size()-1; ++iCluster){

                                 auto cluster = hitClusterArray->at(iCluster+1);
                                 auto pos = cluster.GetPosition();

                                 Double_t clusterPosX = pos.X();
                                 Double_t clusterPosY = pos.Y();
                                 Double_t clusterPosZ = pos.Z();
                                 X_vs_Y->Fill(iniPosX, iniPosY);

                                 for (pp = 0; pp < n1; pp++) {

                                     l1x[pp] = fx(t1,iniVx1);
                                     l1y[pp] = fy(t1,iniVy1);
                                     l1z[pp] = fz(t1,iniVz1);

                                     l1vx[pp] = dxdt(t1,iniVx1,iniVy1,iniVz1);
                                     l1vy[pp] = dydt(t1,iniVx1,iniVy1,iniVz1);
                                     l1vz[pp] = dzdt(t1,iniVx1,iniVy1,iniVz1);

                                     l2x[pp] = fx(t1+h*0.5,iniVx1+0.5*h*l1x[pp]);
                                     l2y[pp] = fy(t1+h*0.5,iniVy1+0.5*h*l1y[pp]);
                                     l2z[pp] = fz(t1+h*0.5,iniVz1+0.5*h*l1z[pp]);

                                     l2vx[pp] = dxdt(t1+h*0.5,iniVx1+l1vx[pp]*0.5*h,iniVy1+l1vy[pp]*h*0.5,iniVz1+l1vz[pp]*h*0.5);
                                     l2vy[pp] = dydt(t1+h*0.5,iniVx1+l1vx[pp]*h*0.5, iniVy1+l1vy[pp]*h*0.5,iniVz1+l1vz[pp]*h*0.5);
                                     l2vz[pp] = dzdt(t1+h*0.5,iniVx1+l1vx[pp]*h*0.5, iniVy1+l1vy[pp]*h*0.5, iniVz1+l1vz[pp]*h*0.5);

                                     l3x[pp] = fx(t1+h*0.5,iniVx1+0.5*h*l2x[pp]);
                                     l3y[pp] = fy(t1+h*0.5,iniVy1+0.5*h*l2y[pp]);
                                     l3z[pp] = fz(t1+h*0.5,iniVz1+0.5*h*l2z[pp]);

                                     l3vx[pp] = dxdt(t1+h*0.5,iniVx1+l2vx[pp]*0.5*h,iniVy1+l2vy[pp]*h*0.5,iniVz1 + l2vz[pp]*h*0.5);
                                     l3vy[pp] = dydt(t1+h*0.5,iniVx1+l2vx[pp]*h*0.5, iniVy1+l2vy[pp]*h*0.5,iniVz1 +l2vz[pp]*h*0.5);
                                     l3vz[pp] = dzdt(t1+h*0.5,iniVx1+l2vx[pp]*h*0.5, iniVy1+l2vy[pp]*h*0.5, iniVz1 + l2vz[pp]*h*0.5);

                                     l4x[pp] = fx(t1+h,iniVx1+l3x[pp]*h);
                                     l4y[pp] = fy(t1+h,iniVy1+l3y[pp]*h);
                                     l4z[pp] = fz(t1+h,iniVz1+l3z[pp]*h);

                                     l4vx[pp] = dxdt(t1+h,iniVx1+l3vx[pp]*h,iniVy1+l3vy[pp]*h,iniVz1+l3vz[pp]*h);
                                     l4vy[pp] = dydt(t1+h,iniVx1+l3vx[pp]*h,iniVy1+l3vy[pp]*h,iniVz1+l3vz[pp]*h);
                                     l4vz[pp] = dzdt(t1+h,iniVx1+l3vx[pp]*h,iniVy1+l3vy[pp]*h,iniVz1+l3vz[pp]*h);


                                     iniVx1 =  iniVx1 + h/6 *( l1vx[pp] + 2*l2vx[pp] + 2*l3vx[pp] + l4vx[pp]);
                                     iniVy1  = iniVy1 + h/6 *(l1vy[pp] + 2*l2vy[pp] + 2*l3vy[pp] + l4vy[pp]);
                                     iniVz1  = iniVz1 + h/6 *(l1vz[pp] + 2*l2vz[pp] + 2*l3vz[pp] + l4vz[pp]);

                                     iniPosX =  iniPosX + h/6 *( l1x[pp]  + 2*l2x[pp] + 2*l3x[pp] + l4x[pp]);
                                     iniPosY  = iniPosY + h/6 *( l1y[pp] + 2*l2y[pp] + 2*l3y[pp] + l4y[pp]);
                                     iniPosZ  = iniPosZ + h/6 *( l1z[pp] + 2*l2z[pp] + 2*l3z[pp] + l4z[pp]);


                                     // Update the cluster positions to reach the next cluster
                                     iniPosX = iniPosX + (clusterPosX - iniPosX) / n1;
                                     iniPosY = iniPosY + (clusterPosY - iniPosY) / n1;
                                     iniPosZ = iniPosZ + (clusterPosZ - iniPosZ) / n1;

                                     state_matrix1(0,pp) = iniPosX;
                                     state_matrix1(1,pp) = iniPosY;
                                     state_matrix1(2,pp) = iniPosZ;
                                     state_matrix1(3,pp) = iniVx1;
                                     state_matrix1(4,pp) = iniVy1;
                                     state_matrix1(5,pp) = iniVz1;




                                  //   kx_vs_ky->Fill(iniPosX, iniPosY);

                                     t1 = t1 + h;

                                 }
                                 Jacobi_matrice(Jacobi_matrix1);

                                 I_matrix(I);

                                 // Calculate propagator matrix using intermediate matrices
                                 // Initialize F as identity matrix
                                 F1.UnitMatrix();

                                 // Compute F1, F2, F3, F4
                                 Double_t dt_2 = 1e-10;
                                 TMatrixD IplusFt (6,6);
                                 IplusFt.Zero();
                                 TMatrixD Ft(6,6);
                                 Double_t h[cols], p[cols];

                                 for (int j = 0; j < 4; j++) {
                                     IplusFt = F1* dt_2;
                                     Ft  = (1e-10)*Jacobi_matrix1*IplusFt;
                                     F1     += Ft *((j==0 || j==3) ?  1.0/6.0 : 1.0/3.0);

                                 }

                                 // Define matrix to hold time derivatives of state vectors
                                 TMatrixD  state_dot_matrix1(rows,cols);

                                 // Calculate time derivatives of state vectors
                                 for (int k = 0; k < cols; k++) {
                                     // Extract state vector at time t. 
                                     TMatrixD state_vector1(6,1);
                                     for (int j = 0; j < 6; j++) {
                                         state_vector1(j,0) = state_matrix1(j,k);
                                     }

                                     // Multiply propagator matrix with state vector to get time derivative of state vector
                                     TMatrixD state_dot_vector1 = F1 * state_vector1;
                                    // X_vs_Y->Fill(state_dot_vector1(0,0), state_dot_vector1(1,0)); 
                                 }



                                 //Update the initial conditions for the next cluster
                                 iniPosX = clusterPosX;
                                 iniPosY = clusterPosY;
                                 iniPosZ = clusterPosZ;


                                 ++clusterCount;
                                 if (clusterCount > 2) {
                                    break; // stop iterating over clusters after the second one
                                 }

                             }


                           auto ik = 1;

                           for (auto iclus = 0; iclus < hitClusterArray->size(); iclus++){

                            //  Get the measurements.
                               auto MeasurementCluster = hitClusterArray->at(iclus);
                               auto measurements = MeasurementCluster.GetPosition();
                               Y_1(0,0) = measurements.X();
                               Y_1(1,0) = measurements.Y();

                               //Perform Kalman here.
                               //Initial state predictions for protons.
                               x_pred = F1 * initial_state ;
                               P_pred =  (F1 *TMatrixD(P, TMatrixD::kMultTranspose,F1))  + Q;

                               //this updates the  predicted states and the covariances.
                               K =  TMatrixD(P_pred, TMatrixD::kMultTranspose,H_1) *  (H_1 * TMatrixD(P_pred, TMatrixD::kMultTranspose,H_1) + R_1).Invert();

                               // Predicted measurement
                               TMatrixD predictedMeasurement = (H_1 * x_pred);

                               // Residual between predicted and actual measurement
                               TMatrixD residual = (Y_1 - predictedMeasurement);
                               x_estimate = x_pred + K * residual;

                               P =(I-K*H_1)*P_pred;
                               TMatrixD output = H_1 * x_estimate;              // this projects the estimated state into the output.
                               initial_state = x_estimate;
                               P.Print();

                               kx_vs_ky->Fill(output(0,0), output(1,0));

                               //Update the initial conditions for the next cluster
                            //   initial_state  = initial_state1;
                              // initial_state  = initial_state1;
                              // initial_state  = initial_state1;

                               ++ik;
                               if (ik > 2) {
                                  break; // stop iterating over clusters after the second one
                               }



                           }

                          }



                       }

                }

                }
            }


        }


/*
c2->cd(2);

//        TGraph *gr1  = new TGraph(n,iniPosX,iniPosY);
  //      TGraph *gr2  = new TGraph(n,secPosX,secPosX);
        gr2->SetMarkerColor(2);
        //gr2->SetMarkerStyle(21);

        // create a multigraph and draw it
        TMultiGraph  *mg  = new TMultiGraph();
        mg->Add(gr1);
        mg->Add(gr2);
        mg->Draw("AP*");








c2->cd(1);
        X_vs_Y->SetMarkerStyle(21);
        X_vs_Y->SetMarkerSize(0.3);
        X_vs_Y->SetLineWidth(3);
        X_vs_Y->SetLineColor(kBlack);
        X_vs_Y->SetMarkerColor(kBlack);
        X_vs_Y->Draw();

*/

c2->cd(2);

        X_vs_Y->SetMarkerStyle(24);
        X_vs_Y->SetMarkerSize(0.5);
        X_vs_Y->SetMarkerColor(kRed);
      //X_vs_Y->Draw();

        kx_vs_ky->SetMarkerStyle(20);
        kx_vs_ky->SetMarkerSize(0.5);
        kx_vs_ky->SetMarkerColor(kBlue);
        //kx_vs_ky->GetXaxis()->SetRange(0, 250);
        //kx_vs_ky->GetYaxis()->SetRangeUser(0, 250);


       X_vs_Y->Draw();

       kx_vs_ky->Draw("same");



/*
c2->cd(3);
        phi_pattern->SetMarkerStyle(15);
        phi_pattern->SetMarkerSize(0.3);
        phi_pattern->Draw();

*/
        return(0);


}

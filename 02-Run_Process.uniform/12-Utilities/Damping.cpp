#include "Damping.hpp"

#include <stdio.h>
#include <iostream>

//Overload Constructor.
Damping::Damping(std::string name, const std::vector<double> parameters,
  const std::vector<double> wc, const std::vector<double> xc) :
  Name(name), Parameters(parameters), FilterFreqs(wc), FilterCoeffs(xc){
    //Does nothing.
}

//Destructor.
Damping::~Damping(){
    //Does nothing.
}

//Gets damping model's name
std::string 
Damping::GetName(){
    return Name;
}

//Gets damping parameters vector
std::vector<double>
Damping::GetParameters(){
  return Parameters;
}

std::vector<double> 
Damping::GetFilterFreqs(){
    return FilterFreqs;
}

std::vector<double>
Damping::GetFilterCoeffs(){
    return FilterCoeffs;
}

//
Eigen::VectorXd 
Damping::GetDampingForce(){
    return fd;
}

//Set the name of the damping model.
void
Damping::SetName(std::string name){
    Name = name;
}

//Sets specifically the damping ratio.
void 
Damping::SetDampingRatio(double dampingRatio){
    Parameters[0] = dampingRatio;
}

//Set the damping parameters.
void 
Damping::SetParameters(std::vector<double> param){
    Parameters = param;
}

void 
Damping::SetFilterFreqs(std::vector<double> freqs){
    FilterFreqs = freqs;
}

void 
Damping::SetFilterCoeffs(std::vector<double> coeffs){
    FilterCoeffs = coeffs;
}

//
void
Damping::CommitDamping(){
    f0C = f0;
    fdC = fd;
    fLC = fL;
}

//
void 
Damping::ReverseDamping(){
    f0 = f0C;
    fd = fdC;
    fL = fLC;
}

void 
Damping::InitialDamping(unsigned int nComp){
    f0.resize(nComp);
    f0.fill(0.0);
    f0C.resize(nComp);
    f0C.fill(0.0);
    fd.resize(nComp);
    fd.fill(0.0);
    fdC.resize(nComp);
    fdC.fill(0.0);
    fL.resize(nComp, FilterFreqs.size());
    fL.fill(0.0);
    fLC.resize(nComp, FilterFreqs.size());
    fLC.fill(0.0);

    // Precompute coefficients
    double dt = Parameters[1];
    alpha.resize(FilterFreqs.size());
    for(unsigned int j = 0; j < FilterFreqs.size(); j++){
        double wcj = FilterFreqs[j];
        
        double yj = 1.0-cos(wcj*dt);
        alpha[j] = -yj + sqrt(yj*yj + 2*yj);
        
        //alpha[j] = 1-exp(-wcj*dt);

        //alpha[j] = wcj*dt / (1+wcj*dt);

        // hall
        //alpha[j] = wcj*dt / (2+wcj*dt);

    }
}

//
void 
Damping::UpdateDamping(Eigen::VectorXd f){
    double z0 = Parameters[0];
    double dt = Parameters[1];

    // Save structural forces
    f0 = f;

    // Reset current damping force vector
    fd.fill(0.0);

    // Loop over filters
    for(unsigned int j = 0; j < FilterFreqs.size(); j++){
        double xj = FilterCoeffs[j];
        double wcj = FilterFreqs[j];
        double aj = alpha[j];
        
        // Exponentially weighted moving average filter (EWMA)
        fL.col(j) = fLC.col(j) + aj*(f0 - fLC.col(j));
        //fL.col(j) = aj * (f0 + f0C) + (1-2*aj) * fLC.col(j);
        
        // Update damping forces
        fd += 2.0*z0*xj*aj/(wcj*dt) * (f0 - fLC.col(j));
        //fd += 4.0*z0*xj*aj/(wcj*dt) * (f0 + f0C - 2.0*fLC.col(j));
    }

    // Subtract previous damping force history 
    //fd -= fdC;
}

double 
Damping::GetStiffnessMultiplier(void){
    double multiplier = 0.0;
    double z0 = Parameters[0];
    double dt = Parameters[1];

    for(unsigned int j = 0; j < FilterFreqs.size(); j++){
        multiplier += 2.0*z0*alpha[j]*FilterCoeffs[j]/(FilterFreqs[j]*dt);
        //multiplier += 4.0*z0*FilterCoeffs[j]/(2.0 + FilterFreqs[j]*dt);
    }

    return 1.0 + multiplier;
}

std::unique_ptr<Damping>
Damping::CopyDamping(){
    return std::make_unique<Damping>(Name, Parameters, FilterFreqs, FilterCoeffs);
}

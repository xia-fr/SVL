#include <cmath>
#include "Lin3DChannel.hpp"
#include "Definitions.hpp"

//Define constant value PI:
const double PI = 3.1415926535897932;

//Overload Constructor.
Lin3DChannel::Lin3DChannel(double h, double b, double tw, double tf, std::unique_ptr<Material> &material, double theta, unsigned int ip) : 
Section("Lin3DChannel"), h(h), b(b), tw(tw), tf(tf), Theta(theta), InsertPoint(ip){
    //Initialize material strain.
    Strain.resize(6);
    Strain << 0.0, 0.0, 0.0, 0.0, 0.0, 0.0;

    //The section material:
    theMaterial = material->CopyMaterial();

    //Transform the rotation Angle into radians.
    Theta = PI*theta/180.0;
}

//Destructor.
Lin3DChannel::~Lin3DChannel(){
    //Does nothing.
}

//Clone the 'Lin3DChannel' section.
std::unique_ptr<Section>
Lin3DChannel::CopySection(){
    return std::make_unique<Lin3DChannel>(h, b, tw, tf, theMaterial, 180.0*Theta/PI, InsertPoint);
}

//Returns the section generalized strain.
Eigen::VectorXd
Lin3DChannel::GetStrain(){
    return Strain;
}

//Returns the section generalized stress.
Eigen::VectorXd
Lin3DChannel::GetStress(){
    Eigen::VectorXd Stress = GetTangentStiffness()*Strain;
    return Stress;
}

//Returns the section axial stiffness.
Eigen::MatrixXd
Lin3DChannel::GetDensity(){
    //Area Properties.
    double A   = GetArea();
    double J   = GetInertiaAxis1();

    //Material density.
    double Rho = theMaterial->GetDensity(); 

    //Returns the section stiffness for 2-dimensions.
    Eigen::MatrixXd SectionDensity(4,4);
    SectionDensity << Rho*A, 0.0 ,  0.0 ,  0.0 ,
                       0.0 , J/A ,  0.0 ,  0.0 ,
                       0.0 , 0.0 , Rho*A,  0.0 ,
                       0.0 , 0.0 ,  0.0 , Rho*A;

    return SectionDensity;
}

//Returns the section axial stiffness.
Eigen::MatrixXd
Lin3DChannel::GetTangentStiffness(){
    //Gets the section centroid.
    double zcm, ycm;
    ComputeSectionCenter(zcm, ycm);

    //Area Properties.
    double A   = GetArea();
    double J   = GetInertiaAxis1();
    double As2 = GetShearArea2();
    double As3 = GetShearArea3();
    double I22 = GetInertiaAxis2();
    double I33 = GetInertiaAxis3();

    //Section shear Stiffness at center of mass.
    double G = theMaterial->GetShearModulus();

    Eigen::MatrixXd Cs(3,3);
    Cs << G*J,  0.0 ,  0.0 ,
          0.0, G*As2,  0.0 ,
          0.0,  0.0 , G*As3;

    //Section flexural Stiffness at center of mass.
    Eigen::MatrixXd E = theMaterial->GetInitialTangentStiffness(); 

    Eigen::MatrixXd Cm(3,3);
    Cm << E(0,0)*A, 0.0       , 0.0       ,
          0.0     , E(0,0)*I22, 0.0       ,
          0.0     , 0.0       , E(0,0)*I33;

    //Compute the rotation matrix.
    Eigen::MatrixXd T = GetLineRotationMatrix(Theta);

    //Compute the translation matrix.
    Eigen::MatrixXd L = GetLineTranslationMatrix(h, b, zcm, ycm, InsertPoint);

    //Computes the global section stiffness matrix.
    Cs = T.transpose()*Cs*T;
    Cm = T.transpose()*L.transpose()*Cm*L*T;

    //Returns the section stiffness for 3-dimensions.
    Eigen::MatrixXd SectionStiffness(6,6);
    SectionStiffness << Cm(0,0),   0.0  , Cm(0,1), Cm(0,2),   0.0  ,   0.0  ,
                          0.0  , Cs(0,0),   0.0  ,   0.0  , Cs(0,1), Cs(0,2),
                        Cm(0,1),   0.0  , Cm(1,1), Cm(2,1),   0.0  ,   0.0  ,
                        Cm(0,2),   0.0  , Cm(2,1), Cm(2,2),   0.0  ,   0.0  ,
                          0.0  , Cs(0,1),   0.0  ,   0.0  , Cs(1,1), Cs(2,1),
                          0.0  , Cs(0,2),   0.0  ,   0.0  , Cs(2,1), Cs(2,2);

    return SectionStiffness;
}

//Returns the section initial tangent stiffness matrix.
Eigen::MatrixXd 
Lin3DChannel::GetInitialTangentStiffness(){
    return GetTangentStiffness();
}

//Returns the section strain at given position.
Eigen::VectorXd 
Lin3DChannel::GetStrainAt(double x3, double x2){
    //The strain vector in local coordinates
    Eigen::VectorXd theStrain(6);
    theStrain.fill(0.0);

    //Gets the section centroid.
    double zcm, ycm;
    ComputeSectionCenter(zcm, ycm);

    //Transform coordinates into section local axis.
    x3 = zcm - x3;
    x2 = x2 - ycm;

    //Checks the coordinate is inside the section
    if(((fabs(x2) >= h/2.0 - tf) & (fabs(x2) <= h/2.0) & (x3 >= zcm - b) & (x3 <= zcm)) | ((x2 >= -h/2.0 + tf) & (x2 <= h/2.0 - tf) & (x3 >= zcm - b) & (x3 <= zcm - b + tw))){
        //Transforms generalised strains from Element to Section local coordinate
        Eigen::VectorXd strain = ComputeLineLocalAxes(h, b, zcm, ycm, Theta, InsertPoint)*Strain;

        // Epsilon = [exx, 0.0, 0.0, exy, 0.0, exz]
        theStrain << strain(0) + x3*strain(2) - x2*strain(3), 
                     0.0, 
                     0.0, 
                     strain(4),
                     0.0, 
                     strain(5);
    }

    return theStrain;
}

//Returns the section stress at given position.
Eigen::VectorXd 
Lin3DChannel::GetStressAt(double x3, double x2){
    //The stress vector in local coordinates
    Eigen::VectorXd theStress(6);
    theStress.fill(0.0);

    //Gets the section centroid.
    double zcm, ycm;
    ComputeSectionCenter(zcm, ycm);

    //Transform coordinates into section local axis.
    x3 = zcm - x3;
    x2 = x2 - ycm;

    //Checks the coordinate is inside the section
    bool isInside = false;
    double Qs2, Qs3, th, tb;
    if((fabs(x2) >= h/2.0 - tf) & (fabs(x2) <= h/2.0) & (x3 >= zcm - b) & (x3 <= zcm)){
        isInside = true;
        Qs2 =  b/2.0*(h*h/4.0 - x2*x2); tb = b; 
        Qs3 = tf/2.0*(zcm*zcm - x3*x3); th = tf; 
    }
    else if((x2 >= -h/2.0 + tf) & (x2 <= h/2.0 - tf) & (x3 >= zcm - b) & (x3 <= zcm - b + tw)){
        isInside = true;
        Qs2 = tf/2.0*(h - tf)*(b - tw) + tw/2.0*(h*h/4.0 - x2*x2); tb = tw; 
        Qs3 =  h/2.0*((b - zcm)*(b - zcm) - x3*x3); th = h; 
    }

    //Computes the generalized section force
    if(isInside){
        //Section geometry properties.
        double A   = GetArea();
        double I22 = GetInertiaAxis2();
        double I33 = GetInertiaAxis3();

        //Transforms generalised stresses from Element to Section local coordinate
        Eigen::VectorXd Forces = ComputeLineLocalAxes(h, b, zcm, ycm, Theta, InsertPoint)*GetStress();

        // Sigma = [Sxx, 0.0, 0.0, txy, 0.0, txz]
        theStress << Forces(0)/A + Forces(2)*x3/I22 - Forces(3)*x2/I33, 
                     0.0, 
                     0.0, 
                     Forces(4)*Qs2/I33/tb, 
                     0.0,
                     Forces(5)*Qs3/I22/th;
    }

    return theStress;
}

//Perform converged section state update.
void 
Lin3DChannel::CommitState(){
    theMaterial->CommitState();
}

//Reverse the section states to previous converged state.
void 
Lin3DChannel::ReverseState(){
    theMaterial->ReverseState();
}

//Brings the section states to its initial state.
void 
Lin3DChannel::InitialState(){
    Strain.fill(0.0);
    theMaterial->InitialState();
}

//Update the section state for this iteration.
void
Lin3DChannel::UpdateState(Eigen::VectorXd strain, unsigned int cond){
    //Update the matrial behavior.
    Eigen::VectorXd mStrain(1);
    mStrain << strain(0);
    theMaterial->UpdateState(mStrain, cond);

    //Update the section strain.
    Strain = strain;
}

//Computes the Lin3DChannel area.
double 
Lin3DChannel::GetArea(){
    return 2.0*tf*b + tw*(h - 2.0*tf);
}

//Computes the Lin3DChannel shear area along axis 2.
double 
Lin3DChannel::GetShearArea2(){
    return h*tw;
}

//Computes the Lin3DChannel shear area along axis 3.
double 
Lin3DChannel::GetShearArea3(){
    return 5.0/6.0*(2.0*b*tf);
}

//Computes the Lin3DChannel torsional inertia.
double 
Lin3DChannel::GetInertiaAxis1(){
    double I22 = GetInertiaAxis2();
    double I33 = GetInertiaAxis3();
    return I22 + I33;
}

//Computes the Lin3DChannel flexural inertia.
double 
Lin3DChannel::GetInertiaAxis2(){
    double Area = GetArea();
    double zcm  = (tf*b*b + 0.5*tw*tw*(h - 2.0*tf))/Area;
    double hc   = zcm - tw;
    double dc   = b - zcm;
    return 1.0/3.0*(h*zcm*zcm*zcm + 2.0*tf*dc*dc*dc - (h - 2.0*tf)*hc*hc*hc);
}

//Computes the Lin3DChannel flexural inertia.
double 
Lin3DChannel::GetInertiaAxis3(){
    return 1.0/12.0*(b*h*h*h - (b - tw)*(h - 2.0*tf)*(h - 2.0*tf)*(h - 2.0*tf));
}

//Gets the section centroid.
void
Lin3DChannel::ComputeSectionCenter(double &zcm, double &ycm){
    //Cross Section-Centroid.
    double Area = GetArea();
    ycm = h/2.0;
    zcm = (tf*b*b + tw*(h - 2.0*tf)*(b - tw/2.0))/Area;
}
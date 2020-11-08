#include <cmath>
#include <iostream>
#include "null2DFrame2.hpp"
#include "Definitions.hpp"

//Define constant tolerance value:
const double TOL = 0.9999995;

//Define VTK cell value for Paraview:
const unsigned int VTKCELL = 3;

//Overload constructor.
null2DFrame2::null2DFrame2(const std::vector<unsigned int> nodes) :
Element("null2DFrame2", nodes, 6, VTKCELL){
    //The element nodes.
    theNodes.resize(2);
}

//Destructor.
null2DFrame2::~null2DFrame2(){
    //Does nothing.
}

//Save the material states in the element.
void 
null2DFrame2::CommitState(void){
    //Does nothing
}

//Update the material states in the element.
void 
null2DFrame2::UpdateState(){
    //Does nothing
}

//Sets the finite element dependance among objects.
void 
null2DFrame2::SetDomain(std::map<unsigned int, std::shared_ptr<Node> > &nodes){
    //Gets the global element connectivity.
    std::vector<unsigned int> conn = GetNodes();

    //Assign the element to mesh node pointer.  
    for(unsigned int i = 0; i < GetNumberOfNodes(); i++){
        theNodes[i] = nodes[conn[i]];
    }
}

//Sets the damping model.
void 
null2DFrame2::SetDamping(const std::shared_ptr<Damping> &damping){
    //The damping model
    theDamping = damping;
}

//Gets the list of total-degree of freedom of this element.
std::vector<unsigned int> 
null2DFrame2::GetTotalDegreeOfFreedom() const{
    //Total number of degree-of-freedom.
    unsigned int nDofs = GetNumberOfDegreeOfFreedom();

    //Reserve memory for the element list of degree-of-freedom.
    std::vector<unsigned int> dofs(nDofs);

    //Construct the element list of degree-of-freedom for assembly.
    for(unsigned int j = 0; j < 2; j++){    
        unsigned int LengthDofs = theNodes[j]->GetNumberOfDegreeOfFreedom();
        std::vector<int> totalDofs = theNodes[j]->GetTotalDegreeOfFreedom();

        for(unsigned int i = 0; i < LengthDofs; i++)
            dofs[i + LengthDofs*j] = totalDofs[i];    
    }

    return dofs;
}

//Returns the section generalised strain at integration point.
Eigen::MatrixXd 
null2DFrame2::GetStrain() const{
    //Considers one integration point.
    Eigen::MatrixXd theStrain(1,3);
    theStrain.fill(0.0);

    return theStrain;
}

//Returns the section generalised stress at integration point.
Eigen::MatrixXd 
null2DFrame2::GetStress() const{
    //Considers one integration point.
    Eigen::MatrixXd theStress(1,3);
    theStress.fill(0.0);

    return theStress;
}

//Returns the section generalised strain-rate at integration point.
Eigen::MatrixXd 
null2DFrame2::GetStrainRate() const{
    //Considers one integration point.
    Eigen::MatrixXd theStrainRate(1,3);
    theStrainRate.fill(0.0);

    return theStrainRate;
}

//Gets the material strain in section at  coordinate (x3,x2).
Eigen::MatrixXd 
null2DFrame2::GetStrainAt(double x3, double x2) const{
    //Considers one integration point.
    Eigen::MatrixXd theStrain(1, 3);
    theStrain.fill(0.0);

    return theStrain;
}

//Gets the material stress in section at  coordinate (x3,x2).
Eigen::MatrixXd 
null2DFrame2::GetStressAt(double x3, double x2) const{
    //Considers one integration point.
    Eigen::MatrixXd theStress(1, 3);
    theStress.fill(0.0);

    return theStress;
}

//Gets the element internal response in VTK format.
Eigen::VectorXd 
null2DFrame2::GetVTKResponse(std::string response) const{   
    //The VTK response vector.
    Eigen::VectorXd theResponse(6);
    theResponse.fill(0.0);

    return theResponse;
}

//Compute the mass matrix of the element.
Eigen::MatrixXd 
null2DFrame2::ComputeMassMatrix(){
    //Consistent mass definition.
    Eigen::MatrixXd MassMatrix(6,6);
    MassMatrix.fill(0.0);

    return MassMatrix;
}

//Compute the stiffness matrix of the element.
Eigen::MatrixXd 
null2DFrame2::ComputeStiffnessMatrix(){
    //Stiffness matrix definition:
    Eigen::MatrixXd StiffnessMatrix(6,6);
    StiffnessMatrix.fill(0.0);

    return StiffnessMatrix;
}

//Compute the damping matrix of the element.
Eigen::MatrixXd 
null2DFrame2::ComputeDampingMatrix(){
    //Damping matrix definition.
    Eigen::MatrixXd DampingMatrix;
    DampingMatrix.resize(6,6);
    DampingMatrix.fill(0.0);
    
    //No material damping contribution is allowed.
    return DampingMatrix;
}

//Compute the PML history matrix for Perfectly-Matched Layer (PML).
Eigen::MatrixXd 
null2DFrame2::ComputePMLMatrix(){
    Eigen::MatrixXd Kpml;
    return Kpml;
}

//Compute the element the internal forces acting on the element.
Eigen::VectorXd 
null2DFrame2::ComputeInternalForces(){
    //Stiffness matrix definition:
    Eigen::VectorXd InternalForces(6);
    InternalForces.fill(0.0);

    return InternalForces;
}

//Compute the elastic, inertial, and vicous forces acting on the element.
Eigen::VectorXd 
null2DFrame2::ComputeInternalDynamicForces(){
    //The Internal dynamic force vector
    Eigen::VectorXd InternalForces(6);
    InternalForces.fill(0.0);

    return InternalForces;
}

//Compute the PML history vector using gauss-integration.
Eigen::VectorXd 
null2DFrame2::ComputePMLVector(){
    //Empty PML vector.
    Eigen::VectorXd Fpml;
    return Fpml;
}

//Compute the surface forces acting on the element.
Eigen::VectorXd 
null2DFrame2::ComputeSurfaceForces(const std::shared_ptr<Load> &surfaceLoad, unsigned int face){
    //Local surface load vector:
    Eigen::VectorXd surfaceForces(6);
    surfaceForces.fill(0.0);

    return surfaceForces;
}

//Compute the body forces acting on the element.
Eigen::VectorXd 
null2DFrame2::ComputeBodyForces(const std::shared_ptr<Load> &bodyLoad, unsigned int k){
    //Local body load vector:
    Eigen::VectorXd bodyForces(6);
    bodyForces.fill(0.0);

    return bodyForces;
}

//Compute the domain reduction forces acting on the element.
Eigen::VectorXd 
null2DFrame2::ComputeDomainReductionForces(const std::shared_ptr<Load> &drm, unsigned int k){
    //TODO: Domain reduction forces not implemented for frame.
    Eigen::VectorXd DRMForces(6);
    DRMForces.fill(0.0);

    return DRMForces;
}
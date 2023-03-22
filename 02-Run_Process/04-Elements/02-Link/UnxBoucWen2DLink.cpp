#include <cmath>
#include "UnxBoucWen2DLink.hpp"
#include "Definitions.hpp"
#include "Profiler.hpp"

//Overload constructor.
UnxBoucWen2DLink::UnxBoucWen2DLink(const std::vector<unsigned int> nodes, std::vector<double> params, std::vector<double> vars, const unsigned int dim, const unsigned int dir, double tol, unsigned int nmax) :
Element("UnxBoucWen2DLink", nodes, 2*dim, VTK_LINEAR_LINE, GROUP_ELEMENT_BWEN), Tol(tol), nMax(nmax), Dimension(dim), Direction(dir){
    //The element nodes.
    theNodes.resize(2);

    //Assign Histeretic curve variables.
    Fy = vars[0];
    Ko = vars[1];

    //Assign Histeretic shape variables.
    alpha  = params[0];
    eta    = params[1];
    beta   = params[2];
    gamma  = params[3];

    //Compute Bouc-Wen backbone curve parameters
    z    = 0.0;
    zn   = 0.0;
    U    = 0.0;
    Un   = 0.0;

    //Initialize stiffness and internal force.
    qbw = 0.0;
    kbw = Ko;
    qbc = 0.0;
    kbc = Ko;
}

//Destructor:
UnxBoucWen2DLink::~UnxBoucWen2DLink(){
    //Does nothing.
}

//Save the internal variable states in the element.
void 
UnxBoucWen2DLink::CommitState(){
    zn = z;
    Un = U;
    qbc = qbw;
    kbc = kbw;
}

//Reverse the internal variables to previous converged state.
void 
UnxBoucWen2DLink::ReverseState(){
    qbw = qbc;
    kbw = kbc;
}

//Brings the internal variables to its initial state in this element.
void 
UnxBoucWen2DLink::InitialState(){
    //Bouc-Wen backbone curve parameters
    z    = 0.0;
    zn   = 0.0;
    U    = 0.0;
    Un   = 0.0;

    //Initialize stiffness and internal force.
    qbw = 0.0;
    kbw = Ko;
    qbc = 0.0;
    kbc = Ko;
}

//Update the material states in the element.
void 
UnxBoucWen2DLink::UpdateState(){
    //Transformation matrix from global to local.
    Eigen::MatrixXd Tr = ComputeRotationMatrix();

    //Computes deformation vector in local coordinates.
    Eigen::VectorXd Ubw = Tr*ComputeRelativeDeformation();
    U = Ubw(Direction);

    //Relative link deformation.
    double dUn = U - Un;

    //Yield force and deformation of hysteretic component.
    double qY = Fy*(1.0 - alpha);
    double uY = qY/Ko;

    //Newton-Raphson iteration.
    double f, df, dz;
    unsigned int k = 0;

    do{
        f  = z - zn - dUn/uY*(1.0 - pow(fabs(z), eta)*(gamma + beta*sign(z*dUn)));
        df = 1.0 + dUn/uY*eta*pow(fabs(z), eta - 1.0)*sign(z)*(gamma + beta*sign(z*dUn));
        dz = f/df;
        z  = z - dz;
        k++;
    } while( (fabs(dz) > Tol) & (k < nMax) );

    //Derivative of internal variable w.r.t displacement.
    double dzdu = 1.0 - pow(fabs(z), eta)*(gamma + beta*sign(z*dUn));

    //Compute consistent stiffness matrix and internal force vector.
    qbw = qY*z + alpha*Ko*U;
    kbw = (1.0 - alpha)*Ko*dzdu + alpha*Ko;
}

//Sets the finite element dependance among objects.
void 
UnxBoucWen2DLink::SetDomain(std::map<unsigned int, std::shared_ptr<Node> > &nodes){
    //Gets the global element connectivity.
    std::vector<unsigned int> conn = GetNodes();

    //Assign the element to mesh node pointer.  
    for(unsigned int i = 0; i < GetNumberOfNodes(); i++){
        theNodes[i] = nodes[conn[i]];
    }
}

//Sets the damping model.
void 
UnxBoucWen2DLink::SetDamping(const std::shared_ptr<Damping>& UNUSED(damping)){
    //does nothing.
}

//Gets the list of total-degree of freedom of this element.
std::vector<unsigned int> 
UnxBoucWen2DLink::GetTotalDegreeOfFreedom() const{
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

//Returns the maximum strain value.
double
UnxBoucWen2DLink::GetMaxStrain(){
    //return maxStrain;
    return 0.0;
}

//Returns the material strain.
Eigen::MatrixXd 
UnxBoucWen2DLink::GetStrain() const{
    //The relative deformation.
    Eigen::MatrixXd theStrain(1,1);
    theStrain << Un;

    return theStrain;
}

//Returns the material stress.
Eigen::MatrixXd 
UnxBoucWen2DLink::GetStress() const{
    //The non-linear internal force.
    Eigen::MatrixXd theStress(1,1);
    theStress << qbw;

    return theStress;
}

//Gets the material strain rate.
Eigen::MatrixXd 
UnxBoucWen2DLink::GetStrainRate() const{
    Eigen::MatrixXd theStrainRate(1,1);
    theStrainRate << 0.0;

    return theStrainRate;
}

//Gets the material strain in section at  coordinate (x3,x2).
Eigen::MatrixXd 
UnxBoucWen2DLink::GetStrainAt(double UNUSED(x3), double UNUSED(x2)) const{
    //Stress at coordinate is define within section.
    Eigen::MatrixXd theStrain(1, 3); 
    theStrain.fill(0.0);

    return theStrain;
}

//Gets the material stress in section at  coordinate (x3,x2).
Eigen::MatrixXd 
UnxBoucWen2DLink::GetStressAt(double UNUSED(x3), double UNUSED(x2)) const{
    //Stress at coordinate is define within section.
    Eigen::MatrixXd theStress(1, 3); 
    theStress.fill(0.0);

    return theStress;
}

//Gets the element internal response in VTK format.
Eigen::VectorXd 
UnxBoucWen2DLink::GetVTKResponse(std::string UNUSED(response)) const{
    //TODO: Stress/Strain responses
    //The VTK response vector.
    Eigen::VectorXd theResponse(18);
    theResponse.fill(0.0);

    return theResponse;
}

//Computes the element energy for a given deformation.
double 
UnxBoucWen2DLink::ComputeEnergy(){
    //TODO: Integrate over element volume to compute the energy
    return 0.0;
}

//Compute the mass matrix of the element.
Eigen::MatrixXd 
UnxBoucWen2DLink::ComputeMassMatrix(){
    //Starts profiling this function.
    PROFILE_FUNCTION();

    //The matrix dimension.
    unsigned int nDim = 2*Dimension;

    //Consistent mass definition.
    Eigen::MatrixXd MassMatrix(nDim, nDim);
    MassMatrix.fill(0.0);

    return MassMatrix;
}

//Compute the stiffness matrix of the element.
Eigen::MatrixXd 
UnxBoucWen2DLink::ComputeStiffnessMatrix(){
    //Starts profiling this function.
    PROFILE_FUNCTION();

    //The vector dimension.
    unsigned int nDim = 2*Dimension;

    //The global axes transformation.
    Eigen::MatrixXd localStiffness(nDim, nDim);
    localStiffness.fill(0.0);

    localStiffness(Direction , Direction) = kbw;
    localStiffness(Dimension + Direction, Dimension + Direction) = kbw;

    //The global axes transformation: 
    Eigen::MatrixXd localAxes = ComputeLocalAxes();

    //Stiffness matrix definition.
    Eigen::MatrixXd StiffnessMatrix = localAxes.transpose()*localStiffness*localAxes;

    return StiffnessMatrix;
}

//Compute damping matrix of the element.
Eigen::MatrixXd 
UnxBoucWen2DLink::ComputeDampingMatrix(){
    //Starts profiling this function.
    PROFILE_FUNCTION();

    //The matrix dimension.
    unsigned int nDim = 2*Dimension;

    //Consistent mass definition.
    Eigen::MatrixXd DampingMatrix(nDim, nDim);
    DampingMatrix.fill(0.0);

    return DampingMatrix;
}

//Compute the PML history matrix for Perfectly-Matched Layer (PML).
Eigen::MatrixXd 
UnxBoucWen2DLink::ComputePMLMatrix(){
    Eigen::MatrixXd Kpml;
    return Kpml;
}

//Compute the element internal forces acting on the element.
Eigen::VectorXd 
UnxBoucWen2DLink::ComputeInternalForces(){
    //Starts profiling this function.
    PROFILE_FUNCTION();

    //The vector dimension.
    unsigned int nDim = 2*Dimension;

    //The global axes transformation.
    Eigen::VectorXd localForces(nDim);
    localForces.fill(0.0);

    localForces(Direction) = -qbw;
    localForces(Dimension + Direction) = qbw;

    //The global axes transformation.
    Eigen::MatrixXd localAxes = ComputeLocalAxes(); 
 
    //Compute the internal elastic force contribution.
    Eigen::VectorXd InternalForces = localAxes.transpose()*localForces;

    return InternalForces;
}

//Compute the elastic, inertial, and viscous forces acting on the element.
Eigen::VectorXd 
UnxBoucWen2DLink::ComputeInternalDynamicForces(){
    //The Internal dynamic force vector
    Eigen::VectorXd InternalForces;

    if( HasFixedNode(theNodes) ){
        unsigned int ndims = 2*Dimension;

        //Allocate memory for velocity/acceleraton. 
        Eigen::VectorXd V(ndims); 
        Eigen::VectorXd A(ndims);

        //Fills the response vectors with velocity/acceleraton values.
        V << theNodes[0]->GetVelocities(), theNodes[1]->GetVelocities();
        A << theNodes[0]->GetAccelerations(), theNodes[1]->GetAccelerations();

        //Compute the inertial/viscous/elastic dynamic force contribution.
        InternalForces = ComputeInternalForces() + ComputeDampingMatrix()*V + ComputeMassMatrix()*A;
    }

    return InternalForces;
}

//Compute the surface forces acting on the element.
Eigen::VectorXd 
UnxBoucWen2DLink::ComputeSurfaceForces(const std::shared_ptr<Load>& UNUSED(surface), unsigned int UNUSED(face)){
    //Starts profiling this function.
    PROFILE_FUNCTION();

    //Local surface load vector.
    Eigen::VectorXd surfaceForces(2*Dimension);
    surfaceForces.fill(0.0);

    return surfaceForces;
}

//Compute the body forces acting on the element.
Eigen::VectorXd 
UnxBoucWen2DLink::ComputeBodyForces(const std::shared_ptr<Load>& UNUSED(body), unsigned int UNUSED(k)){
    //Starts profiling this function.
    PROFILE_FUNCTION();

    //Local body load vector.
    Eigen::VectorXd bodyForces(2*Dimension);
    bodyForces.fill(0.0);

    return bodyForces;
}

//Compute the domain reduction forces acting on the element.
Eigen::VectorXd 
UnxBoucWen2DLink::ComputeDomainReductionForces(const std::shared_ptr<Load>& UNUSED(drm), unsigned int UNUSED(k)){
    //Starts profiling this function.
    PROFILE_FUNCTION();

    //Domain reduction force vector.
    unsigned int nDofs = GetNumberOfDegreeOfFreedom();
    Eigen::VectorXd DRMForces(nDofs);
    DRMForces.fill(0.0);

    return DRMForces;
}

//Compute the local axis of the element.
Eigen::MatrixXd
UnxBoucWen2DLink::ComputeLocalAxes() const{
    //Gets the element coordinates in undeformed configuration.  
    Eigen::VectorXd Xi = theNodes[0]->GetCoordinates();
    Eigen::VectorXd Xj = theNodes[1]->GetCoordinates();

    //Local axis definition.
    Eigen::Vector2d v1 = Xj - Xi;
    v1 = v1/v1.norm();

    //The matrix dimension.
    unsigned int nDim = 2*Dimension;

    //The global axes transformation:
    Eigen::MatrixXd localAxes(nDim, nDim);

    if(Dimension == 2){
        //For Solid element nodes
        localAxes <<  v1(0), v1(1),   0.0,    0.0,
                     -v1(1), v1(0),   0.0,    0.0,
                        0.0,   0.0,  v1(0), v1(1),
                        0.0,   0.0, -v1(1), v1(0);
    }
    else if(Dimension == 3){
        //For Structural element nodes
        localAxes <<  v1(0), v1(1), 0.0,   0.0,    0.0, 0.0,
                     -v1(1), v1(0), 0.0,   0.0,    0.0, 0.0,
                        0.0,   0.0, 1.0,   0.0,    0.0, 0.0,
                        0.0,   0.0, 0.0,  v1(0), v1(1), 0.0,
                        0.0,   0.0, 0.0, -v1(1), v1(0), 0.0,
                        0.0,   0.0, 0.0,   0.0,    0.0, 1.0;
    }
    else{
        //TODO: For non-standard nodes
        localAxes.setIdentity();
    }
    
    return localAxes;
}

//Compute the local axis of the element.
Eigen::MatrixXd
UnxBoucWen2DLink::ComputeRotationMatrix() const{
    //Gets the element coordinates in undeformed configuration.  
    Eigen::VectorXd Xi = theNodes[0]->GetCoordinates();
    Eigen::VectorXd Xj = theNodes[1]->GetCoordinates();

    //Local axis definition.
    Eigen::Vector2d v1 = Xj - Xi;
    v1 = v1/v1.norm();

    //The global axes transformation:
    Eigen::MatrixXd localAxes(Dimension, Dimension);

    if(Dimension == 2){
        //For Solid element nodes
        localAxes <<  v1(0), v1(1),
                     -v1(1), v1(0);
    }
    else if(Dimension == 3){
        //For Structural element nodes
        localAxes <<  v1(0), v1(1), 0.0,
                     -v1(1), v1(0), 0.0,
                        0.0,   0.0, 1.0;
    }
    else{
        //TODO: For non-standard nodes
        localAxes.setIdentity();
    }
    
    return localAxes;
}

//Sign function implementation.
double
UnxBoucWen2DLink::sign(double x) const{
    return (x > 0) ? 1 : ((x < 0) ? -1 : 0);
}

//Update strain in the element.
Eigen::VectorXd 
UnxBoucWen2DLink::ComputeRelativeDeformation() const{  
    //Gets the element displacements.
    Eigen::VectorXd Ui = theNodes[0]->GetDisplacements() + theNodes[0]->GetIncrementalDisplacements();
    Eigen::VectorXd Uj = theNodes[1]->GetDisplacements() + theNodes[1]->GetIncrementalDisplacements();
    
    Eigen::VectorXd deformation = Uj - Ui;

    return deformation;
}

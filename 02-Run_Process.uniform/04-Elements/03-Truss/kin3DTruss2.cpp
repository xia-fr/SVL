#include <cmath>
#include "kin3DTruss2.hpp"
#include "Definitions.hpp"
#include "Profiler.hpp"

//Define constant tolerance value:
const double TOL = 0.9999995;

//Overload constructor.
kin3DTruss2::kin3DTruss2(const std::vector<unsigned int> nodes, std::unique_ptr<Material> &material, const double area) :
Element("kin3DTruss2", nodes, 6, VTK_LINEAR_LINE, GROUP_ELEMENT_TRUSS), A(area){
    //The element nodes.
    theNodes.resize(2);

    //The element material.
    theMaterial = material->CopyMaterial();
}

//Destructor:
kin3DTruss2::~kin3DTruss2(){
    //Does nothing.
}

//Save the material states in the element.
void 
kin3DTruss2::CommitState(){
    if(theMaterial->IsViscous()){
        //Computes strain rate vector.
        Eigen::VectorXd strainrate = ComputeStrainRate();

        //Update material states.
        theMaterial->UpdateState(strainrate, 2);
    }

    theMaterial->CommitState();
}

//Reverse the material states to previous converged state in this element.
void 
kin3DTruss2::ReverseState(){
    theMaterial->ReverseState();
}

//Brings the material state to its initial state in this element.
void 
kin3DTruss2::InitialState(){
    theMaterial->InitialState();
}

//Update the material states in the element.
void 
kin3DTruss2::UpdateState(){
    //Computes strain vector.
    Eigen::VectorXd strain = ComputeStrain();

    //Update material states.
    theMaterial->UpdateState(strain, 1);
}

//Sets the finite element dependance among objects.
void 
kin3DTruss2::SetDomain(std::map<unsigned int, std::shared_ptr<Node> > &nodes){
    //Gets the global element connectivity.
    std::vector<unsigned int> conn = GetNodes();

    //Assign the element to mesh node pointer.  
    for(unsigned int i = 0; i < GetNumberOfNodes(); i++){
        theNodes[i] = nodes[conn[i]];
    }

    //Computes initial length of element. 
    Eigen::VectorXd Xi = theNodes[0]->GetCoordinates();
    Eigen::VectorXd Xj = theNodes[1]->GetCoordinates();

    Lo = (Xj - Xi).norm();
}

//Sets the damping model.
void 
kin3DTruss2::SetDamping(const std::shared_ptr<Damping> &damping){
    //The damping model
    theDamping = damping;
}

//Gets the list of total-degree of freedom of this element.
std::vector<unsigned int> 
kin3DTruss2::GetTotalDegreeOfFreedom() const{
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
kin3DTruss2::GetMaxStrain(){
    //return maxStrain;
    return 0.0;
}

//Returns the material strain at integration points.
Eigen::MatrixXd 
kin3DTruss2::GetStrain() const{
    Eigen::VectorXd Strain = theMaterial->GetStrain();
    Eigen::MatrixXd theStrain(1,1);
    theStrain << Strain(0);

    return theStrain;
}

//Returns the material stress at integration points.
Eigen::MatrixXd 
kin3DTruss2::GetStress() const{
    Eigen::VectorXd Stress = theMaterial->GetTotalStress();
    Eigen::MatrixXd theStress(1,1);
    theStress << Stress(0);

    return theStress;
}

//Returns the material strain-rate at integration points.
Eigen::MatrixXd 
kin3DTruss2::GetStrainRate() const{
    Eigen::VectorXd strainrate = theMaterial->GetStrainRate();
    Eigen::MatrixXd theStrainRate(1,1);
    theStrainRate << strainrate(0);

    return theStrainRate;
}

//Gets the material strain in section at  coordinate (x3,x2).
Eigen::MatrixXd 
kin3DTruss2::GetStrainAt(double UNUSED(x3), double UNUSED(x2)) const{
    //Stress at coordinate is define within section.
    Eigen::MatrixXd theStrain(1, 6);
    theStrain.fill(0.0);

    return theStrain;
}

//Gets the material stress in section at  coordinate (x3,x2).
Eigen::MatrixXd 
kin3DTruss2::GetStressAt(double UNUSED(x3), double UNUSED(x2)) const{
    //Stress at coordinate is define within section.
    Eigen::MatrixXd theStress(1, 6);
    theStress.fill(0.0);

    return theStress;
}

//Gets the element internal response in VTK format.
Eigen::VectorXd 
kin3DTruss2::GetVTKResponse(std::string response) const{
    //The VTK response vector.
    Eigen::VectorXd theResponse(18);

    if (strcasecmp(response.c_str(),"Strain") == 0){
        Eigen::VectorXd Strain = theMaterial->GetStrain();
        theResponse << Strain(0), 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0;
    }
    else if(strcasecmp(response.c_str(),"Stress") == 0){
        Eigen::VectorXd Stress = theMaterial->GetTotalStress();
        theResponse << Stress(0), 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0;
    }

    return theResponse;
}

//Computes the element energy for a given deformation.
double 
kin3DTruss2::ComputeEnergy(){
    //TODO: Integrate over element volume to compute the energy
    return 0.0;
}

//Compute the mass matrix of the element.
Eigen::MatrixXd 
kin3DTruss2::ComputeMassMatrix(){
    //Starts profiling this function.
    PROFILE_FUNCTION();

    //Computes the transformation matrix.
    Eigen::MatrixXd localAxes = ComputeTransformationAxes();
    
    //Consistent mass definition:
    Eigen::MatrixXd MassMatrix(6,6);

    //Gets material properties:
    double rho = theMaterial->GetDensity();

    //Construct mass matrix according to formulation.
    if(MassFormulation){
        //Gets the total mass:
        double mass = rho*A*Lo/2.0;

        MassMatrix <<  mass,    0.0,    0.0,    0.0,    0.0,     0.0,
                        0.0,   mass,    0.0,    0.0,    0.0,     0.0,
                        0.0,    0.0,   mass,    0.0,    0.0,     0.0,
                        0.0,    0.0,    0.0,   mass,    0.0,     0.0,
                        0.0,    0.0,    0.0,    0.0,   mass,     0.0,
                        0.0,    0.0,    0.0,    0.0,    0.0,    mass;
    }
    else{
        //Gets the total mass:
        double mass11 = rho*A*Lo/3.0;
        double mass12 = rho*A*Lo/6.0;    

        MassMatrix << mass11,    0.0,    0.0, mass12,    0.0,    0.0,
                         0.0, mass11,    0.0,    0.0, mass12,    0.0,
                         0.0,    0.0, mass11,    0.0,    0.0, mass12,
                      mass12,    0.0,    0.0, mass11,    0.0,    0.0,
                         0.0, mass12,    0.0,    0.0, mass11,    0.0,
                         0.0,    0.0, mass12,    0.0,    0.0, mass11;
    }

    //Transform Mass matrix into Global Coordinates.
    MassMatrix = localAxes.transpose()*MassMatrix*localAxes;

    return MassMatrix;
}

//Compute the stiffness matrix of the element.
Eigen::MatrixXd 
kin3DTruss2::ComputeStiffnessMatrix(){
    //Starts profiling this function.
    PROFILE_FUNCTION();

    //Gets material tangent matrix.
    Eigen::MatrixXd E = theMaterial->GetTangentStiffness();

    //Computes the current element length.
    double L = ComputeLength();

    //Computes the current axial force.
    double N = ComputeAxialForce();

    //The global axes transformation: 
    Eigen::VectorXd LocalAxes = ComputeLocalAxes();

    //The geometric axes transformation: 
    Eigen::MatrixXd GeometricAxes = ComputeGeometricAxes();

    //The material stiffness matrix.
    Eigen::MatrixXd KL = E(0,0)*A/Lo*LocalAxes*LocalAxes.transpose();

    //The material stiffness matrix.
    Eigen::MatrixXd KNL = N/L*GeometricAxes.transpose()*GeometricAxes;

    //Stiffness matrix definition:
    Eigen::MatrixXd StiffnessMatrix = KL + KNL;

    return StiffnessMatrix;
}

//Compute the damping matrix of the element.
Eigen::MatrixXd 
kin3DTruss2::ComputeDampingMatrix(){
    //Starts profiling this function.
    PROFILE_FUNCTION();

    //Define damping matrix
    Eigen::MatrixXd DampingMatrix(6,6);
    DampingMatrix.fill(0.0);
    
    //TODO: check if this is initial stiffness
    std::string dampName = theDamping->GetName();
    std::vector<double> dampParam = theDamping->GetParameters();

    if(strcasecmp(dampName.c_str(),"Free") == 0){
        //Does nothing.
    }
    else if(strcasecmp(dampName.c_str(),"Rayleigh") == 0){    
        //Compute stiffness and mass matrix.
        Eigen::MatrixXd MassMatrix = ComputeMassMatrix();
        Eigen::MatrixXd StiffnessMatrix = ComputeInitialStiffnessMatrix();

        DampingMatrix += dampParam[0]*MassMatrix + dampParam[1]*StiffnessMatrix;
    }
    else if(strcasecmp(dampName.c_str(),"Caughey") == 0){
        //TODO: implement Caughey damping
    }

    //TODO: Adds material damping contribution.

    return DampingMatrix;
}

//Compute the PML history matrix for Perfectly-Matched Layer (PML).
Eigen::MatrixXd 
kin3DTruss2::ComputePMLMatrix(){
    Eigen::MatrixXd Kpml;
    return Kpml;
}

//Compute the element the internal forces acting on the element.
Eigen::VectorXd 
kin3DTruss2::ComputeInternalForces(){
    //Starts profiling this function.
    PROFILE_FUNCTION();

    //The global axes transformation.
    Eigen::VectorXd LocalAxes = ComputeLocalAxes();

    //The nodal internal force vector in global coordinates.
    Eigen::VectorXd stress = theMaterial->GetStress();
    Eigen::VectorXd InternalForces = stress(0)*A*LocalAxes;

    return InternalForces;
}

//Compute the elastic, inertial, and viscous forces acting on the element.
Eigen::VectorXd 
kin3DTruss2::ComputeInternalDynamicForces(){
    //The Internal dynamic force vector
    Eigen::VectorXd InternalForces;

    if( HasFixedNode(theNodes) ){
        //Allocate memory for velocity/acceleraton. 
        Eigen::VectorXd V(6); 
        Eigen::VectorXd A(6);

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
kin3DTruss2::ComputeSurfaceForces(const std::shared_ptr<Load> &surfaceLoad, unsigned int UNUSED(face)){
    //Starts profiling this function.
    PROFILE_FUNCTION();

    //Local surface load vector:
    Eigen::VectorXd surfaceForces(6);

    //Gets the surface force:
    Eigen::VectorXd qs = surfaceLoad->GetLoadVector();

    //Transformation matrix to local coordinates.
    Eigen::MatrixXd localAxes = ComputeTransformationAxes();
    Eigen::MatrixXd RotationMatrix = localAxes.block(0,0,3,3);

    //Transform load from global to local coordinates.
    qs = RotationMatrix*qs;

    surfaceForces << qs(0)*Lo/2.0, 
                     qs(1)*Lo/2.0, 
                     qs(2)*Lo/2.0, 
                     qs(0)*Lo/2.0, 
                     qs(1)*Lo/2.0, 
                     qs(2)*Lo/2.0;

    //Node load vector in global coordinates.
    surfaceForces = localAxes.transpose()*surfaceForces;

    return surfaceForces;
}

//Compute the body forces acting on the element.
Eigen::VectorXd 
kin3DTruss2::ComputeBodyForces(const std::shared_ptr<Load> &bodyLoad, unsigned int k){
    //Starts profiling this function.
    PROFILE_FUNCTION();

    //Local body load vector:
    Eigen::VectorXd bodyForces(6);

    //Gets material properties:
    double rho = theMaterial->GetDensity();

    //Gets the body force:
    Eigen::VectorXd qb = A*rho*bodyLoad->GetLoadVector(k);

    //Transformation matrix to local coordinates.
    Eigen::MatrixXd localAxes = ComputeTransformationAxes();
    Eigen::MatrixXd RotationMatrix = localAxes.block(0,0,3,3);

    //Transform load into local coordinates.
    qb = RotationMatrix*qb;

    bodyForces << qb(0)*Lo/2.0, 
                  qb(1)*Lo/2.0, 
                  qb(2)*Lo/2.0, 
                  qb(0)*Lo/2.0, 
                  qb(1)*Lo/2.0, 
                  qb(2)*Lo/2.0;

    //Node load vector in global coordinates.
    bodyForces = localAxes.transpose()*bodyForces;

    return bodyForces;
}

//Compute the domain reduction forces acting on the element.
Eigen::VectorXd 
kin3DTruss2::ComputeDomainReductionForces(const std::shared_ptr<Load>& UNUSED(drm), unsigned int UNUSED(k)){
    //Starts profiling this function.
    PROFILE_FUNCTION();

    //TODO: Domain reduction forces are not implemented for Truss.
    Eigen::VectorXd DRMForces(6);
    DRMForces.fill(0.0);

    return DRMForces;
}

//Compute the current length of the element.
double
kin3DTruss2::ComputeLength() const{
    //Gets the element node coordinates. 
    Eigen::VectorXd xi = theNodes[0]->GetCoordinates() + theNodes[0]->GetDisplacements() + theNodes[0]->GetIncrementalDisplacements();
    Eigen::VectorXd xj = theNodes[1]->GetCoordinates() + theNodes[1]->GetDisplacements() + theNodes[1]->GetIncrementalDisplacements();

    //Current length of element:
    double L = (xj - xi).norm(); 

    return L;
}

//Compute/update the local axis of the element.
Eigen::VectorXd
kin3DTruss2::ComputeLocalAxes() const{
    //Gets the element node coordinates. 
    Eigen::VectorXd xi = theNodes[0]->GetCoordinates() + theNodes[0]->GetDisplacements() + theNodes[0]->GetIncrementalDisplacements();
    Eigen::VectorXd xj = theNodes[1]->GetCoordinates() + theNodes[1]->GetDisplacements() + theNodes[1]->GetIncrementalDisplacements();

    //The global axes transformation: 
    Eigen::VectorXd localAxes(6);
    Eigen::Vector3d v1;

    //Local axis 1:
    v1 = xj - xi;
    v1 = v1/v1.norm();

    localAxes << -v1(0), -v1(1), -v1(2), v1(0), v1(1), v1(2);

    return localAxes;
}

//Compute/update the geometric global axis of the element.
Eigen::MatrixXd 
kin3DTruss2::ComputeGeometricAxes() const{
    //The global axes transformation: 
    Eigen::MatrixXd geometricAxes(3,6);
    geometricAxes << 1.0, 0.0, 0.0, -1.0,  0.0,  0.0,
                     0.0, 1.0, 0.0,  0.0, -1.0,  0.0,
                     0.0, 0.0, 1.0,  0.0,  0.0, -1.0;

    return geometricAxes;
}

double 
kin3DTruss2::ComputeAxialForce() const{
    //Gets the element node coordinates. 
    Eigen::VectorXd xi = theNodes[0]->GetCoordinates() + theNodes[0]->GetDisplacements();
    Eigen::VectorXd xj = theNodes[1]->GetCoordinates() + theNodes[1]->GetDisplacements();

    //Current length of element:
    double L = (xj - xi).norm(); 

    Eigen::VectorXd strain(1);
    strain << (L - Lo)/Lo;

    //TODO: (IMPORTANT!!!) Careful here, the TangentStiffness must be computed for this strain.
    //it will work for linear elastic materials.    
    Eigen::MatrixXd E = theMaterial->GetTangentStiffness(); 

    double N = E(0,0)*A*strain(0);

    return N;
}

//Compute/update the local axis of the element.
Eigen::MatrixXd
kin3DTruss2::ComputeTransformationAxes() const{
    //Gets the element node coordinates. 
    Eigen::VectorXd xi = theNodes[0]->GetCoordinates() + theNodes[0]->GetDisplacements() + theNodes[0]->GetIncrementalDisplacements();
    Eigen::VectorXd xj = theNodes[1]->GetCoordinates() + theNodes[1]->GetDisplacements() + theNodes[1]->GetIncrementalDisplacements();

    //The global axes transformation: 
    Eigen::MatrixXd transformationAxes;
    transformationAxes.resize(6,6);

    Eigen::Vector3d v1;
    Eigen::Vector3d v2;
    Eigen::Vector3d v3;

    //Local axis 1:
    v1 = xj - xi;
    v1 = v1/v1.norm();

    //Local Axis 3:
    if(fabs(v1(2)) > TOL){
        v3 << 0.0, v1(2), -v1(1);
        v3 = v3/v3.norm();
    }
    else{
        v3 << v1(1), -v1(0), 0.0;
        v3 = v3/v3.norm();
    }

    //Local Axis 2:
    v2 = v3.cross(v1);
    v2 = v2/v2.norm();

    transformationAxes << v1(0), v1(1), v1(2),  0.0,   0.0,   0.0,
                          v2(0), v2(1), v2(2),  0.0,   0.0,   0.0,
                          v3(0), v3(1), v3(2),  0.0,   0.0,   0.0,
                            0.0,   0.0,   0.0, v1(0), v1(1), v1(2),
                            0.0,   0.0,   0.0, v2(0), v2(1), v2(2),
                            0.0,   0.0,   0.0, v3(0), v3(1), v3(2);
     
    return transformationAxes;
}

//Update strain in the element.
Eigen::VectorXd 
kin3DTruss2::ComputeStrain() const{
    //Current length of element.
    double L = ComputeLength();
    
    //Strain vector definition.
    Eigen::VectorXd strain(1);
    strain << (L - Lo)/Lo;

    return strain;
}

//Update strain rate in the element.
Eigen::VectorXd 
kin3DTruss2::ComputeStrainRate() const{
    //Gets the element coordinates in deformed configuration.  
    Eigen::VectorXd xi = theNodes[0]->GetCoordinates() + theNodes[0]->GetDisplacements() + theNodes[0]->GetIncrementalDisplacements();
    Eigen::VectorXd xj = theNodes[1]->GetCoordinates() + theNodes[1]->GetDisplacements() + theNodes[1]->GetIncrementalDisplacements();

    //Gets the element velocities in undeformed configuration.  
    Eigen::VectorXd Vi = theNodes[0]->GetVelocities();
    Eigen::VectorXd Vj = theNodes[1]->GetVelocities();

    //Local axis-1.
    Eigen::Vector3d u1;
    u1 = xj - xi;
    u1 = u1/u1.norm();

    //Current length of element:
    double rate = u1.dot(Vj - Vi)/Lo; 

    //Strain vector definition:
    Eigen::VectorXd strainrate(1);
    strainrate << rate;

    return strainrate;
}

//Compute the initial stiffness matrix of the element.
Eigen::MatrixXd 
kin3DTruss2::ComputeInitialStiffnessMatrix() const{
    //Gets material tangent matrix.
    Eigen::MatrixXd E = theMaterial->GetInitialTangentStiffness();

    //Gets the element node coordinates. 
    Eigen::VectorXd Xi = theNodes[0]->GetCoordinates();
    Eigen::VectorXd Xj = theNodes[1]->GetCoordinates();

    //Local axis 1:
    Eigen::Vector3d v1 = (Xj - Xi)/Lo;

    //The global axes transformation: 
    Eigen::VectorXd LocalAxes(6);
    LocalAxes << -v1(0), -v1(1), -v1(2), v1(0), v1(1), v1(2);

    //The initial stiffness matrix.
    Eigen::MatrixXd StiffnessMatrix = E(0,0)*A/Lo*LocalAxes*LocalAxes.transpose();

    return StiffnessMatrix;
}

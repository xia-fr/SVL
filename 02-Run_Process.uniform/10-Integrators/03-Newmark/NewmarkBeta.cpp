#include "NewmarkBeta.hpp"
#include "Definitions.hpp"
#include "Profiler.hpp"

//Overload constructor.
NewmarkBeta::NewmarkBeta(std::shared_ptr<Mesh> &mesh, double TimeStep, double mtol, double ktol, double ftol) : 
Integrator(mesh), dt(TimeStep){
    //Allocate memory for total state vector. 
    U.resize(numberOfTotalDofs); U.fill(0.0);
    V.resize(numberOfTotalDofs); V.fill(0.0);
    A.resize(numberOfTotalDofs); A.fill(0.0);

    //Allocate memory for total model matrices. 
    M.resize(numberOfTotalDofs, numberOfTotalDofs); 
    C.resize(numberOfTotalDofs, numberOfTotalDofs);
    K.resize(numberOfTotalDofs, numberOfTotalDofs);

    //Creates the dynamic assembler for this integrator.
    theAssembler = std::make_unique<Assembler>();
    theAssembler->SetMassTolerance(mtol);
    theAssembler->SetForceTolerance(ftol);
    theAssembler->SetStiffnessTolerance(ktol);

    //Assemble the external force vector from previous analysis.
    Fbar = theAssembler->ComputeProgressiveForceVector(mesh);
}

//Default destructor.
NewmarkBeta::~NewmarkBeta(){
    //Does nothing.
}

//Initialize model matrices.
void 
NewmarkBeta::Initialize(std::shared_ptr<Mesh> &mesh){
    //Starts profiling this function.
    PROFILE_FUNCTION();

    //Sets up Initial Condition from previous simulation
    std::map<unsigned int, std::shared_ptr<Node> > Nodes = mesh->GetNodes();  
    for(auto it : Nodes){
        auto &Tag = it.first;

        //Gets the associated nodal degree-of-freedom.
        std::vector<int> TotalDofs = Nodes[Tag]->GetTotalDegreeOfFreedom();

        //Creates the nodal/incremental vector state.
        Eigen::VectorXd Uij = Nodes[Tag]->GetDisplacements();
        Eigen::VectorXd Vij = Nodes[Tag]->GetVelocities();
        Eigen::VectorXd Aij = Nodes[Tag]->GetAccelerations();

        for(unsigned int j = 0; j < TotalDofs.size(); j++){
            U(TotalDofs[j]) = Uij(j);
            V(TotalDofs[j]) = Vij(j);
            A(TotalDofs[j]) = Aij(j);
        }
    }

    //Computes the mass matrix of the model.
    M = theAssembler->ComputeMassMatrix(mesh);

    //Computes the total damping matrix.
    C = theAssembler->ComputeDampingMatrix(mesh);

}

//Sets the load combination to be used.
void 
NewmarkBeta::SetLoadCombination(std::shared_ptr<LoadCombo> &combo){
    theAssembler->SetLoadCombination(combo);
}

//Sets the algorithm  to be used.
void 
NewmarkBeta::SetAlgorithm(std::shared_ptr<Algorithm> &algorithm){
    theAlgorithm = algorithm;
}

//Gets the displacement vector.
const Eigen::VectorXd& 
NewmarkBeta::GetDisplacements(){
    //Starts profiling this function.
    PROFILE_FUNCTION();

    return U;
}    

//Gets the velocity vector.
const Eigen::VectorXd& 
NewmarkBeta::GetVelocities(){
    //Starts profiling this function.
    PROFILE_FUNCTION();

    return V;
}

//Gets the acceleration vector.
const Eigen::VectorXd& 
NewmarkBeta::GetAccelerations(){
    //Starts profiling this function.
    PROFILE_FUNCTION();

    return A;
}

//Gets the perfectly-matched layer history vector.
const Eigen::VectorXd& 
NewmarkBeta::GetPMLHistoryVector(){
    //Starts profiling this function.
    PROFILE_FUNCTION();

    //Empty PML history vector (not used).
    return Ubar;
}

//Computes a new time step.
bool 
NewmarkBeta::ComputeNewStep(std::shared_ptr<Mesh> &mesh, unsigned int k){
    //Starts profiling this function.
    PROFILE_FUNCTION();

    //Gets the shared_ptr information from the weak_ptr object.
    std::shared_ptr<Algorithm> p = theAlgorithm.lock();

    //Computes a new displacement increment.
    bool stop = p->ComputeNewIncrement(mesh, k);

    //Checks the solution has issues.
    if(stop) return stop;

    //Obtains the displacement increment from algorithm.
    Eigen::VectorXd dU = Total2FreeMatrix*(p->GetDisplacementIncrement());

    //Update displacement states.
    U += (dU + SupportMotion);

    //Update acceleration states.
    A = 4.0/dt/dt*dU - 4.0/dt*V - A;
    //A = (1.0-1.0/2.0/0.3025)*A - 1.0/0.3025/dt*V + 1.0/0.3025/dt/dt*dU;

    //Update velocity states.
    V = 2.0/dt*dU - V;
    //V = dt*(1.0-0.6/2.0/0.3025)*A + (1.0-0.6/0.3025)*V + (0.6/0.3025/dt)*dU;

    //Return the integrator status.
    return false;
}

//Gets the reaction force ins this step.
Eigen::VectorXd 
NewmarkBeta::ComputeReactionForce(std::shared_ptr<Mesh> &mesh, unsigned int k){
    //Starts profiling this function.
    PROFILE_FUNCTION();

    //Assemble the total external force vector.
    Eigen::VectorXd Fint = theAssembler->ComputeDynamicInternalForceVector(mesh);

    //Assemble the total external force vector.
    Eigen::VectorXd Fext = theAssembler->ComputeExternalForceVector(mesh, k);

    //Computes the reaction forces.
    Eigen::VectorXd Reaction = Fint - Fext - Fbar;

    return Reaction;
}

//Gets the external force vector for next phase analysis.
Eigen::VectorXd 
NewmarkBeta::ComputeProgressiveForce(std::shared_ptr<Mesh> &mesh, unsigned int k){
//Starts profiling this function.
    PROFILE_FUNCTION();

    //Assemble the total internal force vector.
    Eigen::VectorXd Fext = theAssembler->ComputeExternalForceVector(mesh, k);

    //Update the stage force vector.
    Eigen::VectorXd Force = Fext + Fbar;

    return Force;
}

//Gets the incremental nodal support motion vector.
void
NewmarkBeta::ComputeSupportMotionVector(std::shared_ptr<Mesh> &mesh, Eigen::VectorXd &Feff, double UNUSED(factor), unsigned int k){
    //TODO:Include some if statement that performs the addition only if there is support motion applied.
    //Starts profiling this function.
    PROFILE_FUNCTION();

    //Assembles the incremental support motion displacements.
    SupportMotion = theAssembler->ComputeSupportMotionIncrement(mesh, k-1);

    //Computes the required forces to impose these displacements.
    Eigen::VectorXd Lg = Total2FreeMatrix.transpose()*(K*SupportMotion);

    //Add the contribution to the current effective force vector.
    Feff -= Lg;
}

//Gets the effective force associated to this integrator.
void
NewmarkBeta::ComputeEffectiveForce(std::shared_ptr<Mesh> &mesh, Eigen::VectorXd &Feff, double UNUSED(factor), unsigned int k){
    //Starts profiling this function.
    PROFILE_FUNCTION();

    //Gets the shared_ptr information from the weak_ptr object.
    std::shared_ptr<Algorithm> p = theAlgorithm.lock();

    //Obtains the displacement increment from algorithm.
    Eigen::VectorXd dU = Total2FreeMatrix*(p->GetDisplacementIncrement());

    //Assemble the total external force vector.
    Eigen::VectorXd Fint = theAssembler->ComputeInternalForceVector(mesh);

    //Assemble the total internal force vector.
    Eigen::VectorXd Fext = theAssembler->ComputeExternalForceVector(mesh, k);

    //Computes the effective force vector.
    Fext = Fext + Fbar - Fint + M*(4.0/dt*V + A - 4.0/dt/dt*dU) + C*(V - 2.0/dt*dU);
    //Fext = Fext + Fbar - Fint + M*((1.0/0.3025/dt)*V - (1.0-1.0/2.0/0.3025)*A + (1.0/0.3025/dt/dt)*dU);

    //Impose boundary conditions on effective force vector.
    Feff = Total2FreeMatrix.transpose()*Fext;
}

//Gets the effective stiffness associated to this integrator.
void
NewmarkBeta::ComputeEffectiveStiffness(std::shared_ptr<Mesh> &mesh, Eigen::SparseMatrix<double> &Keff){
    //Starts profiling this function.
    PROFILE_FUNCTION();

    //Computes the stiffness matrix of the model.
    K = theAssembler->ComputeStiffnessMatrix(mesh);

    //Assemble the effective stiffness matrix.
    K = K + 4.0/dt/dt*M + 2.0/dt*C;
    //K = K + 1.0/0.3025/dt/dt*M;

    //Impose boundary conditions on effective stiffness matrix.
    Keff = Eigen::SparseMatrix<double>(Total2FreeMatrix.transpose())*K*Total2FreeMatrix;
}

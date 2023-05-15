#include "NewmarkBeta.hpp"
#include "Definitions.hpp"
#include "Profiler.hpp"

#include <iostream>
#include <stdio.h>
#include <fstream>
#include <chrono>

//#include <Eigen/SparseCholesky>
#include <Spectra/SymGEigsShiftSolver.h>
#include <Spectra/MatOp/SymShiftInvert.h>
#include <Spectra/MatOp/SparseSymMatProd.h>

// Modified to override user selection of corner frequencies, instead automates it by 
// inverse iteration estimate of first eigenvalue

const static Eigen::IOFormat CSVFormat(Eigen::StreamPrecision, Eigen::DontAlignCols, ", ", "\n");

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

    //Additions for EQL//
    // Calculates estimate of fundamental eigenvalue and overwrites
    // corner frequencies for Rayleigh damping in mesh with 
    // cf1 at the fundamental and cf2 at 3xfundamental
    std::map<unsigned int, std::shared_ptr<Damping> > Dampings = mesh->GetDampings();  
    for(auto it : Dampings){
        auto &Tag = it.first;

        // Gets name of relevant damping
        std::string name = Dampings[Tag] -> GetName();
        
        // If using automatic cf rayleigh damping
        if(strcasecmp(name.c_str(),"Autorayleigh") == 0) {
            std::cout << name << std::endl;
            std::cout << "INVERSE ITERATION FOR EIGENVALUES \n" << std::endl;

            //Gets the stiffness matrix of the model.
            K = theAssembler->ComputeStiffnessMatrix(mesh);

            //Impose boundary conditions on matrices
            Eigen::SparseMatrix<double> KK = Eigen::SparseMatrix<double>(Total2FreeMatrix.transpose())*K*Total2FreeMatrix;
            Eigen::SparseMatrix<double> MM = Eigen::SparseMatrix<double>(Total2FreeMatrix.transpose())*M*Total2FreeMatrix;

            // Matrix size info
            Eigen::MatrixXd MMdense = Eigen::MatrixXd(MM);
            std::cout << "Matrix size: [" << MMdense.rows() << "x" << MMdense.rows() << "] \n" << std::endl;

            // Timing info start
            auto start = std::chrono::high_resolution_clock::now();

            // Kx = lMx
            // K = A, M = B
            using OpType = Spectra::SymShiftInvert<double, Eigen::Sparse, Eigen::Sparse>;
            using BOpType = Spectra::SparseSymMatProd<double>;

            OpType op(KK, MM);
            BOpType Bop(MM);

            Spectra::SymGEigsShiftSolver<OpType, BOpType, Spectra::GEigsMode::ShiftInvert> 
                geigs(op, Bop, 5, 10, 0);

            // Initialize and compute
            geigs.init();
            geigs.compute(Spectra::SortRule::LargestMagn);

            // Retrieve results
            Eigen::VectorXd evalues;
            if (geigs.info() == Spectra::CompInfo::Successful) {
                evalues = geigs.eigenvalues();
            }

            // Timing info end
            auto stop = std::chrono::high_resolution_clock::now();
            auto dur = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start);
            std::cout << "Run time: " << dur.count() << "ms\n" << std::endl;

            Eigen::VectorXd ws = evalues.cwiseSqrt();
            std::cout << "Natural frequencies (rad/s):" << std::endl;
            std::cout << ws.reverse() << "\n" << std::endl;

            // Select first eigenvalue that is larger than tolerance level
            // for what is "zero" or "-nan"
            double eigenTol = 1.0e-4;
            double ww = ws[0];
            for (int eix = 1; eix < ws.rows(); eix++){
                if (ws[eix] < eigenTol){
                    break;
                }
                ww = ws[eix];
            }

            // Calculated corner frequencies from first eigenvalue
            //double w1 = sqrt(lambda);
            std::cout << "Selected pivot frequencies:" << std::endl;
            double w1 = ww;
            double w2 = 3*w1;
            std::cout << "w1: " << w1 << " rad/s (" << w1/(2.0*3.14159265358979) << " Hz)" << std::endl;
            std::cout << "w2: " << w2 << " rad/s (" << w2/(2.0*3.14159265358979) << " Hz) \n" << std::endl;

            // Sets the damping parameters according to calculated fundamental
            // dparams[0] := w1 = cf1, dparams[1] := w2 = cf2
            std::vector<double> dparams = Dampings[Tag]->GetParameters();
            dparams[0] = w1;
            dparams[1] = w2;
            dparams[2] = dt;
            
            Dampings[Tag]->SetParameters(dparams);
        
            // Debug check
            //std::vector<double> debugparams = Dampings[Tag]->GetParameters();
            //for(int i = 0; i<2; i++) {
            //    std::cout << debugparams[i] << std::endl;
            //}

            //Debug: Save calculated corner frequencies into a file
            //std::ofstream outfile;
            //outfile.open("autorayleigh_freqs.txt", std::ios_base::app);
            //outfile << "w1: " << w1 << " rad/s (" << w1/(2.0*3.14159265358979) << " Hz), ";
            //outfile << "w2: " << w2 << " rad/s (" << w2/(2.0*3.14159265358979) << " Hz) \n";
        }
        
    }
    /////////////////////

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

    //Update velocity states.
    V = 2.0/dt*dU - V;

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

    //Impose boundary conditions on effective stiffness matrix.
    Keff = Eigen::SparseMatrix<double>(Total2FreeMatrix.transpose())*K*Total2FreeMatrix;
}

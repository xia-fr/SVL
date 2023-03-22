#include <cmath>
#include <vector>
#include "Node.hpp"
#include "Load.hpp"
#include "Element.hpp"
#include "Assembler.hpp"
#include "Definitions.hpp"
#include "Profiler.hpp"

typedef Eigen::Triplet<double> T;

//Default Constructor:
Assembler::Assembler() : MassTolerance(1E-15), StiffnessTolerance(1E-15), ForceTolerance(1E-15){
    //Does nothing.
}

//Destructor:
Assembler::~Assembler(){
    //Does nothing.
}

//Set the mass assembly allowed tolerance.
void 
Assembler::SetMassTolerance(double tol){
    MassTolerance = tol;
}

//Set the force assembly allowed tolerance.
void 
Assembler::SetForceTolerance(double tol){
    ForceTolerance = tol;
}

//Set the stiffness assembly allowed tolerance.
void 
Assembler::SetStiffnessTolerance(double tol){
    StiffnessTolerance = tol;
}

//Sets the load combination to be used.
void 
Assembler::SetLoadCombination(std::shared_ptr<LoadCombo> &combo){
    LoadCombination = combo;
}

//Assemble stiffness matrix.    
Eigen::SparseMatrix<double>
Assembler::ComputeMassMatrix(std::shared_ptr<Mesh> &mesh){
    //Starts profiling this function.
    PROFILE_FUNCTION();

    //Global mass matrix.
    Eigen::SparseMatrix<double> MassMatrix(numberOfTotalDofs, numberOfTotalDofs);

    //Nodal mass matrix contribution.
    Eigen::SparseMatrix<double> NodeMassMatrix(numberOfTotalDofs, numberOfTotalDofs);
    AssembleNodalMass(mesh, NodeMassMatrix);

    //Element mass matrix contribution.
    Eigen::SparseMatrix<double> ElemMassMatrix(numberOfTotalDofs, numberOfTotalDofs);
    AssembleElementMass(mesh, ElemMassMatrix);
    
    //Assemble Nodal and Element COntribution.
    MassMatrix = ElemMassMatrix + NodeMassMatrix;

    return MassMatrix;
}

//Assemble stiffness matrix.    
Eigen::SparseMatrix<double>
Assembler::ComputeStiffnessMatrix(std::shared_ptr<Mesh> &mesh){
    //Starts profiling this function.
    PROFILE_FUNCTION();

    //Global stiffness matrix.
    Eigen::SparseMatrix<double> StiffnessMatrix(numberOfTotalDofs,numberOfTotalDofs);

    //Sparse matrix format.
    std::vector<T> tripletList;
    tripletList.reserve(ConsistentStorage);
    
    //Gets element information from the mesh.
    std::map<unsigned int, std::shared_ptr<Element> > Elements = mesh->GetElements();

    //Assembly stiffness matrix process:
    unsigned int sum = 0;

    for(auto it : Elements){
        auto &Tag = it.first;

        //Gets the element degree-of-freedom connectivity.
        std::vector<unsigned int> dofs = Elements[Tag]->GetTotalDegreeOfFreedom();

        //Gets the Stiffness matrix in global coordinates:
        Eigen::MatrixXd Ke = Elements[Tag]->ComputeStiffnessMatrix();

        //Assemble contribution of each element in mesh.
        for(unsigned int j = 0; j < dofs.size(); j++){
            for(unsigned int i = 0; i < dofs.size(); i++){
                if(fabs(Ke(i,j)) > StiffnessTolerance){
                    tripletList[sum] = T(dofs[i], dofs[j], Ke(i,j));
                    sum++;
                }
            }
        }

    }

    //Builds the stiffness sparse matrix.
    StiffnessMatrix.setFromTriplets(tripletList.begin(), tripletList.begin() + sum);

    return StiffnessMatrix;
}

//Assemble damping matrix.    
Eigen::SparseMatrix<double>
Assembler::ComputeDampingMatrix(std::shared_ptr<Mesh> &mesh){
    //Starts profiling this function.
    PROFILE_FUNCTION();

    //Global damping matrix definition.
    Eigen::SparseMatrix<double> DampingMatrix(numberOfTotalDofs,numberOfTotalDofs);

    //Sparse matrix format.
    std::vector<T> tripletList;
    tripletList.reserve(ConsistentStorage);
    
    //Gets element information from the mesh.
    std::map<unsigned int, std::shared_ptr<Element> > Elements = mesh->GetElements();
    
    //Assembly damping matrix process:
    unsigned int sum = 0;
    
    for(auto it : Elements){
        auto &Tag = it.first;

        //Gets the element degree-of-freedom connectivity.
        std::vector<unsigned int> dofs = Elements[Tag]->GetTotalDegreeOfFreedom();
        
        //Gets the damping matrix in global coordinates:
        Eigen::MatrixXd Ce = Elements[Tag]->ComputeDampingMatrix();
        
        //Assemble contribution of each element in mesh.
        for(unsigned int j = 0; j < dofs.size(); j++){
            for(unsigned int i = 0; i < dofs.size(); i++){
                if(fabs(Ce(i,j)) > StiffnessTolerance){
                    tripletList[sum] = T(dofs[i], dofs[j], Ce(i,j));
                    sum++;
                }
            }
        }
    }
    
    //Builds the damping sparse matrix.
    DampingMatrix.setFromTriplets(tripletList.begin(), tripletList.begin() + sum);

    return DampingMatrix;
}

//Assemble the integrated history matrix for Perfectly-Matched Layer(PML)
Eigen::SparseMatrix<double> 
Assembler::ComputePMLHistoryMatrix(std::shared_ptr<Mesh> &mesh){
    //Starts profiling this function.
    PROFILE_FUNCTION();

    //Global stiffness matrix.
    Eigen::SparseMatrix<double> HistoryMatrix(numberOfTotalDofs,numberOfTotalDofs);

    //Sparse matrix format.
    std::vector<T> tripletList;
    tripletList.reserve(PMLStorage);
    
    //Gets element information from the mesh.
    std::map<unsigned int, std::shared_ptr<Element> > Elements = mesh->GetElements();

    //Assembly stiffness matrix process:
    unsigned int sum = 0;

    for(auto it : Elements){
        auto &Tag = it.first;

        //Gets the element degree-of-freedom connectivity.
        std::vector<unsigned int> dofs = Elements[Tag]->GetTotalDegreeOfFreedom();

        //Gets the Stiffness matrix in global coordinates:
        Eigen::MatrixXd Kpml = Elements[Tag]->ComputePMLMatrix();

        //Assemble contribution of each element in mesh.
        if(Kpml.size() > 0){
            for(unsigned int j = 0; j < dofs.size(); j++){
                for(unsigned int i = 0; i < dofs.size(); i++){
                    if(fabs(Kpml(i,j)) > StiffnessTolerance){
                        tripletList[sum] = T(dofs[i], dofs[j], Kpml(i,j));
                        sum++;
                    }
                }
            }
        }
    }

    //Builds the stiffness sparse matrix.
    HistoryMatrix.setFromTriplets(tripletList.begin(), tripletList.begin() + sum);

    return HistoryMatrix;
}

//Assemble the external force vector accumulated from previous analyses.
Eigen::VectorXd 
Assembler::ComputeProgressiveForceVector(std::shared_ptr<Mesh> &mesh){
    //Starts profiling this function.
    PROFILE_FUNCTION();

    //Sets the total force vector. 
    Eigen::VectorXd ForceVector(numberOfTotalDofs);
    ForceVector.fill(0.0);

    //Gets element information from the mesh.
    std::map<unsigned int, std::shared_ptr<Node> > Nodes = mesh->GetNodes();

    for(auto it : Nodes){
        auto &Tag = it.first;

        //Gets the node degree-of-freedom list.
        std::vector<int> dofs = Nodes[Tag]->GetTotalDegreeOfFreedom();

        //Gets the progressive force from previous analyses:
        Eigen::VectorXd Fprevious = Nodes[Tag]->GetProgressiveForces();

        //Assemble contribution of each node force in mesh.
        for(unsigned int j = 0; j < dofs.size(); j++){
            ForceVector(dofs[j]) += Fprevious(j);
        }
    }

    return ForceVector;
}

//Assembles the internal force vector.
Eigen::VectorXd
Assembler::ComputeInternalForceVector(std::shared_ptr<Mesh> &mesh){
    //Starts profiling this function.
    PROFILE_FUNCTION();

    //Sets the total force vector. 
    Eigen::VectorXd ForceVector(numberOfTotalDofs);
    ForceVector.fill(0.0);

    //Gets element information from the mesh.
    std::map<unsigned int, std::shared_ptr<Element> > Elements = mesh->GetElements();

    for(auto it : Elements){
        auto &Tag = it.first;

        //Gets the element degree-of-freedom connectivity.
        std::vector<unsigned int> dofs = Elements[Tag]->GetTotalDegreeOfFreedom();

        //Gets the Stiffness matrix in global coordinates:
        Eigen::VectorXd Fint = Elements[Tag]->ComputeInternalForces();

        //Assemble contribution of each element in mesh.
        for(unsigned int j = 0; j < dofs.size(); j++){
            if(fabs(Fint(j)) > ForceTolerance){
                ForceVector(dofs[j]) += Fint(j);
            }
        }
    }

    return ForceVector;
}

//Assemble the internal elastic, inertial, and viscous force vector.
Eigen::VectorXd 
Assembler::ComputeDynamicInternalForceVector(std::shared_ptr<Mesh> &mesh){
    //Starts profiling this function.
    PROFILE_FUNCTION();

    Eigen::VectorXd ForceVector(numberOfTotalDofs);
    ForceVector.fill(0.0);

    //Adds the inertial forces contribution associated to the nodes.
    AddNodeInertiaForces(mesh, ForceVector);

    //Adds the elastic, inertial, and viscous forces associated to the elements.
    AddElementDynamicForces(mesh, ForceVector);

    return ForceVector;
}

//Assembles the external force vector.
Eigen::VectorXd
Assembler::ComputeExternalForceVector(std::shared_ptr<Mesh> &mesh, unsigned int k){
    //Starts profiling this function.
    PROFILE_FUNCTION();
  
    //Sets the total force vector. 
    Eigen::VectorXd ForceVector(numberOfTotalDofs);
    ForceVector.fill(0.0);

    //Auxiliary force vector.
    Eigen::VectorXd Fext(numberOfTotalDofs);

    //Gets load and node information from the mesh.
    std::map<unsigned int, std::shared_ptr<Node> > Nodes = mesh->GetNodes();
    std::map<unsigned int, std::shared_ptr<Load> > Loads = mesh->GetLoads();

    std::vector<double> Factors = LoadCombination->GetLoadFactors();
    std::vector<unsigned int> IDs = LoadCombination->GetLoadCombination();

    for(unsigned int it = 0; it < IDs.size(); it++){
        //Initialize external force vector.
        Fext.fill(0.0);

        //Load identifier.
        unsigned int Tag = IDs[it];

        switch (Loads[Tag]->GetClassification()){
            case POINTLOAD_CONCENTRATED_CONSTANT: 
                {   //Static Node Point Load.
                    std::vector<unsigned int> LoadNodes = Loads[Tag]->GetNodes();

                    for(unsigned int j = 0; j < LoadNodes.size(); j++){
                        //Degree-of-freedom list of the node.
                        std::vector<int> totalDofs = Nodes[LoadNodes[j]]->GetTotalDegreeOfFreedom();

                        //Gets the static force vector.
                        Eigen::VectorXd Fn = Loads[Tag]->GetLoadVector();

                        //Assemble the force vector.
                        for(unsigned int i = 0; i < Fn.size(); i++)
                            Fext(totalDofs[i]) = Fn(i); 
                    }    
                    break; 
                }
            case POINTLOAD_CONCENTRATED_DYNAMIC: 
                {   //Dynamic Node Point Load.
                    std::vector<unsigned int> LoadNodes = Loads[Tag]->GetNodes();

                    for(unsigned int j = 0; j < LoadNodes.size(); j++){
                        //Degree-of-freedom list of the node.
                        std::vector<int> totalDofs = Nodes[LoadNodes[j]]->GetTotalDegreeOfFreedom();

                        //Gets the dynamic force vector.
                        Eigen::VectorXd Fn = Loads[Tag]->GetLoadVector(k);

                        //Assemble the force vector.
                        for(unsigned int i = 0; i < Fn.size(); i++)
                            Fext(totalDofs[i]) = Fn(i); 
                    }    
                    break;
                }
            case POINTLOAD_BODY_CONSTANT: 
                {   //Static Node (Body) Point Load.
                    std::vector<unsigned int> LoadNodes = Loads[Tag]->GetNodes();

                    for(unsigned int j = 0; j < LoadNodes.size(); j++){
                        //Gets the static force vector.
                        Eigen::VectorXd Fn = Loads[Tag]->GetLoadVector();

                        //Gets the associated mass to this node.
                        Eigen::VectorXd Mn = Nodes[LoadNodes[j]]->GetMass();

                        //Assemble the force vector.
                        if(Mn.size() != 0){
                            //Degree-of-freedom list of the node.
                            std::vector<int> totalDofs = Nodes[LoadNodes[j]]->GetTotalDegreeOfFreedom();

                            for(unsigned int i = 0; i < Fn.size(); i++)
                                Fext(totalDofs[i]) = Fn(i)*Mn(i);
                        } 
                    }
                    break; 
                }
            case POINTLOAD_BODY_DYNAMIC: 
                {   //Dynamic Node (Body) Point Load.
                    std::vector<unsigned int> LoadNodes = Loads[Tag]->GetNodes();

                    for(unsigned int j = 0; j < LoadNodes.size(); j++){
                        //Gets the static force vector.
                        Eigen::VectorXd Fn = Loads[Tag]->GetLoadVector(k);

                        //Gets the associated mass to this node.
                        Eigen::VectorXd Mn = Nodes[LoadNodes[j]]->GetMass();

                        //Assemble the force vector.
                        if(Mn.size() != 0){
                            //Degree-of-freedom list of the node.
                            std::vector<int> totalDofs = Nodes[LoadNodes[j]]->GetTotalDegreeOfFreedom();

                            for(unsigned int i = 0; i < Fn.size(); i++)
                                Fext(totalDofs[i]) = Fn(i)*Mn(i);
                        } 
                    }
                    break;
                }
            case ELEMENTLOAD_SURFACE_CONSTANT: 
                {   //Static Element Surface Load. 
                    std::vector<unsigned int> LoadFaces    = Loads[Tag]->GetFaces();
                    std::vector<unsigned int> LoadElements = Loads[Tag]->GetElements();

                    std::map<unsigned int, std::shared_ptr<Element> > Elements = mesh->GetElements();

                    for(unsigned int j = 0; j < LoadElements.size(); j++){
                        //Load identifier.
                        unsigned int eTag = LoadElements[j];

                        //Gets the element degree-of-freedom connectivity.
                        std::vector<unsigned int> totalDofs = Elements[eTag]->GetTotalDegreeOfFreedom();

                        //Gets the force vector.
                        Eigen::VectorXd Fn = Elements[eTag]->ComputeSurfaceForces(Loads[Tag], LoadFaces[j]);
    
                        //Assemble the force vector.
                        for(unsigned int i = 0; i < Fn.size(); i++)
                            Fext(totalDofs[i]) += Fn(i); 
                    }
                    break;
                }
            case ELEMENTLOAD_BODY_CONSTANT: 
                {   //Static Element Body (volume) Load.
                    std::vector<unsigned int> LoadElements = Loads[Tag]->GetElements();
                    std::map<unsigned int, std::shared_ptr<Element> > Elements = mesh->GetElements();

                    for(unsigned int j = 0; j < LoadElements.size(); j++){
                        //Load identifier.
                        unsigned int eTag = LoadElements[j];

                        //Gets the element degree-of-freedom connectivity.
                        std::vector<unsigned int> totalDofs = Elements[eTag]->GetTotalDegreeOfFreedom();

                        //Gets the force vector.
                        Eigen::VectorXd Fn = Elements[eTag]->ComputeBodyForces(Loads[Tag]);
    
                        //Assemble the force vector.
                        for(unsigned int i = 0; i < Fn.size(); i++)
                            Fext(totalDofs[i]) += Fn(i); 
                    }
                    break;
                }
            case ELEMENTLOAD_BODY_DYNAMIC: 
                {   //Dynamic Element Body (volume) Load. 
                    std::vector<unsigned int> LoadElements = Loads[Tag]->GetElements();
                    std::map<unsigned int, std::shared_ptr<Element> > Elements = mesh->GetElements();

                    for(unsigned int j = 0; j < LoadElements.size(); j++){
                        //Load identifier.
                        unsigned int eTag = LoadElements[j];

                        //Gets the element degree-of-freedom connectivity.
                        std::vector<unsigned int> totalDofs = Elements[eTag]->GetTotalDegreeOfFreedom();

                        //Gets the force vector.
                        Eigen::VectorXd Fn = Elements[eTag]->ComputeBodyForces(Loads[Tag], k);
    
                        //Assemble the force vector.
                        for(unsigned int i = 0; i < Fn.size(); i++)
                            Fext(totalDofs[i]) += Fn(i); 
                    }
                    break;
                }
            case ELEMENTLOAD_DOMAIN_REDUCTION: 
                {   //Dynamic Element Domain-Reduction Forces.
                    std::vector<unsigned int> elemID = Loads[Tag]->GetElements();
                    std::map<unsigned int, std::shared_ptr<Element> > Elements = mesh->GetElements();

                    //Loop over al DRM elements.
                    for(unsigned int i = 0; i < elemID.size(); i++){
                        //Gets the element degree-of-freedom connectivity.
                        std::vector<unsigned int> totalDofs = Elements[elemID[i]]->GetTotalDegreeOfFreedom();

                        //Gets the force vector.
                        Eigen::VectorXd Fn = Elements[elemID[i]]->ComputeDomainReductionForces(Loads[Tag], k);

                        //Assemble the force vector.
                        for(unsigned int j = 0; j < Fn.size(); j++)
                            Fext(totalDofs[j]) += Fn(j); 
                    }
                    break;
                }
            default:
                {   //Does nothing.
                }
        }

        //Adds the effective/total force.
        ForceVector += Factors[it]*Fext;
    }

    return ForceVector;
}

//Assembles the support motion vector.
Eigen::VectorXd
Assembler::ComputeSupportMotionIncrement(std::shared_ptr<Mesh> &mesh, unsigned int k){
    PROFILE_FUNCTION();
  
    //Sets the total force vector. 
    Eigen::VectorXd SupportMotion(numberOfTotalDofs);
    SupportMotion.fill(0.0);

    //Gets load and node information from the mesh.
    std::map<unsigned int, std::shared_ptr<Node> > Nodes = mesh->GetNodes();
    std::map<unsigned int, std::shared_ptr<Load> > Loads = mesh->GetLoads();

    std::vector<double> Factors = LoadCombination->GetLoadFactors();
    std::vector<unsigned int> IDs = LoadCombination->GetLoadCombination();

    for(unsigned int it = 0; it < IDs.size(); it++){
        //Load identifier.
        unsigned int Tag = IDs[it];

        if(Loads[Tag]->GetClassification() == POINTLOAD_SUPPORT_MOTION){
            //Static Node Point Load.
            std::vector<unsigned int> LoadNodes = Loads[Tag]->GetNodes();

            for(unsigned int j = 0; j < LoadNodes.size(); j++){
                //Degree-of-freedom list of the node.
                std::vector<int> totalDofs = Nodes[LoadNodes[j]]->GetTotalDegreeOfFreedom();

                //Gets the static force vector.
                Eigen::VectorXd dg = Nodes[LoadNodes[j]]->GetSupportMotion(k); 
                if(k > 0)
                    dg -= Nodes[LoadNodes[j]]->GetSupportMotion(k-1);

                //Assemble the force vector.
                for(unsigned int i = 0; i < dg.size(); i++)
                    SupportMotion(totalDofs[i]) += Factors[it]*dg(i);
            }
        }
    }

    return SupportMotion;
}

//Assemble the integrated history vector for Perfectly-Matched Layer (PML).
Eigen::VectorXd 
Assembler::ComputePMLHistoryVector(std::shared_ptr<Mesh> &mesh){
    //Starts profiling this function.
    PROFILE_FUNCTION();

    //Sets the total force vector. 
    Eigen::VectorXd HistoryVector(numberOfTotalDofs);
    HistoryVector.fill(0.0);

    //Gets element information from the mesh.
    std::map<unsigned int, std::shared_ptr<Node> > Nodes = mesh->GetNodes();

    for(auto it : Nodes){
        auto &Tag = it.first;

        //Gets the element degree-of-freedom connectivity.
        std::vector<int> dofs = Nodes[Tag]->GetTotalDegreeOfFreedom();

        //Gets the Stiffness matrix in global coordinates:
        Eigen::VectorXd Fpml = Nodes[Tag]->GetPMLVector();

        //Assemble contribution of each PML in mesh.
        for(unsigned int j = 0; j < dofs.size(); j++){
            if(fabs(Fpml(j)) > ForceTolerance){
                HistoryVector(dofs[j]) += Fpml(j);
            }
        }
    }

    return HistoryVector;
}

//Adds the inertial forces contribution associated to the nodes.
void 
Assembler::AddNodeInertiaForces(std::shared_ptr<Mesh> &mesh, Eigen::VectorXd &DynamicForces){
    //Starts profiling this function.
    PROFILE_FUNCTION();

    //Gets element information from the mesh.
    std::map<unsigned int, std::shared_ptr<Node> > Nodes = mesh->GetNodes();
   
    //Loops over all nodes in mesh.
    for(auto it : Nodes){
        auto &Tag = it.first;

        //Obtains the node inertial force vector.
        Eigen::VectorXd Fn = Nodes[Tag]->GetInertialForces();

        if(Fn.size() != 0){
            //Gets the node degree-of-freedom.
            std::vector<int> dofs = Nodes[Tag]->GetTotalDegreeOfFreedom();

            //Assemble contribution of each Node mass in mesh.
            for(unsigned int k = 0; k < dofs.size(); k++)
                DynamicForces[dofs[k]] += Fn(k);
        }
    }
}

//Adds the elastic, inertial, and viscous forces associated to the elements.
void 
Assembler::AddElementDynamicForces(std::shared_ptr<Mesh> &mesh, Eigen::VectorXd &DynamicForces){
    //Starts profiling this function.
    PROFILE_FUNCTION();

    //Gets element information from the mesh.
    std::map<unsigned int, std::shared_ptr<Element>> Elements = mesh->GetElements();

    //Loops over all elements in mesh.
    for(auto it : Elements){
        auto &Tag = it.first;

        //Gets the element degree-of-freedom connectivity.
        std::vector<unsigned int> dofs = Elements[Tag]->GetTotalDegreeOfFreedom();

        //Obtains the element inertial, elastic and viscous force vector.
        Eigen::VectorXd Fe = Elements[Tag]->ComputeInternalDynamicForces();

        if(Fe.size() != 0){
            //Assemble contribution of each element in mesh.
            for(unsigned int k = 0; k < dofs.size(); k++)
                DynamicForces[dofs[k]] += Fe(k);
        }
    }
}

//Assemble Nodal Mass matrix.    
void 
Assembler::AssembleNodalMass(std::shared_ptr<Mesh> &mesh, Eigen::SparseMatrix<double>& MassMatrix){
    //Starts profiling this function.
    PROFILE_FUNCTION();

    //Sparse matrix format.
    std::vector<T> tripletList;
    tripletList.reserve(LumpedStorage);

    //Gets element information from the mesh.
    std::map<unsigned int, std::shared_ptr<Node> > Nodes = mesh->GetNodes();
    
    //Assembly mass matrix process:
    unsigned int sum = 0;

    for(auto it : Nodes){
        auto &Tag = it.first;

        Eigen::VectorXd Mn = Nodes[Tag]->GetMass();
        if(Mn.size() != 0){
            //Gets the node degree-of-freedom.
            std::vector<int> dofs = Nodes[Tag]->GetTotalDegreeOfFreedom();

            //Assemble contribution of each Node mass in mesh.
            for(unsigned int k = 0; k < dofs.size(); k++){
                if(fabs(Mn(k)) > MassTolerance){
                    tripletList[sum] = T(dofs[k], dofs[k], Mn(k));
                    sum++;
                }
            }
        }
    }

    //Builds the nodal mass sparse matrix.
    MassMatrix.setFromTriplets(tripletList.begin(), tripletList.begin()+sum);
}

//Assemble Element Mass matrix.    
void 
Assembler::AssembleElementMass(std::shared_ptr<Mesh> &mesh, Eigen::SparseMatrix<double>& MassMatrix){
    //Starts profiling this function.
    PROFILE_FUNCTION();

    //Sparse matrix format.
    std::vector<T> tripletList;
    tripletList.reserve(ConsistentStorage);

    //Gets element information from the mesh.
    std::map<unsigned int, std::shared_ptr<Element> > Elements = mesh->GetElements();

    //Assembly mass matrix process:
    unsigned int sum = 0;

    for(auto it : Elements){
        auto &Tag = it.first;

        //Gets the element degree-of-freedom connectivity.
        std::vector<unsigned int> dofs = Elements[Tag]->GetTotalDegreeOfFreedom();

        //Gets the Stiffness matrix in global coordinates:
        Eigen::MatrixXd Me = Elements[Tag]->ComputeMassMatrix();

        //Assemble contribution of each element in mesh.
        for(unsigned int j = 0; j < dofs.size(); j++){
            for(unsigned int i = 0; i < dofs.size(); i++){
                if(fabs(Me(i,j)) > MassTolerance){
                    tripletList[sum] = T(dofs[i], dofs[j], Me(i,j));
                    sum++;
                }
            }
        }
    }

    //Builds the mass sparse matrix.
    MassMatrix.setFromTriplets(tripletList.begin(), tripletList.begin()+sum);
}

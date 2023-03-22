//==============================================================================
//
//                       Seismo Virtual Laboratory
//             Module for Serial and Parallel Analysis of seismic 
//         wave propagation and soil-structure interaction simulation
//         Copyright (C) 2018-2021, The California Institute of Technology
//                         All Rights Reserved.
//
// Commercial use of this program without express permission of the California
// Institute of Technology, is strictly  prohibited. See  file "COPYRIGHT"  in
// main  directory  for  information on  usage  and  redistribution, and for a
// DISCLAIMER OF ALL WARRANTIES.
//
//==============================================================================
//
// Written by:
//   Danilo S. Kusanovic (dkusanov@caltech.edu)
//   Elnaz E. Seylabi    (elnaze@unr.edu)
//
// Supervised by:
//   Domniki M. Asimaki  (domniki@caltech.edu)
//
// References : 
//   [1]
//
// Description:
///This file contains the "Static Analysis object" declarations, and updates 
///information in the mesh, and performs the incremental analysis in a 
///'NumberOfSteps' for which the state variable are computed. 
//------------------------------------------------------------------------------

#ifndef _STATICANALYSIS_HPP_
#define _STATICANALYSIS_HPP_

#include <memory>
#include <Eigen/Dense>

#include "Mesh.hpp"
#include "Analysis.hpp"
#include "Algorithm.hpp"
#include "Integrator.hpp"
#include "LoadCombo.hpp"

/// @author    Danilo S. Kusanovic (dkusanov@caltech.edu)
/// @date      November 25, 2018
/// @version   1.0
/// @file      StaticAnalysis.hpp
/// @class     StaticAnalysis
/// @see       Analysis.hpp Integrator.hpp Algorithm.hpp Mesh.hpp
/// @brief     Class for creating an static analysis and updating the states variables in mesh 
class StaticAnalysis : public Analysis {

    public:
        ///Creates a StaticAnalysis object.
        ///@param mesh Pointer to the Mesh container to extract Node and Element.
        ///@param algorithm Pointer to the Algorithm to solve the linear system.
        ///@param integrator Pointer to the Integrator to evolve the solution.
        ///@param loadcombo Pointer to the LoadCombo to load the finite element model.
        ///@param nSteps The current time step being solved.
        ///@note More details can be found at @ref linkStaticAnalysis.
        ///@see StaticAnalysis::Mesh, StaticAnalysis::theIntegrator.
        StaticAnalysis(std::shared_ptr<Mesh> &mesh, std::shared_ptr<Algorithm> &algorithm, std::shared_ptr<Integrator> &integrator, std::shared_ptr<LoadCombo> &loadcombo, unsigned int nSteps=0);

        ///Destroys this StaticAnalysis object.
        ~StaticAnalysis();

        ///Analyze the current incremental step.
        ///@return Whether or not the analysis was successful.
        ///@note More details can be found at @ref linkStaticAnalysis.
        bool Analyze();

        ///Performs changes in mesh.
        ///param k The time step to be updated.
        ///@note Changes in Node and Material at each Element are committed.
        void UpdateDomain(unsigned int k);

    private:
        ///Total number of time increments.
        unsigned int NumberOfSteps;

        ///The finite element mesh:
        std::shared_ptr<Mesh> theMesh;

        ///The linear system algorithm.
        std::shared_ptr<Algorithm> theAlgorithm;

        ///The static integrator method.
        std::shared_ptr<Integrator> theIntegrator;
};

#endif

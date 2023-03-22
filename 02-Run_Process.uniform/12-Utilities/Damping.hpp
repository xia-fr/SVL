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
///This file contains the "Damping object" declarations to update damping 
///matrix to account for Caughey-type damping models.
//------------------------------------------------------------------------------

#ifndef _DAMPING_HPP_
#define _DAMPING_HPP_

#include <string>
#include <vector>
#include <memory>
#include <Eigen/Dense>

/// @author    Elnaz E. Seylabi (elnaze@unr.edu)
/// @date      July 18, 2018
/// @version   1.0
/// @file      Damping.hpp
/// @class     Damping
/// @see       Element.hpp Mesh.hpp
/// @brief     Class for applying a certain damping model to a group of elements
class Damping{

    public:
        ///Creates a Damping object.
        ///@param name Name of the damping model.
        ///@param parameters Parameters that define the damping model.
        ///@note More details can be found at @ref linkDamping.
        ///@see Damping::Name, Damping::Parameters.
        Damping(std::string name, const std::vector<double> parameters, 
                const std::vector<double> wc,
                const std::vector<double> xc);

        ///Destroys this Damping object.
        ~Damping();

        ///Gets name of the damping model.
        ///@return The name of damping model.
        ///@note More details can be found at @ref linkDamping.
        std::string GetName();

        ///Gets parameters of the damping model.
        ///@return The list of parameters of damping model.
        ///@note More details can be found at @ref linkDamping.
        std::vector<double> GetParameters();

        ///Gets
        ///@return 
        ///@note More details can be found at @ref linkDamping.
        std::vector<double> GetFilterFreqs();

        ///Gets
        ///@return 
        ///@note More details can be found at @ref linkDamping.
        std::vector<double> GetFilterCoeffs();

		///Set the name of the damping model.
        ///@param name The name of damping model.
        ///@note More details can be found at @ref linkDamping.
        void SetName(std::string name);

        ///Set the damping ratio specifically.
        ///@param dampingRatio The damping ratio.
        ///@note More details can be found at @ref linkDamping.
        void SetDampingRatio(double dampingRatio);

        ///Set the damping parameters.
        ///@param param The vector of parameters.
        ///@note More details can be found at @ref linkDamping.
        void SetParameters(std::vector<double> param);

        ///
        ///@param freqs
        ///@note More details can be found at @ref linkDamping.
        void SetFilterFreqs(std::vector<double> freqs);

        ///
        ///@param coeffs
        ///@note More details can be found at @ref linkDamping.
        void SetFilterCoeffs(std::vector<double> coeffs);

        // NEW FOR UNIFORM DAMPING: 

        ///
        ///@param f Current internal (structural) force
        ///@note More details can be found at @ref linkDamping.
        void UpdateDamping(Eigen::VectorXd f);

        /// 
        ///@note More details can be found at @ref linkDamping.
        void CommitDamping();

        /// 
        ///@note More details can be found at @ref linkDamping.
        void ReverseDamping();

        /// 
        ///@note More details can be found at @ref linkDamping.
        void InitialDamping(unsigned int nComp);

        ///Gets damping force
        ///@return 
        ///@note More details can be found at @ref linkDamping.
        Eigen::VectorXd GetDampingForce();
  
        ///Gets stiffness multiplier
        ///@return 
        ///@note More details can be found at @ref linkDamping.
        double GetStiffnessMultiplier(void);

        ///Gets copy of damping
        ///@return 
        ///@note More details can be found at @ref linkDamping.
        std::unique_ptr<Damping> CopyDamping();

    private:
        ///Damping model name.
        std::string Name;
        
        ///Damping model parameters.
        // For uniform damping:
        // Parameters[0] = damping ratio
        // Parameters[1] = dt
        std::vector<double> Parameters;

        /// Vector of cutoff frequencies.
        std::vector<double> FilterFreqs;

        /// Vector of filter coefficients.
        std::vector<double> FilterCoeffs;

        /// Vector of EWMA coefficients.
        std::vector<double> alpha;

        // Uniform damping current force variables
        Eigen::VectorXd f0;
        Eigen::VectorXd fd;
        Eigen::MatrixXd fL;

        // Uniform damping converged force variables
        Eigen::VectorXd f0C;
        Eigen::VectorXd fdC;
        Eigen::MatrixXd fLC;

};

#endif

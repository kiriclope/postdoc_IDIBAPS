#ifndef __MFRATES__
#define __MFRATES__

#include <iostream>
#include <eigen3/Eigen/Dense>

using namespace Eigen ;

void mean_field_rates(int n_pop, double* ext_inputs, double* J, double* &mf_rates) {
  
  mf_rates = new double [n_pop]() ;

  MatrixXd A(n_pop,n_pop) ;
  VectorXd b(n_pop) ;
  
  for(int i=0;i<n_pop;i++) {
    b(i) = ext_inputs[i]*m0 ; 
    for(int j=0;j<n_pop;j++) 
      A(i,j) = J[j+i*n_pop] ; 
  }
  
  VectorXd x(n_pop) ; 
  x = A.colPivHouseholderQr().solve(-b) * 1000. ;
  
  for(int i=0;i<n_pop;i++)
    mf_rates[i] = x(i) ; 
  
}

#endif

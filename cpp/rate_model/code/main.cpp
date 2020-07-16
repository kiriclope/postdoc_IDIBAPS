#include "librairies.h"
#include "globals.h"
#include "net_utils.h"
#include "mat_utils.h"
#include "mean_field.h"

clock_t t1=clock();

int main(int argc , char** argv) {

  unsigned long int i,j,k ;

  // default_random_engine gen  ;
  random_device rd;
  default_random_engine rand_gen( rd() ); 
  mt19937 rand_gen_meyer(rd()) ; 
  // rand_gen_meyer.seed(1) ;

  // for(i=0;i<5;i++)
  //   cout << rand_gen() << " ";
  // cout << endl ;

  // for(i=0;i<5;i++)
  //   cout << rand_gen_meyer() << " ";
  // cout << endl ;

  
  int pre_pop, post_pop ; 
  double t_window=0., percentage=0. ;
  
  double (*transfert_func)(double) = NULL ; 
  transfert_func = &Phi ; 
  // transfert_func = &threshold_linear ; 
  
  string dir ; 
  int n_pop ;
  unsigned long n_neurons ;
  double K ;
  get_args(argc , argv, dir, n_pop, n_neurons, K) ;
  
  double *ext_inputs, *J, *Tsyn ; 
  get_param(n_pop, dir, ext_inputs, J, Tsyn) ;


  if(n_pop==1) {
    J[0] = J0 ;
    ext_inputs[0] = I0 ;
    Tsyn[0] = Tsyn0 ;

    char char_dir[30] ;
    sprintf(char_dir, "I0_%0.2f_J0_%0.2f", I0, -J0) ;
    string str_dir = string(char_dir) ;
    
    dir = str_dir ;
    cout << dir ;
  }   
  
  double *mf_rates ;
  mf_rates = new double [n_pop]() ;
  mean_field_rates(n_pop, ext_inputs, J, mf_rates) ;
  
  cout << "External Inputs : " ;
  for(int i=0;i<n_pop;i++) 
    cout << ext_inputs[i]*m0 << " ";
  cout << endl ;
  
  cout << "J : " ;
  for(int i=0;i<n_pop;i++) { // postsynaptic pop
    for(int j=0;j<n_pop;j++) // presynaptic pop
      cout << J[j+i*n_pop] << " ";
    cout << endl ;
  }

  cout << "Tsyn : " ;
  for(i=0;i<n_pop;i++) // postsynaptic pop
    for(j=0;j<n_pop;j++) // presynaptic pop
      cout << Tsyn[j+i*n_pop] << " " ;
  cout << endl ;
  
  ///////////////////////////////////////////////////////////////////    
  // Random Connectivity, J
  ///////////////////////////////////////////////////////////////////    
  
  int *n_pre ;
  unsigned long *id_pre, *idx_pre ;
  
  string con_path ;
  con_path = "/homecentral/alexandre.mahrach/gdrive/postdoc_IDIBAPS/connectivity/" ;
  con_path += to_string(n_pop) + "pop/N" + to_string(n_neurons) + "/K" + to_string((int)K) ;
  
  get_con_mat(con_path, n_neurons, n_pre, id_pre, idx_pre) ; 
  
  // cout << n_pre[0] << " " << id_pre[0] << " " << idx_pre[0] << endl ;
  
  ///////////////////////////////////////////////////////////////////    
  // Path
  ///////////////////////////////////////////////////////////////////    
  
  string path = "../" ;
  create_dir(dir, path, n_pop, n_neurons, K) ;
  
  n_neurons = n_neurons * 10000 ;
  
  int* n_per_pop ; 
  n_per_pop = new int [n_pop]() ; 

  for(i=0;i<n_pop;i++) 
    n_per_pop[i] = n_neurons/n_pop ; 
  
  int* cum_n_per_pop ; 
  cum_n_per_pop = new int [n_pop+1]() ; 

  cout <<"cum_n_per_pop=" << " " ; 
  for(i=0;i<n_pop+1;i++) { 
    for(j=0;j<i;j++) 
      cum_n_per_pop[i] += n_per_pop[j] ; 
    cout << cum_n_per_pop[i] << " " ;
  }
  cout << endl ;
 
  int *which_pop ; 
  which_pop = (int *) malloc( (unsigned long long) n_neurons * sizeof(int) ) ;

  for(i=0;i<n_pop;i++) 
    for(j=0; j<n_neurons; j++)
      if(j>=cum_n_per_pop[i] && j<cum_n_per_pop[i+1])
	which_pop[j] = i ; 

    ///////////////////////////////////////////////////////////////////    
  // Structured Connectivity, xi
  ///////////////////////////////////////////////////////////////////    

  double *xi ;
  xi = new double [n_neurons]() ; // gaussian vector
  normal_distribution<double> white_noise(0.0, 1.0) ;
  
  if(IF_LOW_RANK) {
    char char_mean_xi[10] ;
    sprintf(char_mean_xi, "%0.2f", MEAN_XI) ;
    string str_mean_xi = string(char_mean_xi) ;

    char char_var_xi[10] ;
    sprintf(char_var_xi, "%0.2f", VAR_XI) ;
    string str_var_xi = string(char_var_xi) ;
    
    path += "/low_rank/xi_" + str_mean_xi + "_mean_" + str_var_xi + "_var" ; 
    string mkdirp = "mkdir -p " ;
    mkdirp += path ;

    const char * cmd = mkdirp.c_str() ;
    const int dir_err = system(cmd) ;

    cout << path << endl ; 

    string sLow = path + "/low_rank_xi.dat" ; 
    ofstream file_low_rank_xi(sLow.c_str(), ios::out | ios::ate);

    for(i=0;i<n_neurons;i++) { 
      xi[i] = (MEAN_XI + sqrt(VAR_XI) * white_noise(rand_gen_meyer)) / sqrt( (double) n_per_pop[which_pop[i]] ) ; // xi_i * xi_j scales as 1/N 
      file_low_rank_xi << xi[i] << " "; 
    }
    
    file_low_rank_xi.close() ;

  }
  
  cout << "low rank vector: " ;
  for(i=0;i<5;i++)
      cout << xi[i] << " " ;
  cout << endl ;

  ///////////////////////////////////////////////////////////////////     
  // Scaling 
  /////////////////////////////////////////////////////////////////// 
  
  for(i=0;i<n_pop;i++) { 
    ext_inputs[i] = sqrt(K)*ext_inputs[i]*m0 ; 
    for(j=0;j<n_pop;j++) 
      J[j+i*n_pop] = J[j+i*n_pop]/sqrt(K) ; 
  }
    
  /////////////////////////////////////////////////////////////////// 
  // Variables 
  ///////////////////////////////////////////////////////////////////    

  double *rates ;
  rates = new double [n_neurons]() ; // instantaneous individual rates

  double *mean_rates ;
  mean_rates = new double [n_pop]() ; // population averaged rate also averaged over TIME_WINDOW
  
  double *filter_rates ;
  filter_rates = new double [n_neurons]() ; // temporal averaged over TIME_WINDOW

  double *kappa ; // overlap averaged over TIME_WINDOW
  kappa = new double [n_pop]() ; 

  // h^(ab)_i=h^b_i, inputs from presynaptic population b to postsynaptic neuron (i,a)
  double **inputs ;
  inputs = new double *[n_pop]() ;
  for(i=0;i<n_pop;i++) // presynaptic population b
    inputs[i] = new double [n_neurons]() ; 

  double **filter_inputs ;
  filter_inputs = new double *[n_pop]() ;
  for(i=0;i<n_pop;i++) // presynaptic population b
    filter_inputs[i] = new double [n_neurons]() ; 

  // htot_i = h^E_i + h^I_i, net input into neuron i
  double *net_inputs ;
  net_inputs = new double [n_neurons]() ;
  
  // S^(ab)_j=S^a_j, synapse from presynaptic neuron (j,b) to postsynaptic population a
  double **synapses ;
  synapses = new double *[n_pop]() ;
  for(i=0;i<n_pop;i++) //postsynaptic pop a
    synapses[i] = new double [n_neurons]() ;
  
  ///////////////////////////////////////////////////////////////////    
  // Files
  ///////////////////////////////////////////////////////////////////    

  string str_mean_rates = path + "/mean_rates.dat" ; 
  ofstream file_mean_rates(str_mean_rates.c_str(), ios::out | ios::ate);
  
  string str_filter_rates = path + "/filter_rates.dat" ; 
  ofstream file_filter_rates(str_filter_rates.c_str(), ios::out | ios::ate);

  string str_inputs = path + "/inputs.dat" ; 
  ofstream file_inputs(str_inputs.c_str(), ios::out | ios::ate);

  string str_overlap = path + "/overlap.dat" ; 
  ofstream file_overlap(str_overlap.c_str(), ios::out | ios::ate);

  ///////////////////////////////////////////////////////////////////    
  // Initial conditions
  ///////////////////////////////////////////////////////////////////    

  cout << "Initialization" << endl ;
  
  uniform_real_distribution<double> unif(0,2) ;
  // normal_distribution<double> gaussianI( 0, 1.0/8.0 ) ;
 
  for(i=0;i<n_neurons;i++) 
    net_inputs[i] = ext_inputs[which_pop[i]] ; 

  double sigma_0 ;
  sigma_0 = unif(rand_gen) ;
    
  for(i=0;i<n_pop;i++) 
    for(j=0;j<n_neurons;j++) {
      pre_pop = which_pop[j] ;
      synapses[i][j] = mf_rates[pre_pop] / (double) n_per_pop[pre_pop] + sigma_0 * white_noise(rand_gen) ; 
      net_inputs[j] += synapses[i][j] ;
    }  

  for(i=0;i<n_neurons;i++) 
    rates[i] = transfert_func(net_inputs[i]) ; 

  cout << "initial rates: " ;
  for(i=0;i<5;i++)
    cout << rates[i] << " " ;
  cout << endl ;
  
  ///////////////////////////////////////////////////////////////////    
  // Dynamics of the network : Rate model
  ///////////////////////////////////////////////////////////////////    
	
  cout << "Main loop :" ;
  cout << " duration " << DURATION << " | DT " << DT ; 
  cout << " | TIME_STEADY " << TIME_STEADY << " | TIME_WINDOW " << TIME_WINDOW << endl ;

  cout << "mean_field_rates: " ;
  for(i=0;i<n_pop;i++)
    cout << mf_rates[i] << " " ;
  cout << endl ;

  double *dt_over_tau ;
  dt_over_tau = new double [n_pop*n_pop]() ;
  for(i=0;i<n_pop;i++)
    for(j=0;j<n_pop;j++)
      dt_over_tau[j+i*n_pop] = DT/Tsyn[j+i*n_pop] ;

  double *one_minus_dt_over_tau ;
  one_minus_dt_over_tau = new double [n_pop*n_pop]() ;
  for(i=0;i<n_pop;i++)
    for(j=0;j<n_pop;j++)
      one_minus_dt_over_tau[j+i*n_pop] = 1.0-dt_over_tau[j+i*n_pop] ; 
  
  for (double t=0.; t<=DURATION; t+=DT) {

    percentage = t/DURATION ; 
    
    // cout << "updating synapses" << endl ;	
    for(i=0;i<n_pop;i++) // postsynaptic pop
      for(j=0;j<n_neurons;j++) { // presynaptic neuron
    	pre_pop = which_pop[j] ;
    	// synapses[i][j] = (1.0-DT/Tsyn[pre_pop+i*n_pop])*synapses[i][j] + (DT/Tsyn[pre_pop+i*n_pop])*rates[j] ; 
    	synapses[i][j] = one_minus_dt_over_tau[pre_pop+i*n_pop]*synapses[i][j] + dt_over_tau[pre_pop+i*n_pop]*rates[j] ; 
      }
    
    // cout << "reseting synaptic inputs" << endl ;
    for(i=0;i<n_pop;i++) // presynaptic pop
      for(j=0;j<n_neurons;j++) // postynaptic neuron
	inputs[i][j] = 0 ;
    
    // cout << "updating postsynaptic inputs" << endl ;// hai = sum_k Sak
    if(IF_LOW_RANK)
      for(i=0;i<n_neurons;i++) { // presynaptic
	post_pop = which_pop[i] ;
	for(j=0;j<n_neurons;j++) {  // postsynaptic neuron
	  pre_pop = which_pop[id_pre[j]] ;
	  inputs[pre_pop][i] += xi[pre_pop]*xi[post_pop] ;
	}
      }
    
    for(i=0;i<n_neurons;i++) { // postsynaptic neuron
      post_pop = which_pop[i] ;
      for(j=idx_pre[i]; j<idx_pre[i]+n_pre[i]; j++) { // sum over presynaptic neurons of i
	pre_pop = which_pop[id_pre[j]] ; 
	// if(id_pre[j]>=cum_n_per_pop[pre_pop] && id_pre[j]<cum_n_per_pop[pre_pop+1]) 
	inputs[pre_pop][i] += J[pre_pop+post_pop*n_pop]*synapses[post_pop][id_pre[j]] ;
      }
    }
        
    // cout << "reseting net inputs" << endl ;
    for(i=0;i<n_neurons;i++) // postsynaptic neuron
      net_inputs[i] = ext_inputs[which_pop[i]] ;
    
    // updating net inputs
    for(i=0;i<n_pop;i++)  // presynaptic pop
      for(j=0;j<n_neurons;j++) {// postsynaptic neuron
  	net_inputs[j] += inputs[i][j] ;
	filter_inputs[i][j] += inputs[i][j] ;    
      }
    
    // Updating Rates
    for(i=0;i<n_neurons;i++) {
      rates[i] = transfert_func(net_inputs[i]) ; 
	
      if(t>=TIME_STEADY) {
  	mean_rates[which_pop[i]] += rates[i] ; 
  	filter_rates[i] += rates[i] ;
	if(IF_LOW_RANK)
	  kappa[which_pop[i]] += xi[i]*rates[i] ; 
      }      
    }
  
    // Writing to files
    if(t_window<TIME_WINDOW) {
      cout << int(percentage*100.0) << "% " ; 
      cout << "\r" ;
      cout.flush() ; 
    }
    else{ 

      cout << int(percentage*100.0) << "% " ; 

      // mean rates
      cout << " t " << t-TIME_STEADY << " rates: ";
      for(i=0;i<n_pop;i++) {
      	cout << mean_rates[i]*DT/TIME_WINDOW/(double)n_per_pop[i] *1000. << " " ;
	if(IF_LOW_RANK) 
	  cout << "| J0 "<< J[0]*sqrt(K) << "| overlap: "<< kappa[i]*DT/TIME_WINDOW *1000.0 / sqrt( (double)n_per_pop[i]) ;
	
      }
      cout << "\r" ; 
      cout.flush() ; 
      // cout << endl ; 

      file_mean_rates << t-TIME_STEADY ; 
      for(i=0;i<n_pop;i++) {
      	file_mean_rates << " " << mean_rates[i]*DT/TIME_WINDOW/(double)n_per_pop[i] *1000. ; 
      	mean_rates[i] = 0 ; 
      } 
      file_mean_rates << endl ; 

      // overlap kappa
      if(IF_LOW_RANK) {
	file_overlap << t-TIME_STEADY ; 
	for(i=0;i<n_pop;i++) {
	  file_overlap << " " << kappa[i]*DT/TIME_WINDOW *1000.0/ sqrt( (double)n_per_pop[i]) ;
	  kappa[i] = 0.0 ;
	}
      }
      file_overlap << endl ;
      
      // J[0] -= 0.1/sqrt(K) ;
      
      // filtered rates over tw
      file_filter_rates << t-TIME_STEADY ;
      for(i=0;i<n_neurons;i++) {
  	file_filter_rates << " " << filter_rates[i]*DT/TIME_WINDOW *1000. ;
  	filter_rates[i] = 0 ;
      }
      file_filter_rates << endl ;

      // filtered inputs over tw
      file_inputs << t-TIME_STEADY ;
      for(i=0;i<n_pop;i++)
	for(j=0;j<n_neurons;j++) {
	  file_inputs << " " << filter_inputs[i][j]*DT/TIME_WINDOW ; 
	  filter_inputs[i][j] = 0 ; 
	} 
      file_inputs << endl ; 
      
      t_window=0. ;
      
    }//endif 

    // printProgress (percentage) ;
    //Defining time window 
    if(t>=TIME_STEADY)
      t_window += DT ;
    
  } //ENDMAINLOOP
  
    ///////////////////////////////////////////////////////////////////

  delete [] idx_pre ; 
  delete [] id_pre ; 
  delete [] n_pre ; 

  delete [] mean_rates ; 
  delete [] rates ; 
  delete [] filter_rates ; 

  delete [] synapses ; 
  delete [] inputs ; 
  delete [] filter_inputs ; 
  delete [] net_inputs ; 
  
  delete [] ext_inputs ; 
  delete [] J ; 

  delete [] xi ; 
  delete [] kappa ;
  
  delete [] n_per_pop ;
  delete [] cum_n_per_pop ;

  delete [] dt_over_tau ;
  delete [] one_minus_dt_over_tau ;
    
  file_mean_rates.close();
  file_filter_rates.close();
  file_inputs.close();
  file_overlap.close();
  
  cout << "done" << endl; 
  
  ///////////////////////////////////////////////////////////////////
  
  cout << "Simulation Done !" << endl;

  clock_t t2=clock();  
  int HOURS=0,MIN=0,SEC=0;
  string str_TIME = path + "/CPU_TIME.txt" ; 
  ofstream TIME(str_TIME.c_str(), ios::out | ios::ate);

  SEC = (t2-t1)/CLOCKS_PER_SEC ;
  HOURS = SEC/3600 ;
  MIN = SEC/60 ;
  SEC = SEC % 60 ;
  cout << "Elapsed Time = " << HOURS << "h " << MIN << "m " << SEC << "s" << endl;
  TIME << "Elapsed Time = " << HOURS << "h " << MIN << "m " << SEC << "s" << endl;
  TIME.close() ;
  return 0;

}

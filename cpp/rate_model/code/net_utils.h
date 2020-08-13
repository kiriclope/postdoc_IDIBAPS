#ifndef __NETUTILS__
#define __NETUTILS__

using namespace:: std ; 

#define VarToString(name) var_to_string(#name, (name))

template <class T>
const char* var_to_string(const char *name, const T& val) {
  // cout << name << " = " << val << endl;
  return name ;
}

void get_args(int argc , char** argv, string &dir, int &n_pop, unsigned long &n_neurons, double &K) {
  if(argv[1] != NULL) {    
    dir = argv[1] ;
    n_pop = (int) atoi(argv[2]) ;
    n_neurons = (unsigned long) atoi(argv[3]) ;
    K = (double) atof(argv[4]) ;
  }
  else {
    cout << "Directory ? " ;
    cin >> dir ;
    cout << "n_pop ? " ;
    cin >> n_pop ;
    cout << "n_neurons ? " ;
    cin >> n_neurons ;
    cout << "K ? " ;
    cin >> K ;
  } 
}

///////////////////////////////////////////////////////////////////////

void get_param(int n_pop, string dir, double * &ext_inputs, double* &J, double* &Tsyn) {

  cout << "reading parameters from : " ;
  string file_name = "../parameters/" + to_string(n_pop) + "pop/" + dir +".txt" ;
  cout << file_name << endl;

  ext_inputs = new double [n_pop]() ;
  J = new double [n_pop*n_pop]() ;   
  Tsyn = new double [n_pop*n_pop]() ; 
  
  string token ;
  string::size_type sz;
  ifstream file(file_name.c_str());
  int i,j;
  
  i=0 ; 
  while(getline(file, token)) {
    j=-1 ; 
    istringstream line(token);
    while(line >> token) {
      if(i==0)
	if(j!=-1) ext_inputs[j] = stod(token, &sz) ; 
      
      if(i==1)
	if(j!=-1) J[j] = stod(token, &sz) ; 

      if(i==2)
	if(j!=-1) Tsyn[j] = stod(token, &sz) ; 
      
      j++ ;
    }
    
    if(file.unget().get() == '\n') {
      i++ ;
    }
  }
  
  /* cout << "ext_inputs" << endl ; */
  /* for(int i=0;i<n_pop;i++)  */
  /*   cout << ext_inputs[i] << " " ; */
  /* cout << endl ;   */

  /* cout << "J" << endl ; */
  /* for(int i=0;i<n_pop;i++) { */
  /*   for(int j=0;j<n_pop;j++) */
  /*     cout << J[j+i*n_pop] << " " ; */
  /*   cout << endl ; */
  /* } */

  /* cout << "Tsyn" << endl ; */
  /* for(int i=0;i<n_pop;i++) { */
  /*   for(int j=0;j<n_pop;j++) */
  /*     cout << Tsyn[j+i*n_pop] << " " ; */
  /*   cout << endl ; */
  /* } */

}

double Phi(double x) { // Gaussian CDF
  return 0.5 * ( 1.0 + erf(x/sqrt(2.0) ) ) ; 
}

double threshold_linear(double x) {
  if(x>0)
    return x ;
  else
    return 0. ;
}

void create_dir(string dir, string &path, int n_pop, unsigned long n_neurons, double K) {

  string mkdirp = "mkdir -p " ;
  path += "simulations/"+ to_string(n_pop)+"pop/" + dir + "/N" + to_string(n_neurons) + "/K" + to_string((int)K) ;

  mkdirp += path ;

  const char * cmd = mkdirp.c_str();
  const int dir_err = system(cmd);

  if(-1 == dir_err)
    cout << "error creating directories" << endl ;

  cout << "Created directory : " ;
  cout << path << endl ;

}

#endif

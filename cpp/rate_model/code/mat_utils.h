#ifndef __MATRIXUTILS__
#define __MATRIXUTILS__

template <class T>
void read_from_file(string path, string file_name, T * &array, size_t array_size) {

  int dum ;
  
  string file_path ;
  file_path = path + "/" + file_name + ".dat" ;  
  cout << file_path << endl ;
  
  struct stat buffer ;
  FILE *file ;

  if (stat (file_path.c_str(), &buffer) == 0) {    
    file = fopen(file_path.c_str(), "rb") ;
    dum = fread(&array[0], sizeof array[0], array_size, file); 
    fclose(file) ; 
  }
  else {
    cout << "ERROR : file not found" << endl ;
    exit(-1) ;
  }
}

///////////////////////////////////////////////////////////////////    

void get_con_mat(string con_path, unsigned long n_neurons, int* &n_pre, unsigned long* &id_pre, unsigned long* &idx_pre) {

  n_neurons = (unsigned long) n_neurons*10000 ;    

  n_pre = new int [n_neurons]() ; 
  read_from_file(con_path, "n_pre", n_pre, n_neurons) ; 
  
  idx_pre = new unsigned long [n_neurons]() ;
  read_from_file(con_path, "idx_pre", idx_pre, n_neurons) ;
  
  unsigned long total_n_pre = 0 ;
  for(unsigned long i=0 ; i<n_neurons; i++)
      total_n_pre += n_pre[i] ; 

  id_pre = (unsigned long *) malloc( (unsigned long) total_n_pre * sizeof(unsigned long) ) ;
  /* id_pre = new unsigned long [total_n_pre]() ; */ // for some reason this is not working ...
  read_from_file(con_path, "id_pre", id_pre, total_n_pre) ;
  
}


/* void avg_n_pre(int n_pop, unsigned long n_neurons) { */

/*   double mean_n_pre ; */

/*   cout << "average n_pre : " ; */
/*   for(int i=0;i<nbpop;i++) */
/*     for(int j=0;j<nbpop;j++) { */
/*       mean_n_pre=0 ; */
/*       for(int k=0; k<n_neurons; k++) */
/* 	mean_n_pre += n_pre[k] ; */
/*       cout << mean_n_pre/(double)Nk[i] << " " ; */
/*     } */
/*   cout << endl ; */
/* } */

#endif

#ifndef __GLOBALVARS__ 
#define __GLOBALVARS__ 

//////////////////////////////////////////
// Simulation globals
//////////////////////////////////////////
#define DT .1 
#define DURATION 0.250E3 // 10.E3  // 

#define TIME_STEADY .100E3 //  2.0E3 // 10.E3 // 
#define TIME_WINDOW .025E3 // 1.00E3 // 10.E3 // 

#define IF_EULER 1
#define IF_RK2 0 

//////////////////////////////////////////
// Network globals
//////////////////////////////////////////

#define J0 -1.0
#define I0 1.0
#define Tsyn0 2.0
#define m0 1.0E-2

//////////////////////////////////////////
// Low-rank globals
//////////////////////////////////////////
#define IF_LOW_RANK 1
#define MEAN_XI 0.0
#define VAR_XI 1.0

#endif

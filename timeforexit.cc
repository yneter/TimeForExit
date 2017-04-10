#include <iostream>
#include <iomanip>
#include <fstream>
#include <stdexcept>
#include <vector>
#include <cstdio>
#include <cstdlib>
#include <cmath>

#define NX 32
#define NFORCES 256

const double UMAX = 10.0;
const int POPSIZE = NFORCES - 1;
const int MAXGENS = 10000;
const int NVARS = NX;
const double PXOVER = 0.8;
const double PMUTATION = 0.05;


double sq(double x) { return x * x; }
double myrand(void) { return (double) rand() / (double) RAND_MAX; }


#ifdef VEX_WALKER

#include <vexcl/vexcl.hpp>

VEX_FUNCTION(double, select_force, (double, value)(int, walk_number)(int, force_number),
	     if ((walk_number % NFORCES) == force_number) return value;
	     return 0.0;
	     );

VEX_FUNCTION(double, update_exit_time, (double, x)(double, previous_exit_time)(double, time),
	     if (x > 1.0) return time;
	     else return previous_exit_time;
	     );

VEX_FUNCTION(int, increment_exit_count, (double, x),
	     if (x > 1.0) return 1;
	     else return 0;
	     );


VEX_FUNCTION(double, restart_walker, (double, x),
	     if (x > 1.0) return 0.0;
	     else if (x < 0.0) return -x;
	     else return x;
	     );


VEX_FUNCTION(double, vex_force, (double, x)(int, x_number)(const double *, forces),
	     int ix = (int) floor( x * (double) NX);
	     int nforce = x_number % NFORCES;
	     if (ix < 0) ix = 0;
	     else if (ix >= NX) ix = NX - 1;
	     return forces[nforce * NX + ix];
	     );	    

VEX_FUNCTION(double, vex_smooth_force, (double, x)(int, x_number)(const double *, forces),	     
	     double xnx = x * (double) NX;
	     int nforce = x_number % NFORCES;	     
	     int ilo = (int) floor(xnx);
	     int ihi = (int) ceil(xnx);
	     double Flo = forces[nforce * NX + ilo];
	     // periodic boundary conditions on force
	     double Fhi = (ihi < NX) ? forces[nforce * NX + ihi ] : forces[nforce * NX];
	     return (xnx - (double)ilo) * Fhi + ((double) ihi - xnx) * Flo;
	     );	    

VEX_FUNCTION(double, mean_exit_time, (double, exit_time)(int, exit_count),
	     return exit_time / (double) exit_count;
	     );	    



class vex_walker {
    const size_t n_walkers;
    vex::Context ctx;
    vex::vector<double> walker_x;           // walker positions 
    vex::vector<double> walker_xnew;        // new walker positions (temporary variable)
    vex::vector<double> walker_exit_time;   // sum of all exit times
    vex::vector<int> walker_exit_count;     // number of exits 
    vex::Random<double> rnd;
    vex::Reductor<double, vex::SUM> sum;
    vex::Reductor<double, vex::MIN> min;
    vex::vector<double> force_gpu;
    std::vector<double> force;
    std::vector<int> escape_count;
    std::vector<double> escape_time;

public : 
    double D;
    double dt;
    double gamma;
    double Temp;
  
    vex_walker(size_t init_size) :
      n_walkers(init_size), 
      ctx(vex::Filter::GPU && vex::Filter::Any),
      walker_x(ctx, n_walkers), 
      walker_xnew(ctx, n_walkers),
      walker_exit_time(ctx, n_walkers),
      walker_exit_count(ctx, n_walkers),
      sum(ctx), 
      min(ctx),
      force_gpu(ctx, NX * NFORCES),
      force(NX * NFORCES),
      escape_count(NFORCES),
      escape_time(NFORCES)
    { 
       if (!ctx) throw std::runtime_error("No devices available.");      
       // Print out list of selected devices:
       std::cout << ctx << std::endl;
    }

    void update_forces(void) { 
       for (int j = 0; j < NFORCES; j++) 
	  for (int i = 0; i < NX; i++)
	     force[j*NX+i] = 1.0 * (double)i/(double)NX;

       vex::copy(force, force_gpu);
    }

    void update_force_from_potential(int force_index, const double *potential) { 
       force[force_index*NX] = -potential[0] * (double)NX;
       for (int n = 1; n < NX; n++) 
	  force[force_index*NX+n] = -(potential[n]- potential[n-1]) * (double)NX;;
       vex::copy(force, force_gpu);
    }

  /****
    void integrate(int N) { 
       walker_exit_time = 0.0;
       double beta = 1.0;
       for (int i = 0; i < N; i++) { 
	 walker_exit_time += vex_potential_exp( rnd(vex::element_index(0, n_walkers), std::rand()), 
						rnd(vex::element_index(0, n_walkers), std::rand()), 
						vex::element_index() );
       }
    }
  ****/
    void run(size_t n_steps) { 
        walker_x = 0.0;
	walker_exit_time = 0.0;
	walker_exit_count = 0;

	double time = 0.0;
	for (size_t i = 0; i < n_steps; i++) { 
	  time += dt;
	  vex::tie(walker_xnew, walker_exit_time, walker_exit_count, walker_x) =
	    std::make_tuple(walker_x + gamma * dt * vex_smooth_force(walker_x, vex::element_index(), vex::raw_pointer(force_gpu)) + sqrt(6.0 * D * dt) * (2.0 * rnd(vex::element_index(0, n_walkers), std::rand()) - 1.0),
			    update_exit_time( walker_xnew, walker_exit_time, time ),
			    walker_exit_count + increment_exit_count( walker_xnew ),
			    restart_walker(walker_xnew)
			    );
	}

	double min_cc = min(walker_exit_count);
	double sum_cc = sum(walker_exit_count);
	double time_cc = sum(walker_exit_time);
	std::cout << "# min count/average count/dt step count/Temp"
		  << min_cc << "  "
		  << sum_cc/ (double) n_walkers << "  "
		  << time_cc / (sum_cc * dt) << "  "
		  << Temp << std::endl;

	for (size_t force_index = 0; force_index < NFORCES; force_index++) { 
	    double tex = sum(select_force(walker_exit_time, vex::element_index(), force_index));
	    int nex = sum(select_force(walker_exit_count, vex::element_index(), force_index));	    
	    escape_count[force_index] = nex;
	    if (nex < 1) { 
	       escape_time[force_index] = 1e10;
	    } else { 
	       escape_time[force_index] = tex/nex;	   	  
	    }
	}	
    }

    double mean_time(int force_index) { 
        return escape_time[force_index];
    }

    double fitness(int force_index) { 
        return exp( (((double)escape_count[force_index]) * (double) NFORCES / (double) n_walkers)/Temp );
    }

    void print_info(void) {
      std::cout << "# n_walkers " <<  n_walkers << std::endl;
      std::cout << "# D " <<  D << std::endl;
      std::cout << "# dt " << dt << std::endl;
      std::cout << "# gamma " << gamma << std::endl;
      std::cout << "# Temp " << Temp << std::endl;
      std::cout << "# NX " << NX << std::endl;
      std::cout << "# NFORCES " << NFORCES << std::endl;
    }  
};

#endif 


class cpu_walker {
    const size_t n_walkers;
    std::vector<double> walker_x;
    std::vector<double> walker_exit_time;
    std::vector<int> walker_exit_count;
    std::vector<double> force;
    std::vector<double> escape_time;
    std::vector<int> escape_count;
public :
    double D;
    double dt;
    double gamma;
private:
    double cpu_force(double x, int x_number) { 
       int ix = (int) floor(x * (double) NX);
       int nforce = x_number % NFORCES;
       if (ix < 0) ix = 0;
       else if (ix >= NX) ix = NX - 1;
       return force[nforce * NX + ix];
    }

    double cpu_smooth_force(double x, int x_number) { 
       double xnx = x * (double) NX;
       int nforce = x_number % NFORCES;	     
       int ilo = (int) floor(xnx);
       int ihi = (int) ceil(xnx);
       double Flo = force[nforce * NX + ilo];
       // periodic boundary conditions on force
       double Fhi = (ihi < NX) ? force[nforce * NX + ihi ] : force[nforce * NX];
       return (xnx - (double)ilo) * Fhi + ((double) ihi - xnx) * Flo;
    }
public :


      cpu_walker(const size_t init_size) :
      n_walkers(init_size),
      walker_x(n_walkers, 0.0),
      walker_exit_time(n_walkers, 0.0),
      walker_exit_count(n_walkers, 0),
      force(NX * NFORCES),
      escape_count(NFORCES),
      escape_time(NFORCES)
    {
       
    }


    void update_forces(void) { 
       for (int j = 0; j < NFORCES; j++) 
	  for (int i = 0; i < NX; i++)
	     force[j*NX+i] = 1.0 * (double)i/(double)NX;
    }

    void update_force_from_potential(int force_index, const double *potential) { 
       force[force_index*NX] = -potential[0] * (double)NX;
       for (int n = 1; n < NX; n++) 
	  force[force_index*NX+n] = -(potential[n]- potential[n-1]) * (double)NX;;
    }

    void run(size_t n_steps) { 
       double time;
       std::fill(walker_x.begin(), walker_x.end(), 0.0);
       std::fill(walker_exit_time.begin(), walker_exit_time.end(), 0.0);
       std::fill(walker_exit_count.begin(), walker_exit_count.end(), 0.0);

       for (size_t j = 0; j < n_walkers; j++) { 
	  time = 0.0;
	  for (size_t i = 0; i < n_steps; i++) { 
	    time += dt;
	    walker_x[j] += gamma * dt * cpu_smooth_force(walker_x[j], j);
	    walker_x[j] += sqrt(6.0 * D * dt) * (2.0 * myrand() - 1.0);
	    if (walker_x[j] < 0.0) walker_x[j] = -walker_x[j];
	    if (walker_x[j] > 1.0) { 
	      walker_exit_time[j] = time;
	      walker_exit_count[j]++;
	      walker_x[j] = 0.0;
	    }
	  }
       }
       
       double min_cc = walker_exit_count[0];
       double av_cc = 0.0;

       for (size_t force_index = 0; force_index < NFORCES; force_index++) { 
	  double tex = 0.0;
	  int nex = 0;
	  for (size_t j = 0; j < n_walkers; j++) { 
	     if (j % NFORCES == force_index) { 
	        tex += walker_exit_time[j];
		nex += walker_exit_count[j];
		av_cc += walker_exit_count[j];
		if (walker_exit_count[j] < min_cc) { 
		   min_cc = walker_exit_count[j];
		}
	     }
	  }
	  escape_count[force_index] = nex;
	  if (abs(nex) < 0.5) { 
	     std::cerr << "no exit events for force " << force_index << std::endl;
	     escape_time[force_index] = 1e10;
	  } else { 
	     escape_time[force_index] = tex/nex;	   	  
	  }
       }
       av_cc /= (double) n_walkers;
       std::cout << "# min/average exit count " << min_cc << "     " << av_cc << std::endl;
       
    }

    double mean_time(int force_index) { 
        return escape_time[force_index];
    }

    double fitness(int force_index) { 
        return ((double)escape_count[force_index]) * (double) NFORCES / (double) n_walkers;
    }

    void print_info(void) {
      std::cout << "# D " <<  D << std::endl;
      std::cout << "# dt " << dt << std::endl;
      std::cout << "# gamma " << gamma << std::endl;
      std::cout << "# NX " << NX << std::endl;
      std::cout << "# NFORCES " << NFORCES << std::endl;
      std::cout << "# UMAX " << UMAX << std::endl;
      std::cout << "# PXOVER " << PXOVER << std::endl;
      std::cout << "# PMUTATION " << PMUTATION << std::endl;
    }  

};


#ifdef VEX_WALKER
typedef vex_walker walk_maker;
#else
typedef cpu_walker walk_maker;
#endif

struct genotype
{
  double gene[NVARS];
  double fitness;
  double upper[NVARS];
  double lower[NVARS];
  double rfitness;
  double cfitness;
};



class simple_GA { 
//    Modified simple GA 
//
//    Original version by Dennis Cormier/Sita Raghavan/John Burkardt.
//
//  Reference:
//
//    Zbigniew Michalewicz,
//    Genetic Algorithms + Data Structures = Evolution Programs,
//    Third Edition,
//    Springer, 1996,
//    ISBN: 3-540-60676-9,
//    LC: QA76.618.M53.
//
   int seed;

   int int_uniform_ab ( int a, int b ) { 
      return a + (rand() % (b - a + 1));
   }

   double real_uniform_ab ( double a, double b ) { 
      return a + (b - a) * (double) rand() / (double) RAND_MAX;
   }

   void Xover ( int one, int two ) {
      //  Select the crossover point.
      int point = int_uniform_ab ( 0, NVARS - 1 );
      //  Swap genes in positions 0 through POINT-1.
      for (int i = 0; i < point; i++ ) {
	 double t = population[one].gene[i];
	 population[one].gene[i] = population[two].gene[i];
	 population[two].gene[i] = t;
      }
   }

public : 
   struct genotype population[POPSIZE+1];
   struct genotype newpopulation[POPSIZE+1]; 

   simple_GA(void) {
   }

   void crossover (void) {
      const double a = 0.0;
      const double b = 1.0;
      int mem;
      int one;
      int first = 0;
      
      for ( mem = 0; mem < POPSIZE; ++mem ) {
	double x = real_uniform_ab ( a, b );
	
	if ( x < PXOVER ) {
	  ++first;
	  
	  if ( first % 2 == 0 ) {
	    Xover ( one, mem );
	  } else {
	    one = mem;
	  }
	}
      }
      return;
   }

// 
//  If the best individual from the new population is better than 
//  the best individual from the previous population, then 
//  copy the best from the new population; else replace the 
//  worst individual from the current population with the 
//  best one from the previous generation                     
//  
   void elitist (void) {
     int i;
     double best, worst;
     int best_mem, worst_mem;
     
     best = population[0].fitness;
     worst = population[0].fitness;
     
     for (i = 0; i < POPSIZE - 1; ++i) {
        if ( population[i+1].fitness < population[i].fitness ) {
	   if ( best <= population[i].fitness ) {
	      best = population[i].fitness;
	      best_mem = i;
	   }
	   
	   if ( population[i+1].fitness <= worst ) {
	      worst = population[i+1].fitness;
	      worst_mem = i + 1;
	   }
	} else {
	  if ( population[i].fitness <= worst ) {
	     worst = population[i].fitness;
	     worst_mem = i;
	  }
	  if ( best <= population[i+1].fitness ) {
	     best = population[i+1].fitness;
	     best_mem = i + 1;
	  }
	}
     }

     if ( population[POPSIZE].fitness <= best ) {
        for ( i = 0; i < NVARS; i++ ) {
	  population[POPSIZE].gene[i] = population[best_mem].gene[i];
	}
	population[POPSIZE].fitness = population[best_mem].fitness;
     } else {
        for ( i = 0; i < NVARS; i++ ) {
	  population[worst_mem].gene[i] = population[POPSIZE].gene[i];
	}
	population[worst_mem].fitness = population[POPSIZE].fitness;
     } 
   }


   void evaluate ( walk_maker &walker, size_t n_steps ) {
      // as we are dealing with random fitnesses we recompute the fitness of the best individual at POPSIZE 
      for (int member = 0; member <= POPSIZE; member++ ) { 
	walker.update_force_from_potential(member, population[member].gene);
      }

      walker.run(n_steps);
      
      for (int member = 0; member <= POPSIZE; member++ ) {
	population[member].fitness = walker.fitness(member);
      }
   }

   void initialize (double lbound, double ubound ) {
      for (int j = 0; j <= POPSIZE; j++ ) {
         for (int i = 0; i < NVARS; i++ ) {
	    population[j].fitness = 0;
	    population[j].rfitness = 0;
	    population[j].cfitness = 0;
	    double i_lbound = lbound * 2.0 * (double) std::min(i, NVARS-1-i) / (double) NVARS;
	    double i_ubound = ubound * 2.0 * (double) std::min(i, NVARS-1-i) / (double) NVARS;
	    //	    double i_lbound = lbound;
	    //	    double i_ubound = ubound;
	    population[j].lower[i] = i_lbound;
	    population[j].upper[i]= i_ubound;
	    if (!j) { 
	      //       population[j].gene[i] = real_uniform_ab (1e-2 * i_lbound, 1e-2 * i_ubound);
	       population[j].gene[i] = real_uniform_ab (i_lbound, i_ubound);	      
	    } else {
	       population[j].gene[i] = (i_ubound + i_lbound) / 2.0;
	    }
	 }
      }
   }  

   void keep_the_best (void) { 
      int cur_best;
      int mem;
      int i;
      
      cur_best = 0;
      population[POPSIZE].fitness = 0;
      
      for ( mem = 0; mem < POPSIZE; mem++ ) {
        if ( population[POPSIZE].fitness < population[mem].fitness ) {
	  cur_best = mem;
	  population[POPSIZE].fitness = population[mem].fitness;
	}
      }
      // 
      //  Once the best member in the population is found, copy the genes.
      //
      for ( i = 0; i < NVARS; i++ ) {
        population[POPSIZE].gene[i] = population[cur_best].gene[i];
      }
      
      return;
   }


   void mutate (void) { 
      const double a = 0.0;
      const double b = 1.0;
      double lbound;
      double ubound;
      double x;

      for (int i = 0; i < POPSIZE; i++ ) {
	 for (int j = 0; j < NVARS; j++ ) {
	    x = real_uniform_ab (a, b);
	    if ( x < PMUTATION ) {	      
	       lbound = population[i].lower[j];
	       ubound = population[i].upper[j];
	       population[i].gene[j] = real_uniform_ab (lbound, ubound);
	    }
	 }
      }
   }
  

   void mutate (double amplitude) { 
      const double a = 0.0;
      const double b = 1.0;
      double lbound;
      double ubound;
      double x;

      for (int i = 0; i < POPSIZE; i++ ) {
	 for (int j = 0; j < NVARS; j++ ) {
	    x = real_uniform_ab (a, b);
	    if ( x < PMUTATION ) {	      
	       lbound = std::max(population[i].lower[j], population[i].gene[j] - amplitude);
	       ubound = std::min(population[i].upper[j], population[i].gene[j] + amplitude);
	       population[i].gene[j] = real_uniform_ab (lbound, ubound);
	    }
	 }
      }
   }

   void report ( int generation ) {
      double avg;
      double best_val;
      double square_sum;
      double stddev;
      double sum;
      double sum_square;

      if ( generation == 0 ) {
	 std::cout << "\n";
	 std::cout << "Value     Generation       Best            Average       Standard \n";
	 std::cout << "Value     number           value           fitness       deviation \n";
	 std::cout << "\n";
      }

      sum = 0.0;
      sum_square = 0.0;

      for (int i = 0; i < POPSIZE; i++ ) {
	 sum = sum + population[i].fitness;
	 sum_square = sum_square + population[i].fitness * population[i].fitness;
      }

      avg = sum / ( double ) POPSIZE;
      square_sum = avg * avg * POPSIZE;
      stddev = sqrt ( ( sum_square - square_sum ) / ( POPSIZE - 1 ) );
      best_val = population[POPSIZE].fitness;

      std::cout << "  " << std::setw(8) << "equal " 
                << "  " << std::setw(8) << generation 
		<< "  " << std::setw(14) << best_val 
		<< "  " << std::setw(14) << avg 
		<< "  " << std::setw(14) << stddev << "\n";

   }

 
   void selector (void) {
      const double a = 0.0;
      const double b = 1.0;
      int i;
      int j;
      int mem;
      double p;
      double sum;
      //
      //  Find the total fitness of the population.
      //
      sum = 0.0;
      for ( mem = 0; mem < POPSIZE; mem++ ) {
	 sum = sum + population[mem].fitness;
      }
      //
      //  Calculate the relative fitness of each member.
      //
      for ( mem = 0; mem < POPSIZE; mem++ ) {
	 population[mem].rfitness = population[mem].fitness / sum;
      }
      // 
      //  Calculate the cumulative fitness.
      //
      population[0].cfitness = population[0].rfitness;
      for ( mem = 1; mem < POPSIZE; mem++ ) {
	 population[mem].cfitness = population[mem-1].cfitness + population[mem].rfitness;
      }
      // 
      //  Select survivors using cumulative fitness. 
      //
      for ( i = 0; i < POPSIZE; i++ ) { 
	 p = real_uniform_ab (a, b);
	 if ( p < population[0].cfitness ) {
	    newpopulation[i] = population[0];      
	 } else {
	    for ( j = 0; j < POPSIZE; j++ ) { 
	       if ( population[j].cfitness <= p && p < population[j+1].cfitness ) {
		  newpopulation[i] = population[j+1]; 
	       }
	    }
	 }
      }
      // 
      //  Overwrite the old population with the new one.
      //
      for ( i = 0; i < POPSIZE; i++ ) {
	 population[i] = newpopulation[i]; 
      }

      return;     
   }

   void print_info(void) { 
      std::cout << "# POPSIXE " << POPSIZE << std::endl;
      std::cout << "# MAXGENS " << MAXGENS << std::endl;
      std::cout << "# NVARS " << NVARS << std::endl;
      std::cout << "# PXOVER " << PXOVER << std::endl;
      std::cout << "# PMUTATION " << PMUTATION << std::endl;
   }

   void print_best(void) { 
      std::cout << "# best gene = " << population[POPSIZE].fitness << "\n";
      for (int i = 0; i < NVARS; i++ ) {
	 std::cout << i << "    " << population[POPSIZE].gene[i] << "  %" << std::endl;
      }
      std::cout << "# with fitness = " << population[POPSIZE].fitness << "\n";
   }
};

int main(int argc, char *argv[]) 
{
    srand(50);
    std::cout.setf(std::ios::unitbuf);
    std::cout << "# UMAX " << UMAX << std::endl;
    const size_t n_walkers = (argc > 1) ? atoi(argv[1]) : 1024;
    size_t n_steps = 2e4;

    walk_maker walker(n_walkers);
    walker.D = 16.0;
    walker.dt = 0.5e-5;
    walker.gamma = 10.0;
    walker.Temp = 0.5;
    walker.print_info();
    std::cout << "# n_steps " << n_steps << std::endl;

    simple_GA ga;
    ga.initialize(-UMAX/2.0, UMAX/2.0);
    ga.print_info();
    ga.evaluate (walker, n_steps);
    ga.report(-1);
    ga.keep_the_best();

    double eps = 1e-3;
    for (int generation = 0; generation < MAXGENS; generation++ ) {
       walker.Temp *= (1. - eps);
       ga.selector ();
       ga.crossover ();
       ga.mutate ();
       ga.report(generation);
       ga.evaluate (walker, n_steps);
       ga.elitist();
       if (!(generation % 20)) {
	  ga.print_best();	  
       }
    }
    ga.print_best();
    return 0;
}




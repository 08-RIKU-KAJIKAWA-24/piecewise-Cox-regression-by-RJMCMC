#include <Rmath.h>
#include <RcppArmadillo.h>


#include <complex.h>
#include <algorithm>
#include <cfloat>
#include <cmath>
#include <iostream>
#include <iterator>
#include <list>
#include <numeric>
#include <tgmath.h>
#include <vector>


// [[Rcpp::depends(RcppArmadillo)]]
using namespace Rcpp;


double log1(int Y){
  double Y1=Y;
  return(log(Y1));
}


double min1(double a, double b){
	double z = 0;
	if(a >= b){
		z = b;
	}else{
		z = a;
	}

	return(z);
}


double max1(double a, double b){
	double z = 0;
	if(a >= b){
		z = a;
	}else{
		z = b;
	}

	return(z);
}


double MaxVec(arma::vec Y){
	int J1 = Y.n_rows;
	double max = Y[0];
	for(int j = 1; j < J1; j++){
		if(Y[j] > max){
			max = Y[j];
		}
	}

	return(max);
}



int SampleDeath(int J){
	double U = as_scalar(arma::randu(1)) * J;
	U = floor(U) + 1;
	return(U);
}



int SampleBirth(arma::vec s, int J){
	arma::vec cumprob(J+1);

	cumprob[0] = s[1] / s[J+1];
	for(int m = 1; m < cumprob.n_rows; m++){
		cumprob[m] = s[m+1] / s[J+1];
	}

	int which = 0;
	double U = as_scalar(arma::randu(1));
	if(U < cumprob[0]){
		which = 0;
	}else{
		for(int m=1; m < cumprob.n_rows; m++){
			which = m;
		}
	}

	return(which);
}


//Log partial likelihood funtion for Piecewise Cox regression model; PCM ---
//'@export
//[[Rcpp::export]]
double logPCM(arma::vec Y,
	          arma::vec I1,
	          arma::vec trt,
	          arma::vec s,
	          arma::vec beta,
	          int J){
	
	double logL = 0;
	double A = 0;
	for(int j=0; j<=J; j++){
		for(int i=0; i<Y.n_rows; i++){

			//分子の計算 ---
			logL = logL + beta(j)*trt(i)*(s(j)<=Y(i) && Y(i)<s(j+1));

			//分母の計算 ---
			A = A + exp(beta[j]*trt[i]*(s(j)<=Y(i) && Y(i)<s(j+1)));
		}
		logL = logL - log(A);
	}
	//std::cout << logL << std::endl;
	return(logL);
}



//'@export
//[[Rcpp::export]]
 List PCMRJMCMC(arma::vec Y,
 				arma::vec I1,
 				arma::vec trt,
 				double Poi,
 				double eta,
 				int Jmax,
 				int iter){

 	//seed value ---
 	std::mt19937 rand_src(12345);

 	//initialize ---
 	int B1 = iter/10;
 	arma::vec Lstore(B1);
 	arma::mat betastore(B1, Jmax+1);
 	arma::mat sstore(B1, Jmax+2);

 	arma::vec beta(Jmax+1);
 	arma::vec s(Jmax+2);
 	beta.zeros();
 	s.zeros();

 	arma::vec betaprop = beta;
 	arma::vec sprop = s;

 	int Spot = 0;
 	double sig = 25;
 	double Var1 = 25;
 	
 	double Birth = 0;
 	double U = 0;
 	double alpha = 0;

 	int J = 1;
 	int add = 0;
 	int del = 0;

 	double m1 = MaxVec(Y);
 	s[1] = m1/2;
 	s[2] = m1;

 	double _BM_ = min1(1, Poi/(J+1));
 	double _DM_ = min1(1, J/Poi);
 	double rho = 1/(_BM_+_DM_);

 	double BM = rho*_BM_;
 	double DM = rho*_DM_;

 	double Newbeta1 = 0;
 	double Newbeta2 = 0;
 	double U1 = 0;
 	
 	double a = 25;
 	double b = 1;

 	//Run reversible jump MCMC -----
 	for(int t=0; t<iter; t++){
 		//std::cout << "[" << t << "]" << " J = " << J << std::endl;
 		//std::cout << "  " << "add:" << BM << std::endl;
 		//std::cout << "  " << "del:" << DM << std::endl;

 		// log-HR sampling ----
 		if(J = 0){
 			betaprop = beta;
 			betaprop[0] = beta[0] + as_scalar(arma::randu(1));

 			//Likelihood ratio ---
 			alpha = logPCM(Y, I1, trt, s, betaprop, J) - logPCM(Y, I1,trt, s, beta, J);

 			//Prior ratio ---
 			alpha = alpha - .5*pow(betaprop[0],2)/Var1 + .5*pow(beta[0],2)/Var1;

 			U = log(as_scalar(arma::randu(1)));
 			if(U < alpha){
 				beta[0] = betaprop[0];
 			}
 		}else{
 			for(int j=0; j<=J; j++){
 				betaprop = beta;
 				betaprop[j] = beta[j] + as_scalar(arma::randu(1));

 				//Likelihood ratio --
 				alpha = logPCM(Y, I1, trt, s, betaprop, J) - logPCM(Y, I1, trt, s, beta, J);

 				//prior ratio ---
 				if(j==0){
 					alpha = alpha - .5*pow(betaprop[j],2)/Var1 - .5*pow(beta[j+1]-betaprop[j],2)/sig + .5*pow(beta[j],2)/Var1 + .5*pow(beta[j+1]-beta[j],2)/sig;
 				}else if(j==J){
 					alpha = alpha - .5*pow(betaprop[j]-beta[j],2)/sig + .5*pow(beta[j]-beta[j-1],2)/sig;
 				}else{
 					alpha = alpha -.5*pow(betaprop[j]-beta[j-1],2)/sig - .5*pow(beta[j+1]-betaprop[j],2)/sig + .5*pow(beta[j]-beta[j-1],2)/sig + .5*pow(beta[j+1]-beta[j],2)/sig;
 				}
 				U = log(as_scalar(arma::randu(1)));
 				if(U < alpha){
 					beta[j] = betaprop[j];
 				}
 			}
 		}

 		// change-point sampling ----
 		if(J > 0){
 			for(int j=1; j<J+1; j++){
 				sprop = s;

 				//propose new value for s ---
 				sprop[j] = s[j-1] + as_scalar(arma::randu(1))*(s[j+1]-s[j-1]);

 				//Likelihood ratio ---
 				alpha = logPCM(Y, I1, trt, sprop, beta, J) - logPCM(Y, I1, trt, s, beta, J);

 				//prior ratio ---
 				//accept = accept + log(s[j+1]-sprop[j]) + log(sprop[j]-s[j-1]) - log(s[j+1]-s[j])-log(s[j]-s[j-1]);
 				alpha = alpha + log(1e-6+(sprop[j]-s[j-1])*(s[j-1]<=eta)) + log(1e-6+(s[j+1]-sprop[j])*(sprop[j]<=eta)) - log(1e-6+(s[j]-s[j-1])*(s[j-1]<=eta)) - log(1e-6+(s[j+1]-s[j])*(s[j]<=eta));

 				//Metropolis-Hastings draw ---
				U = log(as_scalar(arma::randu(1)));

				//Accept or Reject ---
				if (U < alpha){
					s[j] = sprop[j];
				}
 			}
 		}

 		// mean and variance of HR sampling ---
 		int cum = 0;
 		if(J > 0){
 			for(int j=0; j<J; j++){
 				cum = cum + pow(beta[j]-beta[j-1], 2);
 			}
 			sig = 1/R::rgamma(a+.5, b+.5*cum);
 		}else{
 			sig = 1e+4;
 		}

 		//Birth or Death ---
 		U = as_scalar(arma::randu(1));
 		if(J < Jmax){
 			if(J==0){
 				add = 1;
 				del = 0;
 			}
 			else if((U <= BM) || (BM==1)){
 				add = 1;
 				del = 0;
 			}else{
 				add = 0;
 				del = 1;
 			}
 		}else{
 			if(J = Jmax){
 				add = 0;
 				del = 1;
 			}else if((U <= DM) && (DM==1)){
 				add = 0;
 				del = 1;
 			}else{
 				add = 1;
 				del = 0;
 			}
 		}

 		// Birth move ---
 		if(add = 1){
			betaprop.zeros();
            sprop.zeros();

            //Find birth location
            Spot = SampleBirth(s,J)+1; // Spot = j

            //Sample birthed split point
            Birth = as_scalar(arma::randu(1))*(s[Spot]-s[Spot-1])+s[Spot-1];
            //Now we have the interval of the new split in Spot and the actual location in Birth

            //Random Perturbation for detailed balance when dimension matching
            U1 = as_scalar(arma::randu(1));

            //Find new betabdas
            Newbeta1 =  beta[Spot-1] - log((1-U1)/U1)*(s[Spot]-Birth)/(s[Spot]-s[Spot-1]);
            Newbeta2 =  beta[Spot-1] + log((1-U1)/U1)*(Birth-s[Spot-1])/(s[Spot]-s[Spot-1]);

            //Let's add Birth to the Spot location and push back the rest of sprop
            for(int j=0;j<Spot;j++){
                sprop[j]=s[j];
            }
            for(int j=(Spot+1); j<s.n_rows; j++){
                sprop[j]=s[j-1];
            }
            sprop[Spot]=Birth;

            for(int j=0; j<(Spot-1); j++){
                betaprop[j] = beta[j];
            }
            for(int j=(Spot+1); j<(beta.n_rows+1); j++){
                betaprop[j] = beta[j-1];
            }
            betaprop[Spot-1]=Newbeta1;
            betaprop[Spot]=Newbeta2;

            //Now we have our new proposal vector, evaluate it!
            //Like Ratio
            alpha = logPCM(Y,I1,trt,sprop,betaprop,J+1) - logPCM(Y,I1,trt,s,beta,J);

            //Add proposal ratio
            //Poisson
            alpha = alpha + log1(Poi) - log1(J+1) ;

            //change-point proposal
            alpha = alpha + log1(2*J+3) + log1(2*J+2) + log(1e-6+(Birth-s[Spot-1])*(s[Spot-1]<=eta)) + log(1e-6+(s[Spot]-Birth)*(Birth<=eta));
            alpha = alpha - 2*log(eta) - log(1e-6+(s[Spot]-s[Spot-1])*(s[Spot-1]<=eta));
            //alpha = alpha + log1(2*J+3)+log1(2*J+2)+log(Birth-s[Spot-1])+log(s[Spot]-Birth);
            //alpha = alpha - 2*log(m1)-log(s[Spot]-s[Spot-1]);

            //proposal ratio ---
            alpha = alpha + log(1e-6+J) + log(1e-6+min1(1,(J+1)/Poi));
            alpha = alpha - log(J+1) - log(1e-6+min1(1,Poi/(J+1)));

            //log of Jacobian for detailed balance
            alpha = alpha - log(U1*(1-U1)) ;

            //Add proposal ratio for beta
            if(J==0){ //No Change-point
                alpha = alpha - .5*pow(betaprop[Spot]-betaprop[Spot-1],2)/sig - .5*pow(betaprop[Spot-1],2)/Var1;
                alpha = alpha + .5*pow(beta[Spot-1],2)/Var1;

            }else{
                if(Spot==(J+1)){
                    //Birthed in the last interval
                    alpha = alpha - .5*pow(betaprop[Spot]-betaprop[Spot-1],2)/sig - .5*pow(betaprop[Spot-1]-betaprop[Spot-2],2)/sig;
                    alpha = alpha + .5*pow(beta[Spot-1]-beta[Spot-2],2)/sig;

                }else{
                    if(Spot==1){
                        //Birthed is in the first interval
                        alpha = alpha - .5*pow(betaprop[Spot+1]-betaprop[Spot],2)/sig - .5*pow(betaprop[Spot]-betaprop[Spot-1],2)/sig - .5*pow(betaprop[Spot-1],2)/Var1;
                        alpha = alpha + .5*pow(beta[Spot]-beta[Spot-1],2)/sig + .5*pow(beta[Spot-1],2)/Var1;

                    }else{ //中間のとき
                        alpha = alpha - .5*pow(betaprop[Spot+1]-betaprop[Spot],2)/sig - .5*pow(betaprop[Spot]-betaprop[Spot-1],2)/sig -.5*pow(betaprop[Spot-1]-betaprop[Spot-2],2)/sig;
                        alpha = alpha + .5*pow(beta[Spot]-beta[Spot-1],2)/sig + .5*pow(beta[Spot-1]-beta[Spot-2],2)/sig;
                    }
                }
            }

            //Metropolis Hastings
            U = log(as_scalar(arma::randu(1)));

            //Accept/Reject
            if(U < alpha){
                s = sprop;
                beta = betaprop;
                J = J+1;
            }
 		}

 		// Death move ---
 		if(del = 1){
 			//Which one to delete??
            Spot=SampleDeath(J);

            //Setup sprop with deleted value here
            sprop.zeros();
            betaprop.zeros();

            //Fill in Storage
            for(int j=0; j<Spot; j++){
                sprop[j] = s[j];
            }

            //Finish sproposal fill in
            for(int j=Spot; j<(s.n_rows-1);j++){
                sprop[j] = s[j+1];
            }

            //Fill in betabda proposal vector ここ変じゃないか？
            if(J > 1){
                for(int j=0; j<(Spot-1); j++){
                    betaprop[j] = beta[j];
                }

                for(int j=Spot; j<(beta.n_rows-1); j++){
                    betaprop[j] = beta[j+1];
                }

                //New betabda is a weighted average of the old betabdas
                betaprop[Spot-1] = ((s[Spot+1]-s[Spot])*beta[Spot] + (s[Spot]-s[Spot-1])*beta[Spot-1])/(s[Spot+1]-s[Spot-1]);


            }else{
                betaprop[0] = ((s[2]-s[1])*beta[1]+(s[1]-s[0])*beta[0])/(s[2]-s[0]);
            }


            //Now we have our new proposal vectors, evaluate then!
            //Like Ratio
            alpha = logPCM(Y,I1,trt,sprop,betaprop,J-1) - logPCM(Y,I1,trt,s,beta,J);

            //Prior Ratio
            //Poisson
            alpha = alpha - log1(Poi) + log1(1e-6+J);

            //s Prior
            alpha = alpha + 2*log(eta) + log(1e-6+(s[Spot+1]-s[Spot-1])*(s[Spot-1]<=eta));
            alpha = alpha - log1(2*J+1) - log1(2*J) - log(1e-6+(s[Spot]-s[Spot-1])*(s[Spot-1]<=eta)) - log(1e-6+(s[Spot+1]-s[Spot])*(s[Spot]<=eta));
            //alpha = alpha + 2*log(m1) + log(s[Spot+1]-s[Spot-1]) - log1(2*J+1) - log1(2*J) - log(s[Spot]-s[Spot-1])- log(s[Spot+1]-s[Spot]);

            //proposal ratio ---
            alpha = alpha + log(1e-6+J) + log(min1(1,Poi/J));
            alpha = alpha - log(1e-6+(J-1)) + log(min1(1,J/Poi));

            //betabda Prior, we DROPPED one
            if(J==1){
                //Removing will drop the sampler to 0 split points
                alpha = alpha - .5*pow(betaprop[Spot-1],2)/sig;
                alpha = alpha + .5*pow(beta[Spot]-beta[Spot-1],2)/sig + .5*pow(beta[Spot-1],2)/Var1;

            }else{
                if(Spot==1){ //first interval
                    alpha = alpha - .5*pow(betaprop[Spot]-betaprop[Spot-1],2)/sig - .5*pow(betaprop[Spot-1],2)/Var1;
                    alpha = alpha + .5*pow(beta[Spot+1]-beta[Spot],2)/sig + .5*pow(beta[Spot]-beta[Spot-1],2)/sig + .5*pow(beta[Spot-1],2)/Var1;

                }else if(Spot==J){ //last interval
                    alpha = alpha - .5*pow(betaprop[Spot]-betaprop[Spot-1],2)/sig;
                    alpha = alpha + .5*pow(beta[Spot]-beta[Spot-1],2)/sig + .5*pow(beta[Spot-1]-beta[Spot-2],2)/sig;

                }else{ //middle interval
                    alpha = alpha - .5*pow(betaprop[Spot]-betaprop[Spot-1],2)/sig - .5*pow(betaprop[Spot-1]-betaprop[Spot-2],2)/sig;
                    alpha = alpha + .5*pow(beta[Spot+1]-beta[Spot],2)/sig + .5*pow(beta[Spot]-beta[Spot-1],2)/sig + .5*pow(beta[Spot-1]-beta[Spot-2],2)/sig;

                }
            }

            //Random Perturbation for dimension matching
            U1 = as_scalar(arma::randu(1));
            alpha = alpha + log(U1*(1-U1));

            //Metropolis Draw
            U = log(as_scalar(arma::randu(1)));
            //Accept/Reject
            if(U < alpha){
                J = J-1;
                s = sprop;
                beta = betaprop;
            }
 		}
 		
 		std::cout << "  " << "U:" << U << std::endl;
        std::cout << "  " << "accept:" << alpha << std::endl;

 		// Update add or delete info ---
 		_BM_ = min1(1, Poi/(J+1));
 		_DM_ = min1(1, J/Poi);
 		rho = 1/(_BM_+_DM_);
 		BM = rho*_BM_;
 		DM = rho*_DM_;

 		if(t > (iter-B1-1)){
            //Store Values in Matrix
            int StoreInx = t-iter+B1;

            for(int j=0; j<sstore.n_cols; j++){
                sstore(StoreInx,j) = s(j);
            }

            Lstore[StoreInx] = J;
        }
 	}

 	List z1 = List::create(Lstore,betastore,sstore);
    return(z1);
 }



// End of Program --------------------------------------------------------------------------


## packages --------------------------------------------------------------------
library("doParallel")
library("survminer")
library("survival")
library("tidyverse")
library("Rcpp")



## set directry -----
PATH <- "C:\\PhD\\"
setwd(PATH)



# reversible jump MCMC for PCM ------
PCMrjmcmc <- function(Y, I1, trt, Poi, eta, Jmax, iter){
  sourceCpp("PiecewiseCoxModel.cpp")
  return(PCMRJMCMC(Y, I1, trt, Poi, eta, Jmax, iter))
}


# Piece-wise Weibull Hazard -----
rpwh <- function(n, lam.1, lam.2, p, cpt){
  u <- runif(n)
  x <- -log(1-u)
  d <- x-(lam.1*cpt)^p
  
  bf <- (x^(1/p)/lam.1)*(d<=0)
  af <- ((cpt^p+d/(lam.2^p))^(1/p))*(d>0)
  bf[is.nan(bf)] <- 0
  af[is.nan(af)] <- 0
  
  return(bf+af)
}



# ## setting -----
N = 500
ACCRUAL.TIME = 12
FOLLOW.UP = 30
FIRST.SURVIVAL.RATE = 0.5
CPT = 10
HR = 0.5
N.E = N/2
N.C = N/2
TOTAL = ACCRUAL.TIME + FOLLOW.UP
lam = -log(FIRST.SURVIVAL.RATE)/12


set.seed(42)
dat <- data.frame(id = rep(1:N),
                  acctime = runif(n = N, min = 0, max = ACCRUAL.TIME),
                  arm = rep(0:1, each = N/2)) %>%
    mutate(U = ifelse(arm==1,
                      rpwh(n = N.E, lam.1 = lam, lam.2 = HR*lam, p=1, cpt=CPT),
                      rpwh(n = N.C, lam.1 = lam, lam.2 = lam, p=1, cpt=CPT)),
           t = ifelse(acctime+U < TOTAL, U, TOTAL-acctime),
           status = ifelse(t==U, 1, 0))


# reversible jump MCMC ---
survfit(Surv(t, status) ~ arm, data = dat) %>%
    ggsurvplot(risk.table = "abs_pct",
               palette = "lancet",
               conf.int = F,
               ggtheme = theme_gray(),
               title= "各治療群の生存曲線（Kaplan-Meier曲線）",
               xlab="ランダム化からの経過時間（月）",
               ylab="生存割合（%）",
               xlim = c(0,42),
               legend=c(0.8,0.8),
               legend.title="治療群",
               legend.labs=c("対照群（arm = C）","試験群（arm = E）"))

# Cox regression ---
#res.cox <- coxph(Surv(t, status) ~ arm, data = dat)
#loglik <- logPCM(Y=dat$t, I1=dat$status, trt=dat$arm, s=c(0,max(dat$t)), beta=log(HR), J=0)

res <- PCMrjmcmc(dat$t, dat$status, dat$arm, Poi=3, eta=max(dat$t)/2, Jmax=2, iter=100)





# End of Program ***************************************************************
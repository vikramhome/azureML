# COMMAND ----------

par_model_name= dbutils.widgets.get("model_name")

# COMMAND ----------

import azureml.core
from azureml.core import Workspace
from azureml.core.authentication import ServicePrincipalAuthentication
from azureml.core.run import Run
from azureml.core.experiment import Experiment

# Check core SDK version number
print("SDK version:", azureml.core.VERSION)

# COMMAND ----------

import os
import urllib
import pprint
import numpy as np
import shutil
import time

from pyspark.ml import Pipeline, PipelineModel
from pyspark.ml.feature import OneHotEncoder, OneHotEncoderEstimator, StringIndexer, VectorAssembler
from pyspark.ml.classification import LogisticRegression, DecisionTreeClassifier
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder

# COMMAND ----------

# Download AdultCensusIncome.csv from Azure CDN. This file has 32,561 rows.
#basedataurl = "https://amldockerdatasets.azureedge.net"
#datafile = "AdultCensusIncome.csv"
#datafile_dbfs = os.path.join("/dbfs", datafile)

#if os.path.isfile(datafile_dbfs):

#    print("found {} at {}".format(datafile, datafile_dbfs))

#else:
#    print("downloading {} to {}".format(datafile, datafile_dbfs))
#    urllib.request.urlretrieve(os.path.join(basedataurl, datafile), datafile_dbfs)
#    time.sleep(30)

library(readxl)
library(forecast)
library(stringr)
#library(Combine)
library(gtools)
library(data.table)
library(Hmisc)
#library(urca)
library(tseries)
library(vars)
library(reshape)
library(RSQLite)
#library(uroot)
library(egcm)
cond.scale<-function(x) {
	  n.col<-ncol(x)
	  z<-x
	  y<-x
	  #
	  kappa0.<-kappa(x)
	  f.<-rep(0,n.col)
	  for (i in 1:n.col) {
		f.[i]<-floor(max(log10(x[,i])))
		if (f.[i]<0) f.[i]<-0
	  }
	  for (i in 1:n.col) {
		z[,i]<-z[,i]/10^f.[i]  
	  }
	  kappa1.<-kappa(x)
	  #
	  for (i in 1:n.col) {
		if (f.[i]>1) f.[i]<-f.[i]+1
		y[,i]<-y[,i]/10^f.[i]  
	  }
	  kappa2.<-kappa(y)
	  if (kappa2.<kappa1.) {
		kappa.<-kappa2.
		x.out<-y
	  } else {
		kappa.<-kappa1.
		x.out<-z
	  }  
	  if (kappa0.<kappa.) {
		kappa.<-kappa0.
		x.out<-x
	  }  
	  outp.<-list()
	  outp.$x<-x.out
	  outp.$kappa<-kappa.
	  outp.$f.<-f.
	  return(outp.)
	}


	table.trans<-function(x.ts,col.,lag.) {
	  names.<-colnames(x.ts) 
				  n.<-length(col.)
				  n.ts<-ncol(x.ts)
				  if (n.==1) {
					x1<-stats::lag(x.ts,lag.)
					return(x1)
				  }
				  else {
					for (il in 1:n.) {
					  x1<-stats::lag(x.ts[,col.[il]],lag.[il])
					  if (il==1) x.b<-x1 else x.b<-cbind(x.b,x1)
					}
					y1<-cbind(x.ts,x.b) 
					for (il in 1:n.) {
					  y1[,col.[il]]<-y1[,n.ts+il]
					}
					y1<-y1[,-c((n.ts+1):(n.ts+n.))]
					colnames(y1)<-names.
					return(y1)} 
				}

				corr.stat.cointX<-function(out.x) {
				  #
				  #
				  #
				  flag.coint<-FALSE
				  flag.stat<-FALSE
				  index.<-NA
				  #
				  ts.<-out.x
				  x11<-ts.
				  #
				  ncol.ts.<-ncol(ts.)
				  #
				  #
				  if (nrow(test.coint(ts.))>0) flag.coint <- TRUE else
				  {
					flag.stat<-TRUE
					#print(ts.)
					test.out.<-test.stat(ts.)
    
					var.nstat<-as.character(test.out.[test.out.[,2]==0,]$Product)
					index.<-colnames(ts.) %in% var.nstat
					n.col<-ncol(ts.)
					for (i in 1:n.col) {
					  if (i==1) {if (index.[i]==TRUE) x11<-diff(ts.[,i],1) else x11<-ts.[,i]} else 
						if(index.[i]==TRUE) x11<-cbind(x11,diff(ts.[,i],1)) else x11<-cbind(x11,ts.[,i])
					}
					colnames(x11)<-colnames(ts.)
				  }
				  out.p.corr<-list()
				  out.p.corr$flag.coint <-flag.coint 
				  out.p.corr$flag.stat<-flag.stat 
				  out.p.corr$index.<-index.
				  out.p.corr$ts.<-x11
				  return(out.p.corr)
				}

				test.coint<-function(data.) {
				  #
				  data.<-data.[,colnames(data.)!="key"]
				  #
				  d1<-allpairs.egcm(data.,startdate=start(data.))
				  d1.out<-subset(d1,d1$is.cointegrated==TRUE)
				  d1.out<-d1.out[,c(1,2,17,18,20,25)]
				  d1.out<-d1.out[order(d1.out$r.p),]
				  return(d1.out)
				}
				#
				# scaling of time series
				#
				cond.scale<-function(x) {
				  n.col<-ncol(x)
				  z<-x
				  y<-x
				  #
				  kappa0.<-kappa(x)
				  f.<-rep(0,n.col)
				  for (i in 1:n.col) {
					f.[i]<-floor(max(log10(x[,i])))
					if (f.[i]<0) f.[i]<-0
				  }
				  for (i in 1:n.col) {
					z[,i]<-z[,i]/10^f.[i]  
				  }
				  kappa1.<-kappa(x)
				  #
				  for (i in 1:n.col) {
					if (f.[i]>1) f.[i]<-f.[i]+1
					y[,i]<-y[,i]/10^f.[i]  
				  }
				  kappa2.<-kappa(y)
				  if (kappa2.<kappa1.) {
					kappa.<-kappa2.
					x.out<-y
				  } else {
					kappa.<-kappa1.
					x.out<-z
				  }  
				  if (kappa0.<kappa.) {
					kappa.<-kappa0.
					x.out<-x
				  }  
				  outp.<-list()
				  outp.$x<-x.out
				  outp.$kappa<-kappa.
				  outp.$f.<-f.
				  return(outp.)
				}
				#




				############################################################################################################
				#
				#   forecasting environment for VAR/VARX 
				#
				#   M.Willenbacher CI/CS 2018-11-30 Version ?
				#
				############################################################################################################


				library(mice)
				library(RODBC)
				library(DBI)
				library(odbc)


				#
				#

				##
				## input for forecast
				##

				data11<-NA
				data21<-NA
				data31<-NA
				#
				#
				# parameter settings input files
				#
				#

				# application folder
				con <- dbConnect(odbc(), driver = "SQL Server",server = "13.68.235.147",trusted_connection="yes",database = "EOA_DEV",UID = "RUser",PWD = "1ruser!")
				data11 <- dbGetQuery(con,"Select * from imp.data_pricing")
				#data11<-read.csv2("Z:/EOA/GUI/csv_input/DataPriceMDTx.csv")
				names(data11)[1]<-"Period_Fiscal_Year"

				# exogeneous variables (not macroeconomic e.g. raw material prices)
				#
				data21 <- dbGetQuery(con,"Select * from imp.data_raw")
				#data21<-read.csv2("Z:/EOA/GUI/csv_input/DataRawMDTx.csv")

				# exogeneous variables (macroeconomic e.g. ExR and CLI)
				data31 <- dbGetQuery(con,"Select * from imp.data_macro")
				#data31<-read.csv2("Z:/EOA/GUI/csv_input/DataMacroMDTx.csv")

				##############################################################################################
				##############################################################################################

				# # # MONO # # # 

				##############################################################################################
				##############################################################################################



				# selection of variables
				#
				# Modell Supply=Produktion
				#
				s.en.c<-c(1,5,6)

				s.ex.r.c<-c(1,2,3)
				s.ex.m.c<-c(2)

				# variables lag
				l.ex.r.c<-c(1,1,1)
				l.ex.m.c<-c(2)

				data.en=data11
				data.exr=data21
				data.exm=data31
				start.<-c(2010,1)
				s.en=s.en.c
				s.ex.r=s.ex.r.c
				s.ex.m=s.ex.m.c
				l.ex.r=l.ex.r.c
				l.ex.m=l.ex.m.c
				ci.f=0.75
				n.ahead.f=5
				lag.max.f=7
				alpha.jt.f=0.05


				#
				#
				# data extraction
				#
				#
				data.11<-ts(data.en[,-1],start=start.,frequency=12)
				names.en.cf<-colnames(data.11)
				data.11<-data.11[,s.en]
				names.en.cf<-names.en.cf[s.en]
				n.en.f<-length(names.en.cf)

				if (all(is.na(data.exr))==TRUE) {
				  s.ex.r<-0
				  names.ex.r.cf<-NA
				  n.ex.r.f<-0
				  l.ex.r<-0
				} else {
				  data.21<-ts(data.exr[,-1],start=start.,frequency=12)
				  names.ex.r.cf<-colnames(data.21)
				  data.21<-data.21[,s.ex.r]
				  names.ex.r.cf<-names.ex.r.cf[s.ex.r]
				  n.ex.r.f<-length(names.ex.r.cf)
				}  
				#
				if (all(is.na(data.exm))==TRUE) {
				  s.ex.m<-0
				  names.ex.m.cf<-NA
				  n.ex.m.f<-0
				  l.ex.m<-0
				} else {
				  data.31<-ts(data.exm[,-1],start=start.,frequency=12)
				  names.ex.m.cf<-colnames(data.31)
				  data.31<-data.31[,s.ex.m]
				  names.ex.m.cf<-names.ex.m.cf[s.ex.m]
				  n.ex.m.f<-length(names.ex.m.cf)
				}

				n.s.en<-length(s.en)

				if (all(s.ex.r!=0)) n.s.ex.r<-length(s.ex.r) else n.s.ex.r<-0 
				if (all(s.ex.m!=0)) n.s.ex.m<-length(s.ex.m) else n.s.ex.m<-0 
				n.s.ex<-n.s.ex.r+n.s.ex.m
				#
				#  variable selection
				#
				data.<-data.11
				if (all(s.ex.r!=0)) data.<-cbind(data.,data.21)
				if (all(s.ex.m!=0)) data.<-cbind(data.,data.31)
				#
				colnames(data.)[1:n.s.en]<-names.en.cf
				if (all(s.ex.r!=0)) colnames(data.)[(n.s.en+1):(n.s.en+n.s.ex.r)]<-names.ex.r.cf
				if (all(s.ex.m!=0)) colnames(data.)[(n.s.en+n.s.ex.r+1):(n.s.en+n.s.ex.r+n.s.ex.m)]<-names.ex.m.cf
				#
				# 
				#  lagged structure 
				#
				if (n.s.ex==0) data.t<-data. else {
				  colx<-(n.s.en+1):(n.s.en+n.s.ex)
				  if (all(s.ex.r!=0) & all(s.ex.m==0)) lagx<-l.ex.r
				  if (all(s.ex.r==0) & all(s.ex.m!=0)) lagx<-l.ex.m 
				  if (all(s.ex.r!=0) & all(s.ex.m!=0)) lagx<-c(l.ex.r,l.ex.m)
				  data.t<-table.trans(data.,col.=colx,lag.=-lagx)
				  l.max<-max(lagx,n.ahead.f) 
  
				  data.tt<-window(data.t,start=start(data.11)+c(0,max(l.ex.m,l.ex.r)),end=end(data.11)+c(0,l.max),frequency=12)
				  if (n.s.ex!=1) {
					n.tt<-nrow(data.tt)
					for (i in n.tt:1) {
					  full.<-TRUE
					  for (j in (n.s.en+1):(n.s.en+n.s.ex)) {
						if (is.na(data.tt[i,j])==TRUE) full.<-FALSE
					  }
					  if (full.==FALSE) data.tt<-data.tt[-i,] else break
					}
					data.tt<-ts(data.tt,start=start(data.11)+c(0,max(l.ex.m,l.ex.r)),frequency=12)
				  }
				  data.t<-data.tt
				}
				#
				# scaling of datasets 
				#
				data.t.old<-na.omit(data.t)
				data.t.bt<-data.t.old
				scale.<-cond.scale(data.t.old)$f.
				cond.<-cond.scale(data.t.old)$kappa
				for (i in 1:(n.s.en+n.s.ex)) {
				  data.t.old[,i]<-data.t.old[,i]/10^scale.[i]
				  data.t[,i]<-data.t[,i]/10^scale.[i]
				}
				#
				#
				# cointregration or differentiation
				#
				data.adapt<-corr.stat.cointX(data.t.old)
				flag.coint.f<-data.adapt$flag.coint
				flag.stat.f<-data.adapt$flag.stat
				indexc.<-data.adapt$index.
				#
				tsc.en<-data.adapt$ts.[,1:n.s.en]
				tsc.ex<-NA
				if ((n.s.ex>0) & all(is.na(indexc.))==FALSE) {
				  tsc.ex<-data.adapt$ts.[,(1+n.s.en):(n.s.en+n.s.ex)]
				  tsc.ex.orig<-data.t[,(1+n.s.en):(n.s.en+n.s.ex)]
				  names.ts.orig<-colnames(data.t)[(1+n.s.en):(n.s.en+n.s.ex)]
				  if (n.s.ex==1) {
					if (indexc.[n.s.en+1]==TRUE) xn1<-diff(tsc.ex.orig,1) else xn1<-tsc.ex.orig
				  } else {
					for (i in 1:ncol(tsc.ex.orig)) {
					  if (i==1) {if (indexc.[n.s.en+i]==TRUE) xn1<-diff(tsc.ex.orig[,i],1) else xn1<-tsc.ex.orig[,i]} else 
					  {if(indexc.[n.s.en+i]==TRUE) xn1<-cbind(xn1,diff(tsc.ex.orig[,i],1)) else xn1<-cbind(xn1,tsc.ex.orig[,i])}
					}
					colnames(xn1)<-names.ts.orig
				  }
				}  
				if ((n.s.ex>0) & all(is.na(indexc.))==TRUE) {
				  xn1<-data.t[,(1+n.s.en):(n.s.en+n.s.ex)]
				  tsc.ex<-xn1}


				#
				if (all(is.na(tsc.ex))==TRUE) time.st<-start(na.omit(tsc.en)) else time.st<-start(na.omit(cbind(tsc.en,tsc.ex)))
				tsc.en<-window(tsc.en,start=time.st)
				if (n.s.ex>0) {
				  tsc.ex<-window(tsc.ex,start=time.st,end=end(data.11))
				  tsc.ex.new<-na.omit(window(xn1,start=end(data.11)+c(0,1)))
				  yx.<-start(cbind(tsc.en,tsc.ex))
				  tsc.en<-window(tsc.en,start=yx.)
				}


				#############################################################################


				DATES_act <- unique(data.frame(DATE = as.Date(tsc.ex)))

				TSDATA_all <- data.frame(DATE = DATES_act$DATE, MONTH = as.numeric(format(DATES_act$DATE,"%m")), PREFIX = "MONO", tsc.ex, tsc.en)
				TSDATA_M <- subset(TSDATA_all, as.numeric(TSDATA_all$DATE) >= min(as.numeric(DATES_act$DATE)) )


				OutputDataSet_M <- data.frame(DATE = TSDATA_M$DATE, MONTH = TSDATA_M$MONTH, PREFIX = TSDATA_M$PREFIX)
				if ("CLI_AP" %in% colnames(TSDATA_M)) {OutputDataSet_M$CLI_AP <- TSDATA_M$CLI_AP} else {OutputDataSet_M$CLI_AP <- 0}
				if ("CLI_EU" %in% colnames(TSDATA_M)) {OutputDataSet_M$CLI_EU <- TSDATA_M$CLI_AP} else {OutputDataSet_M$CLI_EU <- 0}
				if ("GLOB_CAPACITY" %in% colnames(TSDATA_M)) {OutputDataSet_M$GLOB_CAPACITY <- TSDATA_M$GLOB_CAPACITY} else {OutputDataSet_M$GLOB_CAPACITY <- 0}
				if ("GLOB_DEMAND" %in% colnames(TSDATA_M)) {OutputDataSet_M$GLOB_DEMAND <- TSDATA_M$GLOB_DEMAND} else {OutputDataSet_M$GLOB_DEMAND <- 0}
				if ("ExR_USD_EUR" %in% colnames(TSDATA_M)) {OutputDataSet_M$ExR_USD_EUR <- TSDATA_M$ExR_USD_EUR} else {OutputDataSet_M$ExR_USD_EUR <- 0}
				if ("P_ETHY_AP_EUR" %in% colnames(TSDATA_M)) {OutputDataSet_M$P_ETHY_AP_EUR <- TSDATA_M$P_ETHY_AP_EUR} else {OutputDataSet_M$P_ETHY_AP_EUR <- 0}
				if ("P_ETHY_EU_EUR" %in% colnames(TSDATA_M)) {OutputDataSet_M$P_ETHY_EU_EUR <- TSDATA_M$P_ETHY_EU_EUR} else {OutputDataSet_M$P_ETHY_EU_EUR <- 0}
				if ("P_ETHY_US_EUR" %in% colnames(TSDATA_M)) {OutputDataSet_M$P_ETHY_US_EUR <- TSDATA_M$P_ETHY_US_EUR} else {OutputDataSet_M$P_ETHY_US_EUR <- 0}
				if ("P_OIL_EUR" %in% colnames(TSDATA_M)) {OutputDataSet_M$P_OIL_EUR <- TSDATA_M$P_OIL_EUR} else {OutputDataSet_M$P_OIL_EUR <- 0}
				OutputDataSet_M$E_P_PCI_X_EUR <- TSDATA_M$E_P_PCI_M_EUR
				OutputDataSet_M$A_P_PCI_X_ABC <- TSDATA_M$E_P_PCI_T_EUR
				OutputDataSet_M$B_P_PCI_X_ABC <- TSDATA_M$U_P_PCI_T_USD


				##############################################################################################
				##############################################################################################

				# # # DI # # # 

				##############################################################################################
				##############################################################################################



				# selection of variables
				#
				# Modell Supply=Produktion
				#
				s.en.c<-c(3,2,6)


				s.ex.r.c<-c(3)
				s.ex.m.c<-c(2)
				#
				# variables lag
				l.ex.r.c<-c(0)
				l.ex.m.c<-c(1)
				#
				data.en=data11
				data.exr=data21
				data.exm=data31
				start.<-c(2010,1)
				s.en=s.en.c
				s.ex.r=s.ex.r.c
				s.ex.m=s.ex.m.c
				l.ex.r=l.ex.r.c
				l.ex.m=l.ex.m.c
				ci.f=0.75
				n.ahead.f=4
				lag.max.f=7
				alpha.jt.f=0.05


				#
				#
				# data extraction
				#
				#
				data.11<-ts(data.en[,-1],start=start.,frequency=12)
				names.en.cf<-colnames(data.11)
				data.11<-data.11[,s.en]
				names.en.cf<-names.en.cf[s.en]
				n.en.f<-length(names.en.cf)

				if (all(is.na(data.exr))==TRUE) {
				  s.ex.r<-0
				  names.ex.r.cf<-NA
				  n.ex.r.f<-0
				  l.ex.r<-0
				} else {
				  data.21<-ts(data.exr[,-1],start=start.,frequency=12)
				  names.ex.r.cf<-colnames(data.21)
				  data.21<-data.21[,s.ex.r]
				  names.ex.r.cf<-names.ex.r.cf[s.ex.r]
				  n.ex.r.f<-length(names.ex.r.cf)
				}  
				#
				if (all(is.na(data.exm))==TRUE) {
				  s.ex.m<-0
				  names.ex.m.cf<-NA
				  n.ex.m.f<-0
				  l.ex.m<-0
				} else {
				  data.31<-ts(data.exm[,-1],start=start.,frequency=12)
				  names.ex.m.cf<-colnames(data.31)
				  data.31<-data.31[,s.ex.m]
				  names.ex.m.cf<-names.ex.m.cf[s.ex.m]
				  n.ex.m.f<-length(names.ex.m.cf)
				}

				n.s.en<-length(s.en)

				if (all(s.ex.r!=0)) n.s.ex.r<-length(s.ex.r) else n.s.ex.r<-0 
				if (all(s.ex.m!=0)) n.s.ex.m<-length(s.ex.m) else n.s.ex.m<-0 
				n.s.ex<-n.s.ex.r+n.s.ex.m
				#
				#  variable selection
				#
				data.<-data.11
				if (all(s.ex.r!=0)) data.<-cbind(data.,data.21)
				if (all(s.ex.m!=0)) data.<-cbind(data.,data.31)
				#
				colnames(data.)[1:n.s.en]<-names.en.cf
				if (all(s.ex.r!=0)) colnames(data.)[(n.s.en+1):(n.s.en+n.s.ex.r)]<-names.ex.r.cf
				if (all(s.ex.m!=0)) colnames(data.)[(n.s.en+n.s.ex.r+1):(n.s.en+n.s.ex.r+n.s.ex.m)]<-names.ex.m.cf
				#
				# 
				#  lagged structure 
				#
				if (n.s.ex==0) data.t<-data. else {
				  colx<-(n.s.en+1):(n.s.en+n.s.ex)
				  if (all(s.ex.r!=0) & all(s.ex.m==0)) lagx<-l.ex.r
				  if (all(s.ex.r==0) & all(s.ex.m!=0)) lagx<-l.ex.m 
				  if (all(s.ex.r!=0) & all(s.ex.m!=0)) lagx<-c(l.ex.r,l.ex.m)
				  data.t<-table.trans(data.,col.=colx,lag.=-lagx)
				  l.max<-max(lagx,n.ahead.f) 
  
				  data.tt<-window(data.t,start=start(data.11)+c(0,max(l.ex.m,l.ex.r)),end=end(data.11)+c(0,l.max),frequency=12)
				  if (n.s.ex!=1) {
					n.tt<-nrow(data.tt)
					for (i in n.tt:1) {
					  full.<-TRUE
					  for (j in (n.s.en+1):(n.s.en+n.s.ex)) {
						if (is.na(data.tt[i,j])==TRUE) full.<-FALSE
					  }
					  if (full.==FALSE) data.tt<-data.tt[-i,] else break
					}
					data.tt<-ts(data.tt,start=start(data.11)+c(0,max(l.ex.m,l.ex.r)),frequency=12)
				  }
				  data.t<-data.tt
				}
				#
				# scaling of datasets 
				#
				data.t.old<-na.omit(data.t)
				data.t.bt<-data.t.old
				scale.<-cond.scale(data.t.old)$f.
				cond.<-cond.scale(data.t.old)$kappa
				for (i in 1:(n.s.en+n.s.ex)) {
				  data.t.old[,i]<-data.t.old[,i]/10^scale.[i]
				  data.t[,i]<-data.t[,i]/10^scale.[i]
				}
				#
				#
				# cointregration or differentiation
				#
				data.adapt<-corr.stat.cointX(data.t.old)
				flag.coint.f<-data.adapt$flag.coint
				flag.stat.f<-data.adapt$flag.stat
				indexc.<-data.adapt$index.
				#
				tsc.en<-data.adapt$ts.[,1:n.s.en]
				tsc.ex<-NA
				if ((n.s.ex>0) & all(is.na(indexc.))==FALSE) {
				  tsc.ex<-data.adapt$ts.[,(1+n.s.en):(n.s.en+n.s.ex)]
				  tsc.ex.orig<-data.t[,(1+n.s.en):(n.s.en+n.s.ex)]
				  names.ts.orig<-colnames(data.t)[(1+n.s.en):(n.s.en+n.s.ex)]
				  if (n.s.ex==1) {
					if (indexc.[n.s.en+1]==TRUE) xn1<-diff(tsc.ex.orig,1) else xn1<-tsc.ex.orig
				  } else {
					for (i in 1:ncol(tsc.ex.orig)) {
					  if (i==1) {if (indexc.[n.s.en+i]==TRUE) xn1<-diff(tsc.ex.orig[,i],1) else xn1<-tsc.ex.orig[,i]} else 
					  {if(indexc.[n.s.en+i]==TRUE) xn1<-cbind(xn1,diff(tsc.ex.orig[,i],1)) else xn1<-cbind(xn1,tsc.ex.orig[,i])}
					}
					colnames(xn1)<-names.ts.orig
				  }
				}  
				if ((n.s.ex>0) & all(is.na(indexc.))==TRUE) {
				  xn1<-data.t[,(1+n.s.en):(n.s.en+n.s.ex)]
				  tsc.ex<-xn1}


				#
				if (all(is.na(tsc.ex))==TRUE) time.st<-start(na.omit(tsc.en)) else time.st<-start(na.omit(cbind(tsc.en,tsc.ex)))
				tsc.en<-window(tsc.en,start=time.st)
				if (n.s.ex>0) {
				  tsc.ex<-window(tsc.ex,start=time.st,end=end(data.11))
				  tsc.ex.new<-na.omit(window(xn1,start=end(data.11)+c(0,1)))
				  yx.<-start(cbind(tsc.en,tsc.ex))
				  tsc.en<-window(tsc.en,start=yx.)
				}


				#############################################################################


				DATES_act <- unique(data.frame(DATE = as.Date(tsc.ex)))

				TSDATA_all <- data.frame(DATE = DATES_act$DATE, MONTH = as.numeric(format(DATES_act$DATE,"%m")), PREFIX = "DI", tsc.ex, tsc.en)
				TSDATA_D <- subset(TSDATA_all, as.numeric(TSDATA_all$DATE) >= min(as.numeric(DATES_act$DATE)) )

				OutputDataSet_D <- data.frame(DATE = TSDATA_D$DATE, MONTH = TSDATA_D$MONTH, PREFIX = TSDATA_D$PREFIX)
				if ("CLI_AP" %in% colnames(TSDATA_D)) {OutputDataSet_D$CLI_AP <- TSDATA_D$CLI_AP} else {OutputDataSet_D$CLI_AP <- 0}
				if ("CLI_EU" %in% colnames(TSDATA_D)) {OutputDataSet_D$CLI_EU <- TSDATA_D$CLI_AP} else {OutputDataSet_D$CLI_EU <- 0}
				if ("GLOB_CAPACITY" %in% colnames(TSDATA_D)) {OutputDataSet_D$GLOB_CAPACITY <- TSDATA_D$GLOB_CAPACITY} else {OutputDataSet_D$GLOB_CAPACITY <- 0}
				if ("GLOB_DEMAND" %in% colnames(TSDATA_D)) {OutputDataSet_D$GLOB_DEMAND <- TSDATA_D$GLOB_DEMAND} else {OutputDataSet_D$GLOB_DEMAND <- 0}
				if ("ExR_USD_EUR" %in% colnames(TSDATA_D)) {OutputDataSet_D$ExR_USD_EUR <- TSDATA_D$ExR_USD_EUR} else {OutputDataSet_D$ExR_USD_EUR <- 0}
				if ("P_ETHY_AP_EUR" %in% colnames(TSDATA_D)) {OutputDataSet_D$P_ETHY_AP_EUR <- TSDATA_D$P_ETHY_AP_EUR} else {OutputDataSet_D$P_ETHY_AP_EUR <- 0}
				if ("P_ETHY_EU_EUR" %in% colnames(TSDATA_D)) {OutputDataSet_D$P_ETHY_EU_EUR <- TSDATA_D$P_ETHY_EU_EUR} else {OutputDataSet_D$P_ETHY_EU_EUR <- 0}
				if ("P_ETHY_US_EUR" %in% colnames(TSDATA_D)) {OutputDataSet_D$P_ETHY_US_EUR <- TSDATA_D$P_ETHY_US_EUR} else {OutputDataSet_D$P_ETHY_US_EUR <- 0}
				if ("P_OIL_EUR" %in% colnames(TSDATA_D)) {OutputDataSet_D$P_OIL_EUR <- TSDATA_D$P_OIL_EUR} else {OutputDataSet_D$P_OIL_EUR <- 0}
				OutputDataSet_D$E_P_PCI_X_EUR <- TSDATA_D$E_P_PCI_D_EUR
				OutputDataSet_D$A_P_PCI_X_ABC <- TSDATA_D$U_P_PCI_M_USD
				OutputDataSet_D$B_P_PCI_X_ABC <- TSDATA_D$U_P_PCI_T_USD


				##############################################################################################
				##############################################################################################

				# # # TRI # # # 

				##############################################################################################
				##############################################################################################



				# selection of variables

				#
				# Modell Supply=Produktion
				#
				s.en.c<-c(5,1,4)


				s.ex.r.c<-c(1,2,3)
				s.ex.m.c<-c(2)
				#
				# variables lag
				l.ex.r.c<-c(3,1,1)
				l.ex.m.c<-c(1)
				#l.ex.m.c<-c(2)
				#
				data31<-NA
				data.en=data11
				data.exr=data21
				data.exm=data31
				start.<-c(2010,1)
				s.en=s.en.c
				s.ex.r=s.ex.r.c
				s.ex.m=s.ex.m.c
				l.ex.r=l.ex.r.c
				l.ex.m=l.ex.m.c
				ci.f=0.75
				n.ahead.f=4
				lag.max.f=7
				alpha.jt.f=0.05


				#
				#
				# data extraction
				#
				#
				data.11<-ts(data.en[,-1],start=start.,frequency=12)
				names.en.cf<-colnames(data.11)
				data.11<-data.11[,s.en]
				names.en.cf<-names.en.cf[s.en]
				n.en.f<-length(names.en.cf)

				if (all(is.na(data.exr))==TRUE) {
				  s.ex.r<-0
				  names.ex.r.cf<-NA
				  n.ex.r.f<-0
				  l.ex.r<-0
				} else {
				  data.21<-ts(data.exr[,-1],start=start.,frequency=12)
				  names.ex.r.cf<-colnames(data.21)
				  data.21<-data.21[,s.ex.r]
				  names.ex.r.cf<-names.ex.r.cf[s.ex.r]
				  n.ex.r.f<-length(names.ex.r.cf)
				}  
				#
				if (all(is.na(data.exm))==TRUE) {
				  s.ex.m<-0
				  names.ex.m.cf<-NA
				  n.ex.m.f<-0
				  l.ex.m<-0
				} else {
				  data.31<-ts(data.exm[,-1],start=start.,frequency=12)
				  names.ex.m.cf<-colnames(data.31)
				  data.31<-data.31[,s.ex.m]
				  names.ex.m.cf<-names.ex.m.cf[s.ex.m]
				  n.ex.m.f<-length(names.ex.m.cf)
				}

				n.s.en<-length(s.en)

				if (all(s.ex.r!=0)) n.s.ex.r<-length(s.ex.r) else n.s.ex.r<-0 
				if (all(s.ex.m!=0)) n.s.ex.m<-length(s.ex.m) else n.s.ex.m<-0 
				n.s.ex<-n.s.ex.r+n.s.ex.m
				#
				#  variable selection
				#
				data.<-data.11
				if (all(s.ex.r!=0)) data.<-cbind(data.,data.21)
				if (all(s.ex.m!=0)) data.<-cbind(data.,data.31)
				#
				colnames(data.)[1:n.s.en]<-names.en.cf
				if (all(s.ex.r!=0)) colnames(data.)[(n.s.en+1):(n.s.en+n.s.ex.r)]<-names.ex.r.cf
				if (all(s.ex.m!=0)) colnames(data.)[(n.s.en+n.s.ex.r+1):(n.s.en+n.s.ex.r+n.s.ex.m)]<-names.ex.m.cf
				#
				# 
				#  lagged structure 
				#
				if (n.s.ex==0) data.t<-data. else {
				  colx<-(n.s.en+1):(n.s.en+n.s.ex)
				  if (all(s.ex.r!=0) & all(s.ex.m==0)) lagx<-l.ex.r
				  if (all(s.ex.r==0) & all(s.ex.m!=0)) lagx<-l.ex.m 
				  if (all(s.ex.r!=0) & all(s.ex.m!=0)) lagx<-c(l.ex.r,l.ex.m)
				  data.t<-table.trans(data.,col.=colx,lag.=-lagx)
				  l.max<-max(lagx,n.ahead.f) 
  
				  data.tt<-window(data.t,start=start(data.11)+c(0,max(l.ex.m,l.ex.r)),end=end(data.11)+c(0,l.max),frequency=12)
				  if (n.s.ex!=1) {
					n.tt<-nrow(data.tt)
					for (i in n.tt:1) {
					  full.<-TRUE
					  for (j in (n.s.en+1):(n.s.en+n.s.ex)) {
						if (is.na(data.tt[i,j])==TRUE) full.<-FALSE
					  }
					  if (full.==FALSE) data.tt<-data.tt[-i,] else break
					}
					data.tt<-ts(data.tt,start=start(data.11)+c(0,max(l.ex.m,l.ex.r)),frequency=12)
				  }
				  data.t<-data.tt
				}
				#
				# scaling of datasets 
				#
				data.t.old<-na.omit(data.t)
				data.t.bt<-data.t.old
				scale.<-cond.scale(data.t.old)$f.
				cond.<-cond.scale(data.t.old)$kappa
				for (i in 1:(n.s.en+n.s.ex)) {
				  data.t.old[,i]<-data.t.old[,i]/10^scale.[i]
				  data.t[,i]<-data.t[,i]/10^scale.[i]
				}
				#
				#
				# cointregration or differentiation
				#
				data.adapt<-corr.stat.cointX(data.t.old)
				flag.coint.f<-data.adapt$flag.coint
				flag.stat.f<-data.adapt$flag.stat
				indexc.<-data.adapt$index.
				#
				tsc.en<-data.adapt$ts.[,1:n.s.en]
				tsc.ex<-NA
				if ((n.s.ex>0) & all(is.na(indexc.))==FALSE) {
				  tsc.ex<-data.adapt$ts.[,(1+n.s.en):(n.s.en+n.s.ex)]
				  tsc.ex.orig<-data.t[,(1+n.s.en):(n.s.en+n.s.ex)]
				  names.ts.orig<-colnames(data.t)[(1+n.s.en):(n.s.en+n.s.ex)]
				  if (n.s.ex==1) {
					if (indexc.[n.s.en+1]==TRUE) xn1<-diff(tsc.ex.orig,1) else xn1<-tsc.ex.orig
				  } else {
					for (i in 1:ncol(tsc.ex.orig)) {
					  if (i==1) {if (indexc.[n.s.en+i]==TRUE) xn1<-diff(tsc.ex.orig[,i],1) else xn1<-tsc.ex.orig[,i]} else 
					  {if(indexc.[n.s.en+i]==TRUE) xn1<-cbind(xn1,diff(tsc.ex.orig[,i],1)) else xn1<-cbind(xn1,tsc.ex.orig[,i])}
					}
					colnames(xn1)<-names.ts.orig
				  }
				}  
				if ((n.s.ex>0) & all(is.na(indexc.))==TRUE) {
				  xn1<-data.t[,(1+n.s.en):(n.s.en+n.s.ex)]
				  tsc.ex<-xn1}


				#
				if (all(is.na(tsc.ex))==TRUE) time.st<-start(na.omit(tsc.en)) else time.st<-start(na.omit(cbind(tsc.en,tsc.ex)))
				tsc.en<-window(tsc.en,start=time.st)
				if (n.s.ex>0) {
				  tsc.ex<-window(tsc.ex,start=time.st,end=end(data.11))
				  tsc.ex.new<-na.omit(window(xn1,start=end(data.11)+c(0,1)))
				  yx.<-start(cbind(tsc.en,tsc.ex))
				  tsc.en<-window(tsc.en,start=yx.)
				}


				#############################################################################


				DATES_act <- unique(data.frame(DATE = as.Date(tsc.ex)))

				TSDATA_T <- data.frame(DATE = DATES_act$DATE, MONTH = as.numeric(format(DATES_act$DATE,"%m")), PREFIX = "TRI", tsc.ex, tsc.en)
		
				OutputDataSet_T <- data.frame(DATE = TSDATA_T$DATE, MONTH = TSDATA_T$MONTH, PREFIX = TSDATA_T$PREFIX)
				if ("CLI_AP" %in% colnames(TSDATA_T)) {OutputDataSet_T$CLI_AP <- TSDATA_T$CLI_AP} else {OutputDataSet_T$CLI_AP <- 0}
				if ("CLI_EU" %in% colnames(TSDATA_T)) {OutputDataSet_T$CLI_EU <- TSDATA_T$CLI_AP} else {OutputDataSet_T$CLI_EU <- 0}
				if ("GLOB_CAPACITY" %in% colnames(TSDATA_T)) {OutputDataSet_T$GLOB_CAPACITY <- TSDATA_T$GLOB_CAPACITY} else {OutputDataSet_T$GLOB_CAPACITY <- 0}
				if ("GLOB_DEMAND" %in% colnames(TSDATA_T)) {OutputDataSet_T$GLOB_DEMAND <- TSDATA_T$GLOB_DEMAND} else {OutputDataSet_T$GLOB_DEMAND <- 0}
				if ("ExR_USD_EUR" %in% colnames(TSDATA_T)) {OutputDataSet_T$ExR_USD_EUR <- TSDATA_T$ExR_USD_EUR} else {OutputDataSet_T$ExR_USD_EUR <- 0}
				if ("P_ETHY_AP_EUR" %in% colnames(TSDATA_T)) {OutputDataSet_T$P_ETHY_AP_EUR <- TSDATA_T$P_ETHY_AP_EUR} else {OutputDataSet_T$P_ETHY_AP_EUR <- 0}
				if ("P_ETHY_EU_EUR" %in% colnames(TSDATA_T)) {OutputDataSet_T$P_ETHY_EU_EUR <- TSDATA_T$P_ETHY_EU_EUR} else {OutputDataSet_T$P_ETHY_EU_EUR <- 0}
				if ("P_ETHY_US_EUR" %in% colnames(TSDATA_T)) {OutputDataSet_T$P_ETHY_US_EUR <- TSDATA_T$P_ETHY_US_EUR} else {OutputDataSet_T$P_ETHY_US_EUR <- 0}
				if ("P_OIL_EUR" %in% colnames(TSDATA_T)) {OutputDataSet_T$P_OIL_EUR <- TSDATA_T$P_OIL_EUR} else {OutputDataSet_T$P_OIL_EUR <- 0}
				OutputDataSet_T$E_P_PCI_X_EUR <- TSDATA_T$E_P_PCI_T_EUR
				OutputDataSet_T$A_P_PCI_X_ABC <- TSDATA_T$E_P_PCI_M_EUR
				OutputDataSet_T$B_P_PCI_X_ABC <- TSDATA_T$U_P_PCI_D_USD


				OutputDataSet <- rbind(OutputDataSet_M, OutputDataSet_D, OutputDataSet_T)

		  '
		INSERT INTO R.TSDATA (
			 [DATE_int]
			,[MONTH] 
			,[PREFIX]
			,[CLI_AP]
			,[CLI_EU]
			,[GLOB_CAPACITY] 
			,[GLOB_DEMAND] 
			,[ExR_USD_EUR] 
			,[P_ETHY_AP_EUR]
			,[P_ETHY_EU_EUR]
			,[P_ETHY_US_EUR]
			,[P_OIL_EUR] 
			,[E_P_PCI_x_EUR]
			,[A_P_PCI_x_ABC]
			,[B_P_PCI_x_ABC] )
		EXEC sp_execute_external_script 
		  @language =N'R',
		  @script=@Rscript
		
		UPDATE R.TSDATA 
		SET [DATE] = DATEFROMPARTS(YEAR([DATE_int])+70,MONTH([DATE_int]),1)
		
		UPDATE R.TSDATA
		SET [E_P_PCI_x_EUR] = 1000*[E_P_PCI_x_EUR]

		UPDATE R.TSDATA
		SET [A_P_PCI_x_ABC] = 1000*[A_P_PCI_x_ABC]
		UPDATE R.TSDATA
		SET [B_P_PCI_x_ABC] = 1000*[B_P_PCI_x_ABC]

		--UPDATE R.TSDATA
		--SET [E_P_PCI_x_EUR] = [E_P_PCI_x_EUR]/1000

		--UPDATE R.TSDATA
		--SET [A_P_PCI_x_ABC] = [A_P_PCI_x_ABC]/1000
		--UPDATE R.TSDATA
		--SET [B_P_PCI_x_ABC] = [B_P_PCI_x_ABC]/1000

END

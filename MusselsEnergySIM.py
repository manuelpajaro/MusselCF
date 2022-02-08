# -*- coding: utf-8 -*-
"""
Created on Thu Jul 16 14:21:18 2020

@author: Manuel Pájaro
"""
import subprocess
import sys
def install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])
# install required packages
install("tkcalendar")

import numpy as np
# import scipy as sc
from collections import namedtuple
from scipy.integrate import odeint
from subprocess import call
import pandas as pd
import datetime as dt
import time
# to have multiple plots in the GUI 
import matplotlib 
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates


## Arrhemius correction ##
"""
SST : sea surface temperature (K)
"""
def arrhemius(SST,T1=288.15,Ta=5800):
    ar = np.exp(Ta/T1-Ta/SST)
    return ar

# Shell length L, (mm)
"""
Es = E_gS*W_S  : shell energy
Eg_s           : constant factor to convert (mg) into (J)
"""
def shell_L(Es,Eg_s,aL = 31.5,bL = 0.3764):
    Ws = 0.001*Es/Eg_s # must be in (g) to use the allometric parameters aL and bL
    L = aL*Ws**bL
    return L

# inverse shell_L
"""
L  : shell length (mm)
"""
def inv_shell_L(L,aL = 31.5,bL = 0.3764):
    Ws = (L/aL)**(1/bL)
    return Ws

# clearance rate CR (l/h)
"""
L   : shell length    
"""
def ClearanceRate(L,aCR = 0.00397,bCR = 1.71,maxCR = 6):
    # clearance rate CR (l/h)
    fCR = aCR*(L**bCR)
    CR = min(fCR,maxCR)
    return CR

# Ingestion rate, IR (mg/h)
"""
L   : shell length
POM : particulate organc matter     
"""
def IngestionRate(L,POM):
    # clearance rate CR (l/h)
    CR = ClearanceRate(L)
    # ingestion rate IR (mg/h)
    IR = CR*POM
    return IR

# Absorption efficiency AE
    """
TPM : total particulate matter
POM : particulate organc matter     
"""
def AbsorptionEfficiency(TPM,POM,aAE = 0.95,bAE = 0.18):    
    f = POM/TPM
    AE = aAE - bAE/f
    return AE

# Assimilation rate (J/h)
"""
L   : shell length
TPM : total particulate matter
POM : particulate organc matter     
"""
def assimilation(L,TPM,POM,muPOM = 23.5):
    IR = IngestionRate(L,POM)
    # Absorption efficiency AE
    AE = AbsorptionEfficiency(TPM,POM)
    # Assimilation rate A (J/h)
    A = muPOM*IR*AE
    return A

# Respiration (ml/h)
"""
L   : shell length   
"""
def respiration(L,aResp = 5.6e-5,bResp = 2.101):
    Resp = aResp*L**bResp
    return Resp

# Ammonia excretion ($\mu$g/h)
"""
L   : shell length   
"""
def AmmoniaExcretion(L,aExc = 0.004,bExc = 2.066):
    Exc = aExc*L**bExc
    return Exc

# Maintenance: metabolic expenditure (J/h)
"""
L   : shell length   
"""
def maintenance(L,muO2 = 20.36,muNH4N = 0.0249):
    # respiration Resp (mg/h)
    Resp = 1.428*respiration(L) #conversion from ml to mg
    # ammonia excretion Exc ($\mu$g/h)
    Exc = AmmoniaExcretion(L)
    # metabolic expenditure (J/h)
    M = muO2*Resp + muNH4N*Exc
    return M

# Net production per day NP (J/day)
"""
L   : shell length
SST : sea surface temperature (Kelvin) 15C = 288.15K
TPM : total particulate matter
POM : particulate organc matter     
"""
def NetProduction(L, SST, TPM, POM):
    PN = 24*arrhemius(SST)*(assimilation(L,TPM,POM) - maintenance(L))
    return PN


# fraction of energy allocated for reserves kappaE
"""
age : mussel age, (age = t0 + simulationTime with t0 = 150 days)
"""
def f_kappaE(age,aK = 3,bK = 0.0035):
    kappaE = 1/(1 + aK*np.exp(-bK*age))
    return kappaE

# total energy content φ (J)
"""
Et = E_gT*OW_som  : somatic tissue energy
E                 : Reserves
R                 : Reproduction buffer    
"""
def f_phi(Et,E,R):
    phi = Et + E + R
    return phi

# reserve density E_frac
"""
E      : Reserves   
phi    : Total energy content
"""
def f_Efrac(E,phi):
    Efrac = E/phi
    return Efrac

# gonado-somatic index
"""
R      : Reproduction buffer   
phi    : Total energy content
"""
def f_GSI(R,phi,pOT = 0.85):
    GSI = pOT*R/phi
    return GSI

# DEB equation
"""
Es = E_gS*W_S     : shell energy
Et = E_gT*OW_som  : somatic tissue energy
E                 : Reserves
R                 : Reproduction buffer
PN = A-M          : Net production
state = [Es, Et, E, R]
timeODE  : ode simulation time
Pars  : list of parameters
"""
def DEBeqs(state,timeODE,Pars):
    E = state[-2]
    R = state[-1]
    ######## conditions to reproduction #####
    reservas = 0 # increment of reproduction buffer R
    reproduc = 0 # decrement of reproduction buffer R
    if Pars.Efrac > Pars.Egamet:
        reservas = 1
        if Pars.rad > Pars.minRad and Pars.GSI > Pars.minGSI:
            reproduc = 1
    ########### model equations ##################
    if Pars.PN > 0: #Pars.A > params.M:
        if Pars.phi == Pars.maxPhi:
            dEs = Pars.k*Pars.ks*Pars.PN  
            dEt = Pars.k*(1-Pars.ks)*Pars.PN  
            dE  = (1-Pars.k)*Pars.PN -reservas*Pars.krep*E
            dR  = reservas*Pars.krep*E -reproduc*Pars.kspawn*R
        if Pars.phi < Pars.maxPhi:
            dEs = 0
            dEt = 0
            dE  = Pars.PN -reservas*Pars.krep*E
            dR  = reservas*Pars.krep*E -reproduc*Pars.kspawn*R
    else:
        dEs = 0
        dEt = 0
        dE  = Pars.PN -reservas*Pars.krep*E
        dR  = reservas*Pars.krep*E -reproduc*Pars.kspawn*R
    return [dEs,dEt,dE,dR]


# Load Enviromental Data
# import dates of observations
"""The data is colected between 2017-01-01 and 2019-12-31 (3 years of 365 days)
We will start our simulations in a given day between 2017-01-01 and 2017-12-31
"""
date_df = pd.read_csv('InputData/date.csv')
date_aux = date_df.x
date_data = np.matrix(date_aux)
date_dt = [dt.datetime.strptime(d,'%Y-%m-%d').date() for d in date_aux] 
# obter o índice dunha data
"""
Obtain an index for a date between 2017-01-01 and 2017-12-31
DateSeries : Series with data
Date : string with date format '2017-mm-dd'
"""
def DayIndex(Date,DateSeries=date_aux):
    indexDate = DateSeries[DateSeries == Date].index[0]
    return indexDate

# inport Enviromental Data
"""
Read the Enviromental data
fileName       : string with the file name where data is saved 
                ('Temp.csv', 'POM.csv', 'TPM.csv', 'Rad.csv')
Nrealizations  : number of realizations to load (maximum 1000)
FirstDay       : index of the initial day of simulation (use function DayIndex to obtain)
Ndays          : number of simulation days, by default 2 years which is the maximum
Firsrea        : to pick one of the data files
"""
def ReadEnvDatFile(fileName, Nrealizations = 1000, FirstDay = 0, Ndays = 365*2, Firstrea = 0):
    data_df = pd.read_csv(fileName)
    data_aux = data_df.to_numpy() # para ter unha matriz cos datos
    data = data_aux[Firstrea:Firstrea+Nrealizations,1+FirstDay:1+FirstDay+Ndays]
    return data

# simulation of one mussel for a given vector of enviromental data
"""
SST_data : vector with n values of SST
TPM_data : vector with n values of TPM
POM_data : vector with n values of POM
Rad_data : vector with n values of Rad
EndTime  : total days of simulation
t0       : initial age of mussels
Eg_s     : constant factor to convert (mg) into (J)
Es0      : initial shell energy
Et0      : initial tissue energy
E0       : initial reserves energy
R0       : initial reproduction buffer energy
"""
def SimulateMussel(SST_data,TPM_data,POM_data,Rad_data,EndTime,t0,Eg_s,Es0,Et0,E0,R0):
    ############ DEBmodel Parameters ##############
    ks = 0.7 		 # rate of growth energy allocated to shell. 
    krep = 0.02		 # rate of reserve enery allocated for reprudiction
    Egamet = 0.2 	 # Minimum reserve density for gametogenesis
    minRad = 11.5    # Solar irradiance threshold for spawning.	
    kspawn = 0.1    # Proportion of reproductive tissue released
    minGSI= 0.0005
    params = namedtuple('params', ['PN', 'phi', 'maxPhi', 'k', 'ks', 'Efrac', 'Egamet', 
                              'krep', 'rad', 'minRad', 'GSI', 'minGSI', 'kspawn'])
    # time points for odesolver
    tode = np.linspace(0,1,2)
    
    #### Intialization of maxPhi ######
    maxPhi = 0
    
    ### Time dependent variables initialization ###
    Lv = np.zeros([EndTime+1,1]) # vector of shell length 
    XX = np.zeros([EndTime+1,4]) # matrix with colunms [Es, Et, E, R]
    XX[0,:] = [Es0,Et0,E0,R0]
    Phiv = np.zeros([EndTime+1,1]) # vector of phis
    Phiv[0] = f_phi(Et0,E0,R0)
    PNv = np.zeros([EndTime+1,1]) # vector of PN
    Resp = np.zeros([EndTime+1,1]) # vector of O2 respiration
    Exc = np.zeros([EndTime+1,1]) # vector of amonia excretion
    AE = np.zeros([EndTime+1,1]) # vector of absortion efficience
    IR = np.zeros([EndTime+1,1]) # vector of ingestion rate
    CR = np.zeros([EndTime+1,1]) # vector of clearance rate
    SRresp = np.zeros([EndTime+1,1]) # rate of shell respiration
    for timeDay in range(EndTime):
        ############ variables #######
        Es = XX[timeDay,0]
        Et = XX[timeDay,1]
        E = XX[timeDay,2]
        R = XX[timeDay,3]
        phi = Phiv[timeDay] 	   # Growth condition
        ######### functions of Variables ############ 
        L = shell_L(Es, Eg_s)
        Lv[timeDay] = L
        PN = NetProduction(L,SST_data[timeDay],TPM_data[timeDay],POM_data[timeDay])
        PNv[timeDay+1] = PN
        Resp[timeDay+1] = arrhemius(SST_data[timeDay])*respiration(L)
        Exc[timeDay+1] = arrhemius(SST_data[timeDay])*AmmoniaExcretion(L)
        AE[timeDay+1] = AbsorptionEfficiency(TPM_data[timeDay],POM_data[timeDay])
        IR[timeDay+1] = arrhemius(SST_data[timeDay])*IngestionRate(L,POM_data[timeDay])
        CR[timeDay+1] = arrhemius(SST_data[timeDay])*ClearanceRate(L)
        rad = Rad_data[timeDay]         # Observed solar irradiance. (EnvData) rad in [2.814,24.94225]
        age = t0 + timeDay
        k = 1 - f_kappaE(age)	   # rate of energy allocatd for growth.
        maxPhi = max(maxPhi,phi)
        Efrac = f_Efrac(E, phi)    # fraction of reserves (resrve density in soft tissue)
        GSI = f_GSI(R, phi)        # gonado-somatic index
        Pars = params(PN = PN, phi = phi, maxPhi = maxPhi, k = k, ks = ks, Efrac = Efrac, 
                      Egamet = Egamet, krep = krep, rad = rad, minRad = minRad, GSI = GSI,
                      minGSI = minGSI, kspawn = kspawn)
        SRresp[timeDay+1] = ks*k
        # initial condition [Es, Et, E, R]
        y0 = [Es, Et, E, R] 
        # solve ODEs
        y = odeint(DEBeqs,y0,tode,args=(Pars,)) 
        # Update solution and Phis
        XX[timeDay+1,:] = y[-1,:] 
        Phiv[timeDay+1] = f_phi(y[-1,1],y[-1,2],y[-1,3])
    # Compute the last shell length 
    Lv[EndTime] = shell_L(XX[-1,0], Eg_s)
    # return data into a list
    return [XX,Lv,Phiv,PNv,Resp,Exc,AE,IR,CR,SRresp]


################### simulate Nsimulation mussels
"""
Eg_s     : constant factor to convert (mg) into (J)
DWt0     : initial tissue wheight
Et0      : initial shell wheight
EndTime  : number of days to simulate
Nsimulation  : number of mussels
InitialDate  : start date of simulation
DATES        : matrix with the dates
"""
    
def Sim_n_Mussels(Eg_s,DWt,L0,EndTime,Nsimulation,InitialDate,DATES=date_data):
    # convert units from g to mg
    DWt0=1000*DWt
    DWs0=1000*inv_shell_L(L0)
    #### parameters #####
    ## Energy equivalents:
    muE_OT = 23.9 # J/mgDW energy equivalent for organic flesh compound
    pOT    = 0.85 # Organic content of soft tissue
    ##### Initial Conditions #####
    t0 = 150 
    ## Starting fraction of somatic flesh 
    pst = 2/3
    OTW = DWt0*pOT
    E0 = OTW*(1-pst)*muE_OT
    Et0 = pst*muE_OT*OTW
    Es0 = Eg_s*DWs0
    R0 = 0
    #### Load Dates
    IndexFirstDay = DayIndex(InitialDate)
    Dates_data = DATES[0,IndexFirstDay:IndexFirstDay+EndTime]
    #### Load Enviromental Data
    SST_data_C = ReadEnvDatFile('InputData/Temp.csv', Nsimulation, IndexFirstDay, EndTime)
    TPM_data = ReadEnvDatFile('InputData/TPM.csv', Nsimulation, IndexFirstDay, EndTime)
    POM_data = ReadEnvDatFile('InputData/POM.csv', Nsimulation, IndexFirstDay, EndTime)
    Rad_data = ReadEnvDatFile('InputData/Rad.csv', Nsimulation, IndexFirstDay, EndTime)
    sal_data = ReadEnvDatFile('InputData/sal.csv', Nsimulation, IndexFirstDay, EndTime)
    ### Transform the SST to kelvin
    SST_data = SST_data_C + 273.15 
    ############   simulation to save each realization  ################
    start_time_sim = time.process_time()
    simulation = [] # List: [WS, WT, L, Resp, Exc, AE, IR, CR,SRresp]
    for nsim in range(Nsimulation):
        realization = SimulateMussel(SST_data[nsim,:],TPM_data[nsim,:],POM_data[nsim,:],
                                     Rad_data[nsim,:],EndTime,t0,Eg_s,Es0,Et0,E0,R0) 
        # Each realization is a list: [XX=[Es, Et, E, R], L , Phi, PN, Resp, Exc, AE, IR,CR,SRresp]
        WT = 0.001*realization[2][:,0]/(muE_OT*pOT)   
        WS = 0.001*realization[0][:,0]/Eg_s                            
        simulation.append([WS,WT,realization[1][:,0],realization[4][:,0],realization[5][:,0],
                           realization[6][:,0],realization[7][:,0],realization[8][:,0],realization[9][:,0]])
    ####### simulations as a matrix ##################
    WSm = np.zeros([len(simulation),len(simulation[0][0])])
    WTm = np.zeros([len(simulation),len(simulation[0][1])])
    Lm = np.zeros([len(simulation),len(simulation[0][2])])
    Resp_m = np.zeros([len(simulation),len(simulation[0][3])])
    Exc_m = np.zeros([len(simulation),len(simulation[0][4])])
    AEm = np.zeros([len(simulation),len(simulation[0][5])])
    IRm = np.zeros([len(simulation),len(simulation[0][6])])
    CRm = np.zeros([len(simulation),len(simulation[0][7])])
    SRrespm = np.zeros([len(simulation),len(simulation[0][8])])
    for sim in range(len(simulation)):
        WSm[sim,:] = simulation[sim][0]
        WTm[sim,:] = simulation[sim][1]
        Lm[sim,:] = simulation[sim][2]
        Resp_m[sim,:] = simulation[sim][3]
        Exc_m[sim,:] = simulation[sim][4]
        AEm[sim,:] = 100*simulation[sim][5] # tanto por 1 to (%)
        IRm[sim,:] = simulation[sim][6]
        CRm[sim,:] = simulation[sim][7]
        SRrespm[sim,:] = simulation[sim][8]
    # to avoid numeric error of recomputing the initial L using the functions Shell_L and inv_Shell_L we save the initial value.
    Lm[:,0] = L0 # avoid residual diferences between the L0 and the recomputed one
    cumResp = np.cumsum(24*Resp_m,axis=1)/1000 # from ml to l
    cumRespS = np.cumsum(24*np.multiply(SRrespm,Resp_m),axis=1)/1000 # from ml to l
    cumExc = np.cumsum(24*Exc_m,axis=1)/1e6    # from \mu g to g
    cumIR = np.cumsum(24*IRm,axis=1)/1000      # from mg to g
    cumCR = np.cumsum(24*CRm,axis=1)
    FWt = 4.64*WTm # Fhesh Wet Weight (g)
    IDWs = 0.95*WSm # CaCO3  (inorganic) shell (g)
    ODWs = 0.05*WSm # organic shell matter (g)
    OFaeces = np.cumsum(np.multiply(24*IRm/1000,1-AEm/100),axis=1) # organic Faeces (g)
    TFW = WSm + FWt # total Wet Weight (g)
    Msimulation = [Lm,WSm,IDWs,ODWs,WTm,FWt,TFW,Resp_m,cumResp,cumRespS,
                   Exc_m,cumExc,AEm,IRm,cumIR,CRm,cumCR,OFaeces,
                   SST_data_C,TPM_data,POM_data,Rad_data,sal_data,Dates_data]
    end_time_sim = time.process_time()
    Tspent = end_time_sim - start_time_sim 
    return Tspent,simulation,Msimulation

############# compute differences ######################
"""
X : np.array of dim n
output an (n-1) np.array with X[1:]-X[0]
"""
def differencesX_X0(X):
    dif_X = X[1:]-X[0]
    return dif_X

############# compute differences Alkalinity ######################
"""
X : np.array of dim n
output an (n) np.array with X[0:]-X[0]
"""
def differencesX_Alk(X):
    dif_X = X[0:]-X[0]
    return dif_X

############# compute comulative summs ##################
"""
X : np.array of dim n
listIndex: list of index to summ between then 
output:np array of dim n-1 (sum_listIndex[0]^listIndex[1]X,...,sum_listIndex[0]^listIndex[-1]X)
"""
def ComulativeSumm(X,listIndex):
    sumX = np.zeros(len(listIndex)-1)
    for i in range(len(listIndex)-1):
        sumX[i] = np.sum(X[listIndex[0]:listIndex[i+1]+1])/(listIndex[i]-listIndex[0]+1)
    return sumX

############# Compute Carbon Concentrations #########################
"""
X     : np.array of dim n+1
CR    : Clearence Rate (l/h) (np.array of dim n+1)
cumCR : cummulative Clearence Rate (l) (np.array of dim n+1)
ConOpt: Options to compute concentrations: ['Daily','Cumulative','Final']
output:np arrayof dim n
"""
def ComputeCarbonConcentrations(X,CR,cumCR,ConOpt):
    muCmol = 1e6/12
    if ConOpt == 'Daily':
        Conc = muCmol*np.divide((X[1:]-X[:-1]),24*CR[1:])
    elif ConOpt == 'Cumulative':
        Conc = muCmol*np.divide((X[1:]-X[0]),cumCR[1:])
    elif ConOpt == 'Final':
        Conc = muCmol*np.array(X[1:]-X[0])/np.max(cumCR[1:])
    else:
        print('ConOpt must take a value in [Daily,Cumulative,Final]')
    return Conc

############# Compute Alkalinity Concentrations #########################
"""
X     : np.array of dim n+1
CR    : Clearence Rate (l/h) (np.array of dim n+1)
cumCR : cummulative Clearence Rate (l) (np.array of dim n+1)
ConOpt: Options to compute concentrations: ['Daily','Cumulative','Final']
output:np arrayof dim n
"""
def ComputeAlkConcentrations(X,CR,cumCR,ConOpt):
    if ConOpt == 'Daily':
        AklCon = 1e4*np.divide((X[1:]-X[:-1]),24*CR[1:])
    elif ConOpt == 'Cumulative':
        AklCon = 1e4*np.divide(X[1:],cumCR[1:])
    elif ConOpt == 'Final':
        AklCon = 1e4*np.array(X[1:])/np.max(cumCR[1:])
    else:
        print('ConOpt must take a value in [Daily,Cumulative,Final]')
    return AklCon

################### Compute Carbon Footprint ############################
def ComputeCarbonFootprint(X,Y):
    return 1e3*np.divide(X,Y)

########## simulate Bio-Calcification using seacarb in R #########
"""
Ccon0   : initial CO2 pressure (ppm)
SST     : sea surface temperature (Celsious)
Sal     : salinity
"""
def seacarbBioCal(pCO20,SST,Sal):
    Psi = np.zeros([1,len(SST)])
    # save inputs (vectors)
    INcarb_df = pd.DataFrame(columns=["Sal","SST"])
    for i in range(len(SST)):
        INcarb_df.loc[i]=[Sal[i],SST[i]]
    INcarb_df.to_csv('carbR/input.csv')
    # save inputs (parameters)
    INcarb_df = pd.DataFrame(columns=["pCO2_0"])
    INcarb_df.loc[0]=[pCO20]
    INcarb_df.to_csv('carbR/param.csv')
    # run the BioCal.R script
    call(["Rscript","carbR/BioCal.R"])   
    # load outputs 
    output_df = pd.read_csv('carbR/output.csv')
    Psi = output_df.psi0
    return Psi

################# Carbon table computation ###############
"""
Msimulation : list of (Nsimulation)x(DaysSimulation) matrices:
            Msimulation[0] : L    : shell length (mm)
            Msimulation[1] : WS   : shell weight (g)
            Msimulation[2] : IDWs : CaCO3  (inorganic) shell (g)
            Msimulation[3] : ODWs : organic shell matter (g)
            Msimulation[4] : WT   : Dry tissue weight (g)
            Msimulation[5] : FWt : Fresh tissue Weight (g)
            Msimulation[6] : TFW  : total fresh weight (g)
            Msimulation[7] : Resp : Respitarion (ml/h)
            Msimulation[8] : cumResp : Respitarion (l)
            Msimulation[9] : cumRespS : Shell Respitarion (l)
            Msimulation[10] : Exc  : Amonia excretion ($\mu$g/h)
            Msimulation[11] : cumExc  : Amonia excretion (g)
            Msimulation[12] : AE   : Absorption efficiency (%)
            Msimulation[13] : IR   : Ingestion rate (mg/h)
            Msimulation[14] : cumIR   : Ingestion rate (g)
            Msimulation[15] : CR   : Clearance rate (l/h)
            Msimulation[16] : cumCR   : Clearance rate (l)
            Msimulation[17] : OFaeces : organic Faeces (g)
            Msimulation[18] : SST (ºC) 
            Msimulation[19] : TPM (mg/l)
            Msimulation[20] : POM (mg/l)
            Msimulation[21] : Rad (MJ/(m^2 day))
            Msimulation[22] : Salinity
            Msimulation[23] : Dates
SCoption   : '1' to use one simulation
            'Mean' to use the mean of Rea simulations
Rea       : realization number to use if SCoption='1' or the number of realizations to compute the mean
Pdrdoc    : pDRDOC parameter in the interval [0 1]
Pirdoc    : pIRDOC parameter in the interval [0 1]
pCO2_0    : Initial PCO2 (ppm)
ConOpt    : Options to compute concentrations: ['Daily','Cumulative','Final']
"""
def MusselCarbonContent(Msimulation,SCoption,Rea,Pburial,Pdrdoc,Pirdoc,pCO2_0,ConOpt):
    if SCoption == '1' or Rea == 0:
        # index star in 0
        Rea=Rea-1
        L = Msimulation[0][Rea,:]
        WS = Msimulation[1][Rea,:]
        IDWs = Msimulation[2][Rea,:]
        ODWs = Msimulation[3][Rea,:]
        WT = Msimulation[4][Rea,:]
        FWt = Msimulation[5][Rea,:]
        TFW = Msimulation[6][Rea,:]
        #Resp = Msimulation[7][Rea,:]
        cumResp = Msimulation[8][Rea,:]
        cumRespS = Msimulation[9][Rea,:]
        #Exc = Msimulation[10][Rea,:]
        cumExc = Msimulation[11][Rea,:]
        #AE = Msimulation[12][Rea,:]
        #IR = Msimulation[13][Rea,:]
        #cumIR = Msimulation[14][Rea,:]
        CR = Msimulation[15][Rea,:]
        cumCR = Msimulation[16][Rea,:]
        OFaeces = Msimulation[17][Rea,:]
        SST = Msimulation[18][Rea,:] 
        TPM = Msimulation[19][Rea,:]
        POM = Msimulation[20][Rea,:]
        Rad = Msimulation[21][Rea,:]
        sal = Msimulation[22][Rea,:]
        DAT = Msimulation[23]
    else:
        L = Msimulation[0][0:Rea,:].mean(0)
        WS = Msimulation[1][0:Rea,:].mean(0)
        IDWs = Msimulation[2][0:Rea,:].mean(0)
        ODWs = Msimulation[3][0:Rea,:].mean(0)
        WT = Msimulation[4][0:Rea,:].mean(0)
        FWt = Msimulation[5][0:Rea,:].mean(0)
        TFW = Msimulation[6][0:Rea,:].mean(0)
        #Resp = Msimulation[7][0:Rea,:].mean(0)
        cumResp = Msimulation[8][0:Rea,:].mean(0)
        cumRespS = Msimulation[9][0:Rea,:].mean(0)
        #Exc = Msimulation[10][0:Rea,:].mean(0)
        cumExc = Msimulation[11][0:Rea,:].mean(0)
        #AE = Msimulation[12][0:Rea,:].mean(0)
        #IR = Msimulation[13][0:Rea,:].mean(0)
        #cumIR = Msimulation[14][0:Rea,:].mean(0)
        CR = Msimulation[15][0:Rea,:].mean(0)
        cumCR = Msimulation[16][0:Rea,:].mean(0)
        OFaeces = Msimulation[17][0:Rea,:].mean(0)
        SST = Msimulation[18][0:Rea,:].mean(0)
        TPM = Msimulation[19][0:Rea,:].mean(0)
        POM = Msimulation[20][0:Rea,:].mean(0)
        Rad = Msimulation[21][0:Rea,:].mean(0)
        sal = Msimulation[22][0:Rea,:].mean(0)
        DAT = Msimulation[23]
    ### compute Psi to Bio-Calficication ####
    Psi = seacarbBioCal(pCO2_0,SST,sal)
    # To find index of L~15, L~50 and L~75 or similar
    Lmin = np.min(L)
    Lmax = np.max(L)
    if Lmin <= 15 and Lmax >= 75:
        indL15 = np.min(np.where(L>=15))
        indL50 = np.min(np.where(L>=50))
        indL75 = np.min(np.where(L>=75))
        listIndex = [indL15,indL50,indL75]
    elif Lmin <= 15 and Lmax > 50:
        indL15 = np.min(np.where(L>=15))
        indL50 = np.min(np.where(L>=50))
        indLmax = np.min(np.where(L>=Lmax))
        listIndex = [indL15,indL50,indLmax]
    elif Lmin <= 15 and Lmax <= 50:
        indL15 = np.min(np.where(L>=15))
        indLmax = np.min(np.where(L>=Lmax))
        listIndex = [indL15,indLmax]
    elif Lmin < 50 and Lmax >= 75:
        indLmin = np.min(np.where(L>=Lmin))
        indL50 = np.min(np.where(L>=50))
        indL75 = np.min(np.where(L>=75))
        listIndex = [indLmin,indL50,indL75]
    elif Lmin < 50 and Lmax > 50:
        indLmin = np.min(np.where(L>=Lmin))
        indL50 = np.min(np.where(L>=50))
        indLmax = np.min(np.where(L>=Lmax))
        listIndex = [indLmin,indL50,indLmax]
    elif Lmin < 50 and Lmax <= 50:
        indLmin = np.min(np.where(L>=Lmin))
        indLmax = np.min(np.where(L>=Lmax))
        listIndex = [indLmin,indLmax]
    elif Lmin < 75 and Lmax > 75:
        indLmin = np.min(np.where(L>=Lmin))
        indL75 = np.min(np.where(L>=75))
        listIndex = [indLmin,indL75]
    else:
        indLmin = np.min(np.where(L>=Lmin))
        indLmax = np.min(np.where(L>=Lmax))
        listIndex = [indLmin,indLmax]
        
    #### Names for variables to compute
    VarTableNames = ['Total wet weight (g indv^-1)',
                     'Flesh wet weight (g indv^-1)',
                     'Organic carbon content of mussel flesh (g indv^-1)',
                     'Total Alkalinity change due to mussel flesh (10^-2 eq indv^-1)',
                     'CO_2 eq in mussel flesh (g indv^-1)',
                     'Shell wet weight (g indv^-1)',
                     'Organic carbon content of mussel shell (g indv^-1)',
                     'Total Alkalinity change due to organic mussel shell (10^-2 eq indv^-1)',
                     'CO_2 eq in organic mussel shell (g indv^-1)',
                     'CaCO_3 content in mussel shell (g indv^-1)',
                     'Inorganic carbon in mussel shell (g indv^-1)',
                     'Total Alkalinity change due to inorganic mussel shell (10^-2 eq indv^-1)',
                     'CO_2 eq in inorganic mussel shell (g indv^-1)',
                     'CO_2 released by bio-calcification (g indv^-1)',
                     'Carbon released by bio-calcification (g indv^-1)',
                     'Carbon released by respiration (g indv^-1)',
                     'CO_2 released by respiration (g indv^-1)',
                     'N-NH_4^+ released by excretion (g indv^-1)',
                     'Total Alkalinity change due to excretion (10^-2 eq indv^-1)',
                     'Faeces production (g indv^-1)',
                     'Organic carbon production by faeces (g indv^-1)',
                     'CO_2 releases by faeces degradation (g indv^-1)',
                     'Total Alkalinity change due to faeces egestion (10^-2 eq indv^-1)']
    
    
    ############ construct a dataFrame to save data in excel #############
    #columnsName = ['Variable','15 mm', '50 mm', '75 mm', '50-15 mm', '75-15 mm']
    columnsName = ['Variable']
    for i in range(len(listIndex)):
        columnsName.append("{:.1f}".format(L[listIndex[i]])+' mm')
    for i in range(len(listIndex)-1):
        columnsName.append("{:.1f}".format(L[listIndex[i+1]])+'-'+"{:.1f}".format(L[listIndex[0]])+' mm')
    pd_data = pd.DataFrame(columns=columnsName)
    
    ########## construct VarLENindex (ex. 15 50 75) and VarLENindex1 (ex 50-15, 75-15)
    VarLENindex = []
    VarLENindex1 = []
    # var1 Total wet weight 
    dif_TotalWetWeight = differencesX_X0(TFW[listIndex])
    VarLENindex.append(TFW[listIndex])
    VarLENindex1.append(dif_TotalWetWeight)
    # var2 Flesh wet weight
    dif_FleshWetWeight = differencesX_X0(FWt[listIndex])
    VarLENindex.append(FWt[listIndex])
    VarLENindex1.append(dif_FleshWetWeight)
    # var3 Organic carbon content of mussel flesh (C_OT)
    OrganicCarbonFlesh = -0.455*WT
    dif_OrganicCarbonFlesh = differencesX_X0(OrganicCarbonFlesh[listIndex])
    VarLENindex.append(OrganicCarbonFlesh[listIndex])
    VarLENindex1.append(dif_OrganicCarbonFlesh)
    # var4 Total Alkalinity change due to mussel flesh (Alk_OT)
    NULLvarLEN = []
    for i in range(len(listIndex)):
        NULLvarLEN.append('')
    AlkalinityMusselFlesh = -dif_OrganicCarbonFlesh*(44.3+7.8)/100/12*0.31*100
    AlkalinityMusselFlesh_v = -differencesX_Alk(OrganicCarbonFlesh)*(44.3+7.8)/100/12*0.31*100
    VarLENindex.append(NULLvarLEN)
    VarLENindex1.append(AlkalinityMusselFlesh)
    # var5 CO_2 eq in mussel flesh (CO2_OT)
    CO2Flesh = OrganicCarbonFlesh*(44/12)
    dif_CO2Flesh = differencesX_X0(CO2Flesh[listIndex])
    VarLENindex.append(CO2Flesh[listIndex])
    VarLENindex1.append(dif_CO2Flesh)
    # var6 Shell wet weight
    VarLENindex.append(WS[listIndex])
    VarLENindex1.append(differencesX_X0(WS[listIndex]))
    # var7 Organic carbon content of mussel shell (C_OS)
    OrganicCarbonShell = -ODWs*0.507
    dif_OrganicCarbonShell = differencesX_X0(OrganicCarbonShell[listIndex])
    VarLENindex.append(OrganicCarbonShell[listIndex])
    VarLENindex1.append(dif_OrganicCarbonShell)
    # var8 Total Alkalinity change due to organic mussel shell (Alk_OS)
    AlkalinityOrganicShell = -dif_OrganicCarbonShell/12*0.31*100
    AlkalinityOrganicShell_v = -differencesX_Alk(OrganicCarbonShell)/12*0.31*100
    VarLENindex.append(NULLvarLEN)
    VarLENindex1.append(AlkalinityOrganicShell)
    # var9 CO_2 eq in organic mussel shell (CO2_OS)
    CO2OrganicShell = OrganicCarbonShell*(44/12)
    dif_CO2OrganicShell = differencesX_X0(CO2OrganicShell[listIndex])
    VarLENindex.append(CO2OrganicShell[listIndex])
    VarLENindex1.append(dif_CO2OrganicShell)
    # var10 CaCO_3 content in mussel shell
    dif_CaCO3Shell = differencesX_X0(IDWs[listIndex])
    VarLENindex.append(IDWs[listIndex])
    VarLENindex1.append(dif_CaCO3Shell)
    # var11 Inorganic carbon in mussel shell (C_IS)
    InorganicCarbonShell = -0.12*IDWs
    dif_InorganicCarbonShell = differencesX_X0(InorganicCarbonShell[listIndex])
    VarLENindex.append(InorganicCarbonShell[listIndex])
    VarLENindex1.append(dif_InorganicCarbonShell)
    # var12 Total Alkalinity change due to inorganic mussel shell (Alk_IS)
    AlkalinityInorganicShell = dif_InorganicCarbonShell*2/12*100
    AlkalinityInorganicShell_v = differencesX_Alk(InorganicCarbonShell)*2/12*100
    VarLENindex.append(NULLvarLEN)
    VarLENindex1.append(AlkalinityInorganicShell)
    # var13 CO_2 eq in inorganic mussel shell (CO2_IS)
    CO2InorganicShell = 44/12*InorganicCarbonShell
    dif_CO2InorganicShell = differencesX_X0(CO2InorganicShell[listIndex])
    VarLENindex.append(CO2InorganicShell[listIndex])
    VarLENindex1.append(dif_CO2InorganicShell)
    # var14 CO_2 released by bio-calcification
    CO2BioCalcification = np.zeros(np.size(CO2InorganicShell))
    CO2BioCalcification[1:] = np.cumsum(-44/12*Psi*(InorganicCarbonShell[1:]-InorganicCarbonShell[:-1]))
    dif_CO2BioCalcification = differencesX_X0(CO2BioCalcification[listIndex])
    VarLENindex.append(CO2BioCalcification[listIndex])
    VarLENindex1.append(dif_CO2BioCalcification)
    # var15 Carbon released by bio-calcification
    CarbonBioCalcification = np.zeros(np.size(InorganicCarbonShell))
    CarbonBioCalcification[1:] = np.cumsum(-Psi*(InorganicCarbonShell[1:]-InorganicCarbonShell[:-1]))
    dif_CarbonBioCalcification = differencesX_X0(CarbonBioCalcification[listIndex])
    VarLENindex.append(CarbonBioCalcification[listIndex])
    VarLENindex1.append(dif_CarbonBioCalcification)
    ##### new variables
    # var16 Carbon released by respiration (g) (C_Resp)
    CarbonResp = cumResp/22.4*0.85*12*(1-Pdrdoc)
    dif_CarbonResp = differencesX_X0(CarbonResp[listIndex])
    VarLENindex.append(CarbonResp[listIndex])
    VarLENindex1.append(dif_CarbonResp)
    # var17 CO_2 released by respiration (g) (CO2_Resp)
    CO2respiration = CarbonResp*44/12
    dif_CO2respiration = differencesX_X0(CO2respiration[listIndex])
    VarLENindex.append(CO2respiration[listIndex])
    VarLENindex1.append(dif_CO2respiration)
    # var18 NH_4^+ released by excretion (g)
    dif_NH4excretion = differencesX_X0(cumExc[listIndex])
    VarLENindex.append(cumExc[listIndex])
    VarLENindex1.append(dif_NH4excretion)
    # var19 Total Alkalinity change due to excretion (Alk_Exc)
    AlkalinityExcretion = dif_NH4excretion/14*100
    AlkalinityExcretion_v = 2*cumExc/14*100
    VarLENindex.append(NULLvarLEN)
    VarLENindex1.append(AlkalinityExcretion)
    # var20 Faeces production (g)
    dif_FaecesProduction = differencesX_X0(OFaeces[listIndex])
    VarLENindex.append(OFaeces[listIndex])
    VarLENindex1.append(dif_FaecesProduction)
    # var21 Organic carbon production by faeces (g) (C_OFaeces)
    OrganicCarbonFaeces = -38/100*OFaeces
    dif_OrganicCarbonFaeces = differencesX_X0(OrganicCarbonFaeces[listIndex])
    VarLENindex.append(OrganicCarbonFaeces[listIndex])
    VarLENindex1.append(dif_OrganicCarbonFaeces)
    # var22 CO_2 releases by faeces degradation (g) (CO2_OFaeces)
    CO2faeces = (44/12)*OrganicCarbonFaeces
    dif_CO2faeces = differencesX_X0(CO2faeces[listIndex])
    VarLENindex.append(CO2faeces[listIndex])
    VarLENindex1.append(dif_CO2faeces)
    # var23 Total Alkalinity change due to faeces degradation (Alk_OFaeces)
    ### C_IRDOC
    CBFaeces = (Pburial+Pirdoc*(1-Pburial))*OrganicCarbonFaeces
    dif_CBFaeces = differencesX_X0(CBFaeces[listIndex])
    AlkalinityFaeces = -dif_CBFaeces/12/6.3*100
    AlkalinityFaeces_v = -CBFaeces/12/6.3*100
    VarLENindex.append(NULLvarLEN)
    VarLENindex1.append(AlkalinityFaeces)    
    # construc the dataFrame with data
    if len(VarLENindex1[0])==1:
        for indN in range(len(VarTableNames)):
            pd_data.loc[indN] = [VarTableNames[indN]]+list(VarLENindex[indN])+[VarLENindex1[indN][0]]
    else:
        for indN in range(len(VarTableNames)):
            pd_data.loc[indN] = [VarTableNames[indN]]+list(VarLENindex[indN])+list(VarLENindex1[indN])
    ###### change index to save as excel
    IndexData = np.arange(1,len(pd_data)+1)
    pd_data=pd_data.set_index(pd.Index(IndexData))
    # Variables to return
    ### carbon content in ammonia excretion (C_Exc)
    #CarbonExc = -12/14*6.7*cumExc
    ### CO2_Exc
    #CO2Exc = 44/12*CarbonExc
    ### C_RespS
    CarbonRespS = cumRespS/22.4*0.85*12*(1-Pdrdoc)
    ### CO2_RespS
    CO2RespS = 44/12*CarbonRespS
    ### C_DRDOC
    Cdrdoc = -12*0.85/22.4*Pdrdoc*cumResp
    ### CO2_DRDOC
    CO2drdoc = 44/12*Cdrdoc
    ### C_sDRDOC
    Csdrdoc = -12*0.85/22.4*Pdrdoc*cumRespS
    ### CO2_sDRDOC
    CO2sdrdoc = 44/12*Csdrdoc
    ### CO2_IRDOC
    CO2BFaeces = 2*44/12*CBFaeces
    
    #################  Total amounts #####################
    ### Alk_T
    TotalAlk = (AlkalinityMusselFlesh_v + AlkalinityInorganicShell_v + 
                AlkalinityOrganicShell_v + AlkalinityExcretion_v + AlkalinityFaeces_v)
    ###  C_T
    TotalC = (OrganicCarbonFlesh-OrganicCarbonFlesh[0] 
              + InorganicCarbonShell-InorganicCarbonShell[0]
              + OrganicCarbonShell-OrganicCarbonShell[0] 
              + CarbonResp + CBFaeces)
    ### CO2_T
    TotalCO2 = 44/12*(OrganicCarbonFlesh-OrganicCarbonFlesh[0] 
                      + OrganicCarbonShell-OrganicCarbonShell[0] 
                      + CarbonResp + Cdrdoc + CBFaeces) + CO2BioCalcification
    ### FCO2_T 
    TotalFleshCO2 = 44/12*(OrganicCarbonFlesh-OrganicCarbonFlesh[0] 
                           + Cdrdoc-Csdrdoc + CBFaeces) + CO2respiration-CO2RespS
    ### SCO2_T
    TotalShellCO2 = TotalCO2 - TotalFleshCO2
    ### Total carbon footprint TCFmussel (g/kg)
    TCFmussel = ComputeCarbonFootprint(TotalCO2,TFW)
    ### Total Flesh carbon footprint TCFflesh (g/kg)
    TCFflesh = ComputeCarbonFootprint(TotalFleshCO2,FWt)
    ### Total Shell carbon footprint TCFshell (g/kg)
    TCFshell = ComputeCarbonFootprint(TotalShellCO2,WS)
    
    ### CO2_seq sequestration
    CO2_seq = CO2BioCalcification + 44/12*(CarbonResp + Cdrdoc + CBFaeces)
    ### FCO2_seq
    FleshCO2_seq = CO2respiration-CO2RespS + 44/12*(Cdrdoc-Csdrdoc + CBFaeces)
    ### SCO2_seq
    ShellCO2_seq = CO2_seq - FleshCO2_seq
    ### Total carbon footprint CFmussel (g/kg)
    CFmussel = ComputeCarbonFootprint(CO2_seq,TFW)
    ### Flesh carbon footprint CFflesh (g/kg)
    CFflesh = ComputeCarbonFootprint(FleshCO2_seq,FWt)
    ### Shell carbon footprint CFshell (g/kg)
    CFshell = ComputeCarbonFootprint(ShellCO2_seq,WS)
    
    ##########  lists of variables ##########
    Alkalinity = [AlkalinityMusselFlesh_v,AlkalinityInorganicShell_v,AlkalinityOrganicShell_v,
                  AlkalinityExcretion_v,AlkalinityFaeces_v,TotalAlk]
    Carbon = [OrganicCarbonFlesh,OrganicCarbonShell,InorganicCarbonShell,CarbonResp,
              Cdrdoc,CarbonRespS,Csdrdoc,OrganicCarbonFaeces,CBFaeces,TotalC]
    CO2 = [CO2Flesh,CO2OrganicShell,-CO2InorganicShell,CO2BioCalcification,CO2respiration,
           CO2drdoc,CO2RespS,CO2sdrdoc,CO2faeces,CO2BFaeces,
           TotalCO2,TotalFleshCO2,TotalShellCO2,CO2_seq,FleshCO2_seq,ShellCO2_seq]
    CarbonFootprint = [TCFmussel,TCFflesh,TCFshell,CFmussel,CFflesh,CFshell]
    
    #################### concentrations #####################
    ### [Alk]
    ConcentrationAlk = []
    for i in range(len(Alkalinity)-1):
        ConcentrationAlk.append(ComputeAlkConcentrations(Alkalinity[i],CR,cumCR,ConOpt))
    # total alkalinity concentration
    ConcentrationAlk.append(np.sum(ConcentrationAlk,axis=0))
    ### [C]
    ConcentrationC = []
    for i in range(len(Carbon)-1):
        ConcentrationC.append(ComputeCarbonConcentrations(Carbon[i],CR,cumCR,ConOpt))
    # total carbon concentration
    CTcon     = np.sum(ConcentrationC[:5],axis=0) + ConcentrationC[-1]
    ConcentrationC.append(CTcon) 
    # inputs SST, TPM, POM, Rad, Dates
    Dates = np.squeeze(np.asarray(DAT)) # from 1xn dim to n dim 
    INPUTS = [SST, TPM, POM, Rad, sal, Dates] 
    CarbonOUT = [L,Alkalinity,ConcentrationAlk,Carbon,CO2,CarbonFootprint,ConcentrationC,Psi,INPUTS,listIndex,pd_data]
    return CarbonOUT

########## simulate seacarb module #########
"""
Ccon0   : initial CO2 pressure (ppm)
seacarbIN : 5xn matrix with:
            seacarbIN[0,:] : L shell length mm
            seacarbIN[1,:] : [C] carbon concentration micromol/l
            seacarbIN[2,:] : [Alk] alkalinity concentration 1e-6 eq/l
            seacarbIN[3,:] : SST  sea surface temperature (Celsious)
            seacarbIN[4,:] : Sal
            seacarbIN[5,:] : Dates
"""
def seacarbPY(Ccon0,seacarbIN):
    OUTcarb = np.zeros([4,len(seacarbIN[0,:])])
    # seacarb inputs
    DIC=(seacarbIN[1,:])*1e-6
    ALK=(seacarbIN[2,:])*1e-6
    S=seacarbIN[4,:] 
    T=seacarbIN[3,:]
    # save inputs (vectors)
    INcarb_df = pd.DataFrame(columns=["Alk","DIC","Sal","SST"])
    for i in range(len(DIC)):
        INcarb_df.loc[i]=[ALK[i],DIC[i],S[i],T[i]]
    INcarb_df.to_csv('carbR/input.csv')
    # save inputs (parameters)
    INcarb_df = pd.DataFrame(columns=["pCO2_0"])
    INcarb_df.loc[0]=[Ccon0]
    INcarb_df.to_csv('carbR/param.csv')
    # run the carb.R script
    call(["Rscript","carbR/carb.R"])   
    # load outputs 
    output_df = pd.read_csv('carbR/output.csv')
    OUTcarb[0,:]=output_df.pH
    OUTcarb[1,:]=output_df.pCO2
    OUTcarb[2,:]=output_df.pH_0
    OUTcarb[3,:]=output_df.pCO2_0
    return OUTcarb

 
### save as excel ###
"""
Save each input matrix variable in a different excel sheet:
Msimulation : list of (Nsimulation)x(DaysSimulation) matrices:
            Msimulation[0] : L    : shell length (mm)
            Msimulation[1] : WS   : shell weight (g)
            Msimulation[2] : IDWs : CaCO3  (inorganic) shell (g)
            Msimulation[3] : ODWs : organic shell matter (g)
            Msimulation[4] : WT   : Dry tissue weight (g)
            Msimulation[5] : FWt : Fresh tissue Weight (g)
            Msimulation[6] : TFW  : total fresh weight (g)
            Msimulation[7] : Resp : Respitarion (ml/h)
            Msimulation[8] : cumResp : Respitarion (l)
            Msimulation[9] : cumRespS : Shell Respitarion (l)
            Msimulation[10] : Exc  : Amonia excretion ($\mu$g/h)
            Msimulation[11] : cumExc  : Amonia excretion (g)
            Msimulation[12] : AE   : Absorption efficiency (%)
            Msimulation[13] : IR   : Ingestion rate (mg/h)
            Msimulation[14] : cumIR   : Ingestion rate (g)
            Msimulation[15] : CR   : Clearance rate (l/h)
            Msimulation[16] : cumCR   : Clearance rate (l)
            Msimulation[17] : OFaeces : organic Faeces (g)
            Msimulation[18] : SST (ºC) 
            Msimulation[19] : TPM (mg/l)
            Msimulation[20] : POM (mg/l)
            Msimulation[21] : Rad (MJ/(m^2 day))
            Msimulation[22] : Salinity
            Msimulation[23] : Dates
URLname    : directory and name to save data: 'Simulations/Mexillon_0101.xlsx'
"""
def SaveDataExcel(Msimulation,URLname):
    SheetName = ['L (mm)','DWs (g)','IDWs (g)','ODWs (g)','DWt (g)','FWt (g)','TFW (g)',
                 'Resp (ml per h)','Resp (l)','RespS (l)','Exc (microg per h)','Exc (g)','AE (%)',
                 'OIR (mg per h)','OIR (g)','CR (l per h)','CR (l)','OFaeces (g)',
                 'SST', 'TPM', 'POM', 'Rad', 'sal', 'Date']
    with pd.ExcelWriter(URLname) as writer:
        IndexData = np.arange(1,len(Msimulation[0][:,0])+1)
        ColumnName0 = np.arange(0,len(Msimulation[0][0,:]))
        ColumnName1 = np.arange(1,len(Msimulation[0][0,:]))
        for sheetN in range(len(SheetName)-6):
            pd.DataFrame(Msimulation[sheetN],index=IndexData,columns=ColumnName0).to_excel(writer,sheet_name=SheetName[sheetN])
        for sheetN in range(len(SheetName)-6,len(SheetName)-1):
            pd.DataFrame(Msimulation[sheetN],index=IndexData,columns=ColumnName1).to_excel(writer,sheet_name=SheetName[sheetN])
        pd.DataFrame(Msimulation[-1],index=[1],columns=ColumnName1).to_excel(writer,sheet_name=SheetName[-1])

def SaveDataCarbonExcel(CarbonOUT,URL,name):
    #save data
    URLname=URL+name
    SheetName = ['L (mm)','Alkalinity (10^-2 mol)','Alkalinity (10^-6 mol per kg)',
                 'Carbon (g)','CO2 (g)','Relative CO2 Budget (g per kg)','Carbon 10^-6 mol per kg','Psi','ModelInputs']
    NamesAlk = ['MusselFlesh','InorganicShell','OrganicShell','Ammonium Excretion','BudgetFaeces','Total']           
    NamesAlkcon = ['MusselFlesh','InorganicShell','OrganicShell','Ammonium Excretion','BudgetFaeces','Total']                   
    NamesC = ['OrganicFlesh','OrganicShell','InorganicShell','Respiration','DRDOC',
              'Shell Respiration','Shell DRDOC','OrganicFaeces','BudgetFaeces','Total']
    NamesCO2 = ['OrganicFlesh','OrganicShell','InorganicShell','Calcification','Respiration','DRDOC',
                'Shell Respiration','Shell DRDOC','OrganicFaeces','BudgetFaeces',
                'TotalBudget','FleshBudget','ShellBudget','TotalFootprint','FleshFootprint',
                'ShellFootprint']
    NamesCF = ['Mussel','Flesh','Shell','CFmussel','CFflesh','CFshell']
    NamesInputs = ['SST','TPM','POM','Rad','sal','Dates']
    IndexData = [[1],NamesAlk,NamesAlkcon,NamesC,NamesCO2,NamesCF,NamesC,[1],NamesInputs]
    ColumnName = np.arange(0,len(CarbonOUT[0][:]))
    ColumnNameS = np.arange(1,len(CarbonOUT[0][:]))
    with pd.ExcelWriter(URLname) as writer:
        for sheetN in range(len(SheetName)):
            data = np.matrix(CarbonOUT[sheetN])
            if sheetN == 2 or sheetN >= 6:
                pd.DataFrame(data,index=IndexData[sheetN],columns=ColumnNameS).to_excel(writer,sheet_name=SheetName[sheetN])
            else:
                pd.DataFrame(data,index=IndexData[sheetN],columns=ColumnName).to_excel(writer,sheet_name=SheetName[sheetN])
    # save table
    # save as excel (Now we do not save it, since all variables are saved before)
    #with pd.ExcelWriter(URL+'CarbonTable'+name) as writer:
    #    CarbonOUT[-1].to_excel(writer,sheet_name='CO2Data')
    #### save total concentrations #####
    with pd.ExcelWriter(URL+'TotalConcentrations'+name) as writer:
        namesCon = ['L','[C]','[Alk]','SST','Sal','Dates']
        data = np.matrix([CarbonOUT[0][1:],CarbonOUT[6][-1],CarbonOUT[2][-1],
                          CarbonOUT[8][0],CarbonOUT[8][-2],CarbonOUT[8][-1]])
        pd.DataFrame(data,index=namesCon,columns=ColumnNameS).to_excel(writer,sheet_name='TotalConcentrations')  

def SaveSeacarbExcel(seacarbOUT,seacarbIN,URLname):
    SheetName = ['pH','pCO2','ModelInputs']
    NamespH = ['pH','pH0','pH-pH0']           
    NamespCO2 = ['pCO2','pCO20','pCO2-pCO20'] 
    NamesInputs = ['L','[C]','[Alk]','SST','Sal','Dates']
    IndexData = [NamespH,NamespCO2,NamesInputs]
    ColumnNameS = np.arange(1,len(seacarbOUT[0])+1)
    with pd.ExcelWriter(URLname) as writer:
        for sheetN in range(len(SheetName)):
            if sheetN < len(SheetName)-1:
                data = np.matrix([seacarbOUT[sheetN,:],seacarbOUT[sheetN+2,:],seacarbOUT[sheetN,:]-seacarbOUT[sheetN+2,:]])
                pd.DataFrame(data,index=IndexData[sheetN],columns=ColumnNameS).to_excel(writer,sheet_name=SheetName[sheetN])
            else:
                data = seacarbIN
                pd.DataFrame(data,index=IndexData[sheetN],columns=ColumnNameS).to_excel(writer,sheet_name=SheetName[sheetN])
            
#### load excel with simulation data
"""
SIMname  : directory and name where the data is 
"""
def LoadSimulation(SIMname):
    SheetName = ['L (mm)','DWs (g)','IDWs (g)','ODWs (g)','DWt (g)','FWt (g)','TFW (g)',
                 'Resp (ml per h)','Resp (l)','RespS (l)','Exc (microg per h)','Exc (g)','AE (%)',
                 'OIR (mg per h)','OIR (g)','CR (l per h)','CR (l)','OFaeces (g)',
                 'SST', 'TPM', 'POM', 'Rad', 'sal', 'Date']
    SINfile = pd.ExcelFile(SIMname)
    Msimulation = []
    for sheetN in range(len(SheetName)):
        varN = pd.read_excel(SINfile, sheetN)
        Msimulation.append(varN.to_numpy()[:,1:len(varN.loc[0])])
    return Msimulation
#### load excel with carbon concentrations data 
"""
SIMname  : directory and name where the data is 

return: Array 6xn where rows are: ['L','[C]','[Alk]','SST','Sal','Dates']
"""
def LoadSeacarbIN(SIMname):
    SINfile = pd.ExcelFile(SIMname)
    varN = pd.read_excel(SINfile)
    seacarbIN = varN.to_numpy()[:,1:len(varN.loc[0])]
    return seacarbIN

########################### Plot Results ##############################
"""
Msimulation : list of (Nsimulation)x(DaysSimulation) matrices:
            Msimulation[0] : L    : shell length (mm)
            Msimulation[1] : WS   : shell weight (g)
            Msimulation[2] : IDWs : CaCO3  (inorganic) shell (g)
            Msimulation[3] : ODWs : organic shell matter (g)
            Msimulation[4] : WT   : Dry tissue weight (g)
            Msimulation[5] : FWt : Fresh tissue Weight (g)
            Msimulation[6] : TFW  : total fresh weight (g)
            Msimulation[7] : Resp : Respitarion (ml/h)
            Msimulation[8] : cumResp : Respitarion (l)
            Msimulation[9] : cumRespS : Shell Respitarion (l)
            Msimulation[10] : Exc  : Amonia excretion ($\mu$g/h)
            Msimulation[11] : cumExc  : Amonia excretion (g)
            Msimulation[12] : AE   : Absorption efficiency (%)
            Msimulation[13] : IR   : Ingestion rate (mg/h)
            Msimulation[14] : cumIR   : Ingestion rate (g)
            Msimulation[15] : CR   : Clearance rate (l/h)
            Msimulation[16] : cumCR   : Clearance rate (l)
            Msimulation[17] : OFaeces : organic Faeces (g)
            Msimulation[18] : SST (ºC) 
            Msimulation[19] : TPM (mg/l)
            Msimulation[20] : POM (mg/l)
            Msimulation[21] : Rad (MJ/(m^2 day))
            Msimulation[22] : Salinity
            Msimulation[23] : Dates
NreaPlot : int Number of realizations or realization number
ReaOption: str '1' -> only plot 1 realization (the NreaPlot-th)
               'multiple' -> plot NreaPlot realization
VarOption: str in ['L','DWs','IDWs','ODWs','DWt','FWt','TFW','Resp','cumResp','cumRespS',
                 'Exc','cumExc','AE','OIR','cumOIR','CR','cumCR','OFaeces','all']
"""
def PlotSIN(Msimulation,NreaPlot,ReaOption,VarOption):
    date_dt = []
    for i in range(np.size(Msimulation[-1])):
        date_dt.append(dt.datetime.strptime(Msimulation[-1][0,i],'%Y-%m-%d').date())
    TT = date_dt
    # Options for TT label
    T_format = '%m-%d' # full date '%Y-%m-%d'
    T_interval = int(np.round(len(TT)/10)) # to have 10 ticklabel
    TextTitle = 'Seeding date (M-D): ' + '{:%m-%d}'.format(TT[0])
    LabelName = ['$L$ (mm)','$DW_S$ (g)','$IDW_S$ (g)','$ODW_S$ (g)','$DW_T$ (g)','$FW_T$ (g)','$TFW$ (g)',
                 '$Resp$ (ml/h)','$Resp$ (l)','$Resp_S$ (l)','$Exc$ ($\mu$g/h)','$Exc$ (g)','$AE$ (%)',
                 '$OIR$ (mg/h)','$OIR$ (g)','$CR$ (l/h)','$CR$ (l)','$OFaeces$ (g)']
    if VarOption == 'all':
        for i in range(len(LabelName)):
            if ReaOption == '1':
                plt.figure()
                plt.gca().xaxis.set_major_formatter(mdates.DateFormatter(T_format))
                plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval=T_interval))
                plt.plot(TT,Msimulation[i][NreaPlot-1,1:],'k-',linewidth=2,label=LabelName[i])
                plt.gcf().autofmt_xdate()
                plt.xlabel('time (days)')
                plt.ylabel(LabelName[i])
                plt.title(TextTitle)
                plt.legend()
            else:
                plt.figure()
                plt.gca().xaxis.set_major_formatter(mdates.DateFormatter(T_format))
                plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval=T_interval))
                for rea in range(NreaPlot):
                    plt.plot(TT,Msimulation[i][rea,1:])
                plt.gcf().autofmt_xdate()
                plt.xlabel('time (days)')
                plt.ylabel(LabelName[i])
                plt.title(TextTitle)
        plt.show()
    else:
        OptionVar = ['L','DWs','IDWs','ODWs','DWt','FWt','TFW','Resp','cumResp','cumRespS',
                 'Exc','cumExc','AE','OIR','cumOIR','CR','cumCR','OFaeces','all']
        VP = OptionVar.index(VarOption)
        if ReaOption == '1':
            plt.figure()
            plt.gca().xaxis.set_major_formatter(mdates.DateFormatter(T_format))
            plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval=T_interval))
            plt.plot(TT,Msimulation[VP][NreaPlot-1,1:],'k-',linewidth=2,label=LabelName[VP])
            plt.gcf().autofmt_xdate()
            plt.xlabel('time (days)')
            plt.ylabel(LabelName[VP])
            plt.title(TextTitle)
            plt.legend()
            plt.show()
        else:
            plt.figure()
            plt.gca().xaxis.set_major_formatter(mdates.DateFormatter(T_format))
            plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval=T_interval))
            for rea in range(NreaPlot):
                plt.plot(TT,Msimulation[VP][rea,1:])
            plt.gcf().autofmt_xdate()
            plt.xlabel('time (days)')
            plt.ylabel(LabelName[VP])
            plt.title(TextTitle)
            plt.show()
#plot simulation inputs
def PlotSINin(Msimulation,NreaPlot,ReaOption,VarOption):
    date_dt = []
    for i in range(np.size(Msimulation[-1])):
        date_dt.append(dt.datetime.strptime(Msimulation[-1][0,i],'%Y-%m-%d').date())
    TT = date_dt
    # Options for TT label
    T_format = '%m-%d' # full date '%Y-%m-%d'
    T_interval = int(np.round(len(TT)/10)) # to have 10 ticklabel
    TextTitle = 'Seeding date (M-D): ' + '{:%m-%d}'.format(TT[0])
    LabelName = ['$SST$ ($^{\circ}$C)','$TPM$ (mg/l)','$POM$ (mg/l)','$Rad$ (MJ/(m$^2$day))','Salinity']
    StartInd = 18 # fisrt index of input data in Msimulation
    if VarOption == 'all':
        for i in range(len(LabelName)):
            if ReaOption == '1':
                plt.figure()
                plt.gca().xaxis.set_major_formatter(mdates.DateFormatter(T_format))
                plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval=T_interval))
                plt.plot(TT,Msimulation[StartInd+i][NreaPlot-1,:],'k-',linewidth=2,label=LabelName[i])
                plt.gcf().autofmt_xdate()
                plt.xlabel('time (days)')
                plt.ylabel(LabelName[i])
                plt.title(TextTitle)
                plt.legend()
            else:
                plt.figure()
                plt.gca().xaxis.set_major_formatter(mdates.DateFormatter(T_format))
                plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval=T_interval))
                for rea in range(NreaPlot):
                    plt.plot(TT,Msimulation[StartInd+i][rea,:])
                plt.gcf().autofmt_xdate()
                plt.xlabel('time (days)')
                plt.ylabel(LabelName[i])
                plt.title(TextTitle)
        plt.show()
    else:
        OptionVar = ['SST','TPM','POM','Rad','sal']
        VP = OptionVar.index(VarOption)
        if ReaOption == '1':
            plt.figure()
            plt.gca().xaxis.set_major_formatter(mdates.DateFormatter(T_format))
            plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval=T_interval))
            plt.plot(TT,Msimulation[StartInd+VP][NreaPlot-1,:],'k-',linewidth=2,label=LabelName[VP])
            plt.gcf().autofmt_xdate()
            plt.xlabel('time (days)')
            plt.ylabel(LabelName[VP])
            plt.title(TextTitle)
            plt.legend()
            plt.show()
        else:
            plt.figure()
            plt.gca().xaxis.set_major_formatter(mdates.DateFormatter(T_format))
            plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval=T_interval))
            for rea in range(NreaPlot):
                plt.plot(TT,Msimulation[StartInd+VP][rea,:])
            plt.gcf().autofmt_xdate()
            plt.xlabel('time (days)')
            plt.ylabel(LabelName[VP])
            plt.title(TextTitle)
            plt.show()

# plot Carbon Output
def plotCarbonOut(CarbonOUT,XlabelOp,options):
    L = CarbonOUT[0]
    date_dt = []
    for i in range(np.size(Msimulation[-1])):
        date_dt.append(dt.datetime.strptime(CarbonOUT[-3][-1][i],'%Y-%m-%d').date())
    TT = date_dt
    # Options for TT label
    T_format = '%m-%d' # full date '%Y-%m-%d'
    T_interval = int(np.round(len(L)/10)) # to have 10 ticklabel
    TextTitle = 'Seeding date (M-D): ' + '{:%m-%d}'.format(TT[0])
    OptionVar = ['Alkalinity ($10^{-2}$ mol)','Alkalinity ($\mu$mol/kg)',
                 'Carbon (g)','CO$_2$ (g)','CO$_2$ (g/kg)','Carbon ($\mu$mol/kg)','$\Psi$']
    NamesAlk = ['MusselFlesh','InorganicShell','OrganicShell','Ammonium Excretion','BudgetFaeces','Total']           
    NamesAlkcon = ['MusselFlesh','InorganicShell','OrganicShell','Ammonium Excretion','BudgetFaeces','Total']                   
    NamesC = ['OrganicFlesh','OrganicShell','InorganicShell','Respiration','DRDOC',
              'Shell Respiration','Shell DRDOC','OrganicFaeces','BudgetFaeces','Total']
    NamesCO2 = ['OrganicFlesh','OrganicShell','InorganicShell','Calcification','Respiration','DRDOC',
                'Shell Respiration','Shell DRDOC','OrganicFaeces','BudgetFaeces',
                'Total','TotalFlesh','TotalShell','Footprint','FleshFootprint',
                'ShellFootprint']
    NamesCF = ['Mussel','Flesh','Shell','CFmussel','CFflesh','CFshell']
    VarNames = [NamesAlk,NamesAlkcon,NamesC,NamesCO2,NamesCF,NamesC,'$\Psi$']
    if options == 'Alk':
        indP = 0
    elif options == '[Alk]':
        indP = 1
    elif options == 'cumC':
        indP = 2
    elif options == 'cumCO2':
        indP = 3
    elif options == 'rCO2':
        indP = 4
    elif options == '[C]':
        indP = 5
    else:
        indP = 6
    if XlabelOp == 'L':
        X = CarbonOUT[0][1:]
        Istart = 1  # where we start to plot to avoid the 0 at initial condition
        if options == '[Alk]' or options == '[C]' or options == 'Psi':
            Istart = 0
        Nxlabel = '$L$ (mm)'
        
        # plot the variables by group
        if L[0]<50 and L[-1]>50:
            indL50 = np.min(np.where(L>=50)) - 1 # length of L starts in 0 
            if indP < 6:
                for i in range(len(VarNames[indP])):
                    plt.figure()
                    plt.plot(X,CarbonOUT[1+indP][i][Istart:],'k-',linewidth=2)
                    if np.min(CarbonOUT[1+indP][i][Istart:])*np.max(CarbonOUT[1+indP][i][Istart:]) < 0:
                        plt.plot(X,np.zeros(np.size(X)),'r--',linewidth=1)
                    plt.plot([X[indL50],X[indL50]],[np.min(CarbonOUT[1+indP][i]),CarbonOUT[1+indP][i][indL50+Istart]],'b:',linewidth=1)
                    plt.plot(X[:indL50],np.ones(np.size(X[:indL50]))*CarbonOUT[1+indP][i][indL50+Istart],'b:',linewidth=1)
                    plt.xlabel(Nxlabel)
                    plt.ylabel(OptionVar[indP])
                    plt.title(VarNames[indP][i]+'\n'+TextTitle)
                plt.show()
            else: # plot Psi
                plt.figure()
                plt.plot(X,CarbonOUT[1+indP],'k-',linewidth=2)
                if np.min(CarbonOUT[1+indP])*np.max(CarbonOUT[1+indP]) < 0:
                    plt.plot(X,np.zeros(np.size(X)),'r--',linewidth=1)    
                plt.plot([X[indL50],X[indL50]],[np.min(CarbonOUT[1+indP]),CarbonOUT[1+indP][indL50+Istart]],'b:',linewidth=1)
                plt.plot(X[:indL50],np.ones(np.size(X[:indL50]))*CarbonOUT[1+indP][indL50+Istart],'b:',linewidth=1)
                plt.xlabel(Nxlabel)
                plt.ylabel(OptionVar[indP])
                plt.title(VarNames[indP]+'\n'+TextTitle)
                plt.show()
        else:
            if indP < 6:
                for i in range(len(VarNames[indP])):
                    plt.figure()
                    plt.plot(X,CarbonOUT[1+indP][i][Istart:],'k-',linewidth=2)
                    if np.min(CarbonOUT[1+indP][i][Istart:])*np.max(CarbonOUT[1+indP][i][Istart:]) < 0:
                        plt.plot(X,np.zeros(np.size(X)),'r--',linewidth=1)
                    plt.xlabel(Nxlabel)
                    plt.ylabel(OptionVar[indP])
                    plt.title(VarNames[indP][i]+'\n'+TextTitle)
                plt.show()
            else: # plot Psi
                plt.figure()
                plt.plot(X,CarbonOUT[1+indP],'k-',linewidth=2)
                plt.xlabel(Nxlabel)
                plt.ylabel(OptionVar[indP])
                plt.title(VarNames[indP]+'\n'+TextTitle)
                plt.show()    
    else:
        X = TT
        Istart = 1  # where we start to plot to avoid the 0 at initial condition
        if options == '[Alk]' or options == '[C]':
            Istart = 0
        Nxlabel = '$T$ (days)'
        if L[0]<50 and L[-1]>50:
            indL50 = np.min(np.where(L>=50)) - 1 # length of L starts in 0
            # plot the variables by group
            if indP < 6:
                for i in range(len(VarNames[indP])):
                    plt.figure()
                    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter(T_format))
                    plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval=T_interval))
                    plt.plot(X,CarbonOUT[1+indP][i][Istart:],'k-',linewidth=2)
                    if np.min(CarbonOUT[1+indP][i][Istart:])*np.max(CarbonOUT[1+indP][i][Istart:]) < 0:
                        plt.plot(X,np.zeros(np.size(X)),'r--',linewidth=1)
                    plt.plot([X[indL50],X[indL50]],[np.min(CarbonOUT[1+indP][i]),CarbonOUT[1+indP][i][indL50+Istart]],'b:',linewidth=1)
                    plt.plot(X[:indL50],np.ones(np.size(X[:indL50]))*CarbonOUT[1+indP][i][indL50+Istart],'b:',linewidth=1)
                    plt.gcf().autofmt_xdate()
                    plt.xlabel(Nxlabel)
                    plt.ylabel(OptionVar[indP])
                    plt.title(VarNames[indP][i]+'\n'+TextTitle)
                plt.show()
            else:
                plt.figure()
                plt.gca().xaxis.set_major_formatter(mdates.DateFormatter(T_format))
                plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval=T_interval))
                plt.plot(X,CarbonOUT[1+indP],'k-',linewidth=2)
                if np.min(CarbonOUT[1+indP])*np.max(CarbonOUT[1+indP]) < 0:
                    plt.plot(X,np.zeros(np.size(X)),'r--',linewidth=1)
                plt.plot([X[indL50],X[indL50]],[np.min(CarbonOUT[1+indP]),CarbonOUT[1+indP][indL50+Istart]],'b:',linewidth=1)
                plt.plot(X[:indL50],np.ones(np.size(X[:indL50]))*CarbonOUT[1+indP][indL50+Istart],'b:',linewidth=1)
                plt.gcf().autofmt_xdate()
                plt.xlabel(Nxlabel)
                plt.ylabel(OptionVar[indP])
                plt.title(VarNames[indP]+'\n'+TextTitle)
                plt.show()
        else:
            if indP < 6:
                for i in range(len(VarNames[indP])):
                    plt.figure()
                    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter(T_format))
                    plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval=T_interval))
                    plt.plot(X,CarbonOUT[1+indP][i][Istart:],'k-',linewidth=2)
                    if np.min(CarbonOUT[1+indP][i][Istart:])*np.max(CarbonOUT[1+indP][i][Istart:]) < 0:
                        plt.plot(X,np.zeros(np.size(X)),'r--',linewidth=1)
                    plt.gcf().autofmt_xdate()
                    plt.xlabel(Nxlabel)
                    plt.ylabel(OptionVar[indP])
                    plt.title(VarNames[indP][i]+'\n'+TextTitle)
                plt.show()
            else:
                plt.figure()
                plt.gca().xaxis.set_major_formatter(mdates.DateFormatter(T_format))
                plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval=T_interval))
                plt.plot(X,CarbonOUT[1+indP],'k-',linewidth=2)
                if np.min(CarbonOUT[1+indP])*np.max(CarbonOUT[1+indP] ) < 0:
                    plt.plot(X,np.zeros(np.size(X)),'r--',linewidth=1)
                plt.gcf().autofmt_xdate()
                plt.xlabel(Nxlabel)
                plt.ylabel(OptionVar[indP])
                plt.title(VarNames[indP]+'\n'+TextTitle)
                plt.show()

# plot Carbon Input
def PlotCin(CarbonOUT,XlabelOp,options):
    L = CarbonOUT[0]
    date_dt = []
    for i in range(np.size(Msimulation[-1])):
        date_dt.append(dt.datetime.strptime(CarbonOUT[-3][-1][i],'%Y-%m-%d').date())
    TT = date_dt
    # Options for TT label
    T_format = '%m-%d' # full date '%Y-%m-%d'
    T_interval = int(np.round(len(L)/10)) # to have 10 ticklabel
    TextTitle = 'Seeding date (M-D): ' + '{:%m-%d}'.format(TT[0])
    LabelName = ['$SST$ ($^{\circ}$C)','$TPM$ (mg/l)','$POM$ (mg/l)','$Rad$ (MJ/(m$^2$day))','Salinity']
    if XlabelOp == 'L':
        X = CarbonOUT[0][1:]
        Nxlabel = '$L$ (mm)'
        # plot the variables 
        if options == 'all':
            for i in range(len(LabelName)):
                plt.figure()
                plt.plot(X,CarbonOUT[-3][i],'k-',linewidth=2)
                plt.xlabel(Nxlabel)
                plt.ylabel(LabelName[i])
                plt.title(TextTitle)
            plt.show()
        else:
            OptionVar = ['SST','TPM','POM','Rad','sal']
            VP = OptionVar.index(options) 
            plt.figure()
            plt.plot(X,CarbonOUT[-3][VP],'k-',linewidth=2)
            plt.xlabel(Nxlabel)
            plt.ylabel(LabelName[VP])
            plt.title(TextTitle)
            plt.show()
    else:
        X = TT
        Nxlabel = '$T$ (days)'
        # plot the variables by group
        if options == 'all':
            for i in range(len(LabelName)):
                plt.figure()
                plt.gca().xaxis.set_major_formatter(mdates.DateFormatter(T_format))
                plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval=T_interval))
                plt.plot(X,CarbonOUT[-3][i],'k-',linewidth=2)
                plt.gcf().autofmt_xdate()
                plt.xlabel(Nxlabel)
                plt.ylabel(LabelName[i])
                plt.title(TextTitle)
            plt.show()
        else:
            OptionVar = ['SST','TPM','POM','Rad','sal']
            VP = OptionVar.index(options) 
            plt.figure()
            plt.gca().xaxis.set_major_formatter(mdates.DateFormatter(T_format))
            plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval=T_interval))
            plt.plot(X,CarbonOUT[-3][VP],'k-',linewidth=2)
            plt.gcf().autofmt_xdate()
            plt.xlabel(Nxlabel)
            plt.ylabel(LabelName[VP])
            plt.title(TextTitle)
            plt.show()

# plot Seacarb Output
def plotSeacarbOut(seacarbOUT,seacarbIN,XlabelOp,options):
    L = seacarbIN[0,:]
    date_dt = []
    for i in range(len(seacarbIN[-1,:])):
        date_dt.append(dt.datetime.strptime(seacarbIN[-1,i],'%Y-%m-%d').date())
    TT = date_dt
    # Options for TT label
    T_format = '%m-%d' # full date '%Y-%m-%d'
    T_interval = int(np.round(len(L)/10)) # to have 10 ticklabel
    TextTitle = 'Seeding date (M-D): ' + '{:%m-%d}'.format(TT[0])
    OptionVar = ['pH','pCO2']
    NamespH = ['pH','pH$_0$', '$\Delta$pH']
    NamespCO2 = ['pCO2','pCO2$_0$', '$\Delta$pCO2']
    VarNames = [NamespH,NamespCO2]
    # construct the variables to plot
    SCplotVAR = np.zeros([6,len(seacarbIN[0,:])])
    SCplotVAR[0,:] = seacarbOUT[0,:]
    SCplotVAR[1,:] = seacarbOUT[2,:]
    SCplotVAR[2,:] = seacarbOUT[0,:]-seacarbOUT[2,:]
    SCplotVAR[3,:] = seacarbOUT[1,:]
    SCplotVAR[4,:] = seacarbOUT[3,:]
    SCplotVAR[5,:] = seacarbOUT[1,:]-seacarbOUT[3,:]
    if options == 'pH':
        indP = 0
    else:
        indP = 1
    if XlabelOp == 'L':
        if L[0]<50 and L[-1]>50:
            indL50 = np.min(np.where(L>=50)) - 1 # length of L starts in 0
            X = seacarbIN[0,:]
            Nxlabel = '$L$ (mm)'
            # plot the variables by group
            for i in range(len(VarNames[indP])):
                plt.figure()
                plt.plot(X,SCplotVAR[3*indP+i,:],'k-',linewidth=2)
                if np.min(SCplotVAR[3*indP+i,:])*np.max(SCplotVAR[3*indP+i,:]) < 0:
                    plt.plot(X,np.zeros(np.size(X)),'r--',linewidth=1)
                plt.plot([X[indL50],X[indL50]],[np.min(SCplotVAR[3*indP+i,:]),SCplotVAR[3*indP+i,indL50]],'b:',linewidth=1)
                plt.plot(X[:indL50],np.ones(np.size(X[:indL50]))*SCplotVAR[3*indP+i,indL50],'b:',linewidth=1)
                plt.xlabel(Nxlabel)
                plt.ylabel(OptionVar[indP])
                plt.title(VarNames[indP][i]+'\n'+TextTitle)
            plt.show()
        else:
            X = seacarbIN[0,:]
            Nxlabel = '$L$ (mm)'
            # plot the variables by group
            for i in range(len(VarNames[indP])):
                plt.figure()
                plt.plot(X,SCplotVAR[3*indP+i,:],'k-',linewidth=2)
                if np.min(SCplotVAR[3*indP+i,:])*np.max(SCplotVAR[3*indP+i,:]) < 0:
                    plt.plot(X,np.zeros(np.size(X)),'r--',linewidth=1)
                plt.xlabel(Nxlabel)
                plt.ylabel(OptionVar[indP])
                plt.title(VarNames[indP][i]+'\n'+TextTitle)
            plt.show()
    else:
        X = TT
        Nxlabel = '$T$ (days)'
        if L[0]<50 and L[-1]>50:
            indL50 = np.min(np.where(L>=50)) - 1 # length of L starts in 0
            # plot the variables by group
            for i in range(len(VarNames[indP])):
                plt.figure()
                plt.gca().xaxis.set_major_formatter(mdates.DateFormatter(T_format))
                plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval=T_interval))
                plt.plot(X,SCplotVAR[3*indP+i,:],'k-',linewidth=2)
                if np.min(SCplotVAR[3*indP+i,:])*np.max(SCplotVAR[3*indP+i,:]) < 0:
                    plt.plot(X,np.zeros(np.size(X)),'r--',linewidth=1)
                plt.plot([X[indL50],X[indL50]],[np.min(SCplotVAR[3*indP+i,:]),SCplotVAR[3*indP+i,indL50]],'b:',linewidth=1)
                plt.plot(X[:indL50],np.ones(np.size(X[:indL50]))*SCplotVAR[3*indP+i,indL50],'b:',linewidth=1)
                plt.gcf().autofmt_xdate()
                plt.xlabel(Nxlabel)
                plt.ylabel(OptionVar[indP])
                plt.title(VarNames[indP][i]+'\n'+TextTitle)
            plt.show()
        else:
            # plot the variables by group
            for i in range(len(VarNames[indP])):
                plt.figure()
                plt.gca().xaxis.set_major_formatter(mdates.DateFormatter(T_format))
                plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval=T_interval))
                plt.plot(X,SCplotVAR[3*indP+i,:],'k-',linewidth=2)
                if np.min(SCplotVAR[3*indP+i,:])*np.max(SCplotVAR[3*indP+i,:]) < 0:
                    plt.plot(X,np.zeros(np.size(X)),'r--',linewidth=1)
                plt.gcf().autofmt_xdate()
                plt.xlabel(Nxlabel)
                plt.ylabel(OptionVar[indP])
                plt.title(VarNames[indP][i]+'\n'+TextTitle)
            plt.show()
# plot Seacarb Input
def PlotSeacarbIn(seacarbIN,XlabelOp,options):
    L = seacarbIN[0,:]
    date_dt = []
    for i in range(len(seacarbIN[-1,:])):
        date_dt.append(dt.datetime.strptime(seacarbIN[-1,i],'%Y-%m-%d').date())
    TT = date_dt
    # Options for TT label
    T_format = '%m-%d' # full date '%Y-%m-%d'
    T_interval = int(np.round(len(L)/10)) # to have 10 ticklabel
    TextTitle = 'Seeding date (M-D): ' + '{:%m-%d}'.format(TT[0])
    LabelName = ['[C] $\mu$mol/kg','[Alk] $\mu$mol/kg','$SST$ ($^{\circ}$C)','Salinity']
    if XlabelOp == 'L':
        X = L
        Nxlabel = '$L$ (mm)'
        # plot the variables 
        if options == 'all':
            for i in range(4):
                plt.figure()
                plt.plot(X,seacarbIN[1+i,:],'k-',linewidth=2)
                plt.xlabel(Nxlabel)
                plt.ylabel(LabelName[i])
                plt.title(TextTitle)
            plt.show()
        else:
            OptionVar = ['[C]','[Alk]','SST','sal']
            VP = OptionVar.index(options) 
            plt.figure()
            plt.plot(X,seacarbIN[1+VP,:],'k-',linewidth=2)
            if np.min(seacarbIN[1+VP,:])*np.max(seacarbIN[1+VP,:]) < 0:
                plt.plot(X,np.zeros(np.size(X)),'r--',linewidth=1)
            plt.xlabel(Nxlabel)
            plt.ylabel(LabelName[VP])
            plt.title(TextTitle)
            plt.show()
    else:
        X = TT
        Nxlabel = '$T$ (days)'
        # plot the variables by group
        if options == 'all':
            for i in range(4):
                plt.figure()
                plt.gca().xaxis.set_major_formatter(mdates.DateFormatter(T_format))
                plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval=T_interval))
                plt.plot(X,seacarbIN[1+i,:],'k-',linewidth=2)
                if np.min(seacarbIN[1+i,:])*np.max(seacarbIN[1+i,:]) < 0:
                    plt.plot(X,np.zeros(np.size(X)),'r--',linewidth=1)
                plt.gcf().autofmt_xdate()
                plt.xlabel(Nxlabel)
                plt.ylabel(LabelName[i])
                plt.title(TextTitle)
            plt.show()
        else:
            OptionVar = ['[C]','[Alk]','SST','sal']
            VP = OptionVar.index(options) 
            plt.figure()
            plt.gca().xaxis.set_major_formatter(mdates.DateFormatter(T_format))
            plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval=T_interval))
            plt.plot(X,seacarbIN[1+VP,:],'k-',linewidth=2)
            if np.min(seacarbIN[1+VP,:])*np.max(seacarbIN[1+VP,:]) < 0:
                plt.plot(X,np.zeros(np.size(X)),'r--',linewidth=1)
            plt.gcf().autofmt_xdate()
            plt.xlabel(Nxlabel)
            plt.ylabel(LabelName[VP])
            plt.title(TextTitle)
            plt.show()
               
# Close all figures
def ClosePlotSIN():
    plt.close('all')           
    

#################################################################
########################### GUI #################################
#################################################################
import tkinter as tk
from tkcalendar import DateEntry
from tkinter import filedialog
from tkinter import messagebox


# globally declare the simulation and Nsimulation variables 
simulation = []
Msimulation = []
CarbonOUT = []
seacarbIN = []
seacarbOUT = []

# GUI icon
MusselICO = "InputData/Mussel.ico"

root = tk.Tk()
root.title("Mussel Simulator")
root.iconbitmap(MusselICO)
#root.geometry("400x600")
# color fondo
CF = '#f0f8ff'
# Label color
LC = '#d9fefa'
root.configure(background=CF)
# Specify button color
BColor = '#00b3ca'

frameM = tk.LabelFrame(root,text="Mussel Growth",font=("Helvetica", 14),bg='White', padx=10,pady=10)
frameM.grid(row=0,column=0,rowspan=2,padx=20, pady=10)
frame1 = tk.LabelFrame(frameM, text="Simulation Inputs:",font=("Helvetica", 14), bg=CF,padx=10,pady=10)
frame1.grid(row=1,column=0,padx=10, pady=10)
frame2 = tk.LabelFrame(frameM,text="Simulate Mussels:",font=("Helvetica", 14),bg=CF,padx=10,pady=10)
frame2.grid(row=2,column=0,padx=10, pady=10)
frame3 = tk.LabelFrame(frameM,text="Simulation Outputs:",font=("Helvetica", 14),bg=CF,padx=10,pady=10)
frame3.grid(row=3,column=0,padx=10, pady=10)
frameC = tk.LabelFrame(root,text="Carbon Budget",font=("Helvetica", 14),bg='White', padx=10,pady=10)
frameC.grid(row=0,column=1,padx=20, pady=10)
framePH = tk.LabelFrame(root,text="Seawater Carbonate Chemistry",font=("Helvetica", 14),bg='White', padx=10,pady=10)
framePH.grid(row=1,column=1,padx=20, pady=10)

############# Number of simulations ############
# Create Text Boxes
SINoptions = [1,10,100,1000]
SINclicked = tk.IntVar()
SINclicked.set(SINoptions[0])
Nsimulation = tk.OptionMenu(frame1, SINclicked, *SINoptions)
Nsimulation.grid(row=1,column=1,padx=(0,10), pady=(10,0))   
#Create Text Box Labels
Nsimulation_label = tk.Label(frame1, text="Number of Simulations",bg=CF)
Nsimulation_label.grid(row=1,column=0,padx=(10,0),pady=(10,0))

########## Total days of Simulation ###############
DSoptions = list(np.arange(150,451,10))
DSclicked = tk.IntVar()
DSclicked.set(DSoptions[0])
Dsimulation = tk.OptionMenu(frame1, DSclicked, *DSoptions)
Dsimulation.grid(row=2,column=1,padx=(0,10), pady=(10,0))   
#Create Text Box Labels
Dsimulation_label = tk.Label(frame1, text="Simulation Time (days)",bg=CF)
Dsimulation_label.grid(row=2,column=0,padx=(10,0),pady=(10,0))


############# Start Date selection  (01-01-2017 to 31-12-2027) ################
FristDate = dt.date(year=2016,month=12,day=31)
LastDate = dt.date(year=2018,month=1,day=1)
cal = DateEntry(frame1, width=12, year=2017, month=1, day=1,
                selectmode='day', locale='en_US',
                background='darkblue', foreground='white', borderwidth=2,
                mindate=FristDate, maxdate=LastDate, date_pattern ='y-mm-dd')
cal.grid(row=3,column=1,padx=10, pady=10)
#Create Text Box Labels
cal_label = tk.Label(frame1, text="Simulation Initial Date",bg=CF)
cal_label.grid(row=3,column=0,padx=(10,0),pady=(10,0))

################# Initial Meat weight ##############
MeatW0 = tk.Entry(frame1,width=20)
MeatW0.insert(tk.END, '0.0186')
MeatW0.grid(row=4,column=1)
MeatW0_label = tk.Label(frame1,text="Initial Dry Tissue Weight (g)",bg=CF)
MeatW0_label.grid(row=4,column=0,padx=(10,0),pady=(10,0))


################# Initial Shell length ##############
ShellL0 = tk.Entry(frame1,width=20)
ShellL0.insert(tk.END, '15.00')
ShellL0.grid(row=5,column=1)
ShellL0_label = tk.Label(frame1,text="Initial Shell Length (mm)",bg=CF)
ShellL0_label.grid(row=5,column=0,padx=(10,0),pady=(10,0))

################# Initial Shell Length ##############
#### parameters #####
## Energy equivalents:
muE_OS = 20.32 
muE_IS = 2    # enery equivalent for inorganic shell compound (J/mg)
pOS    = 0.05 # Organic content of shell
# Unitariy energy demand of shell and meat growth.	
Eg_OS = muE_OS
Eg_IS = muE_IS
Eg_s  = Eg_OS*pOS+Eg_IS*(1-pOS) # volume-specific growth cost for shell
# compute the condition index
CI = tk.Label(frame1,text="",width=20)
CI.grid(row=6,column=1,pady=(10,0))

def ObtainL():
    CI0 = 100*4.64*float(MeatW0.get())/(4.64*float(MeatW0.get())+inv_shell_L(float(ShellL0.get())))
    CI = tk.Label(frame1,text="CI = "+ "{:.2f}".format(CI0) + " (%)",width=20)
    CI.grid(row=6,column=1,pady=(10,0)) 
    # popup showing an alert if CI>45
    if CI0 > 45:
        messagebox.showwarning("Condition Index", "The condition index should be lower than 45% ") 
    if CI0 < 20:
        messagebox.showwarning("Condition Index", "The condition index should be higher than 20% ") 

CI_button = tk.Button(frame1, text="Check Condition Index",width=20,
                           bg=BColor,comman=ObtainL)
CI_button.grid(row=6,column=0,padx=(10,0),pady=(10,0))


################### Simulate Mussels #######################
SimState = tk.Label(frame2,text="",width=20)
SimState.grid(row=8,column=1)

def SimMussel(Eg_s,DWt,L0,EndTime,Nsimulation,InitialDate):
    # simulation start
    SimState = tk.Label(frame2,text="Running ... ",width=20)
    SimState.grid(row=8,column=1)
    global simulation,Msimulation
    Tspent,simulation,Msimulation=Sim_n_Mussels(Eg_s,DWt,L0,EndTime,Nsimulation,InitialDate)
    ## message end simulation
    if Tspent < 10:
        SimMessage = "Time spent: " +"{:.4f}".format(Tspent)+" s"
    else:
        SimMessage = "Time spent: " +"{:.2f}".format(Tspent)+" s"
    SimState = tk.Label(frame2,text=SimMessage,width=20)
    SimState.grid(row=8,column=1)

    
Sim_button = tk.Button(frame2, text="Start Simulation",width=20,bg=BColor,
                       comman=lambda: SimMussel(Eg_s,float(MeatW0.get()),float(ShellL0.get()),DSclicked.get(),SINclicked.get(),cal.get()))
Sim_button.grid(row=8,column=0,padx=(10,0))


############################## save results #########################
def SimSave(Msimulation,InitialDate):
    Fold2Save = filedialog.askdirectory(parent=frame3,initialdir="/...",title='Please select a directory')
    # Name to save file
    SimName = "Simulation_ID"+ InitialDate+ time.strftime("_%Y%m%d_%H%M%S")
    # Save date
    URLname = Fold2Save + "/" + SimName + ".xlsx"
    SaveDataExcel(Msimulation,URLname)
    # popup showing where data where saved
    messagebox.showinfo("Results Saved", "Simulations have been saved in: \n" + Fold2Save)    
Save_button = tk.Button(frame3, text="Save Simulation",width=20,bg=BColor,comman=lambda: SimSave(Msimulation,cal.get()))
Save_button.grid(row=11,column=0,padx=(10,0))                                       
        
def PlotSINop(Msimulation,Nsimulation):
    # new window to select the plot options
    #global PlotOp,NsimPlot
    PlotOp = tk.Tk()
    PlotOp.title("Plot Options")
    #PlotOp.geometry("400x400")
    PlotOp.configure(background=CF)
    PlotOp.iconbitmap(MusselICO)
    ### frames to outputs and inputs
    frameSINout = tk.LabelFrame(PlotOp,text="Plot Outputs:",font=("Helvetica", 14),bg=CF,padx=10,pady=10)
    frameSINout.grid(row=5,column=0,columnspan=2,padx=20, pady=(10,20))
    frameSINin = tk.LabelFrame(PlotOp,text="Plot Inputs:",font=("Helvetica", 14),bg=CF,padx=10,pady=10)
    frameSINin.grid(row=6,column=0,columnspan=2,padx=20, pady=(10,20))
    # Select option to plot 
    if Nsimulation == 1:
        OptionNumb = ['1']
    else:
        OptionNumb = ['1','Multiple']
    POclicked = tk.StringVar(PlotOp)
    POclicked.set(OptionNumb[0])
    NreaPlot = tk.OptionMenu(PlotOp,POclicked,*OptionNumb)
    NreaPlot.grid(row=2,column=1,padx=(0,10), pady=(10,0))
    NreaPlot_label = tk.Label(PlotOp,text="Realizations Options",bg=CF)
    NreaPlot_label.grid(row=2,column=0,padx=(10,0),pady=(10,0))
    # number of realizations per grahp
    NsimPlot = tk.Scale(PlotOp,from_ = 1,to=Nsimulation,orient=tk.HORIZONTAL)
    NsimPlot.grid(row=3,column=1,pady=(10,0))
    NsimPlot_label = tk.Label(PlotOp,text="Realization selection",bg=CF)
    NsimPlot_label.grid(row=3,column=0,padx=(10,0),pady=(10,0))
    # Select variable option to plot 
    OptionVar = ['L','DWs','IDWs','ODWs','DWt','FWt','TFW','Resp','cumResp','cumRespS',
                 'Exc','cumExc','AE','OIR','cumOIR','CR','cumCR','OFaeces','all']
    VARclicked = tk.StringVar(frameSINout)
    VARclicked.set(OptionVar[0])
    varPlot = tk.OptionMenu(frameSINout,VARclicked,*OptionVar)
    varPlot.grid(row=4,column=1,padx=(0,10), pady=(10,0))
    varPlot_label = tk.Label(frameSINout,text="Select Output",bg=CF)
    varPlot_label.grid(row=4,column=0,padx=(10,0),pady=(10,0))
    # button to plot the selected realizations
    PlotSIN_button = tk.Button(frameSINout, text="Plot Selected Realizations",
                               height=2, width=30,bg=BColor,font=("Helvetica", 12),
                           comman=lambda: PlotSIN(Msimulation,NsimPlot.get(),POclicked.get(),VARclicked.get()))
    PlotSIN_button.grid(row=8,column=0,padx=(20,20),pady=(10,0),columnspan=2)
    # Select input to plot 
    OptionVarIn = ['SST','TPM','POM','Rad','sal','all']
    VARinclicked = tk.StringVar(frameSINin)
    VARinclicked.set(OptionVarIn[0])
    varInPlot = tk.OptionMenu(frameSINin,VARinclicked,*OptionVarIn)
    varInPlot.grid(row=9,column=1,padx=(0,10), pady=(10,0))
    varInPlot_label = tk.Label(frameSINin,text="Select Input",bg=CF)
    varInPlot_label.grid(row=9,column=0,padx=(10,0),pady=(10,0))
    # button to plot the selected realizations inputs
    PlotSINin_button = tk.Button(frameSINin, text="Plot Selected Inputs",
                               height=2, width=30,bg=BColor,font=("Helvetica", 12),
                           comman=lambda: PlotSINin(Msimulation,NsimPlot.get(),POclicked.get(),VARinclicked.get()))
    PlotSINin_button.grid(row=11,column=0,padx=(20,20),pady=(10,0),columnspan=2)
    # button to close the plots
    closePlotSIN_button = tk.Button(PlotOp, text="Close all Figures",
                               height=2, width=30,bg=BColor,font=("Helvetica", 12),
                           comman = ClosePlotSIN)
    closePlotSIN_button.grid(row=12,column=0,padx=(20,20),pady=(10,20),columnspan=2)
    PlotOp.mainloop()

########### Plot simulation button ############
PlotSINop_button = tk.Button(frame3, text="Plot Simulation",width=20,bg=BColor,
                           comman=lambda: PlotSINop(Msimulation,SINclicked.get()))
PlotSINop_button.grid(row=11,column=1,padx=(10,0))


#######################################################
#################### Carbon Content ###################
#######################################################

frameCO = tk.LabelFrame(frameC,text="Computation Options",font=("Helvetica", 14),bg=CF, padx=10,pady=10)
frameCO.grid(row=1,column=0,padx=10, pady=10)

########### Seletc realization to work ###############
Coptions = ["LOAD","CURRENT"]
Cclicked = tk.StringVar()
Cclicked.set(Coptions[0])
carbonO = tk.OptionMenu(frameCO, Cclicked, *Coptions)
carbonO.grid(row=1,column=1,padx=(0,10), pady=(10,0))   
carbonO_label = tk.Label(frameCO, text="Select Simulation",bg=CF)
carbonO_label.grid(row=1,column=0,padx=(10,0),pady=(10,0))

#### Load simulation if it is necesary #####
SelectedRea = tk.Label(frameCO,text='',width=20)
SelectedRea.grid(row=2,column=1,pady=(10,0))

def LoadSim(simulation,loadOption):
    SelectedRea = tk.Label(frameCO,text='',width=20)
    SelectedRea.grid(row=2,column=1,pady=(10,0))
    if loadOption == "LOAD":
        File2open = filedialog.askopenfilename(parent=frameCO,initialdir="/...",
                    title='Please select file of simulations',filetypes = (("xlsx files","*.xlsx"),("all files","*.*")))
        global Msimulation
        Msimulation = LoadSimulation(File2open)
        SelectedRea = tk.Label(frameCO,text="File loaded",width=20)
        SelectedRea.grid(row=2,column=1,pady=(10,0))
    else:
        # check if we have simulate some mussel
        if len(simulation)==0:
            # popup showing where data where saved
            messagebox.showwarning("CURRENT simulation", "The Mussel Growth module should be run prior to use this option.")    
            SelectedRea = tk.Label(frameCO,text='No Simulation',width=20)
            SelectedRea.grid(row=2,column=1,pady=(10,0))
        else:               
            SelectedRea = tk.Label(frameCO,text='Current Simulation',width=20)
            SelectedRea.grid(row=2,column=1,pady=(10,0))

SelectedRea_button = tk.Button(frameCO, text="Check Selection",width=20,
                           bg=BColor,comman=lambda:LoadSim(simulation,Cclicked.get()))
SelectedRea_button.grid(row=2,column=0,padx=(10,0),pady=(10,0))


# Start simulation of Carbon
def ObtainLinterval(Msimulation,SCoption,Rea,frameCin):
    if SCoption == '1' or Rea == 0:
        # index star in 0
        L = Msimulation[0][Rea-1,:]
    else:
        L = Msimulation[0][0:Rea,:].mean(0)
    Lmin = np.min(L)
    Lmax = np.max(L)
    ShellInt = tk.Label(frameCin,text="Lmin = "+ "{:.2f}".format(Lmin) + " (mm)",width=20)
    ShellInt.grid(row=5,column=1,pady=(10,0))
    ShellInt = tk.Label(frameCin,text="Lmax = "+ "{:.2f}".format(Lmax) + " (mm)",width=20)
    ShellInt.grid(row=6,column=1,pady=(10,0))

def SimulateCarbon(Msimulation,SCoption,Rea,Pburial,Pdrdoc,Pirdoc,pCO2_0,ConOpt):
    global CarbonOUT
    CarbonOUT = MusselCarbonContent(Msimulation,SCoption,Rea,Pburial,Pdrdoc,Pirdoc,pCO2_0,ConOpt)
    ### close the window
    ### simCop.destroy()

def SimCarSave(CarbonOUT):
    Fold2Save = filedialog.askdirectory(parent=simCop,initialdir="/...",title='Please select a directory')
    # Name to save file
    SimName = 'CarbonSim_' + CarbonOUT[8][-1][0] + time.strftime("_%Y%m%d_%H%M%S") + ".xlsx"
    # Save date
    URLname = Fold2Save + "/"  
    SaveDataCarbonExcel(CarbonOUT,URLname,SimName)
    # popup showing where data where saved
    messagebox.showinfo("Results Saved", "Simulations have been saved in: \n" + Fold2Save,parent=simCop)    

def PlotCarbon(CarbonOUT):
    # new window to select the plot options
    PlotC = tk.Tk()
    PlotC.title("Carbon Plot Options")
    #PlotC.geometry("400x400")
    PlotC.configure(background=CF)
    PlotC.iconbitmap(MusselICO)
    ### frames to outputs and inputs
    frameCPout = tk.LabelFrame(PlotC,text="Plot Outputs:",font=("Helvetica", 14),bg=CF,padx=10,pady=10)
    frameCPout.grid(row=5,column=0,columnspan=2,padx=20, pady=(10,20))
    frameCPin = tk.LabelFrame(PlotC,text="Plot Inputs:",font=("Helvetica", 14),bg=CF,padx=10,pady=10)
    frameCPin.grid(row=6,column=0,columnspan=2,padx=20, pady=(10,20))  
    # Select option to plot 
    OptionNumb = ['L','T']
    POclicked = tk.StringVar(PlotC)
    POclicked.set(OptionNumb[0])
    labelPlot = tk.OptionMenu(PlotC,POclicked,*OptionNumb)
    labelPlot.grid(row=1,column=1,padx=(0,10), pady=(10,0))
    labelPlot_label = tk.Label(PlotC,text="X-axis Label Options",bg=CF)
    labelPlot_label.grid(row=1,column=0,padx=(10,0),pady=(10,0))
    # Select variable option to plot 
    OptionVar = ['Alk','[Alk]','cumC','cumCO2','rCO2','[C]','Psi']
    cVARclicked = tk.StringVar(frameCPout)
    cVARclicked.set(OptionVar[0])
    varPlot = tk.OptionMenu(frameCPout,cVARclicked,*OptionVar)
    varPlot.grid(row=2,column=1,padx=(0,10), pady=(10,0))
    varPlot_label = tk.Label(frameCPout,text="Select Output",bg=CF)
    varPlot_label.grid(row=2,column=0,padx=(10,0),pady=(10,0))
    # button to plot the selected realizations
    PlotC_button = tk.Button(frameCPout, text="Plot Selection",
                               height=2, width=30,bg=BColor,font=("Helvetica", 12),
                           comman=lambda: plotCarbonOut(CarbonOUT,POclicked.get(),cVARclicked.get()))
    PlotC_button.grid(row=11,column=0,padx=(20,20),pady=(10,0),columnspan=2)
    # Select input to plot 
    OptionVarCIn = ['SST','TPM','POM','Rad','sal','all']
    VARCinclicked = tk.StringVar(frameCPin)
    VARCinclicked.set(OptionVarCIn[0])
    varCInPlot = tk.OptionMenu(frameCPin,VARCinclicked,*OptionVarCIn)
    varCInPlot.grid(row=9,column=1,padx=(0,10), pady=(10,0))
    varCInPlot_label = tk.Label(frameCPin,text="Select Input",bg=CF)
    varCInPlot_label.grid(row=9,column=0,padx=(10,0),pady=(10,0))
    # button to plot the selected realizations inputs
    PlotCin_button = tk.Button(frameCPin, text="Plot Selected Inputs",
                               height=2, width=30,bg=BColor,font=("Helvetica", 12),
                           comman=lambda: PlotCin(CarbonOUT,POclicked.get(),VARCinclicked.get()))
    PlotCin_button.grid(row=11,column=0,padx=(20,20),pady=(10,0),columnspan=2)
    # close all figures
    closePlotSIN_button = tk.Button(PlotC, text="Close all Figures",
                               height=2, width=30,bg=BColor,font=("Helvetica", 12),
                           comman = ClosePlotSIN)
    closePlotSIN_button.grid(row=12,column=0,padx=(20,20),pady=(10,20),columnspan=2)
    PlotC.mainloop()

def computeCarbonVar(Msimulation):
    # new window to select the simulation options
    global simCop
    simCop = tk.Tk()
    simCop.title("Carbon Simulation Options")
    #simCop.geometry("400x400")
    simCop.configure(background=CF)
    simCop.iconbitmap(MusselICO)
    ################# Simulation Inputs ############################
    frameCin = tk.LabelFrame(simCop,text="Simulation Inputs:",font=("Helvetica", 14),bg=CF,padx=10,pady=10)
    frameCin.grid(row=1,column=0,columnspan=4,padx=20, pady=(10,20))
    # Select option to simulate 
    Nsimulation = len(Msimulation[0][:,0])
    if Nsimulation == 1:
        OptionSC = ['1']
    else:
        OptionSC = ['1','Mean']
    SCclicked = tk.StringVar(simCop)
    SCclicked.set(OptionSC[0])
    NreaSC = tk.OptionMenu(frameCin,SCclicked,*OptionSC)
    NreaSC.grid(row=1,column=1,padx=(0,10), pady=(10,0))
    NreaSC_label = tk.Label(frameCin,text="Input Options",bg=CF)
    NreaSC_label.grid(row=1,column=0,padx=(10,0),pady=(10,0))
    # number of realization selection
    NsimSC = tk.Scale(frameCin,from_ = 1,to=Nsimulation,orient=tk.HORIZONTAL)
    NsimSC.grid(row=2,column=1,rowspan=2,pady=(10,0))
    NsimSC_label = tk.Label(frameCin,text="Realization selection",bg=CF)
    NsimSC_label.grid(row=2,column=0,rowspan=2,padx=(10,0),pady=(10,0))
    # length selection
    ShellInt = tk.Label(frameCin,text="Lmin",width=20)
    ShellInt.grid(row=5,column=1,pady=(10,0))
    ShellInt = tk.Label(frameCin,text="Lmax",width=20)
    ShellInt.grid(row=6,column=1,pady=(10,0))
    ShellINT_button = tk.Button(frameCin, text="Check Shell Length",width=20,
                               bg=BColor,comman= lambda: ObtainLinterval(Msimulation,SCclicked.get(),NsimSC.get(),frameCin))
    ShellINT_button.grid(row=5,column=0,padx=(10,0),pady=(10,0))
    ################# parameters 3-4 columns ################################
    ################# Pburial ##############
    Pburial = tk.Entry(frameCin,width=20)
    Pburial.insert(tk.END, '0.08')
    Pburial.grid(row=1,column=3)
    Pburial_label = tk.Label(frameCin,text="Pburial value",bg=CF)
    Pburial_label.grid(row=1,column=2,padx=(30,0),pady=(10,0))
    ################# pDRDOC ##############
    pDRDOC = tk.Entry(frameCin,width=20)
    pDRDOC.insert(tk.END, '0.1')
    pDRDOC.grid(row=2,column=3)
    pDRDOC_label = tk.Label(frameCin,text="pDRDOC value",bg=CF)
    pDRDOC_label.grid(row=2,column=2,padx=(30,0),pady=(10,0))
    ################# pIRDOC ##############
    pIRDOC = tk.Entry(frameCin,width=20)
    pIRDOC.insert(tk.END, '0.1')
    pIRDOC.grid(row=3,column=3)
    pIRDOC_label = tk.Label(frameCin,text="pIRDOC value",bg=CF)
    pIRDOC_label.grid(row=3,column=2,padx=(30,0),pady=(10,0))
    ################# Initial pCO2 in ppm ##############
    Ccon0 = tk.Entry(frameCin,width=20)
    Ccon0.insert(tk.END, '412')
    Ccon0.grid(row=5,column=3)
    Ccon0_label = tk.Label(frameCin,text="Initial CO\u2082 pressure (ppm)",bg=CF)
    Ccon0_label.grid(row=5,column=2,padx=(30,0),pady=(10,0))
    ################# Initial Alkalinity  10^-6 eq. per l##############
    #Alkcon0 = tk.Entry(frameCin,width=20)
    #Alkcon0.insert(tk.END, '2350')
    #Alkcon0.grid(row=6,column=3)
    #Alkcon0_label = tk.Label(frameCin,text="Initial [Alkalinity] (10\u207B\u2076 eq/l)",bg=CF)
    #Alkcon0_label.grid(row=6,column=2,padx=(30,0),pady=(10,0))
    ######################## Simulate Carbon budget  ##############
    frameCsim = tk.LabelFrame(simCop,text="Simulate Carbon budget:",font=("Helvetica", 14),bg=CF,padx=10,pady=10)
    frameCsim.grid(row=5,column=0,columnspan=4,padx=20, pady=(10,20))
    ############# concentrations options
    OptionCon = ['Daily','Cumulative','Final']
    ConClicked = tk.StringVar(simCop)
    ConClicked.set(OptionCon[1])
    CarbonCon = tk.OptionMenu(frameCsim,ConClicked,*OptionCon)
    CarbonCon.config(width=20)
    CarbonCon.grid(row=2,column=1,padx=(10,10), pady=(10,0))
    CarbonCon_label = tk.Label(frameCsim,text="Concentrations Options",bg=CF)
    CarbonCon_label.grid(row=2,column=0,padx=(10,0),pady=(10,0))
    ############# simulate button
    RUNcarbon_button = tk.Button(frameCsim, text="Obtain Carbon Variables",
                               height=3, width=30,bg=BColor,font=("Helvetica", 12),
                           comman=lambda: SimulateCarbon(Msimulation,SCclicked.get(),NsimSC.get(),
                                                         float(Pburial.get()),float(pDRDOC.get()),float(pIRDOC.get()),
                                                         float(Ccon0.get()),ConClicked.get()))
    RUNcarbon_button.grid(row=2,column=2,padx=(10,0),pady=(10,0),columnspan=2)
    ############################## save results #########################
    frameCout = tk.LabelFrame(simCop,text="Simulation Outputs:",font=("Helvetica", 14),bg=CF,padx=10,pady=10)
    frameCout.grid(row=10,column=0,columnspan=4,padx=20, pady=(10,20))
    # Button save data
    Save_button = tk.Button(frameCout, text="Save Simulation",height=3,
                            width=30,bg=BColor,font=("Helvetica", 12),
                            comman=lambda: SimCarSave(CarbonOUT))
    Save_button.grid(row=2,column=0,columnspan=2,padx=(15,15),pady=(10,0))                                       
    # Button save data
    Plot_button = tk.Button(frameCout, text="Plot Simulation",height=3,
                            width=30,bg=BColor,font=("Helvetica", 12),
                            comman=lambda: PlotCarbon(CarbonOUT))
    Plot_button.grid(row=2,column=2,columnspan=2,padx=(15,15),pady=(10,0))   
    simCop.mainloop()
    
 
      
SimC_button = tk.Button(frameCO, text="Start Simulation",width=40,height=2,bg=BColor,
                        font=("Helvetica", 10),comman = lambda: computeCarbonVar(Msimulation))
SimC_button.grid(row=8,column=0,columnspan=2,padx=(10,0),pady=10)


frameCout = tk.LabelFrame(frameC,text="Outputs",font=("Helvetica", 14),bg=CF, padx=10,pady=10)
frameCout.grid(row=1,column=0,padx=10, pady=10)


###########################################################
####################        SeaCarb     ###################
###########################################################

framePH0 = tk.LabelFrame(framePH,text="Input Data",font=("Helvetica", 14),bg=CF, padx=10,pady=10)
framePH0.grid(row=1,column=0,padx=10, pady=10)


#### Load carbon concentrations #####
SelectedCarbCon = tk.Label(framePH0,text='',width=20)
SelectedCarbCon.grid(row=2,column=1,pady=(10,0))

def LoadSimCarb():
    SelectedCarbCon = tk.Label(framePH0,text='',width=20)
    SelectedCarbCon.grid(row=2,column=1,pady=(10,0))
    File2open = filedialog.askopenfilename(parent=frameCO,initialdir="/...",
                    title='Please select file of simulations',filetypes = (("xlsx files","*.xlsx"),("all files","*.*")))
    global seacarbIN
    seacarbIN = LoadSeacarbIN(File2open)
    SelectedCarbCon = tk.Label(framePH0,text="File loaded",width=20)
    SelectedCarbCon.grid(row=2,column=1,pady=(10,0))
    

SelectedCarbCon_button = tk.Button(framePH0, text="Load Concentrations Data",width=20,
                           bg=BColor,comman=lambda:LoadSimCarb())
SelectedCarbCon_button.grid(row=2,column=0,padx=(10,0),pady=(10,0))



################### seacabr simulation #########################
SimSeaState = tk.Label(framePH0,text="",width=20)
SimSeaState.grid(row=8,column=1)

def seacarbSIM(Ccon0,seacarbIN):
    SimSeaState = tk.Label(frame2SC,text="Running ... ",width=20)
    SimSeaState.grid(row=1,column=1)
    global seacarbOUT
    start_time_sim = time.process_time()
    seacarbOUT = seacarbPY(Ccon0,seacarbIN)
    end_time_sim = time.process_time()
    Tspent = end_time_sim - start_time_sim 
    ## message end simulation
    if Tspent < 10:
        SimMessage = "Time spent: " +"{:.4f}".format(Tspent)+" s"
    else:
        SimMessage = "Time spent: " +"{:.2f}".format(Tspent)+" s"
    SimState = tk.Label(frame2SC,text=SimMessage,width=20)
    SimState.grid(row=1,column=1)
    

############################## save seacarb results #########################
def SeaCSave(seacarbOUT,seacarbIN):
    Fold2Save = filedialog.askdirectory(parent=frame3,initialdir="/...",title='Please select a directory')
    # Name to save file
    SimName = "SeaCarb"+ seacarbIN[-1,0]+ time.strftime("_%Y%m%d_%H%M%S")
    # Save date
    URLname = Fold2Save + "/" + SimName + ".xlsx"
    SaveSeacarbExcel(seacarbOUT,seacarbIN,URLname)
    # popup showing where data where saved
    messagebox.showinfo("Results Saved", "Simulations have been saved in: \n" + Fold2Save,parent=simSeaC) 
    
##################### plot seacarb results ###########################
def PlotSeaCarb(seacarbOUT,seacarbIN):
    # new window to select the plot options
    PlotSC = tk.Tk()
    PlotSC.title("Seacarb Plot Options")
    #PlotC.geometry("400x400")
    PlotSC.configure(background=CF)
    PlotSC.iconbitmap(MusselICO)
    ### frames to outputs and inputs
    frameSCPout = tk.LabelFrame(PlotSC,text="Plot Outputs:",font=("Helvetica", 14),bg=CF,padx=10,pady=10)
    frameSCPout.grid(row=5,column=0,columnspan=2,padx=20, pady=(10,20))
    frameSCPin = tk.LabelFrame(PlotSC,text="Plot Inputs:",font=("Helvetica", 14),bg=CF,padx=10,pady=10)
    frameSCPin.grid(row=6,column=0,columnspan=2,padx=20, pady=(10,20))  
    # Select option to plot 
    OptionNumb = ['L','T']
    POclicked = tk.StringVar(PlotSC)
    POclicked.set(OptionNumb[0])
    labelPlot = tk.OptionMenu(PlotSC,POclicked,*OptionNumb)
    labelPlot.grid(row=1,column=1,padx=(0,10), pady=(10,0))
    labelPlot_label = tk.Label(PlotSC,text="X-axis Label Options",bg=CF)
    labelPlot_label.grid(row=1,column=0,padx=(10,0),pady=(10,0))
    # Select variable option to plot 
    OptionVar = ['pH','pCO2']
    cVARclicked = tk.StringVar(frameSCPout)
    cVARclicked.set(OptionVar[0])
    varPlot = tk.OptionMenu(frameSCPout,cVARclicked,*OptionVar)
    varPlot.grid(row=2,column=1,padx=(0,10), pady=(10,0))
    varPlot_label = tk.Label(frameSCPout,text="Select Output",bg=CF)
    varPlot_label.grid(row=2,column=0,padx=(10,0),pady=(10,0))
    # button to plot the selected realizations
    PlotC_button = tk.Button(frameSCPout, text="Plot Selection",
                               height=2, width=30,bg=BColor,font=("Helvetica", 12),
                           comman=lambda: plotSeacarbOut(seacarbOUT,seacarbIN,POclicked.get(),cVARclicked.get()))
    PlotC_button.grid(row=11,column=0,padx=(20,20),pady=(10,0),columnspan=2)
    # Select input to plot 
    OptionVarCIn = ['[C]','[Alk]','SST','sal','all']
    VARCinclicked = tk.StringVar(frameSCPin)
    VARCinclicked.set(OptionVarCIn[0])
    varCInPlot = tk.OptionMenu(frameSCPin,VARCinclicked,*OptionVarCIn)
    varCInPlot.grid(row=9,column=1,padx=(0,10), pady=(10,0))
    varCInPlot_label = tk.Label(frameSCPin,text="Select Input",bg=CF)
    varCInPlot_label.grid(row=9,column=0,padx=(10,0),pady=(10,0))
    # button to plot the selected realizations inputs
    PlotCin_button = tk.Button(frameSCPin, text="Plot Selected Inputs",
                               height=2, width=30,bg=BColor,font=("Helvetica", 12),
                           comman=lambda: PlotSeacarbIn(seacarbIN,POclicked.get(),VARCinclicked.get()))
    PlotCin_button.grid(row=11,column=0,padx=(20,20),pady=(10,0),columnspan=2)
    # close all figures
    closePlotSIN_button = tk.Button(PlotSC, text="Close all Figures",
                               height=2, width=30,bg=BColor,font=("Helvetica", 12),
                           comman = ClosePlotSIN)
    closePlotSIN_button.grid(row=12,column=0,padx=(20,20),pady=(10,20),columnspan=2)
    PlotSC.mainloop()
    
    
#################### seacarb module #####################
def computeSeaC(seacarbIN):
    # new window to select the simulation options
    global simSeaC,frame2SC
    simSeaC = tk.Tk()
    simSeaC.title("Seacarb module")
    #simCop.geometry("400x400")
    simSeaC.configure(background=CF)
    simSeaC.iconbitmap(MusselICO)
    frame2SC = tk.LabelFrame(simSeaC,text="Seacarb Simulation:",font=("Helvetica", 14),bg=CF,padx=10,pady=10)
    frame2SC.grid(row=5,column=0,columnspan=2,padx=10, pady=10)
    frame3SC = tk.LabelFrame(simSeaC,text="Seacarb Outputs:",font=("Helvetica", 14),bg=CF,padx=10,pady=10)
    frame3SC.grid(row=6,column=0,columnspan=2,padx=10, pady=(10,20))
    ################# Initial pCO2 in ppm ##############
    Ccon0 = tk.Entry(simSeaC,width=20)
    Ccon0.insert(tk.END, '412')
    Ccon0.grid(row=3,column=1)
    Ccon0_label = tk.Label(simSeaC,text="Initial CO\u2082 pressure (ppm)",bg=CF)
    Ccon0_label.grid(row=3,column=0,padx=(10,0),pady=(10,0))
    ################# Initial Alkalinity  10^-6 eq. per l##############
    #Alkcon0 = tk.Entry(simSeaC,width=20)
    #Alkcon0.insert(tk.END, '2350')
    #Alkcon0.grid(row=4,column=1)
    #Alkcon0_label = tk.Label(simSeaC,text="Initial [Alkalinity] (10\u207B\u2076 eq/l)",bg=CF)
    #Alkcon0_label.grid(row=4,column=0,padx=(10,0),pady=(10,0))
    ################### simulation part #####################################  
    SimSeaState = tk.Label(frame2SC,text="",width=20)
    SimSeaState.grid(row=1,column=1)
    SimPH_button = tk.Button(frame2SC, text="Start Simulation",width=20,bg=BColor,
                         comman=lambda:seacarbSIM(float(Ccon0.get()),seacarbIN))
    SimPH_button.grid(row=1,column=0,padx=(10,0),pady=10)
    Save_button = tk.Button(frame3SC, text="Save Simulation",width=20,bg=BColor,comman=lambda: SeaCSave(seacarbOUT,seacarbIN))
    Save_button.grid(row=11,column=0,padx=(10,0))
    Plot_button = tk.Button(frame3SC, text="Plot Simulation",width=20,bg=BColor,
                            comman=lambda: PlotSeaCarb(seacarbOUT,seacarbIN))
    Plot_button.grid(row=11,column=1,padx=(10,0))
    simSeaC.mainloop()  
 
    

SimSeaC_button = tk.Button(framePH0, text="Start Simulation",width=40,height=2,bg=BColor,
                        font=("Helvetica", 10),comman = lambda: computeSeaC(seacarbIN))
SimSeaC_button.grid(row=8,column=0,columnspan=2,padx=(10,0),pady=10)


root.mainloop()





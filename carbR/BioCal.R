input <- read.csv(file='./carbR/input.csv')
param <- read.csv(file='./carbR/param.csv')

input$Alk0 = 67*input$Sal*1e-6
input$pCO20 = param$pCO2_0
input$flag = 24

library(seacarb)
psi0=psi(flag=input$flag, var1=input$pCO20, var2=input$Alk0, T=input$SST, S=input$Sal)

output <- data.frame(psi0) 
write.csv(output,'./carbR/output.csv')
input <- read.csv(file='./carbR/input.csv')
param <- read.csv(file='./carbR/param.csv')

library(seacarb)
input$flag0 = 24
input$pCO2_0 = param$pCO2_0
input$Alk0 = 67*input$Sal*1e-6

carb0=carb(flag=input$flag0, var1=input$pCO2_0,var2=input$Alk0,T=input$SST,S=input$Sal)

input$flag = 15
input$DIC0 = carb0$DIC
 
out=carb(flag=input$flag, var1=input$Alk+input$Alk0, var2=input$DIC+input$DIC0, S=input$Sal, T=input$SST) 
out0=carb(flag=input$flag, var1=input$Alk0, var2=input$DIC0, S=input$Sal, T=input$SST) 

pH = out$pH
pCO2 = out$pCO2[1:length(input$Alk)]
pH_0 = out0$pH
pCO2_0 = out0$pCO2[1:length(input$Alk)]
output <- data.frame(pH,pCO2,pH_0,pCO2_0) 
write.csv(output,'./carbR/output.csv')


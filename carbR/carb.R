input <- read.csv(file='./carbR/input.csv')
param <- read.csv(file='./carbR/param.csv')

library(seacarb)
carb0=carb(flag=24, var1=param$pCO2_0,var2=param$Alk_0*1e-6,T=input$SST[1],S=input$Sal[1])

input$flag = 15
input$Alk0 = input$Alk[1]
input$DIC0 = carb0$DIC
 
out=carb(flag=input$flag, var1=input$Alk, var2=input$DIC, S=input$Sal, T=input$SST) 
out0=carb(flag=input$flag, var1=input$Alk0, var2=input$DIC0, S=input$Sal, T=input$SST) 

pH = out$pH
pCO2 = out$pCO2[1:length(input$Alk)]
pH_0 = out0$pH
pCO2_0 = out0$pCO2[1:length(input$Alk)]
output <- data.frame(pH,pCO2,pH_0,pCO2_0) 
write.csv(output,'./carbR/output.csv')


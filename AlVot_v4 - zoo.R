setwd("~/Maestria/Patrones/Proyecto")

library(mltools)
library(caret)
library(rje)

#///////////////////Tratamiento/////////////////////////////////////////////////////

# <- read.csv("~/Maestria/Patrones/Proyecto/Datasets/crx.data", header=FALSE, na.strings="?")
# <- read.csv("~/Maestria/Patrones/Proyecto/Datasets/zoo.data", header=FALSE)
# <- read.csv("~/Maestria/Patrones/Proyecto/Datasets/heart.csv")
data <- read.csv("~/Maestria/Patrones/Proyecto/Datasets/zoo.data", header=FALSE)

set.seed(31) #73

clase <- data[,18]
data <- data[,c(-1,-18)]
data <- data[,-13] #Para solo categoricos zoo

#Con data frame/////////////////////

#data_n <- data.frame(data)


#for(i in 1:5)
#{
#  if(i == 5)
#    i <- 9
#  data_n[,i] <- as.character(data[,i])
#}

#//////////////////////////////////

#Para clases de heart:
#for (i in 1:nrow(data))
#{
#  if(data[i,ncol(data)] == 0)
#    data[i,ncol(data)] <- 1
#  else
#    data[i,ncol(data)] <- 2
#}

head(data)
#/////////////////////Parametros////////////////////////////////////////////////////

clases_vect <- c(1,2,3,4,5,6,7) #Para zoo: 1 - 7; para heart 1, 2

#////////////////////////////Parametros constantes//////////////////////////////////

cv <- 10
porcent <- 0.95
umb <- 0.27

cp_i <- 5 #numero de atributos inferior
cp_s <- 5 #numero de atributos superior

n_clases <- length(clases_vect)

#////////////////////////////////////Normalizacion/////////////////////////////////

#data <- as.data.frame(data)
#norma <- function(x) {(x - min(x))/(max(x) - min(x))}
#data_n[c(5,6,7,8,10)] <- lapply(data[c(5,6,7,8,10)], norma) #Normalizar numericos: data[c(1,4,5,8,10,12)] <- lapply(data[c(1,4,5,8,10)], norma)

for(i in 1:nrow(data))
{
  for (j in 1:5) #5:10
  {
    #if (j == 5 || j == 7 || j == 8 || j == 10)
    #{
    maxi <- max(data[,j])
    mini <- min(data[,j])
    data[i,j] <- (data[i,j] - mini)/(maxi - mini)
    #}
  }
}

#summary(data)
#clase_col <- clase


rm(maxi)
rm(mini)
#////////////////////////////////////One hot encoding///////////////////////////////

data <- as.data.frame(data)

columnas_categ <- c(1,2,3,4,5,6,7,8,9,10,11,12,13,14,15) # Para zoo: 13, para heart: 2, 3, 6, 7, 9, 11, 12, 13
data[columnas_categ] <- lapply(data[columnas_categ], factor)
dmy <- dummyVars(" ~ .", data = data)
data <- data.frame(predict(dmy, newdata = data))

data <- as.matrix(data)

rm(dmy)
rm(columnas_categ)

#//////////////////Crossvalidation//////////////////////////////////////////////////

cross_valid_res <- matrix(nrow = cv, ncol = 1)
matrices_conf <- list()

train_size <- round(porcent*nrow(data))
test_size <- round((1-porcent)*nrow(data))
obj_n_size <- nrow(data) - round(porcent*nrow(data))

for (m in 1:cv)
{
  rand <- sample(1:nrow(data), round(porcent*nrow(data)))
  
  data_train <- data[rand,] #data_norm_oh para one hot; data para usarlos sin trans
  data_test <- data[-rand,] #data_norm_oh
  
  data_target_train <- clase[rand] #General: data[rand, clase_col]
  data_target_test <- clase[-rand] #General: data[-rand, clase_col]
  
  #//////////////////Sistema de conjuntos de apoyo/////////////////////////////////////
  
  ps <- powerSet(1:ncol(data), m = cp_s) #atributos! general: clase_col
  sis_c_a <- list()
  obj_n <- list()
  
  i <- 1
  j <- 1
  
  while (i<=length(ps))
  {
    if(length(ps[[i]]) >= cp_i)#>=
    {
      sis_c_a[[j]] <- data_train[,c(ps[[i]])]
      obj_n[[j]] <- data_test[,c(ps[[i]])]
      j <- j + 1
    }
    i <- i + 1
  }
  length(obj_n)
  #for para toda la prueba!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
  
  clasificacion <- matrix(0, nrow = obj_n_size, ncol = 1)
  
  #Funcion de semejanza
  
  #Para heart: (abs(sis_c_a[[j]][[l]][i] - obj_n[[j]][[l]][k]) <= umb) 
  #Para zoo: (sis_c_a[[j]][[l]][i] == obj_n[[j]][[l]][k])
  
  for (k in 1:obj_n_size) #obj_n_size
  {
    semej <- matrix(nrow = train_size, ncol = length(sis_c_a)) #MA
    
    for (i in 1:train_size)
    {
      for (j in 1:length(sis_c_a))
      {
        s <- 0
        for(l in 1:as.numeric(length(sis_c_a[[j]])/train_size))
        {
          if (  typeof(sis_c_a[[j]][[i,l]]) == "character" && typeof(obj_n[[j]][[k,l]]) == "character")
          {
            if ((sis_c_a[[j]][[i,l]] == obj_n[[j]][[k,l]])) #Funcion de semejanza ATENCION
              s <- s + 1
          }
          else
          {
            if ((abs(sis_c_a[[j]][[i,l]] - obj_n[[j]][[k,l]]) <= umb)) 
              s <- s + 1
          }
          
        }
        if(s == (length(obj_n[[j]])/test_size))
          semej[i,j] <- 1 #pesos Función de evaluación parcial!!
        else
          semej[i,j] <- 0 #pesos
      }
    }
    #}  
    #Evaluacion por clase dado un conjunto de apoyo fijo:
    suma <- matrix(0, nrow = n_clases, ncol = 1)
    
    semej_red <- matrix(nrow = n_clases, ncol = length(sis_c_a))
    
    for (i in 1:length(sis_c_a))
    {
      for (j in 1:train_size)
      {
        for (l in 1:n_clases)
        {
          if(data_target_train[j] == clases_vect[l])
          {
            suma[l] <- suma[l] + semej[j,i]
          }
        }
        
      }
      
      for (j in 1:n_clases)
      {
        semej_red[j,i] <- suma[j]/n_clases
      }
    }
    
    #Evaluación por clase para todo el sistema de conjuntos de apoyo
    
    clases <- matrix(0, nrow = n_clases, ncol = 1)
    
    for (i in 1:length(sis_c_a))
    {
      for (j in 1:n_clases)
      {
        clases[j,1] <- clases[j,1] + semej_red[j,i]
      }
      
    }
    
    
    #Regla de decision
    maxi <- 0
    
    for (i in 1:n_clases)
    {
      if (clases[i,1] > maxi)
      {
        maxi <- clases[i,1]
        c <- i # menos uno para heart desease!! ATENCION
      }
      else-if (clases[i,1] == maxi)
      {
        c <- 0
        #break
      }
      else
        c
    }
    
    clasificacion[k] <- c
    #k<-10
  }
  
  #Validacion
  
  conf_mat <- matrix(0, n_clases, n_clases)
  #i<-19
  for(i in 1:obj_n_size)#obj_n_size
  {
    #Sumar uno en la matriz de confucion si las clases empiezan con 0
    if(clasificacion[i] == data_target_test[i])
      conf_mat[clasificacion[i],clasificacion[i]] <- conf_mat[clasificacion[i],clasificacion[i]] + 1
    else-if(clasificacion[i] != data_target_test[i])
      conf_mat[clasificacion[i],data_target_test[i]] <- conf_mat[clasificacion[i],data_target_test[i]] + 1
    else
      i
  }
  
  conf_mat
  
  accuracy <- function(x){sum(diag(x)/(sum(rowSums(x))))*100}
  #accuracy <- function(x){sum(diag(x)/obj_n_size)*100} #con abstenciones
  
  accuracy(conf_mat)
  
  matrices_conf[[m]] <- conf_mat
  cross_valid_res[m] <- accuracy(conf_mat)
  print(m)
}

suma <- 0

for (i in 1:cv)
{
  suma <- suma + cross_valid_res[i]
}

media <- suma/cv
install.packages("tidyverse")
install.packages("stats")
install.packages("factoextra")
install.packages("raster")
install.packages("Rcpp")
install.packages("terra", type = 'source')
install.packages("sp", type = 'source')
install.packages("rgdal")
install.packages("raster")
install.packages("rgeos")
install.packages("dplyr")
install.packages("reshape2")
install.packages("ggplot2")
install.packages("randomForest")
install.packages("e1071")
install.packages("caret")
install.packages("caTools")
install.packages("prettymapr")
install.packages("jpeg")
install.packages("mapplots")
install.packages("sf")
install.packages("e1071")
install.packages("caret")
library(e1071)
library(caret)
library(sf)
library(jpeg)
library(mapplots)
require(prettymapr)
require(caTools)
require(randomForest)
require(e1071)
require(caret)
library(reshape2)
library(dplyr) ### trabalhar com tabela
library(rgeos)## trabalhar com arquivo shap
library(tidyverse) # Para manipular os dados
library(stats) # Para PCA
library(factoextra) # Para criar alguns gráficos
library(raster) # Para trabalhar com raster
library(Rcpp)
library(rgdal)
library(sp)
library(terra)
library(ggplot2) ## trabalhar com graficos

##########################################################################################################################################################################################################################################

# Carregar imagens

img = stack("dados/Recorte/B2_recorte.tif", 
               "dados/Recorte/B3_recorte.tif",
               "dados/Recorte/B4_recorte.tif", 
               "dados/Recorte/B5_recorte.tif")

names(img) = c("B2","B3", "B4", "B5")

print(img)

plotRGB(img, r = 4, g = 3, b = 2,axes = T, stretch = 'lin', main = "Landsat 2 cor verdadeira")

writeRaster(x=img, filename= "dados/Recorte/Imagem_original_all.tif")

### Carregando dados amostrais da cobertura do solo

amostra = readOGR("dados/Recorte/Amostra/Dados amostrais.shp")
View(data.frame(amostra))

### Juntar as feições de cada classe da amostra

unidos_shp = gUnaryUnion(spgeom = amostra, id = amostra$Classes, checkValidity = 2L)
unidos_shp


## extrair as amostras unidas e criar um date.frame para a utilização do treinamento posteriormente

atributo = extract(x = img, y = unidos_shp)

## criar um data.frame para cada classe utilizando o atributo criado

names(unidos_shp)

agua = data.frame(Classe = "Agua", atributo[1])
area_urbana = data.frame(Classe = "Area urbana", atributo[2])
floresta = data.frame(Classe = "Floresta", atributo[3])
solo = data.frame(Classe = "Solo exposto", atributo[4])
veg_rasteira = data.frame(Classe = "Vegetacao rasteira", atributo[5])

### juntar todas a data.frame em uma unica

amostras_final = rbind(agua, area_urbana,floresta,solo, veg_rasteira)

write.csv(amostras_final, "dados/Recorte/Amostra/amostra_final.csv")

# Contar a quantidade de dados por categoria
contagem = table(amostra$Classes)

# Mostrar a contagem
print(contagem)

## calcular o expectro de reflectancia de cada classe

##agrupar classe 

agrupado = group_by(amostras_final, Classe)

print(agrupado)

### media de cada classes

media_ref = summarise_each(agrupado, mean)

print(media_ref)

### calculo do especto

refs = t(media_ref [ ,2:5])
cores = c ("blue","pink", "green","brown", "orange")
comp_onda = c(490, 560, 660, 705)

matplot(x= comp_onda, y=refs, type = "l", lwd = 2, lty = 1, xlab = "Comprimento de ondas(nm)", ylab = "Reflectancia x 10000", col = cores, ylim = c(0,40000))

legend('top', legend = media_ref$Classe, col = cores, lty = 1, ncol = 2, lwd = 2) 


### calculo da matrix de correlação entre as bandas

# Extrair os valores das bandas e criar uma matriz de dados
data_matrix = data.frame (amostras_final$B2, amostras_final$B3, amostras_final$B4, amostras_final$B5)

names(data_matrix) = c("B2","B3", "B4", "B5")

# Calcular a matriz de correlação
cor_matrix = cor(data_matrix)

# Imprimir a matriz de correlação
print(cor_matrix)

# Carregar os arquivos de treino e validação

# Checar a class (strucutre dos meus dados não pode ser texto)

str(amostras_final)

amostras_final$Classe = as.factor(amostras_final$Classe) 

#Separação dos dados em treinamento e validação

set.seed(1234) # mantem o mesmo resultado
amostras_treino = sample.split(amostras_final$Classe, SplitRatio = 0.7)
(amostras_treino)

##Separar em dados de treino e teste (criar o dataframe de treino e teste)

train = amostras_final[amostras_treino,]
valid = amostras_final[amostras_treino == F, ]

write.csv(train, "dados/Recorte/Amostra/Amostras_treino.csv")
write.csv(valid, "dados/Recorte/Amostra/Amostras_teste.csv")

### Classificação pelo RandomForest

train$Classe = as.factor(train$Classe)
valid$Classe = as.factor(valid$Classe)

set.seed(1234) # para gerar arvore de decisão aleatoria com mesmo resultado 

RF = randomForest(Classe~., data = train, ntree = 100, mtry = 3, importance = T) # classe relacionada com todas as bandas

varImpPlot(RF) # O gráfico Mean Decrease Accuracy expressa quanta precisão o modelo perde ao excluir cada variável. Quanto mais a precisão é prejudicada, mais importante é a variável para o sucesso da classificação. As variáveis são apresentadas em ordem decrescente de importância. A diminuição média no coeficiente de Gini é uma medida de como cada variável contribui para a homogeneidade dos nós e folhas na floresta aleatória resultante. Quanto maior o valor da diminuição média da precisão ou da diminuição média do escore de Gini, maior será a importância da variável no modelo.

importance(RF)

# Support Vector Machines - SVM 

set.seed(1234) 
SVM = svm(Classe~., kernel = 'polynomial', data = train)


## Validação dos modelos 

pred.RF = predict(RF, valid)
pred.SVM = predict(SVM, valid)

## Criacao da matriz de confusão 

CM.RF = confusionMatrix(data = pred.RF, reference = valid$Classe)
CM.SVM = confusionMatrix(data = pred.SVM, reference = valid$Classe)

print(CM.RF)
print(CM.SVM)

#Salvando os modelos

saveRDS(object = RF, file = "dados/Recorte/Classsificacao_rf.rds")
saveRDS(object = SVM, file = "dados/Recorte/Classsificacao_SVM.rds")

##########################################################################################################################################################################

# Usando PCA

# Calcular a Matriz de covariância

Bandas = data.frame (amostras_final$B2, amostras_final$B3, amostras_final$B4, amostras_final$B5)
nomes_colunas = c("B2", "B3", "B4", "B5") 
colnames(Bandas) =  nomes_colunas # alterar o nome das colunas

print("Bandas:")
print(Bandas)

Bandas_centralizados = scale(Bandas, center = TRUE, scale = FALSE) # a média de cada variável foi subtraída

cov_matrix = cov(Bandas_centralizados) #calcula a matriz de covariância de um conjunto de dados. A matriz de covariância é uma medida estatística que descreve como as variáveis em um conjunto de dados mudam juntas
print("Matriz de Covariância:")
print(cov_matrix)

# Calcular autovalores e autovetores

eigen_resultados = eigen(cov_matrix) # Os autovetores são os vetores próprios que definem as direções dos componentes principais.
autovalores = eigen_resultados$values # Os autovalores representam a quantidade de variância explicada por cada componente principal
autovetores = eigen_resultados$vectors # Os autovetores são os vetores próprios que definem as direções dos componentes principais.

# Ordenar autovalores e autovetores

ordem = order(autovalores, decreasing = TRUE)
autovalores = autovalores[ordem]
autovetores = autovetores[, ordem]

print("Autovalores:")
print(autovalores)

print("Autovetores:")
print(autovetores)

# Calcular os componentes principais

componentes_principais = autovetores[, 1:4]

# Transformar os dados

dados_transformados = Bandas_centralizados %*% componentes_principais

# Calcular a porcentagem da variância retida para cada vetor

percentage_retained_B2 = (autovalores[1] / sum(autovalores)) * 100
percentage_retained_B3 = (autovalores[2] / sum(autovalores)) * 100
percentage_retained_B4 = (autovalores[3] / sum(autovalores)) * 100
percentage_retained_B5 = (autovalores[4] / sum(autovalores)) * 100

# Tabela Autovetores e Porcentagem de Variância Explicada para as Componentes Principais

Autovetores = c("CP1", "CP2", "CP3", "CP4")
B2 = c(autovetores[, 1])
B3 = c(autovetores[, 2])
B4 = c(autovetores[, 3])
B5 = c(autovetores[, 4])
Variancia_Explicada = c(percentage_retained_B2, percentage_retained_B3,
                        percentage_retained_B4, percentage_retained_B5)  

Tabela_resumo = data.frame(Autovetores, B2, B3, B4, B5, Variancia_Explicada )
print(Tabela_resumo)

# Criar o gráfico biplot 
plot(1, type = "n", xlab = "", ylab = "", xlim = c(-1, 1), ylim = c(-1, 1))

for (i in seq_along(autovalores[1:4])) {
  arrows(0, 0, autovetores[1, i], autovetores[2, i], angle = 20, length = 0.1, col = "red")
} # Adicionar vetores para as variáveis originais

text(autovetores[1, ], autovetores[2, ], labels = colnames(dados_transformados ), pos = 3, cex = 0.7, col = "blue") # Adicionar rótulos às observações


#Separação dos dados em treinamento e validação

dados_pca = data.frame(amostras_final$Classe, dados_transformados)
nomes_colunas = c("Classe", "B2", "B3", "B4", "B5") 
colnames(dados_pca) =  nomes_colunas # alterar o nome das colunas
dados_pca = dados_pca

str(dados_pca) #verificando se todos os dados são numeros

set.seed(1234) # mantem o mesmo resultado
dados_pca_treino = sample.split(dados_pca$Classe, SplitRatio = 0.7)


##Separar em dados de treino e teste (criar o dataframe de treino e teste)

train_pca = dados_pca[dados_pca_treino,]
valid_pca = dados_pca[dados_pca_treino == F, ]

write.csv(train, "dados/Recorte/Amostra/dados_pca_treino1.csv")
write.csv(valid, "dados/Recorte/Amostra/dados_pca_teste1.csv")

### Classificação pelo RandomForest

train_pca$Classe = as.factor(train_pca$Classe)
valid_pca$Classe = as.factor(valid_pca$Classe)

set.seed(1234) # para gerar arvore de decisão aleatoria com mesmo resultado 

RF_pca = randomForest(Classe~., data = train_pca, ntree = 100, mtry = 3, importance = T) # classe relacionada com todas as bandas

varImpPlot(RF_pca) # O gráfico Mean Decrease Accuracy expressa quanta precisão o modelo perde ao excluir cada variável. Quanto mais a precisão é prejudicada, mais importante é a variável para o sucesso da classificação. As variáveis são apresentadas em ordem decrescente de importância. A diminuição média no coeficiente de Gini é uma medida de como cada variável contribui para a homogeneidade dos nós e folhas na floresta aleatória resultante. Quanto maior o valor da diminuição média da precisão ou da diminuição média do escore de Gini, maior será a importância da variável no modelo.

importance(RF_pca)

# Support Vector Machines - SVM 

set.seed(1234) 
SVM_pca = svm(Classe~., kernel = 'polynomial', data = train_pca)


## Validação dos modelos 

pred.RF_pca = predict(RF_pca, valid_pca)
pred.SVM_pca = predict(SVM_pca, valid_pca)

## Criacao da matriz de confusão 

CM.RF_pca = confusionMatrix(data = pred.RF_pca, reference = valid_pca$Classe)
CM.SVM_pca = confusionMatrix(data = pred.SVM_pca, reference = valid_pca$Classe)

print(CM.RF_pca)
print(CM.SVM_pca)

print(CM.RF)
print(CM.SVM)

##########################################################################################################################################################################

## predição para o raster

RF.raster = predict(img,RF)
SVM.raster = predict(img,SVM)


RF_pca.raster = predict(img,RF_pca)
SVM_pca.raster = predict(img,SVM_pca)


# Plotagem colorida 
#plotagem da imagem RF

cores = c ("blue","pink", "green","brown", "orange")
classes = c ("Agua", "Area urbana", "Floresta", "Solo exposto", "Vegetacao rasteira")

jpeg(filename = "dados/Recorte/classificacao1.jpeg", width = 15, height = 15, res = 200, units = 'in')

plot(RF.raster, legend = FALSE, col = cores, main = "Classificação RF",
     cex.axis = 1.5, cex.main = 1.5)

legend('topleft', legend = classes, fill = cores, border = FALSE, cex =2)

addnortharrow(cols = c("black", 'black'), scale = 0.755)

dev.off()

#plotagem da imagem SVM

cores = c ("blue","pink", "green","brown", "orange")
classes = c ("Agua", "Area urbana", "Floresta", "Solo exposto", "Vegetacao rasteira")

jpeg(filename = "dados/Recorte/classificacao_SVM.jpeg", width = 15, height = 15, res = 200, units = 'in')

plot(SVM.raster, legend = FALSE, col = cores, main = "Classificação SVM",
     cex.axis = 1.5, cex.main = 1.5)

legend('topleft', legend = classes, fill = cores, border = FALSE, cex =2)

addnortharrow(cols = c("black", 'black'), scale = 0.755)

dev.off()

#plotagem da imagem RF_pca

cores = c ("blue","pink", "green","brown", "orange")
classes = c ("Agua", "Area urbana", "Floresta", "Solo exposto", "Vegetacao rasteira")

jpeg(filename = "dados/Recorte/classificacao_RFpca1.jpeg", width = 15, height = 15, res = 200, units = 'in')

plot(RF_pca.raster, legend = FALSE, col = cores, main = "Classificação RF_pca",
     cex.axis = 1.5, cex.main = 1.5)

legend('topleft', legend = classes, fill = cores, border = FALSE, cex =2)

addnortharrow(cols = c("black", 'black'), scale = 0.755)

dev.off()

#plotagem da imagem SVM_pca

cores = c ("blue","pink", "green","brown", "orange")
classes = c ("Agua", "Area urbana", "Floresta", "Solo exposto", "Vegetacao rasteira")

jpeg(filename = "dados/Recorte/classificacao_SVM_pca.jpeg", width = 15, height = 15, res = 200, units = 'in')

plot(SVM_pca.raster, legend = FALSE, col = cores, main = "Classificação SVM_pca",
     cex.axis = 1.5, cex.main = 1.5)

legend('topleft', legend = classes, fill = cores, border = FALSE, cex =2)

addnortharrow(cols = c("black", 'black'), scale = 0.755)

dev.off()

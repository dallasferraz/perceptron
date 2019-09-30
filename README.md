# perceptron
A simple single-layer perceptron constructed in R.

## Example

In this section, I will deploy and explain how a single-layer perceptron algorithm works. It is constructed for binary classification of items in inventory. The fictional company of the example uses several types of raw materials, with different acquisition costs and different levels of item usage. Briefly, the more a raw material is used and the more costly it is, the more likely this item is to be labelled as "prime". On the other hand, if the item is not costly nor used frequently in production, the item is labelled as "standard".

This distinction is loosely based in the Pareto or ABC curve, which classifies items according to its relevance degree, taking into consideration the value the item has to the organization. However, the proposed model for this example is different because its classification considers more than one factor at once (both cost and item usage), and classifies items into two different groups, rather than in three groups (A, B and C).

It is also worth mentioning that classifying the items in stock is useful for companies due to the fact that it helps managers decide which stock control policy they will adopt for each item, and also where the items should be placed physically in the warehouse. For instance, "prime" items probably should be placed near the entrance of the plant, since it avoids unnecessary movement in the warehouse (the personnel responsible for retrieving the raw material would not have to go all the way to the rear in order to get an item with high itemUsage level). Consequently, less costly items and items used more sparingly should be placed in a more distant position in the warehouse.

For this example, the fictional company holds data history of **250 different items** and information regarding their level of importance for the logistics manager, according to subjective classification. In general, prime item's *cost per pallet* range from **\$650** up to **\$1000**, whereas standard items range from **\$100** to **\$600**. Regarding item usage, prime items are used in a rate not lower than **23 pallets/trimester** and not higher than **35 pallets/trimester**. On the other hand, standard items are never used in a rate below **4 pallets/trimester** and above **22 pallets/trimester*.

**In this scenario, the CEO of the company would like to classify new raw materials according to the neural network, by using previous information available.**

## Single-layer perceptron

The code for the proposed neural network is part of a family of solutions known as Machine Learning. This approach uses previous data and weights in order to start learning how to classify new data, a process called *training*. During training, an error function helps the algorithm adjust weights, which in turn process input data and outputs a class. Several approaches in literature describe the pros and cons of every type of error functions available, but it suffices to say that the data are fed to the system, which outputs a value corresponding to a certain class. If the difference between the classification and the real classification is larger than a threshold value, the error unction tweaks the weights and restarts the process, until there is no more training data available, or convergence is reached.

After training phase, *testing* phase comes into play and some of the initial data that was not used during training is fed to the system, so it is possible for the programmer to assess system efficiency. This step consists of inputting this piece of history data in the already trained neural network and then counting how many entries the system misclassifies. After this final step, the neural network is either validated and can finally perform real tasks by classifying new data or it is then scratched and the whole system has to be reconstructed.

## Constructing the algorithm in R

Although there are several libraries containing neural networks, this project is dedicated to show how the particularities of a single-layer perceptron work in every step of the code. This work is based on the original Java project by [Sadawi (2014)](https://github.com/nsadawi/perceptron), but it displays characteristics not found in the original material.

Initially, I start by creating an activation function, which will return 0 whenever input data is below the threshold `theta` and 1 otherwise:

```{r}
activationFunction <- function (theta, weight, cost, itemUsage){
  sumCells <- cost*weight[1] + itemUsage*weight[2] + weight[3]
  
  if (sumCells>= theta){return(1)}
  else {return(0)}
}
```

Other variables used as parameters in `activationFunction()` are `weight`, `cost` and `itemUsage`. Variable `weight` is responsible for carrying the weight vector into the function, whereas `cost` represents acquisition cost for the pallet of a given raw material. Naturally, `itemUsage` represents item usage for the same entry.

Next, I created the function `perceptron()`:

```{r}
perceptron <- function(maxIter, learningRate, numInstances, theta){
  
# Como pode ser visto nos argumentos da função, maxIter diz respeito ao número máximo de tentativas (ou seja, iterações) que o algoritmo pode tentar, antes de declarar uma mensagem de que não convergiu para o usuário. Este cenário é possível, quando se trata de um dataset contendo elementos difíceis ou impossíveis de serem separados em classes diferentes, pelo algoritmo. A variável learningRate é responsável por informar a taxa percentual de mudança dos pesos a cada iteração, enquanto que numInstances contém informações sobre a quantidade de dados disponíveis para as etapas de treino e teste. Por fim, esta função também faz uso do parâmetro theta
  
#--------------------------------------
  
#No mundo real, dados a respeito das matérias-primas estariam armazenados em bancos de dados, como MySQL, MS Access, Excel etc. Para este exemplo, utilizaremos dados gerados in silico, ou seja, aleatoriamente, porém simulando serem dados reais. Serão necessários 4 vetores: 2 para treino e 2 para teste, e 2 para cada atributo (no caso, custo e giro de estoque). as dimensões do vetor de treino será fornecida na execução do algoritmo, enquanto que o vetor de teste foi escolhido como sendo de 100 itens diferentes
  
  costTraining <- vector(length = numInstances)
  costTest <- vector(length = 100)
  
  itemUsageTraining <- vector(length = numInstances)
  itemUsageTest <- vector(length = 100)
  
#--------------------------------------
  
#Juntamente com os vetores contendo os valores das características de treino e teste, também é necessário que existam vetores contendo as classificações como "prime" (0) e "standard" (1), com as mesmas dimensões dos vetores de treino e teste
  
  classificationTraining <- vector(length = numInstances)
  classificationTest <- vector(length = 100)
  
#--------------------------------------
  
#Um outro vetor necessário armazenará os valores da etapa de teste
  
  testOutput <- vector(length = 100)
  
#--------------------------------------
  
#Já o vetor para guardar os pesos das conexões das entradas com o neurônio também tem que contemplar o bias como um peso gerado aleatoriamente
  
  weight <- vector(length = 3)
  
  weight[1]<-runif(1,min=0,max=1) #w1
  weight[2]<-runif(1,min=0,max=1) #w2
  weight[3]<-runif(1,min=0,max=1) #bias
  
#--------------------------------------
  
#As variáveis de erro, necessárias nas etapas intermediárias, bem como as de contagem de iteração e de eficiência devem ser declaradas com seus respectivos valores iniciais para a primeira iteração
  
  localError <- NA
  globalError <- NA
  right <- 0
  wrong <- 0
  
  iteration <- 1
  
#--------------------------------------
  
#Após a criação dos vetores que irão conter os dados de treino, é preciso que se preencham os valores adequados. Para a primeira metade dos componentes gerados in silico, forçaremos que eles sejam classificados como "prime", e exatamente por essa razão, é necessário que seus valores nos atributos correspondam à descrição dada pela empresa, em relação ao custo do pallet e ao giro de estoque. O mesmo também acontece para os produtos "standard", apenas que a classificação será 1 para todos os itens que compõem a segunda metade do vetor que guarda as informações
  
  for (i in 1:(numInstances/2)){
    costTraining[i] <- runif(1, min=650, max=1000)
    itemUsageTraining[i] <- runif(1, min=23, max=35)
    classificationTraining[i] <- 0
  }
  
  for (i in (1+numInstances/2):numInstances){
    costTraining[i] <- runif(1, min=100, max=600)
    itemUsageTraining[i] <- runif(1, min=4, max=22)
    classificationTraining[i] <- 1
  }
  
#--------------------------------------
  
#A etapa de normalização dos vetores ocorre com uma simples conversão de valores, uma vez que os atributos apresentam-se em grandezas muito diversas. Este sistema normaliza os dados ao limitá-los proporcionalmente no intervalo continuo entre -1 e 1.

  for (i in 1:numInstances){
    costTraining[i]<-((1000/2)-costTraining[i])/(1000/2)
    itemUsageTraining[i]<-((35/2)-itemUsageTraining[i])/(35/2)
  }
  
#--------------------------------------
  
#As iterações de fato ocorrem seguindo passos imperativos: primeiramente, o erro global é setado como valendo 0. A finalização desse laço representa o fim da etapa de treinos, e o algoritmo pode então passar para a etapa de testes, com os 100 pontos restantes
  
  repeat{
    globalError<-0
    
    for (p in 1:(numInstances)){
      #armazenando na variável output, para cada iteração, o valor obtido com a função de ativação
      output <- activationFunction (theta,weight,costTraining[p],
                                    itemUsageTraining[p])
      
      #cálculo do erro local: a diferença entre a classificação real e o que foi obtido através da função de ativação
      localError<-classificationTraining[p]-output
      
      #atualizando os valores dos pesos a partir do erro da iteração para cada atributo
      weight[1]<-weight[1]+learningRate*localError*costTraining[p]
      weight[2]<-weight[2]+learningRate*localError*itemUsageTraining[p]
      weight[3]<-weight[3]+learningRate*localError 
      
      #calculando o erro acumulado global
      globalError<-globalError+localError*localError
    }
    #avançando um passo na iteração
    iteration <- iteration+1
    
    #A função só continua quando o erro global for 0 mais uma vez ou exceder o máximo de iterações
    if(globalError==0 | iteration>maxIter){break()}
  }
  
#--------------------------------------
  
#Como dito anteriormente, serão gerados também in silico os 100 itens de matéria-prima que dizem respeito à etapa de teste. Tal como na etapa de treino, os valores corresponderão às descrições dadas pela empresa (que, numa situação real, não precisariam ser gerados in silico, apenas capturados do banco de dados e tratados para o uso no R). Os 50 primeiros valores correspondem a componentes "prime" e os outros 50 aos componentes "standard"
  
  for (i in 1:50){
    costTest[i] <- runif(1, min=350, max=1000)
    itemUsageTest[i] <- runif(1, min=18, max=35)
    classificationTest[i] <- 0
  }
  
  for (i in 51:100){
    costTest[i] <- runif(1, min=100, max=600)
    itemUsageTest[i] <- runif(1, min=4, max=22)
    classificationTest[i] <- 1
  }
  
  #normalizando também esses vetores
  for (i in 1:max(length(costTest), length(itemUsageTest))){
    costTest[i]<-((1000/2)-costTest[i])/(1000/2)
    itemUsageTest[i]<-((35/2)-itemUsageTest[i])/(35/2)
  }
  
#--------------------------------------
  
#Os 100 itens de teste agora já estão normalizados e serão classificados pela RNA já treinada. Isso acontecerá dentro de um laço de repetição que utilizará a função de ativação, agora com os valores de teste. Após a obtenção dos valores a cada iteração, os pontos serão coloridos de acordo com o grau de acerto da rede: pontos corretamente classificados serão representados em azul no gráfico, enquanto que pontos erroneamente classificados serão classificados em vermelho
  
  for (i in 1:100){
    
    testOutput[i]<- activationFunction (theta, weight, costTest[i], itemUsageTest[i])

    if(testOutput[i]==classificationTest[i]){
      classificationTest[i] <- "blue"
      right <- right + 1;
    }
    else{classificationTest[i]<-"red"}
    wrong <- wrong + 1;
  }

#--------------------------------------  

#Para uma melhor compreensão gráfica, faz-se necessário a definição da reta que divide o gráfico onde os pontos correspondentes aos itens de teste serão representados. Os coeficientes da reta podem ser obtidos através de relações entre os pesos, uma vez que o Perceptron simples consiste especificamente em encontrar uma reta que consiga dividir (classificar) o gráfico em duas partes distintas
  
  a <- (-weight[3]/weight[2])
  b <- (-weight[1]/weight[2])
  
#--------------------------------------
  
#Além da representação gráfica, informações sobre as iterações, o erro global e a eficiência da RNA também podem ser impressas no console do R

  print (paste("Iterações:",iteration-1))
  
  #prints global error
  print (paste("Erro global:",globalError))
  
  #efficiency
  print (paste("Classificações erradas:",100-right))
  print (paste("Classificações corretas:",right))
  
#--------------------------------------
  
#O gráfico conterá, na parte superior, a equação da reta, bem como um plot de pontos azuis e vermelhos, dependendo da quantidade de itens classificados corretamente ou erradamente pela RNA
  
  plot(costTest, itemUsageTest,xlim=c(-1,1),ylim=c(-1,1), col=classificationTest,
       type="p", main=paste("RETA: (", signif(weight[1],digits = 2),"* X1 ) + (",
                            signif(weight[2],digits = 2),"* X2 ) + ",
                            signif(weight[3],digits = 2),"= 0"), cex = .8, 
       , pch = 16, lwd = .1 )
  abline(a,b)
}
```

Após a preparação da função `perceptron()`, resta utilizá-la com os parâmetros adequados. Neste caso, utilizaremos os 150 itens restantes para treino (lembrando que os 100 primeiros foram utilizados na fase de teste, já construídos no corpo da função), bem como 10000 iterações limite, taxa de aprendizado de 10% e limiar 0 na função de ativação:

```{r}
perceptron (10000,0.1,150,0)
```

Enfim, se o decisor julgar que a rede está bem treinada, poderá utilizá-la para classificar novos itens que nunca antes foram classificados por nenhum especialista. O RNA torna-se, portanto, especialista em classificar componentes como "prime" ou "standard".

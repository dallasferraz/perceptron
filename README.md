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
  
# As can be seen in the parameters of the function, maxIter is related to the maximum number of attempts (iterations) the algorithm can undergo before it prints out a message warning convergence was not possible. This possibility exists under a scenario where the dataset containing the information of both classes are intertwined and not easily separable by the algorithm. The variable learningRate is responsible for informing the percent rate of weight change every iteration, whereas numInstances holds information regarding available data for training and testing phases. Finally, theta is also a parameter for this function
  
#--------------------------------------

# In reality, data regarding raw materials are stored in databases (MySQL, MC Access, SQL server, Oracle and so on). For this example, I generated in silico data, which is to say that I generated randomly data in order to simulate history records for the items. 4 vectors are necessary: 2 vectors for training, 2 vectors for testing (one for each for each factor). The dimensions are supplied during the execution of the code, while the testing vector are composed of 100 different items.
  
  costTraining <- vector(length = numInstances)
  costTest <- vector(length = 100)
  
  itemUsageTraining <- vector(length = numInstances)
  itemUsageTest <- vector(length = 100)
  
#--------------------------------------

# It is also necessary to create vectors containing the classifications of the aforementioned items, 0 being "prime" and 1 being "standard"
  
  classificationTraining <- vector(length = numInstances)
  classificationTest <- vector(length = 100)
  
#--------------------------------------
  
# Another necessary vector will store the values of the testing phase  
  
  testOutput <- vector(length = 100)
  
#--------------------------------------

# Weight vector is also necessary and it holds information about the weights themselves but also the bias
  
  weight <- vector(length = 3)
  
  weight[1]<-runif(1,min=0,max=1) #w1
  weight[2]<-runif(1,min=0,max=1) #w2
  weight[3]<-runif(1,min=0,max=1) #bias
  
#--------------------------------------
  
# Error variables are necessary in intermediate steps, as well as variables for iteration counting and efficiency measurement  
  
  localError <- NA
  globalError <- NA
  right <- 0
  wrong <- 0
  
  iteration <- 1
  
#--------------------------------------

# After creating all the necessary vectors, I filled the values of each one according to their roles. The first half of the in silico generated history data was forced to represent values that would have them classified as "prime" by any specialist according to their cost per pallet and usage level. The same also happens to "standard" items. Each group was also classified as 0 and 1, accordingly
  
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

# Normalizing the data is necessary in order to avoid one variable being more relevant than the other to the system. It normalizes data to range from -1 to 1

  for (i in 1:numInstances){
    costTraining[i]<-((1000/2)-costTraining[i])/(1000/2)
    itemUsageTraining[i]<-((35/2)-itemUsageTraining[i])/(35/2)
  }
  
#--------------------------------------

# Iterations happen in this order: first, global error is set as 0. By the end of this loop, training phase is over. Only after that the algorithm goes to testing phase, with 100 remaining items
  
  repeat{
    globalError<-0
    
    for (p in 1:(numInstances)){
	  # for each iteration, variable output stores the value of the activation function
      output <- activationFunction (theta,weight,costTraining[p],
                                    itemUsageTraining[p])
      # local error calculation: real classification value minus calculated value via activation function
      localError<-classificationTraining[p]-output
      
      # updating weight values
      weight[1]<-weight[1]+learningRate*localError*costTraining[p]
      weight[2]<-weight[2]+learningRate*localError*itemUsageTraining[p]
      weight[3]<-weight[3]+learningRate*localError 
      
      # calculating cumulative global error
      globalError<-globalError+localError*localError
    }
    # new iteration
    iteration <- iteration+1
    
    # the loop only goes forward if global error equals 0 or number of iterations excedes the maximum number of iterations allowed
    if(globalError==0 | iteration>maxIter){break()}
  }
  
#--------------------------------------
# As stated before, 100 in silico items are generated for testing phase. Like in the training phase, the variable values for these items are based on the description provided by the company (in a real situation, they would not have to be generated in silico, only retrieved from the database). The first 50 values correspond to "prime" items, whereas the last 50 correspond to "standard" items
  
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
  
  # normalizing these vectors as well
  for (i in 1:max(length(costTest), length(itemUsageTest))){
    costTest[i]<-((1000/2)-costTest[i])/(1000/2)
    itemUsageTest[i]<-((35/2)-itemUsageTest[i])/(35/2)
  }
  
#--------------------------------------

# The 100 testing items are now normalized and will be classified according to the trained neural network. This happens inside a loop that utilizes the activation function (this turn, using testing values instead of training values). After this, each entry will turn into a point and plotted in a graph, and the color they'll be plotted in depends on whether it was classified correctly or misclassified by the neural network. Correctly classified points are in blue and misclassified points are plotted in red

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

# For a better visualization, it was necessary to plot a line dividing the groups of points plotted in the graph. The coefficients for the line stem from the relation between the weights in the weight vector
  
  a <- (-weight[3]/weight[2])
  b <- (-weight[1]/weight[2])
  
#--------------------------------------

# R also allows for the user to visualize information regarding the number of iterations, global error and the efficiency of the network

  print(paste("Number of iterations:",iteration-1))
  
  # prints global error
  print(paste("Global error:",globalError))
  
  # efficiency
  print(paste("WRONG:",100-right))
  print(paste("CORRECT:",right))
  
#--------------------------------------

# The graph displays the equation of the line, as well as the blue and red points, depending on how many were classified correctly 

  
  plot(costTest, itemUsageTest,xlim=c(-1,1),ylim=c(-1,1), col=classificationTest,
       type="p", main=paste("LINE: (", signif(weight[1],digits = 2),"* X1 ) + (",
                            signif(weight[2],digits = 2),"* X2 ) + ",
                            signif(weight[3],digits = 2),"= 0"), cex = .8, 
       , pch = 16, lwd = .1 )
  abline(a,b)
}
```

After the construction of the `perceptron()` function, the user can now choose values for the parameters. I utilized, just as an example, 150 items for training (the first 100 are used for testing), 10000 iterations as limit and a learning rate of 10%. 0 is the threshold of the activation function.

```{r}
perceptron (10000,0.1,150,0)
```

The results can be seen below:

![perceptron](https://raw.githubusercontent.com/dallasferraz/perceptron/master/perceptron.png)
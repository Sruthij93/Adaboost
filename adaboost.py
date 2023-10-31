import numpy as np
from sklearn import tree

def error_calc(Y, Y_pred, w_i):   
        """calculates the error"""
        e = 0
        for i in range(len(Y)):
                if(Y[i] != Y_pred[i]):
                    e += w_i[i]

        return e/sum(w_i)  

def new_weights(alpha, w_i, Y, Y_pred):
        """to calculate the new weights"""
        new_wi = []
        z = 0
        for i in range(len(Y)):                
                w = w_i[i] * np.exp(-1* alpha * Y[i] * Y_pred[i])
                new_wi.append(w)

        #  normalizing the weights so they add up to 1
        z = sum(new_wi)
        new_wi = new_wi/z  
          
        return new_wi    

def update_dataset(X, Y, Y_pred, w_i):
        """updating the data set to get a new distribution of samples based on the updated sample weights"""
        new_data =[]
        new_Y = []
        
        # b will contain the weight of the correctly classified sample
        b = min(w_i)

        for j in range(len(X)):
            if(w_i[j]==b):        # in case it has the least weight, append the sample as is (just once)
                new_data.append(X[j])
                new_Y.append (Y[j])
                
            elif (w_i[j] != b):   # the misclassified weights will be a multiple of the weights correctly classified
                c = int(w_i[j]/b) # calculating how many times the incorrect sample has to be duplicated in the new dataset
                for n in range(c):
                    new_data.append(X[j])
                    new_Y.append(Y[j])
               
        return new_data, new_Y


def adaboost_train(X, Y, max_iter):

        # initializing weights of all samples to 1/N
        w_i = np.around((np.ones(len(Y)) * 1/len(Y)) , 4) 

        f_k_main = []
        alpha = []
        X_og = X
        Y_og = Y

        for k in range(max_iter):

                # train using a decision tree stump
                f_k = tree.DecisionTreeClassifier(max_depth=1)
                model = f_k.fit(X, Y)
                f_k_main.append(model)
                Y_pred = model.predict(X_og)

                print("for iteration", k+1, ", Y = ", Y_og)
                print("for iteration", k+1, ", Y_pred = ", Y_pred, "\n")
                
                # calculate error using weights for misclassified samples
                error_wl = error_calc(Y_og, Y_pred, w_i)
                print("error = ", error_wl)

                # calculate the adaptive parameter
                alpha_k = 0.5 * np.log((1-error_wl)/error_wl)
                print("Alpha = ", alpha_k)
                alpha.append(alpha_k)

                # update the weights
                w_i = new_weights(alpha_k, w_i, Y_og, Y_pred)
                print("weights = ", w_i)
                print("sum of weights =", sum(w_i))

                #update the data set by duplicating the samples as per weight    
                new_data, new_Y= update_dataset(X_og, Y_og, Y_pred, w_i)
                X = new_data
                Y = new_Y

                print("____________________________________ \n \n")

        print("Alphas =", alpha, "\n")

        return f_k_main, alpha

def adaboost_test(X, Y, f, alpha):
        accuracy = 0 
        acc = []
        Y_pred = [] 
        for i in range(len(X)):
                y_temp = 0
                for j in range(len(f)):
                        x = np.array(X[i])
                        y = f[j].predict(x.reshape(1,-1))
                        y = alpha[j] * y
                        y_temp +=sum(y)
                        
                s = np.sign(y_temp).astype(int)
                # print("s = ", s)
                Y_pred.append(s)
        # print("Y_pred = ", Y_pred)
        
        for i in range(len(Y)):
            if(Y[i] == Y_pred[i]):
                    accuracy += 1      
        accuracy = np.around(accuracy/len(Y) * 100, 3)           
    
        return accuracy

   
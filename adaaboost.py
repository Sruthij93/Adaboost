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
        # using the formula to calculate the new weight for each sample
        for i in range(len(Y)):                
                w = w_i[i] * np.exp(-1* alpha * Y[i] * Y_pred[i])
                new_wi.append(w)
        #  normalizing the weights so they add up to 1
        z = sum(new_wi)
        new_wi = new_wi/z
        # print("z= ", z)    
          
        return new_wi
        
def update_dataset(X, Y, Y_pred, w_i):
        """updating the data set to get a distribution based on the updated sample weights"""
        new_data =[]
        new_Y = []
        new_Y_pred = []
        # b will contain the weight of the correctly classified sample
        b = min(w_i)
        # print("b = ", b)

        for j in range(len(X)):
            if(w_i[j]==b):
                new_data.append(X[j])
                new_Y.append (Y[j])
                

            elif (w_i[j] != b):  # the misclassified weights will be a multiple of the weights correctly classified
                c = int(w_i[j]/b) # calculating how many times the incorrect sample has to be duplicated in the new dataset
                for n in range(c):
                    new_data.append(X[j])
                    new_Y.append(Y[j])
                    
                
                
        return new_data, new_Y




def adaboost_train(X, Y, max_iter):

        w_i = np.around((np.ones(len(Y)) * 1/len(Y)) , 4)
        #print(w_i)
        f_k_main = []
        alpha = []
        X_og = X
        Y_og = Y

        
        for k in range(max_iter):
                
                # print("for the", k, "iteration, X = ", X)
                # print("for the", k, "iteration, Y = ", Y, "\n")
                f_k = tree.DecisionTreeClassifier(max_depth=1)
                model = f_k.fit(X, Y)
                f_k_main.append(model)
                Y_pred = model.predict(X_og)
                print("for the", k, "iteration, Y = ", Y_og)
                print("for the", k, "iteration, Y_pred = ", Y_pred, "\n")
                # print(y_pred)
                
           
                error_wl = error_calc(Y_og, Y_pred, w_i)
                print("error = ", error_wl)

                alpha_k = 0
                alpha_k = 0.5 * np.log((1-error_wl)/error_wl)
                print("Alpha = ", alpha_k)
                alpha.append(alpha_k)

                w_i = new_weights(alpha_k, w_i, Y_og, Y_pred)
                print("weights = ", w_i)
                print("sum of weights =", sum(w_i))

                new_data, new_Y= update_dataset(X_og, Y_og, Y_pred, w_i)
                # print("new data= ", new_data)
                # print("new labels= ", new_Y)
                # w_i = np.around((np.ones(len(new_Y)) * 1/len(new_Y)) , 4)

                X = new_data
                Y = new_Y
                print("____________________________ \n \n")

        print("alpha =", alpha)
        # predictions = []
        # accuracy = []
        # for alpha, model in zip(alpha, f_k_main):
        #     prediction = alpha*model.predict(X_og) # We use the predict method for the single decisiontreeclassifier models in the list
        #     print("prediction = ", prediction)
        #     predictions.append(prediction)
        #     accuracy.append(np.sum(np.sign(np.sum(np.array(predictions),axis=0))==Y_og[:])/len(predictions[0]))        
        # print("accuracy = ", accuracy)
        return f_k_main, alpha
        

                
def adaboost_test(X, Y, f, alpha):

    accuracy = 0 
    acc = []
    
    Y_pred = []
    y = []
    print(f)    
    for i in range(len(alpha)):
        model = f[i].predict(X)
        y_pred = alpha[i] * model
        Y_pred.append(y_pred) 
        y.append(np.sum(Y_pred))
    print(y)      

#     for i in range(len(X)):
#             y_temp = 0
#             for j in range(len(f)):
#                     x = np.array(X[i])
#                     y = f[j].predict(x.reshape(1,-1))
#                     y = alpha[j] * y
#                     y_temp +=sum(y)
                    
#             s = np.sign(y_temp).astype(int)
#             # print("s = ", s)
#             Y_pred.append(s)
    print("Y_pred = ", Y_pred)

    accuracy = 0
    for i in range(len(Y)):
            if(Y[i] == Y_pred[i]):
                    accuracy += 1      
    accuracy = np.around(accuracy/len(Y) * 100, 3)           
    
    return accuracy
    
          


X = [[-2,-2],[-3,-2],[-2,-3],[-1,-1],[-1,0],[0,-1],[1,1],[1,0],[0,1],[2,2],[3,2],[2,3]]
Y=[-1,-1,-1,1,1,1,-1,-1,-1,1,1,1]
f, alpha = adaboost_train(X,Y,5)
acc = adaboost_test(X,Y,f,alpha)
print("Accuracy:", acc)

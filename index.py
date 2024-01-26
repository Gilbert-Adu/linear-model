import numpy as np

dataset = [[50.1, 178.5],
           [48.3, 173.6],
           [45.2, 164.8],
           [44.7, 163.7],
           [44.5, 168.3],
           [42.7, 165],
           [39.5, 155.4],
           [38, 155.8]]
def correlation(dataset):
    X=[]
    Y = []

    for data in dataset:
        X.append(data[0])
        Y.append(data[1])

    return [X, Y]
def linear_model(dataset):

    #create a linear model
    x_sum, y_sum = 0, 0
    x_sqrd, y_sqrd = 0, 0
    xy = 0
    n = len(dataset)

    for data in dataset:
        x_sum += data[0]
        y_sum += data[1]

        x_sqrd += data[0] ** 2
        y_sqrd += data[1] ** 2

        xy += data[0] * data[1]
    

    #gradient = covariance / self variance
    #constant = y_mean - (x_mean * gradient)
    #correlation_coefficient = covariance / (standard dev(x) * standard dev(y))
    gradient = ((n * xy) - (x_sum * y_sum)) / ((n * x_sqrd) - (x_sum ** 2))
    constant = ((y_sum * x_sqrd)  - (x_sum * xy)) / ((n * x_sqrd) - (x_sum ** 2))
    
    X, Y = np.array(correlation(dataset)[0]), np.array(correlation(dataset)[1])

    correlation_coefficient = np.corrcoef(X, Y)[0, 1]
    return [round(gradient, 3), round(constant, 3), correlation_coefficient]


def predict_(x, dataset):
    #model = [gradient, y-intercept]
    model = linear_model(dataset)
    return (model[0] * x) + model[1]


sales = [[2003, 31.2], 
         [2004, 34.6],
         [2005, 28.9],
         [2006, 37.7],
         [2007, 41.3],
         [2008, 45.1]]

["weight", "miles per gallon"]
fuel_prices = [[1.3, 29],
               [1.4, 24],
               [1.5, 23],
               [1.8, 21],
               [2.1, 17],
               [2.4, 15]]

view = predict_(2,fuel_prices)
print(view)
#print("Miles per gallon will be: " + str(predict_(2, fuel_prices)))

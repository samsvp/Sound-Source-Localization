import numpy as np

def stacker(x,y,test_data,regressor):

    x_split = np.split(x,5)
    y_split = np.split(y,5)

    full_minitest_pred = []
    test_data_pred_list= []

    for i in range(len(x_split)):

        x_test = x_split[i]
        y_test = y_split[i]

        n = 0 

        placeholder_x = np.zeros_like(x_test)
        placeholder_y = np.zeros_like(y_test)


        for j in range(len(x_split)):

            if j != i:

                if n == 0:

                    x_train = np.concatenate((placeholder_x,x_split[j]))
                    y_train = np.concatenate((placeholder_y,y_split[j]))

                    n = n + 1


                else:

                    x_train = np.concatenate((x_train,x_split[j]))
                    y_train = np.concatenate((y_train,y_split[j]))

        regressor     = regressor.fit(x_train,y_train)
        y_minitest_pred   = regressor.predict(x_test)
        test_data_pred      = regressor.predict(test_data)

        full_minitest_pred.append(y_minitest_pred)
        test_data_pred_list.append(test_data_pred)


    full_y_minitest = np.array(full_minitest_pred).flatten()
    y_pred_plus     = np.array(test_data_pred_list).T

    y_pred_cv = np.array([])


    for i in range(len(y_pred_plus[:,0])):

        media = np.mean(y_pred_plus[i,:])

        y_pred_cv = np.append(y_pred_cv,media)


    full_y_minitest = full_y_minitest.reshape((len(full_y_minitest),1))
    y_pred_cv = y_pred_cv.reshape((len(y_pred_cv),1))



    return full_y_minitest,y_pred_cv
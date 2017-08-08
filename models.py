# Author: Najeem Adeleke, PhD
# Model: Predicts Page count

import matplotlib.pyplot as plt
import math
import pandas as pd
from datetime import datetime
from sklearn import model_selection as MS
from sklearn.tree import DecisionTreeRegressor as DTR
from sklearn.naive_bayes import GaussianNB as GNB
from sklearn.ensemble import RandomForestRegressor as RFR
from sklearn.neighbors import KNeighborsRegressor as KNR

def main():
    # ----------------------------
    # Training data
    # ----------------------------
    # Loading training data
    trainingDataFile = 'Training_set.csv'
    trainingData = pd.read_csv(trainingDataFile)

    # Obtaining unique cases of events (Note: This remains the same for both training and test data)
    myEventSet = []
    for x in trainingData.events:
        if x not in myEventSet:
            myEventSet.append(x)
    print('Unique events are as follows: \n', myEventSet,'\n')


    # Event string value reassignment based on unique event cases in 'myEventSet'
    newEvents = []
    for x in trainingData.events:
        for i in range(len(myEventSet)):
            if x == myEventSet[i]:
                newEvents.append(i)

    # Converting datetime to Seconds and saving day of the week
    day = []
    numDateTrainData = []
    for i in range(len(trainingData.date)):
        date_obj = datetime.strptime(str(trainingData.date[i]), '%Y-%m-%d')
        numDateTrainData.append(date_obj.timestamp())
        day.append(date_obj.weekday())

    #print(trainingData.date)
    dictReqCount = {}
    for i in range(len(trainingData.date)):
        if day[i] not in dictReqCount.keys():
            dictReqCount[day[i]] = []
        dictReqCount[day[i]].append(trainingData.request_count[i])
    #print(dictReqCount)

    dictAvgReqCount = {}
    for key,val in dictReqCount.items():
        dictAvgReqCount[key] = sum(val)/len(val)
    #print(dictAvgReqCount)

    maxValue = max(dictAvgReqCount.values())
    maxKey = [key for key,val in dictAvgReqCount.items() if val == maxValue]
    print('Day #{} of the week has the max mean request count'.format(maxKey[0]))

    minValue = min(dictAvgReqCount.values())
    minKey = [key for key, val in dictAvgReqCount.items() if val == minValue]
    print('Day #{} of the week has the min mean request count'.format(minKey[0]))


    # Assembling feature arrays
    features_trainingData = []
    for i in range(len(numDateTrainData)):
        row = [numDateTrainData[i], day[i], trainingData.calendar_code[i], trainingData.site_count[i], trainingData.max_temp[i], trainingData.min_temp[i], trainingData.precipitation[i], newEvents[i]];
        features_trainingData.append(row)

    #for i in range(len(features_trainingData)):
    #    print(len(features_trainingData[i]))

    #Y = list(trainingData.request_count)
    Y = trainingData.request_count
    X = features_trainingData

    #print('length of Y =', len(Y))
    #print(features_trainingData)

    # Models that work on both continuous and discrete data
    scoring = 'neg_mean_squared_error'
    models = [DTR(),GNB(),RFR(),KNR()]
    '''models = [[DTR(), DTR(max_depth=2), DTR(max_depth=5)],
              [GNB(), GNB(priors=None)],
              [RFR(), RFR(), RFR()],
              [KNR(), KNR(), KNR()]]
              '''
    seed = 7
    kfold = MS.KFold(n_splits=10, random_state=seed)
    i = 0
    mErr = []
    for model in models:
        results = MS.cross_val_score(model, X, Y, cv=kfold, scoring=scoring)
        mErr.append(results.mean())
        i += 1
    #print(mErr)

    best_model_index = 0
    maxAbsErrInd = math.fabs(mErr[0])
    for i in range(1, len(mErr)):
        if (math.fabs(mErr[i]) < maxAbsErrInd):
            best_model_index = i
            maxAbsErrInd = math.fabs(mErr[i])
    print('\nModel #%d (i.e. %s) performed best' %(best_model_index, str(models[best_model_index]).split('(')[0]))

    # -------------------------------------------------------
    # Test Data
    # -------------------------------------------------------
    # Loading test data
    testDataFile = 'Test_set.csv'
    testData = pd.read_csv(testDataFile)

    # Event string reassignment using myEventSet from training data
    newEvents = []
    for x in testData.events:
        for i in range(len(myEventSet)):
            if x == myEventSet[i]:
                newEvents.append(i)

    # Converting datetime to Seconds and determining days of the week
    day = []
    numDateTestData = []
    for i in range(len(testData.date)):
        date_obj = datetime.strptime(str(testData.date[i]), '%Y-%m-%d')
        numDateTestData.append(date_obj.timestamp())
        day.append(date_obj.weekday())

    # Assembling feature arrays
    features_testData = []
    for i in range(len(numDateTestData)):
        row = [numDateTestData[i], day[i], testData.calendar_code[i], testData.site_count[i], testData.max_temp[i],
               testData.min_temp[i], testData.precipitation[i], newEvents[i]];
        features_testData.append(row)

    # Test data features
    X_test = features_testData

    # Test data prediction
    bestModel = models[best_model_index]
    Y_pred = bestModel.fit(X, Y).predict(X_test)
    Y_pred_train = bestModel.fit(X, Y).predict(X)
    print('\nThe predicted values for request count using the test data is as follows:\n',Y_pred)

    output_file = open('predicted_request_count.csv','w')
    for i in range(len(Y_pred)):
        output_file.write(str(Y_pred[i])+'\n')
    output_file.close()

    # Plot the results
    plt.figure(1)
    plt.scatter(numDateTrainData, Y, c="darkorange", label="Training data")
    plt.scatter(numDateTestData, Y_pred, c="cornflowerblue", label="Test data model prediction")
    plt.scatter(numDateTrainData, Y_pred_train, c="red", label="Training data model prediction")
    plt.xlabel("Numerical Date")
    plt.ylabel("Page Count")
    plt.title("Best Model")
    plt.legend()
    plt.show()

if __name__ == "__main__":

    print('\n------------Simulation Commenced-------------')
    main()
    print('------------Simulation Completed-------------')

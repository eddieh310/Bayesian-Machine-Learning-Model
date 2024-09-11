import pandas
import math


def main():
    new_data = pandas.read_excel('Data.xlsx') # read data from excel spreadsheet into a dataframe
    # first we must divide data randomy into 75% training data and 25% test data
    # to do so, we can use pandas dataframe.sample, as it randomly splits excel data
    training_data = new_data.sample(frac=.75)  # this takes a random 75 percent of the data set and assigns it to training data
    test_data = new_data.drop(training_data.index) # this sets test_data equal to the other 25% of data

    sum_list = [] # list that will hold the sums of each x value, each entry in the list is tuple, where first value is sum of class 0, and second is sum of class 1
    mean_list = [] # list that will hold the means of each x value, each entry in the list is tuple, where first value is mean of class 0, and second  is mean of class 1
    values_list = [] # list of lists , where each tuple is the value, and class
    var_list = [] # list to hold the variance of each x, each entry in the list is tuple, where first value is var of class 0, and second is var of class 1
    total_list = [] # list to hold the total # of each class for each x value

    #each list will be populated in order, therefore the index of each list will correspond to a given column, or a given x value within training data

    for i in range(30):
        sum_list.append([0,0]) # intialize list to hold all 0s
        total_list.append([0,0])
    
    for i in range(30):
        values_list.append([]) # intialize list to hold all empty lists

    for idx, row in training_data.iterrows():  # this for loop will iterate over the dataframe, here we will fill sum_list
        tempvar = row
        sum_list, total_list = update_s_t_list(sum_list,total_list,row) # call helper method to update sum list
        values_list = update_values_list(values_list,row) # call helper method to update values list
   # print(tempvar)
    s_idx = 0 # temp index used for sum -> mean loop 
    for sums in sum_list:
        mean_list.append((sums[0] / total_list[s_idx][0], sums[1] / total_list[s_idx][1])) # calculate and append mean for each x1 .. x30
        s_idx += 1
    
    v_idx = 0
    for val_list in values_list: # loop for calculating standard deviations
        var_list = update_var_list(val_list, var_list, mean_list,v_idx,total_list) # call helper to update variance list
        v_idx += 1
    
    # now, we will test our classifier against the test data

    result_list = []  # list to hold the results of our classifier
    expected_list = [] # list of the values we expect for each attribute vector in the test_data
    z_cout = 0
    o_cout = 0 # 2 counter variables, to hold number of 0s and 1s respectively, these will be used in classifier
    class_total = 0 # counter for total number of class items

    for _, row in training_data.iterrows():
        class_total += 1
        if row['Class (malignant = 0 , benign = 1)'] == 0:
            z_cout += 1
        else:
            o_cout += 1
    
    z_prob = z_cout / class_total
    o_prob = o_cout / class_total # we now have probability of each class within the training data as well

    for _, row in test_data.iterrows():
        attr_vector = [row['x1'],row['x2'],row['x3'],row['x4'],row['x5'],row['x6'],row['x7'],row['x8'],row['x9'],row['x10'],row['x11'],row['x12'],row['x13'],row['x14'],row['x15'],row['x16'],
                       row['x17'],row['x18'],row['x19'],row['x20'],row['x21'],row['x22'],row['x23'],row['x24'],row['x25'],row['x26'],row['x27'],row['x28'],row['x29'],row['x30']]
        # run classifier on each attribute, and update expected list with expected value
        result_list.append(classifier(attr_vector,mean_list,var_list,z_prob,o_prob))
        expected_list.append(int(row['Class (malignant = 0 , benign = 1)']))  
    # calculating the accuracy results of classifier
    test_idx = 0
    correct_count = 0
    for x in result_list:
        if x == expected_list[test_idx]:
            correct_count +=1
        test_idx += 1
    print("The Accuracy is:", correct_count/test_idx, ", or ",100 * (correct_count/test_idx),"%")
  #  print(result_list)
  #  print(z_prob)
   # print(o_prob)

    # now we will test the sample input given in section 5 of the handout
    sample_vector =  [13.0, 15.0, 85.0, 500.0, 0.1, 0.15, 0.1, 
                      0.05, 0.2, 0.08, 0.5, 1.5, 
                      4.0, 70.0, 0.01, 0.02, 0.02, 0.01,
                    0.015, 0.002, 14.0, 20.0, 90.0, 600.0, 
                    0.2, 0.25, 0.2, 0.1, 0.3, 0.1]
    print("The result of the given sample_vector is: ", classifier(sample_vector,mean_list,var_list,z_prob,o_prob))

    # any additional vectors can be tested here if needed

def classifier(attr_vector, mean_list, var_list, z_prob, o_prob): 
    #classifier method, takes attribute vector, mean list, variance list, our values, and probability of 0 or 1 within test data, then returns either 0 or 1
    class_zero = 1 # probabilty of class zero
    class_one = 1 # probability of class one
    idx = 0
  #  print("mean l: ",mean_list)
   # print("var l: ",var_list)
    for attr in attr_vector: 
        # here we will plug in the attribute vecotr into a Gaussian distribution pdf
        # for each value in the attribute vector , we want to compute its conditional probability
        # well take each of these values, multiply them, multiply that by the probabilty of each class, then we have our classification
        # first we update class 0 probability, then repeat for class 1 probability
        value_1 = 1 / (math.sqrt(2 * math.pi * var_list[idx][0])) # first part of the formula, where you take 1 over the square rooot of 2 * pi * variance
        value_2 = math.exp(-((attr-mean_list[idx][0]) ** 2) / (2 * var_list[idx][0])) # second part of forumula, where you raise e to the negative power of x - mu sqaured over 2 * variance
        class_zero *= value_1 * value_2
        value_3 = 1 / (math.sqrt(2 * math.pi * var_list[idx][1])) # repeat with same attribute but with class 1
        value_4 = math.exp(-((attr-mean_list[idx][1]) ** 2) / (2 * var_list[idx][1])) 
        class_one *= value_3 * value_4
        idx +=1
   # print("c0:",class_zero)
   # print("c1:",class_one)
    class_zero *= z_prob # must multiply by class probabilites
    class_one *= o_prob
    #print(class_zero)
    #print(class_one)

    if class_zero > class_one:
        return 0
    else:
        return 1

def update_var_list(val_list,var_list:list,mean_list,v_idx,total_list):
    cmt = mean_list[v_idx]  # current mean tuple that we care about 
    vnce0 = 0 # variable to hold current variance for 0 and 1 class
    vnce1 = 0
    for val in val_list:
        # first, must check if val is 0 or 1 class
        if val[1] == 0:
            # class is 0
            vnce0 += pow(val[0]-cmt[0],2) 
        else:
            #class is 1
            vnce1 += pow(val[0] - cmt[1],2)
      #  vnce += pow(val-cm,2)
    vnce0 = vnce0 / (total_list[v_idx][0]-1)  # formula for variance
    vnce1 = vnce1 / (total_list[v_idx][1]-1)
    var_list.append((vnce0,vnce1))
    return var_list  #append variance to variance list and return 
     
def update_values_list(values_list,row):
    # Update all values_list entries to append tuples of the form (x-value, class)
    values_list[0].append((row['x1'], row['Class (malignant = 0 , benign = 1)']))
    values_list[1].append((row['x2'], row['Class (malignant = 0 , benign = 1)']))
    values_list[2].append((row['x3'], row['Class (malignant = 0 , benign = 1)']))
    values_list[3].append((row['x4'], row['Class (malignant = 0 , benign = 1)']))
    values_list[4].append((row['x5'], row['Class (malignant = 0 , benign = 1)']))
    values_list[5].append((row['x6'], row['Class (malignant = 0 , benign = 1)']))
    values_list[6].append((row['x7'], row['Class (malignant = 0 , benign = 1)']))
    values_list[7].append((row['x8'], row['Class (malignant = 0 , benign = 1)']))
    values_list[8].append((row['x9'], row['Class (malignant = 0 , benign = 1)']))
    values_list[9].append((row['x10'], row['Class (malignant = 0 , benign = 1)']))
    values_list[10].append((row['x11'], row['Class (malignant = 0 , benign = 1)']))
    values_list[11].append((row['x12'], row['Class (malignant = 0 , benign = 1)']))
    values_list[12].append((row['x13'], row['Class (malignant = 0 , benign = 1)']))
    values_list[13].append((row['x14'], row['Class (malignant = 0 , benign = 1)']))
    values_list[14].append((row['x15'], row['Class (malignant = 0 , benign = 1)']))
    values_list[15].append((row['x16'], row['Class (malignant = 0 , benign = 1)']))
    values_list[16].append((row['x17'], row['Class (malignant = 0 , benign = 1)']))
    values_list[17].append((row['x18'], row['Class (malignant = 0 , benign = 1)']))
    values_list[18].append((row['x19'], row['Class (malignant = 0 , benign = 1)']))
    values_list[19].append((row['x20'], row['Class (malignant = 0 , benign = 1)']))
    values_list[20].append((row['x21'], row['Class (malignant = 0 , benign = 1)']))
    values_list[21].append((row['x22'], row['Class (malignant = 0 , benign = 1)']))
    values_list[22].append((row['x23'], row['Class (malignant = 0 , benign = 1)']))
    values_list[23].append((row['x24'], row['Class (malignant = 0 , benign = 1)']))
    values_list[24].append((row['x25'], row['Class (malignant = 0 , benign = 1)']))
    values_list[25].append((row['x26'], row['Class (malignant = 0 , benign = 1)']))
    values_list[26].append((row['x27'], row['Class (malignant = 0 , benign = 1)']))
    values_list[27].append((row['x28'], row['Class (malignant = 0 , benign = 1)']))
    values_list[28].append((row['x29'], row['Class (malignant = 0 , benign = 1)']))
    values_list[29].append((row['x30'], row['Class (malignant = 0 , benign = 1)']))
    return values_list

def update_s_t_list(sum_list, total_list, row):
    # Check the class: 0 for malignant, 1 for benign, update data structures accordingly 
    if row['Class (malignant = 0 , benign = 1)'] == 0:
        # Class 0, update the first index of each tuple (sum and count for class 0)
        sum_list[0][0] += row['x1']
        total_list[0][0] += 1
        
        sum_list[1][0] += row['x2']
        total_list[1][0] += 1
        
        sum_list[2][0] += row['x3']
        total_list[2][0] += 1
        
        sum_list[3][0] += row['x4']
        total_list[3][0] += 1
        
        sum_list[4][0] += row['x5']
        total_list[4][0] += 1
        
        sum_list[5][0] += row['x6']
        total_list[5][0] += 1
        
        sum_list[6][0] += row['x7']
        total_list[6][0] += 1
        
        sum_list[7][0] += row['x8']
        total_list[7][0] += 1
        
        sum_list[8][0] += row['x9']
        total_list[8][0] += 1
        
        sum_list[9][0] += row['x10']
        total_list[9][0] += 1
        
        sum_list[10][0] += row['x11']
        total_list[10][0] += 1
        
        sum_list[11][0] += row['x12']
        total_list[11][0] += 1
        
        sum_list[12][0] += row['x13']
        total_list[12][0] += 1
        
        sum_list[13][0] += row['x14']
        total_list[13][0] += 1
        
        sum_list[14][0] += row['x15']
        total_list[14][0] += 1
        
        sum_list[15][0] += row['x16']
        total_list[15][0] += 1
        
        sum_list[16][0] += row['x17']
        total_list[16][0] += 1
        
        sum_list[17][0] += row['x18']
        total_list[17][0] += 1
        
        sum_list[18][0] += row['x19']
        total_list[18][0] += 1
        
        sum_list[19][0] += row['x20']
        total_list[19][0] += 1
        
        sum_list[20][0] += row['x21']
        total_list[20][0] += 1
        
        sum_list[21][0] += row['x22']
        total_list[21][0] += 1
        
        sum_list[22][0] += row['x23']
        total_list[22][0] += 1
        
        sum_list[23][0] += row['x24']
        total_list[23][0] += 1
        
        sum_list[24][0] += row['x25']
        total_list[24][0] += 1
        
        sum_list[25][0] += row['x26']
        total_list[25][0] += 1
        
        sum_list[26][0] += row['x27']
        total_list[26][0] += 1
        
        sum_list[27][0] += row['x28']
        total_list[27][0] += 1
        
        sum_list[28][0] += row['x29']
        total_list[28][0] += 1
        
        sum_list[29][0] += row['x30']
        total_list[29][0] += 1
    else:
        # Class 1, update the second index of each tuple (sum and count for class 1)
        sum_list[0][1] += row['x1']
        total_list[0][1] += 1
        
        sum_list[1][1] += row['x2']
        total_list[1][1] += 1
        
        sum_list[2][1] += row['x3']
        total_list[2][1] += 1
        
        sum_list[3][1] += row['x4']
        total_list[3][1] += 1
        
        sum_list[4][1] += row['x5']
        total_list[4][1] += 1
        
        sum_list[5][1] += row['x6']
        total_list[5][1] += 1
        
        sum_list[6][1] += row['x7']
        total_list[6][1] += 1
        
        sum_list[7][1] += row['x8']
        total_list[7][1] += 1
        
        sum_list[8][1] += row['x9']
        total_list[8][1] += 1
        
        sum_list[9][1] += row['x10']
        total_list[9][1] += 1
        
        sum_list[10][1] += row['x11']
        total_list[10][1] += 1
        
        sum_list[11][1] += row['x12']
        total_list[11][1] += 1
        
        sum_list[12][1] += row['x13']
        total_list[12][1] += 1
        
        sum_list[13][1] += row['x14']
        total_list[13][1] += 1
        
        sum_list[14][1] += row['x15']
        total_list[14][1] += 1
        
        sum_list[15][1] += row['x16']
        total_list[15][1] += 1
        
        sum_list[16][1] += row['x17']
        total_list[16][1] += 1
        
        sum_list[17][1] += row['x18']
        total_list[17][1] += 1
        
        sum_list[18][1] += row['x19']
        total_list[18][1] += 1
        
        sum_list[19][1] += row['x20']
        total_list[19][1] += 1
        
        sum_list[20][1] += row['x21']
        total_list[20][1] += 1
        
        sum_list[21][1] += row['x22']
        total_list[21][1] += 1
        
        sum_list[22][1] += row['x23']
        total_list[22][1] += 1
        
        sum_list[23][1] += row['x24']
        total_list[23][1] += 1
        
        sum_list[24][1] += row['x25']
        total_list[24][1] += 1
        
        sum_list[25][1] += row['x26']
        total_list[25][1] += 1
        
        sum_list[26][1] += row['x27']
        total_list[26][1] += 1
        
        sum_list[27][1] += row['x28']
        total_list[27][1] += 1
        
        sum_list[28][1] += row['x29']
        total_list[28][1] += 1
        
        sum_list[29][1] += row['x30']
        total_list[29][1] += 1
    
    return sum_list, total_list

main()

import numpy as np
import matplotlib.pyplot as plt
import random
import math
import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import precision_score, recall_score, f1_score

from sklearn.metrics import recall_score

from math import comb


d_cens = 35.6


def npv_score(y_true, y_pred):
    """
    A function provides npv given the prediction and the truth 
    """
    tn, _, fn, _ = confusion_matrix(y_true = y_true,
    y_pred = y_pred).ravel()
    return tn/(tn + fn)

def specificity_score(y_true, y_pred):
    """
    A function provides specificty given the prediction and the truth 
    """
    tn, fp, _, _ = confusion_matrix(y_true = y_true,
    y_pred = y_pred).ravel()
    return tn/(tn + fp)

# q = 0.001 false postive rate

gamma_matrix = np.load('F_N_d_round.npy')
eta_matrix = np.load('Eta_N_d_round.npy')
#psi_matrix = np.load('Psi_N_d_round.npy')
#theta_matrix = np.load('Theta_N_d_round.npy')

def gamma(n,d,q=0.001):
    """
    the probability to be a negative pool result that a pool size of n with
    d positive cases
    """
    if d > 10:
        return 0
    if d == 0:
        return 1 - q
    return gamma_matrix[n-1,d-1]

def eta(n,d):
    """
    the probability that group test result is positive but individual test is 
    negative
    """
    if d > 10:
        return gamma(1,1)
    return eta_matrix[n-1,d-1]

def fnr(x):
    return 1/(1 + math.exp(-12.5 * (x - 35.8)))

def generate_Vload():
    pi = [0.32, 0.53, 0.15]
    mu = [20.14, 29.35, 34.78]
    sigma = [3.60, 2.96, 1.32]
    rand = random.random()
    if rand < pi[0]:
        return np.random.normal(mu[0], sigma[0])
    elif rand < pi[0] + pi[1]:
        return np.random.normal(mu[1], sigma[1])
    else:
        return np.random.normal(mu[2], sigma[2])



def data_gen(n, p):
    res = np.zeros(shape = (n, 3))
    rand = np.random.uniform(size = n)
    for i in range(n):
        res[i, 0] = i
        res[i, 1] = 1 if rand[i] < p else 0
        res[i, 2] = generate_Vload() if res[i, 1] == 1 else 0
    return res


def group_ct(data):
    if sum(data[:, 1]) < 1:
        return -1
    else:
        temp = [2 ** (-i[2]) for i in data if i[2] > 0]
        n = data.shape[0]
        return -math.log2(1/n * sum(temp))


def false_negative_per_person(n, p):
    if n == 1:
        return gamma(1, 1)
    temp = [(eta(n, d) + gamma(n, d)) * math.comb(n-1, d-1) * (1 - p) ** (n-d) * p ** d for d in range(1, n+1)]
    return sum(temp)
    

def test_false_neg(n,p):
    temp = [gamma(n, d) * math.comb(n, d) * p ** d * (1-p) ** (n-d) for d in range(1, n + 1)]
    return sum(temp)/(1 - (1-p) ** n)



def num_per_person(n, p):
    if n == 1:
        return 1
    temp = [(1 + 1/n - gamma(n, d)) * comb(n, d) * (p/(1-p))**d for d in range(1, n+1)]
    temp = sum(temp)
    temp = 1 - gamma(n, 0) + 1/n + temp
    return temp * (1-p) ** n

def opt_batch(p, thre = 0.18):
    temp = [[n, num_per_person(n, p)] for n in range(1, 101) if test_false_neg(n,p) < thre]
    temp1 = [gamma(n, int(n * p) + 1) for n in range(1, 101)]
    temp = np.array(temp)
    ind = np.argmin(temp[:, 1])
    
    return int(temp[ind][0])



def p_on_negative_batch(p, batch_size):
    typeII = gamma(batch_size, int(batch_size * p)+1, q = 0.001)
    typeI = 0.001
    r = typeII * (1 - (1-p) ** batch_size)/((1 - typeI) * (1-p)**batch_size + typeII * (1-(1-p)**batch_size))
    return p*r/(1 - (1-p) ** batch_size)


def p_on_positive_batch(p, batch_size):
    typeII = gamma(batch_size, int(batch_size * p) + 1, q = 0.001)
    typeI = 0.001
    r = (1 - typeII) * (1 - (1-p) ** batch_size)/(typeI * (1-typeI) ** batch_size + (1-typeII) * (1-(1-p) ** batch_size))
    return p * r /(1- (1-p) ** batch_size)

def neg_pos_batch_split(data, batch_size, typeI):
    neg_batch = []
    pos_batch = []
    test_consum = np.ceil(len(data)/batch_size)
    for temp_batch in np.array_split(data, test_consum):
        batch_ct = group_ct(temp_batch)
        if batch_ct == -1:
            if np.random.uniform(0, 1) < typeI:
                pos_batch.append(temp_batch)
            else:
                neg_batch.append(temp_batch)
        else:
            if np.random.uniform(0, 1) < fnr(batch_ct):
                neg_batch.append(temp_batch)
            else:
                pos_batch.append(temp_batch)

    neg_batch = np.concatenate(neg_batch) if len(neg_batch) > 0 else np.array([])
    pos_batch = np.concatenate(pos_batch) if len(pos_batch) > 0 else np.array([])
    return (neg_batch, pos_batch, test_consum)




def helpfunction(data, p, batch_size, typeI):
    p0, p1 = p_on_negative_batch(p, batch_size), p_on_positive_batch(p, batch_size)
    n0, n1 = opt_batch(p0), opt_batch(p1)
    if data == np.array([]):
        return(np.array([]), np.array([]), p0, p1, n0, n1)

    temp0, temp1, temp_con = neg_pos_batch_split(data, batch_size, typeI)

    return (temp0, temp1, temp_con, p0, p1, n0, n1)



def conventional_test(data, typeI, repeat = 1, seq = True):
    if seq == True:
        consum = 0
        test_result = np.zeros((len(data), 2), dtype = np.int)
        random_table = np.random.uniform(0, 1, (len(data),repeat))

        for i in range(len(data)):
            temp = 0
            j = 0
            ct = data[i, 2]
            while j < repeat and temp == 0:
                random_num = random_table[i, j]
                consum += 1
                if ct > 0:
                    temp = 1 if random_num > fnr(ct) else 0
                else:
                    temp = 1 if random_num < typeI else 0
                j += 1
            test_result[i, 0] = data[i, 0]
            test_result[i, 1] = temp
        return test_result, consum
    else:
        test_result = np.zeros((len(data), 2), dtype = np.int)
        random_table = np.random.uniform(0, 1, (len(data),repeat))
        for i in range(len(data)):
            temp = 0
            ct = data[i, 2]
            for j in range(repeat):
                temp_random = random_table[i, j]
                if ct > 0:
                    temp_1 = 1 if temp_random > fnr(ct) else 0
                elif ct == 0:
                    temp_1 = 1 if temp_random < typeI else 0
                temp += temp_1

            temp = 1 if temp >= repeat/2 else 0
            test_result[i, 0] = data[i, 0]
            test_result[i, 1] = temp

        return test_result, len(data) * repeat









def seq_test(data, stop_rule, p, batch_size,repeat_neg = False, typeI = 0.001, repeat = 1, prob_threshold = 1, seq = True, node = False):
    temp_list, neg_list, pos_list, neg_node, pos_node = [], [], [],[],[]
    consum = 0
    temp = {
        'data': data,
        'NB_Num': 0,
        'PB_Num': 0,
        'p': p,
        'batch_size': batch_size,
        'node':''
    }

    temp_list.append(temp)
    new_list = []
    neg_array = []
    pos_array = []
    while len(temp_list) > 0:
        for i in temp_list:
            temp0, temp1, temp_con, p0, p1, n0, n1 = helpfunction(i['data'], i['p'],i['batch_size'],typeI)
            if repeat_neg == True:
                if len(temp0) > 0:
                    temp00, temp01, temp_con_1, _, _, _, _ = helpfunction(temp0, i['p'], i['batch_size'], typeI)
                    temp0 = temp00
                

                temp_pos = []

                if len(temp01) > 0:
                    temp_pos.append(temp01)

                if len(temp1) > 0:
                 temp_pos.append(temp1)

                if len(temp_pos) > 0:
                    temp1 = np.concatenate(temp_pos)

                consum += temp_con_1
                
            temp0 = {
                    'data': temp0,
                    'NB_Num': i['NB_Num'] + 1,
                    'PB_Num': i['PB_Num'],
                    'p': p0,
                    'batch_size': n0,
                    'node': i['node'] + '-'
                }
            

            temp1 = {
                'data': temp1,
                'NB_Num': i['NB_Num'],
                'PB_Num': i['PB_Num'] + 1,
                'p': p1,
                'batch_size': n1,
                'node': i['node'] + '+'
            }

            if len(temp0['data']) > 0:
                if temp0['NB_Num'] >= stop_rule:
                    neg_list.append(temp0)
                else:
                    new_list.append(temp0)

            if len(temp1['data']) > 0:
                if temp1['PB_Num'] >= stop_rule or temp1['p'] >= prob_threshold:
                    pos_list.append(temp1)
                else:
                    new_list.append(temp1)

            consum += temp_con
            
        temp_list = new_list
        new_list = []
    for j in neg_list:
        neg_array.append(j['data'])
        temp = [[x[0], x[1], x[2], j['node']] for x in j['data']]
        neg_node.append(temp)
    if len(neg_array) >= 1:
        neg_array = np.concatenate(neg_array)
        neg_array[:, 1] = 0
   
        neg_array = neg_array[:,[0,1]]

    for k in pos_list:
        pos_array.append(k['data'])
        temp = [[x[0], x[1], x[2], k['node']] for x in k['data']]
        pos_node.append(temp)
    if len(pos_array) >= 1:
        pos_array = np.concatenate(pos_array)

    #print(neg_array)

   



    

    individual_test, individual_con = conventional_test(
        pos_array, typeI, repeat, seq
    )

    pos_array = individual_test
    consum += individual_con
    
    if len(neg_array) == 0:
        result = pos_array
       
    else:
        result = np.concatenate((pos_array, neg_array))
    #print(neg_array)
    result = result[result[:,0].argsort()]
    result = result.astype('int')
    pos_node.extend(neg_node)
    pos_node = sum(pos_node, [])
    pos_node.sort()
    if node == False:
        return (result, consum, individual_con)
    else:
        return (result, consum, individual_con, pos_node)





def test_result(data, seq_test, **kwargs):
    """
    a helper function provides convenient results for a given test method with its **kwargs

    Input:
        data (array or list of arrays)
        seq_test (test_method object): could be seq_test, matrix_test and other test_method objects
    Output:
        result (DataFrame): a dataframe contains important evaluation metrics for the test method 
    """
    if isinstance(data, list) == False:
          
        pred,consum, ind_con = seq_test(data, **kwargs)
        result = {'acc': np.mean(pred[:,1] == data[:,1]),
        'sens': recall_score(data[:,1], pred[:,1]),
        'spec': specificity_score(data[:,1], pred[:,1]),
        'PPV': precision_score(data[:, 1], pred[:,1]),
        'NPV': npv_score(data[:, 1], pred[:,1]),
        'test_consum': consum,
        'ind_consum': ind_con,
        'batch_consum': consum - ind_con}
        return result
    else:
        length = len(data)
        acc = np.zeros(length)
        sens = np.zeros(length)
        spec = np.zeros(length)
        ppv = np.zeros(length)
        npv = np.zeros(length)
        test_consum = np.zeros(length)
        ind_consum = np.zeros(length)
        batch_consum = np.zeros(length)
        for i in range(length):
             
            pred,consum, ind_con = seq_test(data[i], **kwargs)
            
            acc[i] = np.mean(pred[:,1] == data[i][:,1])
            sens[i] = recall_score(data[i][:,1], pred[:,1])
            spec[i] = specificity_score(data[i][:,1], pred[:,1])
            ppv[i] = precision_score(data[i][:,1], pred[:,1])
            npv[i] = npv_score(data[i][:,1], pred[:,1])
            test_consum[i] = consum
            ind_consum[i] = ind_con
            batch_consum[i] = consum-ind_con

        result = {'acc': acc,
        'sens': sens,
        'spec': spec,
        'PPV': ppv,
        'NPV': npv,
        'test_consum': test_consum,
        'ind_consum': ind_consum,
        'batch_consum': batch_consum}
        return pd.DataFrame(result)



def name_fun(n):
    """
    input: stopping rule
    output: finish nodes
    """

    output = []
    temp = ['']
    for i in range(2*n - 1):
        temp_cur = []
        for j in temp:
            candidate_pos = j + '+'
            candidate_neg = j + '-'
            if str.count(candidate_pos, '+') >= n:
                output.append(candidate_pos)
            else:
                temp_cur.append(candidate_pos)
            if str.count(candidate_neg, '-') >= n:
                output.append(candidate_neg)
            else:
                temp_cur.append(candidate_neg)

        temp = temp_cur

    neg_symbol = [x for x in output if str.count(x, '-') == n]
    pos_symbol = [x for x in output if str.count(x, '+') == n]

    return output, neg_symbol, pos_symbol



def d_test(data, level, p , batch_size, typeI = 0.001, repeat = 1,
prob_threshold = 1, node = False):
    consum = 0
    neg_array = []
    pos_array = []
    target = data
    i = 0
    while len(target) > 0 and i < level:
        
        temp0, temp1, temp_con, p0, p1, n0, n1 = helpfunction(target,p,
        batch_size, typeI)
        p = p1
        batch_size = n1
        consum += temp_con
        i += 1
        if len(temp0) > 0:
            neg_array.append(temp0)
        if len(temp1) > 0:
            pos_array.append(temp1)
            target = temp1
        else:
            break
    if len(neg_array) >= 1:
        neg_array = np.concatenate(neg_array)
        neg_array[:, 1] = 0
        neg_array = neg_array[:, [0, 1]]
    if len(target) >=1:
        individual_test, individual_con = conventional_test(
            target, typeI, repeat
        )
        consum += individual_con
        pos_array = individual_test

    if len(neg_array) == 0:
        result = pos_array
    else: 
        result = np.concatenate((pos_array, neg_array))
        result = result[result[:,0].argsort()]
        result = result.astype('int')
    return (result, consum, individual_con)

        
    
            





# # prove o

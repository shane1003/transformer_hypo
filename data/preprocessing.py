import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler

def data_preprocessing(data, options):
    """
    data preprocess with 3 functions
    
    Args:
        data : loaded data from data_load
        opriots : option for CGM, CHO, Insulin
    
    Returns:
        scaler : CGM scaler
        _X : data original
    """

    _X = []
    CGM, scaler = CGM_preprocessing(data['CGM'].to_numpy().reshape(-1, 1), options['CGM'])
    df_CGM = pd.DataFrame(CGM)
    data['CGM'] = df_CGM[0]

    X = data_drop(data)
    #using all (CGM, CHO, Insulin)
    if options['feature'] != 1:
        for x in X:
            x = np.array(x, dtype=np.float64) #full for main X[data_idx][0,1,2] == [0: CGM, 1: CHO, 2: Insulin]
            x[:, 1] = CHO_preprocessing(x[:, 1], options['CHO'])
            x[:, 2] = INS_preprocessing(x[:, 2], options['Insulin'])

            _X.append(x)
    #using only CGM
    else:
        for x in X:
            x = np.array(x, dtype=np.float64)
            _X.append(x[:, 0])

    return _X, scaler

def CGM_preprocessing(data, option):
    """
    scaling the data according to options
    
    Args:
        data: column data named 'CGM'
        option: argument
            0: not using scaler
            1: using Minmaxscaler
            2: using Stardardscaler

    Returns:
        scaler and scaled data(pandas)
    """
    if option == 0:
        return data, None
    
    elif option == 1:
        scaler = MinMaxScaler(feature_range=(0,1))
        data = scaler.fit_transform(data)
        return data, scaler

    elif option == 2:
        scaler = StandardScaler()
        data = scaler.fit_transform(data)
        return data, scaler

    else:
        print("Error : Unexpected Option!")

def CHO_preprocessing(data, option):
    #print(data)
    #print(data.shape)
    """
    preprocess carbohydrate data according to options

    Args:
        data: column data named 'CHO'
        option: argument
            0: fill the empty data with 0
            1: fill the empty data using formula

    Returns:
        Preprocessed data(pandas)
    """

    if option == 0:
        data[np.where(np.isnan(data))[:]] = 0
    elif option == 1:
        data[np.where(np.isnan(data))[:]] = 0
        def CHO_intake_index(CHOs):
            """
            get CHO intake index

            Args :
                sequence : data['CHO']

            Returns:
                CHO_intake_idx : CHO intake indices
            """


            CHO_intake_idx = []

            for i, CHO in enumerate(CHOs):
                if CHO != 0:
                    CHO_intake_idx.append(i)
            
            CHO_intake_idx.append(len(CHOs))

            return CHO_intake_idx

        def CHO_Absrtion_Fomula(C_in, t):
            """
            CHO absortion formula

            Args :
                C_in : CHO intake
                t : time
            
            Returns : 
                R_a_t : Glucose absortion rate
            """
            C_bio = 0.8
            t_max_G = 60
                
            R_a_t = ( C_in * C_bio * t * np.exp( (-1)*t / t_max_G ) ) / ( t_max_G * t_max_G )
                
            return R_a_t
        
        def Glucose_Absortion_5min(C_in, C_index, t):
            """
            Generate Glucose Absortion Amount for 5 min

            Args : index, intake
                C_in : CHO intake value
                C_index : CHO intake time(index)
                t : time (index 5 min gap)

            Returns :
                Absortion_amount : CHO Absortion amount for 5min(per min)
            """ 

            if (t - C_index) == 0:
                return 0

            else:            
                Absortion_amount = 0
                for iter in range(0, 5):
                    Absortion_amount = Absortion_amount + CHO_Absrtion_Fomula(C_in, (t - C_index) * 5 - iter)

                return Absortion_amount

        def CHO_generated(CHOs):
            """
            """

            CHO_gen = []
            CHO_intake_indices = CHO_intake_index(data)
            #print(CHO_intake_indices)

            for i in range(0, CHO_intake_indices[0]): #앞에 채워있지 않은 CHO 0으로 기입
                CHO_gen.append(0)

            for i, intake_idx in enumerate(CHO_intake_indices[:-1]):
                for j in range(intake_idx, CHO_intake_indices[i + 1]):
                    CHO_gen.append(Glucose_Absortion_5min(CHOs[intake_idx], intake_idx, j))
            
            return CHO_gen

        #print(CHO_generated(data))
        generated_CHO = CHO_generated(data)
        return generated_CHO
        
    else:
        print("Error : Unexpected Option!")

    return data

def INS_preprocessing(data, option):
    """
    preprocess insulin data according to options

    Args :
        data : column data named 'Insulin'
        option : argument
            0 : fill the empty data with 0
            1 : fill the empty data with minimum(0.1000000000000106)

    Returns :
        preprocessed data(pandas)
    """

    if option == 0:
        data[np.where(np.isnan(data))[:]] = 0
    elif option == 1:
        data[np.where(np.isnan(data))[:]] = 0.10000000149011612
    else:
        print("Error : Unexpected Option!")

    return data

def data_drop(data): 
    """
    1. split the data to multi seqeunce and remove unnecessary data(Depeding on CHO whether is included or not)
    2. make the X and Y before split to train and test
    
    Args :
        data : preprocessed data with functions(pandas)

    Returns :
        feature and label sequence data divded by required size.
    """

    data.loc[ (data['CHO'] != 0) & (data['Sequence'] == 'S'), 'Sequence'] = 'SC'
    data.loc[ (data['CHO'] != 0) & (data['Sequence'] != 'S') & (data['Sequence'] != 'SC') , 'Sequence'] = 'C'

    S_seq = data.index[ (data['Sequence']=='S') | (data['Sequence']=='SC')].tolist()
    C_seq = data.index[ (data['Sequence']=='C') | (data['Sequence']=='SC')].tolist()
    SC_seq = data.index[ (data['Sequence']=='SC')].tolist()

    seq_indices = []

    for seq_idx in range(0, len(S_seq) - 1):
        tmp = []
        for CHO_idx in range(0, len(C_seq)):
            if C_seq[0] >= S_seq[seq_idx + 1] :
                break
            else :
                tmp.append(C_seq[0])
                del C_seq[0]

        seq_indices.append(tmp)
    seq_indices.append(C_seq)

    X = []

    data = data.drop(labels=['Sequence'], axis = 1)

    for idx in range(0, len(seq_indices)):
        if len(seq_indices[idx]) == 0 :
            continue
        elif seq_indices[idx][0] == S_seq[idx] :
            try:
                X.append(data.iloc[seq_indices[idx][0]:S_seq[idx+1]].values.tolist())
            except:
                X.append(data.iloc[seq_indices[idx][0]:].values.tolist())
        else :
            try:
                X.append(data.iloc[seq_indices[idx][0]:S_seq[idx+1]].values.tolist())
            except:
                X.append(data.iloc[seq_indices[idx][0]:].values.tolist())

    return X
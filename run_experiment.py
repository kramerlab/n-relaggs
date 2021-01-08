import numpy as np

import time

import tensorflow as tf

import pandas as pd

import arff

from rdm.db import DBVendor, DBConnection, DBContext, RSDConverter, mapper, AlephConverter,TreeLikerConverter,OrangeConverter

from rdm.validation import cv_split


from rdm.wrappers import RSD
from rdm.wrappers import Aleph
from rdm.wrappers import TreeLiker
from rdm.wrappers import Wordification

from sklearn import preprocessing

from n_relaggs import batch_generator,nrelaggs_model,relaggs_generator,relaggs_model,seq_generator,context_converter

from sklearn.model_selection import StratifiedKFold


dataset = 'trains'
target_label = 'direction'
target_table = 'trains'
target_attr_value = "east"

#aleph/rsd/treeliker/wordification/relaggs/nrelaggs
algorithm = nrelaggs

#hyperparameters
predictor_layers_=[(100,),(50,),(100,50)]
loss='hinge'
feature_generation_=[1.,0.5,0.75]
feature_selection_=[1.,0.5,0.75]


connection = DBConnection(
        'guest',  # User
        'relational',  # Password
        'relational.fit.cvut.cz',  # Host
        dataset,  # Database
        vendor=DBVendor.MySQL
)
    
context = DBContext(connection, target_table=target_table, target_att=target_label)


#scores
aurocs = []
accuracies = []


fold_num = 0
for train_context, test_context in cv_split(context, folds=5, random_seed=1):
    fold_num += 1
    
    print("FOLD",fold_num)
    
    
    #ALEPH
    if algorithm == "aleph":
    
        start = time.time()
        conv = AlephConverter(context, target_att_val=target_attr_value)
        aleph = Aleph()
        train_arff, features = aleph.induce('induce_features',conv.positive_examples(),
                                            conv.negative_examples(),
                                            conv.background_knowledge(),printOutput=False)

        data = arff.loads(str(train_arff))
        entries = []
        targets = []

        for entry in data['data']:
            en = list(entry)
            features_target = en[-1]
            features_train = en[0:len(en)-1]
            features_train = [1 if x == "+" else 0 for x in features_train]
            entries.append(features_train)
            targets.append(features_target)


        tmp_learner = 'aleph'
        test_arff = mapper.domain_map(features, tmp_learner, train_context, test_context,format="csv",positive_class=target_attr_value)
        test_ins = test_arff.split("\n")

        entries_test = []
        targets_test = []

        for entry in test_ins:
            en = entry.strip().split(",")
            if en[-1] != '':
                features_target = en[-1]
                features_train = en[0:len(en)-1]
                features_train = [1 if x == "+" else 0 for x in features_train]
                entries_test.append(features_train)
                targets_test.append(features_target)


        targets_test = ['positive' if x == target_attr_value else 'negative' for x in targets_test]


        train_features = pd.DataFrame(entries).to_numpy()
        train_targets = pd.DataFrame(targets).to_numpy()
        test_features = pd.DataFrame(entries_test).to_numpy()
        test_targets = pd.DataFrame(targets_test).to_numpy()

        le = preprocessing.LabelEncoder()
        le.fit(train_targets)
        targets_train_encoded = le.transform(train_targets)
        targets_test_encoded = le.transform(test_targets)

        end = time.time()
        run_time = end - start
        train_data = (train_features,targets_train_encoded)
        test_data = (test_features,targets_test_encoded)

        
    #RSD
    elif algorithm == "rsd":
    
        start = time.time()
        conv = RSDConverter(train_context)
        rsd = RSD()
        features, train_arff, _ = rsd.induce(conv.background_knowledge(),
                                             examples=conv.all_examples(),cn2sd=False)


        data = arff.loads(str(train_arff))
        entries = []
        targets = []

        for entry in data['data']:
            en = list(entry)
            features_target = en[-1]
            features_train = en[0:len(en)-1]
            features_train = [1 if x == "+" else 0 for x in features_train]
            entries.append(features_train)
            targets.append(features_target)


        tmp_learner = 'rsd'
        test_arff = mapper.domain_map(features, tmp_learner, train_context, test_context,format="csv")
        test_ins = test_arff.split("\n")

        entries_test = []
        targets_test = []

        for entry in test_ins:
            en = entry.strip().split(",")
            if en[-1] != '':
                features_target = en[-1]
                features_train = en[0:len(en)-1]
                features_train = [1 if x == "+" else 0 for x in features_train]
                entries_test.append(features_train)
                targets_test.append(features_target)


        train_features = pd.DataFrame(entries).to_numpy()
        train_targets = pd.DataFrame(targets).to_numpy()
        test_features = pd.DataFrame(entries_test).to_numpy()
        test_targets = pd.DataFrame(targets_test).to_numpy()

        le = preprocessing.LabelEncoder()
        le.fit(train_targets)
        targets_train_encoded = le.transform(train_targets)
        targets_test_encoded = le.transform(test_targets)


        end = time.time()
        run_time = end - start
        train_data = (train_features,targets_train_encoded)
        test_data = (test_features,targets_test_encoded)
    
    
    
    #Treeliker
    elif algorithm == "treeliker":
    
        start = time.time()
        conv = TreeLikerConverter(train_context)
        conv2 = TreeLikerConverter(test_context)
        treeliker = TreeLiker(conv.dataset(), conv.default_template(),conv2.dataset())
        train_arff, test_arff = treeliker.run()
        wtag=False
        entries = []
        targets = []
        entries_test = []
        targets_test = []

        for entry in train_arff.split("\n"):
            if wtag:
                en = entry.split(",")
                if len(en)>1:
                    en = [x.replace(" ","") for x in en]
                    targets.append(en[-1])
                    en = [1 if "+" in x else 0 for x in en]
                    entries.append(en[0:len(en)-1])
            if "@data" in entry:
                wtag=True

        wtag=False
        for entry in test_arff.split("\n"):
            if wtag:
                en = entry.split(",")
                if len(en) > 1:
                    en = [x.replace(" ","") for x in en]
                    targets_test.append(en[-1])
                    en = [1 if "+" in x else 0 for x in en]
                    entries_test.append(en[0:len(en)-1])

            if "@data" in entry:
                wtag=True


        train_features = pd.DataFrame(entries).to_numpy()
        train_targets = pd.DataFrame(targets).to_numpy()
        test_features = pd.DataFrame(entries_test).to_numpy()
        test_targets = pd.DataFrame(targets_test).to_numpy()

        le = preprocessing.LabelEncoder()
        le.fit(train_targets)
        targets_train_encoded = le.transform(train_targets)
        targets_test_encoded = le.transform(test_targets)


        end = time.time()
        run_time = end - start
        train_data = (train_features,targets_train_encoded)
        test_data = (test_features,targets_test_encoded)

        
    #Wordification
    elif algorithm == "wordification":
    
        start = time.time()
        corange = OrangeConverter(train_context)
        torange = OrangeConverter(test_context)
        wordification = Wordification(corange.target_Orange_table(), corange.other_Orange_tables(), train_context)
        wordification.run(1)
        wordification.calculate_weights()
        train_arff = wordification.to_arff()
        wordification_test = Wordification(torange.target_Orange_table(), torange.other_Orange_tables(), test_context)
        wordification_test.run(1)
        wordification_test.calculate_weights()

        idfs = wordification.idf
        docs  = wordification_test.resulting_documents
        classes = [str(a) for a in wordification_test.resulting_classes]
        feature_names = wordification.word_features
        feature_vectors = []
        for doc in docs:
            doc_vec = []
            for feature in feature_names:
                cnt = 0
                for x in doc:
                    if x  == feature:
                        cnt+=1
                idf = cnt * idfs[feature]
                doc_vec.append(idf)
            feature_vectors.append(doc_vec)
        print(feature_vectors,classes)

        test_arff = wordification_test.to_arff()

        entries = []
        targets = []
        entries_test = []
        targets_test = []
        wtag = False

        for entry in train_arff.split("\n"):
            if wtag:
                en = entry.split(",")
                if len(en)>1:
                    en = [x.replace(" ","") for x in en]

                    targets.append(en[-1])
                    entries.append([float(x) for x  in en[0:len(en)-1]])
            if "@DATA" in entry:
                wtag=True

        wtag=False

        targets_test = classes
        entries_test = feature_vectors


        train_features = pd.DataFrame(entries).to_numpy()
        train_targets = pd.DataFrame(targets).to_numpy()
        test_features = pd.DataFrame(entries_test).to_numpy()
        test_targets = pd.DataFrame(targets_test).to_numpy()

        le = preprocessing.LabelEncoder()
        le.fit(train_targets)
        targets_train_encoded = le.transform(train_targets)
        targets_test_encoded = le.transform(test_targets)


        end = time.time()
        run_time = end - start
        train_data = (train_features,targets_train_encoded)
        test_data = (test_features,targets_test_encoded)


        
    #relaggs/nrelaggs
    else:
        converter = context_converter(train_context, test_context, verbose=0)
        train_data = converter.get_train()
        test_data = converter.get_test()
        plan = converter.get_plan()
        run_time = converter.get_time()
        
        
        
    #build models
    if algorithm == "relaggs":
        relaggs_data = relaggs_generator(train_data[0],plan)
        X = relaggs_data.get_data()
        y = train_data[1]
        
        relaggs_data = relaggs_generator(test_data[0],plan)
        X_test = relaggs_data.get_data()
        y_test = test_data[1]
        
    elif algorithm != "nrelaggs":
        X = train_data[0]
        y = train_data[1][:,np.newaxis]
        
        X_test = test_data[0]
        y_test = test_data[1][:,np.newaxis]
        
   

    if algorithm != "nrelaggs":
        best_params = [predictor_layers_[0]]
        best_score = [0.]
        
        #Parameter-Selection
        for predictor_layers in predictor_layers_:
            cur_score = 0.
            skf = StratifiedKFold(n_splits=3)
            for train_index, test_index in skf.split(X,y):
                X_train, X_val = X[train_index], X[test_index]
                y_train, y_val = y[train_index], y[test_index]

                train_batch = seq_generator(X_train,y_train)
                val_batch = seq_generator(X_val,y_val)

                rel_model = relaggs_model(train_batch.get_sizes(),predictor_layers,loss)

                rel_model.train_model(train_batch,100)

                acc,auroc = rel_model.evaluate_model(val_batch,y_val)
                cur_score += auroc
                tf.keras.backend.clear_session()

            if cur_score > best_score:
                best_score = cur_score
                best_params = [predictor_layers]

        

        train_batch = seq_generator(X,y)
        test_batch = seq_generator(X_test,y_test)
        rel_model = relaggs_model(train_batch.get_sizes(),best_params[0],loss)
        rel_model.train_model(train_batch,100)     
        acc,auroc = rel_model.evaluate_model(test_batch,y_test)
        tf.keras.backend.clear_session()
        
        print("BEST-PARAMS: ", best_params)
        print("ACC:",acc," AUROC:",auroc)

        accuracies.append(acc)
        aurocs.append(auroc)
        
        
    if algorithm == "nrelaggs":
    
    best_params = [predictor_layers_[0],feature_generation_[0],feature_selection_[0]]
    best_score = [0.]
    
    for predictor_layers in predictor_layers_:
        for feature_generation in feature_generation_:
            for feature_selection in feature_selection_:
                cur_score = 0.
                skf = StratifiedKFold(n_splits=3)
                for train_index, test_index in skf.split(train_data[1], train_data[1]):
                    X_train = [train_data[0][i] for i in train_index]
                    X_val = [train_data[0][i] for i in test_index]
                    y_train, y_val = train_data[1][train_index], train_data[1][test_index]
                    
                    train_batch = batch_generator(X_train,y_train)
                    val_batch = batch_generator(X_val,y_val)
                    
                    
                    rel_model = nrelaggs_model(train_batch.get_sizes(),plan,predictor_layers,
                                               loss,feature_generation,feature_selection)
                    
                    rel_model.train_model(train_batch,100)
                    
                    acc,auroc = rel_model.evaluate_model(val_batch,y_val)
                    cur_score += auroc
                    tf.keras.backend.clear_session()
                
                if cur_score > best_score:
                    best_score = cur_score
                    best_params = [predictor_layers,feature_generation,feature_selection]
            
    

    train_batch = batch_generator(train_data[0],train_data[1])
    test_batch = batch_generator(test_data[0],test_data[1])
    
    rel_model = nrelaggs_model(train_batch.get_sizes(),plan,best_params[0],
                               loss,best_params[1],best_params[2])
    
    rel_model.train_model(train_batch,100)
    acc,auroc = rel_model.evaluate_model(test_batch,test_data[1])
    tf.keras.backend.clear_session()
    
    print("BEST-PARAMS: ", best_params)
    print("ACC:",acc," AUROC:",auroc)
    
    accuracies.append(acc)
    aurocs.append(auroc)
    
print("RESULT for ",algorithm)
print("ACC:",accuracies/5.," AUROC:",aurocs/5.)

import Orange
import numpy as np
import os
import tqdm
import time

from PropStar.propStar import *
from PropStar.neural import *  ## DRMs
from PropStar.learning import *  ## starspace
from PropStar.vectorizers import *  ## ConjunctVectorizer

from sklearn import preprocessing
from sklearn.metrics import accuracy_score, roc_auc_score

import pickle


#Changed to resemble the splits made by the py-rdm package
def preprocess_and_split_2(X, num_fold=10, target_attribute=None, random_seed=None):

    
    Y = X[target_attribute]
    if len(np.unique(Y)) > 40:
        tmp1 = [float(x) for x in Y]
        tmp = []
        for j in tmp1:
            if j > np.mean(tmp1):
                tmp.append(1)
            else:
                tmp.append(0)
        Y = tmp

    Y = np.array(Y)
    le = preprocessing.LabelEncoder()
    Y = le.fit_transform(Y)

    domain = Orange.data.Domain([],Orange.data.ContinuousVariable.make("target"))
    input_list = Orange.data.Table.from_numpy(domain,np.zeros((Y.shape[0],0)),Y)

    random_seed = random.randint(0, 10**6) if random_seed is None else random_seed
    cv = Orange.evaluation.CrossValidation(k=num_fold, random_state=random_seed, stratified=True)
    cv_indices = cv.get_indices(input_list)

    for train_index, test_index in cv_indices:
        yield train_index, test_index



def run_prop(example_sql,target_table,target_attribute,learners=["starspace"],learning_rates=[0.001],
    epochss=[10],dropouts=[0.1],num_featuress=[30000],hidden_sizes=[16],negative_samples_limits=[10],
    negative_search_limits=[10],representation_types=["tfidf"],random_seed=None,result_file=None,num_fold=10):
    
    variable_types_file = open(
        "./PropStar/variable_types.txt")  ## types to be considered.
    variable_types = [
        line.strip().lower() for line in variable_types_file.readlines()
    ]
    variable_types_file.close()

     ## IMPORTANT: a tmp folder must be possible to construct, as the intermediary embeddings are stored here.
    directory = "tmp"

    if not os.path.exists(directory):
        os.makedirs(directory)

    tables, fkg, primary_keys = table_generator(
                    example_sql, variable_types)


    experiment_grid = []
    for learner in learners:
        if learner == "DRM":
            for epochs in epochss:
                for learning_rate in learning_rates:
                    for hidden_size in hidden_sizes:
                        for dropout in dropouts:
                            for representation_type in representation_types:
                                for num_features in num_featuress:
                                    experiment_grid.append([learner,epochs,learning_rate,None,hidden_size,
                                                            dropout,None,representation_type,num_features])
        elif learner == "starspace":
            for epochs in epochss:
                for learning_rate in learning_rates:
                    for negative_samples_limit in negative_samples_limits:
                        for hidden_size in hidden_sizes:
                            for negative_search_limit in negative_search_limits:
                                for representation_type in representation_types:
                                    for num_features in num_featuress:
                                        experiment_grid.append([learner,epochs,learning_rate,negative_samples_limit,
                                                                hidden_size,None,negative_search_limit,
                                                                representation_type,num_features])


        else:
            continue


    split_gen = preprocess_and_split_2(
                tables[target_table],
                num_fold=num_fold,
                target_attribute=target_attribute,
                random_seed=random_seed)
    total_perf = []
    total_perf_roc = []
    fold_nr = 0
    for train_index, test_index in split_gen:
        fold_nr += 1
        best_perf = 0.
        best_pars = experiment_grid[0]

        print("FOLD ",fold_nr)
        print()
        if not result_file is None:
            with open(result_file, 'a') as f:
                f.write("FOLD {}\n\n".format(fold_nr))
        
        #gridsearch for hyperparameters
        for pars in tqdm.tqdm(experiment_grid):
            perf = []
            val_splits = preprocess_and_split_2(
                tables[target_table].iloc[train_index, :],
                num_fold=3,
                target_attribute=target_attribute,
                random_seed=1)

            for train_idx_, test_idx_ in val_splits:
                train_idx = train_index[train_idx_]
                test_idx = train_index[test_idx_]

                train_features, train_classes, vectorizer = generate_relational_words(
                                    tables,
                                    fkg,
                                    target_table,
                                    target_attribute,
                                    relation_order=(1, 2),
                                    indices=train_idx,
                                    vectorization_type=pars[7],
                                    num_features=pars[8])
                test_features, test_classes = generate_relational_words(
                                    tables,
                                    fkg,
                                    target_table,
                                    target_attribute,
                                    relation_order=(1, 2),
                                    vectorizer=vectorizer,
                                    indices=test_idx,
                                    vectorization_type=pars[7],
                                    num_features=pars[8])

                le = preprocessing.LabelEncoder()
                le.fit(train_classes.values)

                train_classes = le.transform(train_classes)
                test_classes = le.transform(test_classes)

                if pars[0] == "DRM":
                    model = E2EDNN(num_epochs=pars[1],
                                    learning_rate=pars[2],
                                    hidden_layer_size=pars[4],
                                    dropout=pars[5])
                        
                    ## standard fit predict
                    model.fit(train_features, train_classes)
                    preds = model.predict(test_features)
                    acc1 = accuracy_score(preds, test_classes)
                    perf.append(acc1)

                elif pars[0] == "starspace":
                    model = starspaceLearner(epoch=pars[1],
                                                learning_rate=pars[2],
                                                neg_search_limit=pars[3],
                                                dim=pars[4],
                                                max_neg_samples=pars[6])

                    ## standard fit predict
                    model.fit(train_features, train_classes)
                    preds = model.predict(test_features, clean_tmp=False)

                    if len(preds) == 0:
                        perf.append(0)
                        continue

                    try:
                        acc1 = accuracy_score(test_classes, preds)

                        perf.append(acc1)

                    except Exception as es:
                        print(es)
                        continue

            cur_perf = np.round(np.mean(perf), 4)
            if cur_perf > best_perf:
                best_perf = cur_perf
                best_pars = pars
        print("FOLD ",fold_nr," RESULTS")
        print("|".join(str(x) for x in best_pars))
        if not result_file is None:
            with open(result_file, 'a') as f:
                f.write("|".join(str(x) for x in best_pars))
                f.write("\n")

        start = time.time()

        train_features, train_classes, vectorizer = generate_relational_words(
                        tables,
                        fkg,
                        target_table,
                        target_attribute,
                        relation_order=(1, 2),
                        indices=train_index,
                        vectorization_type=best_pars[7],
                        num_features=best_pars[8])
        test_features, test_classes = generate_relational_words(
                        tables,
                        fkg,
                        target_table,
                        target_attribute,
                        relation_order=(1, 2),
                        vectorizer=vectorizer,
                        indices=test_index,
                        vectorization_type=best_pars[7],
                        num_features=best_pars[8])

        le = preprocessing.LabelEncoder()
        le.fit(train_classes.values)

        train_classes = le.transform(train_classes)
        test_classes = le.transform(test_classes)

        if best_pars[0] == "DRM":
            model = E2EDNN(num_epochs=best_pars[1],
                            learning_rate=best_pars[2],
                            hidden_layer_size=best_pars[4],
                            dropout=best_pars[5])
            
            ## standard fit predict
            model.fit(train_features, train_classes)
            preds = model.predict(test_features)

            

            acc1 = accuracy_score(preds, test_classes)
            print("ACCURACY:",acc1)
            if not result_file is None:
                with open(result_file, 'a') as f:
                    f.write("ACCURACY: {}\n".format(acc1))
            total_perf.append(acc1)

            if len(np.unique(test_classes)) == 2:
                preds = model.predict(test_features,
                                        return_proba=True)


                roc = roc_auc_score(test_classes, preds)
                print("ROC:",roc)
                if not result_file is None:
                    with open(result_file, 'a') as f:
                        f.write("ROC: {}\n".format(roc))
                total_perf_roc.append(roc)

            else:

                total_perf_roc.append(0.5)

        elif best_pars[0] == "starspace":
            model = starspaceLearner(epoch=best_pars[1],
                                        learning_rate=best_pars[2],
                                        neg_search_limit=best_pars[3],
                                        dim=best_pars[4],
                                        max_neg_samples=best_pars[6])

            ## standard fit predict
            model.fit(train_features, train_classes)
            preds = model.predict(test_features, clean_tmp=False)


            if len(preds) == 0:
                total_perf.append(0)
                total_perf_roc.append(0)
                continue

            try:
                acc1 = accuracy_score(test_classes, preds)
                print("ACCURACY:",acc1)
                if not result_file is None:
                    with open(result_file, 'a') as f:
                        f.write("ACCURACY: {}\n".format(acc1))
                total_perf.append(acc1)

                preds_scores = model.predict(
                                    test_features,
                                    clean_tmp=True,
                                    return_int_predictions=False,
                                    return_scores=True)  ## use scores for auc.

                if len(np.unique(test_classes)) == 2:
                    roc = roc_auc_score(test_classes, preds_scores)
                    total_perf_roc.append(roc)
                    print("ROC:",roc)
                    if not result_file is None:
                        with open(result_file, 'a') as f:
                            f.write("ROC: {}\n".format(roc))
                else:
                    total_perf_roc.append(0.5)

            except Exception as es:
                print(es)
                continue

        run_time = time.time() - start
        print("TIME: ",run_time)
        if not result_file is None:
            with open(result_file, 'a') as f:
                f.write("Time: {}\n".format(run_time))
    print("OVERALL RESULTS")
    print("ACC:",np.round(np.mean(total_perf), 4))
    print("ROC:",np.round(np.mean(total_perf_roc), 4))
    if not result_file is None:
        with open(result_file, 'a') as f:
            f.write("OVERALL RESULTS\n")
            f.write("ACC: {}\n".format(np.mean(total_perf)))
            f.write("ROC: {}\n\n".format(np.mean(total_perf_roc)))
            f.write("ACCs\n")
            f.write(",".join([str(a) for a in total_perf]))
            f.write("\n AUROCS\n")
            f.write(",".join([str(a) for a in total_perf_roc]))
            f.write("\n")

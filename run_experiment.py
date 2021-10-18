
from rdm.db import DBVendor, DBConnection, DBContext

from experiment_essentials import *
#If you want to use propStar or propDRM
from prop_star import *




dataset = 'trains'
target_label = 'direction'
target_table = 'trains'
target_attr_value = "east"

#aleph/rsd/treeliker/wordification/relaggs/nrelaggs/nrelaggs_fix/propstar/propdrm
algorithm = "nrelaggs"

#hyperparameters
predictor_layers=[(100,),(50,),(100,50)]
loss='hinge'
feature_generation=[1.,0.5,0.75]
feature_selection=[1.,0.5,0.75]

#for propStar/propDRM
learning_rates=[0.001,0.01,0.0001]
num_featuress=[10000,30000,50000]
hidden_sizes=[8,16,32]


connection = DBConnection(
        'guest',  # User
        'relational',  # Password
        'relational.fit.cvut.cz',  # Host
        dataset,  # Database
        vendor=DBVendor.MySQL
)
    
context = DBContext(connection, target_table=target_table, target_att=target_label)

#Sql-File for propStar/propDRM
sql_file = "Data/trains/trains.sql"



if algorithm in ["aleph","rsd","treeliker","wordification","relaggs"]:
    transform(algorithm,context,target_attr_value,seed=1,result_file="results_transformation.txt",transformations="tmp_transformation",fold_nums=10)
    experiment(algorithm,transformations="tmp_transformation",result_file="results_prediction.txt",predictor_layers,loss,feature_generation,feature_selection,fold_nums=10,epochs=100)

if algorithm in ["nrelaggs","nrelaggs_fix"]:
    transform("relaggs",context,target_attr_value,seed=1,result_file="results_transformation.txt",transformations="tmp_transformation",fold_nums=10)
    experiment(algorithm,transformations="tmp_transformation",result_file="results_prediction.txt",predictor_layers,loss,feature_generation,feature_selection,fold_nums=10,epochs=100)

if algorithm == "propstar":
    run_prop(sql_file,target_table,target_label,learners=["starspace"],learning_rates,
            epochss=[10],dropouts=[0.1],num_featuress,hidden_sizes,negative_samples_limits=[10],
            negative_search_limits=[10],representation_types=["tfidf"],random_seed=1,result_file="results_prediction.txt",num_fold=10)

if algorithm == "propdrm":
    run_prop(sql_file,target_table,target_label,learners=["DRM"],learning_rates,
            epochss=[10],dropouts=[0.1],num_featuress,hidden_sizes,negative_samples_limits=[10],
            negative_search_limits=[10],representation_types=["tfidf"],random_seed=1,result_file="results_prediction.txt",num_fold=10)



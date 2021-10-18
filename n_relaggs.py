import numpy as np

import pandas as pd

from sklearn.preprocessing import LabelBinarizer
from sklearn.preprocessing import MinMaxScaler

import time
import copy

import pickle

import tensorflow as tf
import math
import os

from tensorflow.python.keras.engine.base_layer import Layer
from tensorflow.python.keras import backend
from tensorflow.python.ops import math_ops
from tensorflow.python.framework import tensor_shape
from sklearn.metrics import roc_auc_score, roc_curve,auc,r2_score

class context_converter():
    
    def __init__(self, train_context, test_context=None, max_unique=20, verbose=0):
        self.train_context = train_context
        self.test_context = test_context
        self.max_unique = max_unique
        self.verbose = verbose
        
        
        self.build_structure_()
        
    def get_train(self):
        return self.data_train[0], self.data_train[1]
    
    def get_test(self):
        return self.data_test[0], self.data_test[1]
    
    def get_plan(self):
        return self.plan
    
    def get_time(self):
        return self.train_time + self.test_time
        
    def build_structure_(self):
        self.train_tables = self.preprocess_tables_()
        if self.test_context is not None:
            self.test_tables = self.preprocess_tables_(is_test=True)
        else:
            self.test_tables = None
        if self.verbose > 0:
            print("tables done")
        
        self.data_train, self.plan, self.train_time = self.gen_data_(is_test=False)
        if self.test_context is not None:
            self.data_test, _, self.test_time = self.gen_data_(is_test=True)
        else:
            self.data_test = (None, None)
            self.test_time = 0.
        
    
    #generate aggregation plan
    def generate_agg_plan_(self):
        data_tables = [self.train_context.target_table]

        connections = list(self.train_context.connected.keys())

        def rec_walk(cur,vis):
            out = ([],cur,cur)
            other_out = []
            for con in connections:
                old,new = con
                if old == cur[0]:
                    if new not in vis:
                        len_new = len(data_tables)
                        data_tables.append(new)
                        out[0].append((new,len_new))
                        other_out += rec_walk((new,len_new),vis+[new])

            out = [out]+other_out
            if len(out[0][0]) == 0:
                out = []
            return out
        plan = rec_walk((self.train_context.target_table,0),[self.train_context.target_table])
        return data_tables,plan
        
        
    #generate Dataset
    def gen_data_(self,is_test=False):
        data_tables, plan = self.generate_agg_plan_()

        outputs = ([],[])
        
        if is_test:
            context = self.test_context
            numpy_tables = self.test_tables
        else:
            context = self.train_context
            numpy_tables = self.train_tables

        total_number = len(numpy_tables[context.target_table]["target"])
        done_number = 0
        times = []
        start = time.time()
        for c,c_y in enumerate(numpy_tables[context.target_table]["target"]):

            outputs[1].append(c_y)
            out_data_ids = [[] for i in data_tables]
            out_data = [[] for i in data_tables]
            out_ids = [[] for i in data_tables]

            out_data_ids[0] = [c]

            for step in plan:
                cur = step[1]
                for prev in step[0]:
                    cur_id_name, prev_id_name = context.connected[(cur[0],prev[0])][0]

                    id_counter = -1
                    for cur_entry in out_data_ids[cur[1]]:
                        id_counter += 1
                        if cur_entry == -1:
                            prev_entries = [-1]
                        else:
                            cur_id = numpy_tables[cur[0]][cur_id_name][cur_entry]
                            prev_entries = list(np.where(numpy_tables[prev[0]][prev_id_name] == cur_id)[0])
                            if len(prev_entries) == 0:
                                prev_entries = [-1]
                        out_data_ids[prev[1]] += prev_entries
                        out_ids[prev[1]] += [id_counter for i in prev_entries]

            for c,data_ids in enumerate(out_data_ids):
                out_data[c] = numpy_tables[data_tables[c]]["value"][data_ids]
                out_ids[c] = np.array(out_ids[c])

            outputs[0].append( (out_data,out_ids) )

            done_number += 1
            end = time.time()
            times.append(end - start)
            start = end
            if self.verbose > 0:
                print("data: {}/{} in {:6.5f}s/entry".format(done_number,total_number,np.mean(times)),end="\r")

        if self.verbose > 0:
            print()
            
        plan2 = []
        for step in plan[::-1]:
            new_step = ([],step[1][1],step[2][1])
            for entry in step[0]:
                new_step[0].append(entry[1])
            plan2.append(new_step)

        outputs = (outputs[0],np.array(outputs[1]))
        plan= plan2
        return outputs,plan,np.sum(times)
    
        
        
    #preprocess values
    def preprocess_tables_(self,is_test=False):
        numpy_tables = dict()
        transformers = dict()
        kept_labels = dict()
        context = self.train_context
        
        if is_test:
            transformers = self.transformers
            kept_labels = self.kept_labels
            context = self.test_context
        
        for table in context.tables:
            
            table_entries = 0
            
            if not is_test:
                transformers[table] = dict()
                kept_labels[table] = dict()
            numpy_tables[table] = dict()

            table_ids = set([context.pkeys[table]])
            if table in context.fkeys:
                table_ids = table_ids.union(context.fkeys[table])

            table_values = []

            arrs = {col: [] for col in context.cols[table]}
            for x in context.orng_tables[table]:
                for dom in arrs:
                    arrs[dom].append(x[dom].value)

            for dom in arrs:
                arrs[dom] = np.array(arrs[dom])[:,np.newaxis]
                
                if dom in table_ids:
                    numpy_tables[table][dom] = arrs[dom].astype('U')
                    table_entries = arrs[dom].shape[0]
                
                elif not np.issubdtype(arrs[dom].dtype, np.number):
                    if len(np.unique(arrs[dom])) > self.max_unique:
                        if is_test:
                            keeps = kept_labels[table][dom]
                        else:
                            uniques,counts = np.unique(arrs[dom],return_counts=True)
                            keeps = np.flip(uniques[np.argsort(counts)])[:self.max_unique]
                            kept_labels[table][dom] = keeps
                        arrs[dom][np.isin(arrs[dom],keeps,invert=True)] = 'other'

                    if is_test:
                        cur_trans = transformers[table][dom]
                    else:
                        cur_trans = LabelBinarizer()
                        cur_trans.fit(arrs[dom])
                        transformers[table][dom] = cur_trans
                    arrs[dom] = cur_trans.transform(arrs[dom])
                    if table == context.target_table and dom == context.target_att:
                        numpy_tables[table]["target"] = arrs[dom]
                    else:
                        table_values.append(arrs[dom])
                else:
                    if is_test:
                        cur_trans = transformers[table][dom]
                    else:
                        cur_trans = MinMaxScaler()
                        cur_trans.fit(arrs[dom])
                        transformers[table][dom] = cur_trans
                    arrs[dom] = cur_trans.transform(arrs[dom])
                    if table == context.target_table and dom == context.target_att:
                        numpy_tables[table]["target"] = arrs[dom]
                    else:
                        table_values.append(arrs[dom])

            if len(table_values) >= 1:
                table_values = np.concatenate(table_values,axis=1)
            else:
                table_values = np.zeros((table_entries,0))
            numpy_tables[table]["value"] = table_values
            numpy_tables[table]["value"] = np.concatenate([numpy_tables[table]["value"],
                                                      np.zeros((1,numpy_tables[table]["value"].shape[1]))])
            
        if not is_test:
            self.transformers = transformers
            self.kept_labels = kept_labels
            
        return numpy_tables

    
    
    
    
    
    
    
    
    
    

class batch_generator(tf.keras.utils.Sequence):
    def __init__(self, x_set, y_set,batch_size=32):
        'Initialization'
        self.x, self.y = x_set, y_set
        self.batch_size = batch_size
        
        self._gen_batches()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return self.num_batches

    def __getitem__(self, idx):
        'Generate one batch of data'
        
        return self.batches[idx][0],self.batches[idx][1]
        #,[None]
    
    def get_sizes(self):
        return self.data_shapes, self.output_size, self.target_shape
    
    def _gen_batches(self):
        self.batches = []
        self.num_batches = math.ceil(len(self.x)/self.batch_size)
        for i in range(self.num_batches):
            X = self.x[i*self.batch_size:(i+1)*self.batch_size]
            
            x_data = [[] for _ in X[0][0]]
            x_ids = [[] for _ in X[0][1]]
            id_offsets = [0 for _ in X[0][1]]
            
            for x in X:
                for c,data in enumerate(x[0]):
                    x_data[c].append(data)

                for c,ids in enumerate(x[1]):
                    if len(x_ids[c]) == 0 or x_ids[c][0].shape[0] == 0:
                        x_ids[c].append(ids)
                        if len(ids) > 0:
                            id_offsets[c] += ids[-1]+1
                    else:
                        offset = id_offsets[c]
                        ids_ = np.array([i+offset for i in ids])
                        x_ids[c].append(ids_)
                        if len(ids) > 0:
                            id_offsets[c] += ids[-1]+1
                    
            
            x_data_ = [np.concatenate(data) for data in x_data]
            x_ids_ = [np.concatenate(ids) for ids in x_ids]
            x_batch = x_data_ + x_ids_
            self.batches.append((x_batch,
                                 self.y[i*self.batch_size:(i+1)*self.batch_size]))
            
            if i == 0:
                self.data_shapes = [d.shape[1] for d in x_data_]
                self.output_size = len(x_batch)
                self.target_shape = self.y.shape[1]
                
                
                
                
                
class nrelaggs_model():
    
    def __init__(self,input_sizes,aggregation_plan,
                 predictor_layers=(100,),loss='hinge',
                 feature_generation=1.,feature_selection=1.,
                 is_regression=False,verbose=0):
        
        self.input_sizes = input_sizes
        self.aggregation_plan = aggregation_plan
        self.predictor_layers = predictor_layers
        self.loss = loss
        self.feature_generation = feature_generation
        self.feature_selection = feature_selection
        self.is_regression = is_regression
        self.verbose = verbose
        
        
        self.model = self.build_model_()
        
    def build_model_(self):
        inputs_data = [tf.keras.layers.Input(shape=(s,)) for s in self.input_sizes[0]]
        inputs_ids = [tf.keras.layers.Input(shape=(1,),dtype="int32") 
                      for _ in range(len(self.input_sizes[0]),self.input_sizes[1])]
        inputs = inputs_data + inputs_ids

        inputs_ids = [tf.reshape(ids_in,[-1]) for ids_in in inputs_ids]

        current_shapes = [s for s in self.input_sizes[0]]

        entry_num = 0
        for entry in self.aggregation_plan:
            entry_num += 1
            to_concat = []
            new_shape = 0
            for agg_candidate in entry[0]:
                agg_shape = int(current_shapes[agg_candidate]*self.feature_generation)
                feature_gen = tf.keras.layers.Dense(agg_shape, 
                                                    activation='relu', 
                                                    name='features_{}_{}'.format(entry_num,agg_candidate))(inputs_data[agg_candidate])

                aggregate_sum = tf.math.segment_sum(feature_gen,inputs_ids[agg_candidate],
                                                    name='sum_{}_{}'.format(entry_num,agg_candidate))
                aggregate_mean = tf.math.segment_mean(feature_gen,inputs_ids[agg_candidate],
                                                    name='mean_{}_{}'.format(entry_num,agg_candidate))
                aggregate_min = tf.math.segment_min(feature_gen,inputs_ids[agg_candidate],
                                                    name='min_{}_{}'.format(entry_num,agg_candidate))
                aggregate_max = tf.math.segment_max(feature_gen,inputs_ids[agg_candidate],
                                                    name='max_{}_{}'.format(entry_num,agg_candidate))

                aggregate_total = tf.concat([aggregate_sum,aggregate_mean,aggregate_min,aggregate_max],1,
                                            name='concat_{}_{}'.format(entry_num,agg_candidate))
                agg_shape = int(agg_shape * 4 * self.feature_selection)
                feature_select = tf.keras.layers.Dense(agg_shape, 
                                                       activation='relu',
                                                       name='select_{}_{}'.format(entry_num,agg_candidate))(aggregate_total)
                to_concat.append(feature_select)
                new_shape += agg_shape

            to_concat.append(inputs_data[entry[1]])
            new_shape += current_shapes[entry[1]]

            inputs_data[entry[2]] = tf.concat(to_concat,1,
                                              name='concat-all_{}'.format(entry_num))
            current_shapes[entry[2]] = new_shape


      
        
        prdeictor_layers = [inputs_data[0]]
        for predictor_size in self.predictor_layers:
            prdeictor_layers.append(tf.keras.layers.Dense(predictor_size, activation='relu')(prdeictor_layers[-1]))
        x_out = tf.keras.layers.Dense(self.input_sizes[2])(prdeictor_layers[-1])

        model = tf.keras.Model(inputs=inputs, outputs=x_out)
        
        if self.is_regression:
            model.compile(optimizer='Adam', loss=self.loss, metrics=['mean_squared_error'])
        else:
            model.compile(optimizer='Adam', loss=self.loss, metrics=['acc'])
        
        #embedding = tf.keras.Model(inputs=model.inputs,
        #                           outputs=model.layers[-1 * (len(self.predictor_layers)+1)])

        return model
    
    def train_model(self,data,epochs=100,early_stopping=True,patience=5):
        callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=patience,restore_best_weights=True)
        if early_stopping:
            callbacks = [callback]
        else:
            callbacks = []
            
        self.model.fit(data,epochs=epochs,callbacks=callbacks,verbose=self.verbose)
        
    def evaluate_model(self,data,y_true):
        test_loss,accuracy = self.model.evaluate(data,verbose=self.verbose)
        y_pred = self.model.predict(data)
        if self.input_sizes[2] > 1:
            auroc = []
            auroc.append(roc_auc_score(y_true, y_pred, multi_class='ovo', average='macro'))
            auroc.append(roc_auc_score(y_true, y_pred, multi_class='ovo', average='weighted'))
            auroc.append(roc_auc_score(y_true, y_pred, multi_class='ovr', average='macro'))
            auroc.append(roc_auc_score(y_true, y_pred, multi_class='ovr', average='weighted'))
        else:
            auroc = roc_auc_score(y_true, y_pred)
        return accuracy, auroc

    def predict(self,data):
        y_pred = self.model.predict(data,verbose=self.verbose)
        return y_pred
    
    def get_embedding(self,data):
        embedding = tf.keras.Model(inputs=self.model.inputs,
                                   outputs=self.model.layers[-1 * (len(self.predictor_layers)+3)].output)
        x_emb = embedding.predict(data)
        return x_emb
    
    def evaluate_regression(self,data,y_true):
        test_loss,mse = self.model.evaluate(data,verbose=self.verbose)
        y_pred = self.model.predict(data)
        r2 = r2_score(y_true, y_pred)
        return mse, r2
    
    
    
    
    
    
    
    
    
class relaggs_generator(tf.keras.utils.Sequence):
    def __init__(self, x_set,plan):
        'Initialization'
        self.x = x_set
        self.plan = plan
        
        self._gen_relaggs()
    
    def get_data(self):
        return self.x_data
    
    def _gen_relaggs(self):
        x_data = []
            
        for x in self.x:
                
            data_entries = [data for data in x[0]]
            for entry in self.plan:
                for to_agg in entry[0]:
                    aggregates = []
                    for j in range(np.max(x[1][to_agg])+1):
                        agg_base = data_entries[to_agg][np.where(x[1][to_agg] == j)]
                        cur_aggregates = []
                        cur_aggregates.append(np.mean(agg_base,axis=0)[np.newaxis])
                        cur_aggregates.append(np.sum(agg_base,axis=0)[np.newaxis])
                        cur_aggregates.append(np.max(agg_base,axis=0)[np.newaxis])
                        cur_aggregates.append(np.min(agg_base,axis=0)[np.newaxis])
                        cur_aggregates.append(np.std(agg_base,axis=0)[np.newaxis])
                            
                        aggregates.append(np.concatenate(cur_aggregates,axis=1))
                        
                    aggregates = np.concatenate(aggregates)
                    data_entries[entry[2]] = np.concatenate([data_entries[entry[1]],aggregates],axis=1)
                        
            x_data.append(data_entries[0])
                
        x_data = np.concatenate(x_data)
        self.x_data = x_data
    
    
    
class seq_generator(tf.keras.utils.Sequence):
    def __init__(self, x_set, y_set,batch_size=32):
        'Initialization'
        self.x, self.y = x_set, y_set
        self.batch_size = batch_size
        
        self._gen_batches()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return self.num_batches

    def __getitem__(self, idx):
        'Generate one batch of data'
        
        return self.batches[idx][0],self.batches[idx][1]
        #,[None]
    
    def get_sizes(self):
        return self.data_shapes, self.target_shape
    
    def _gen_batches(self):
        self.batches = []
        self.num_batches = math.ceil(len(self.x)/self.batch_size)
        for i in range(self.num_batches):
            self.batches.append((self.x[i*self.batch_size:(i+1)*self.batch_size],
                                 self.y[i*self.batch_size:(i+1)*self.batch_size]))
            
            if i == 0:
                self.data_shapes = self.x.shape[1]
                self.target_shape = self.y.shape[1]
    
    
    
    
    
    
    
class relaggs_model():
    
    def __init__(self,input_sizes,predictor_layers=(100,),loss='hinge',verbose=0,is_regression=False):
        
        self.input_sizes = input_sizes
        self.predictor_layers = predictor_layers
        self.loss = loss
        self.verbose = verbose
        self.is_regression = is_regression
        
        
        self.model = self.build_model_()
        
    def build_model_(self):
        
        inputs = [tf.keras.layers.Input(shape=(self.input_sizes[0],))]
        
        predictor_layers = [inputs[0]]
        for predictor_size in self.predictor_layers:
            predictor_layers.append(tf.keras.layers.Dense(predictor_size, activation='relu')(predictor_layers[-1]))
        x_out = tf.keras.layers.Dense(self.input_sizes[1])(predictor_layers[-1])

        model = tf.keras.Model(inputs=inputs, outputs=x_out)
        if self.is_regression:
            model.compile(optimizer='Adam', loss=self.loss, metrics=['mean_squared_error'])
        else:
            model.compile(optimizer='Adam', loss=self.loss, metrics=['acc'])

        return model
    
    def train_model(self,data,epochs=100,early_stopping=True,patience=5):
        callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=patience,restore_best_weights=True)
        if early_stopping:
            callbacks = [callback]
        else:
            callbacks = []
            
        self.model.fit(data,epochs=epochs,callbacks=callbacks,verbose=self.verbose)

    def evaluate_model(self,data,y_true):
        test_loss,accuracy = self.model.evaluate(data,verbose=self.verbose)
        y_pred = self.model.predict(data,verbose=self.verbose)
        if self.input_sizes[1] > 1:
            auroc = []
            auroc.append(roc_auc_score(y_true, y_pred, multi_class='ovo', average='macro'))
            auroc.append(roc_auc_score(y_true, y_pred, multi_class='ovo', average='weighted'))
            auroc.append(roc_auc_score(y_true, y_pred, multi_class='ovr', average='macro'))
            auroc.append(roc_auc_score(y_true, y_pred, multi_class='ovr', average='weighted'))
        else:
            auroc = roc_auc_score(y_true, y_pred)
        return accuracy, auroc

    def predict(self,data):
        y_pred = self.model.predict(data,verbose=self.verbose)
        return y_pred
    
    def evaluate_regression(self,data,y_true):
        test_loss,mse = self.model.evaluate(data,verbose=self.verbose)
        y_pred = self.model.predict(data)
        r2 = r2_score(y_true, y_pred)
        return mse, r2
    
    
    
    
    
    
    
class context_converter_big():
    
    def __init__(self, context, save_folder=None, limits=None, max_unique=20, stop_tables=[], save_step=100, verbose=0,numpy_folder=None):
        self.context = context
        self.max_unique = max_unique
        self.verbose = verbose
        self.stop_tables = stop_tables
        self.save_folder = save_folder
        self.save_step = save_step
        self.limits = limits
        self.numpy_folder = numpy_folder
        
        if not self.save_folder is None:
            if not os.path.exists(self.save_folder):
                if self.verbose > 0:
                    print("Generated folder: {}".format(self.save_folder))
                os.makedirs(self.save_folder)
            else:
                if self.verbose > 0:
                    print("Already there: {}".format(self.save_folder))
        
        self.build_structure_()
        
    def get_data(self):
        return self.data[0], self.data[1]
    
    def get_plan(self):
        return self.plan
    
    def get_time(self):
        return self.run_time
        
    def build_structure_(self):
        self.tables = self.preprocess_tables_()
        if self.verbose > 0:
            print("tables done")
        
        self.data, self.plan, self.run_time = self.gen_data_()
        
    
    #generate aggregation plan
    def generate_agg_plan_(self):
        if not self.save_folder is None:
            plan_path = "{}/agg_plan.p".format(self.save_folder)
            if os.path.isfile(plan_path) and os.access(plan_path, os.R_OK):
                agg_plan = pickle.load(open(plan_path,"rb"))
                return agg_plan[0],agg_plan[1]

        data_tables = [self.context.target_table]

        connections = list(self.context.connected.keys())

        def rec_walk(cur,vis):
            out = ([],cur,cur)
            other_out = []
            if cur[0] in self.stop_tables:
                return []
            for con in connections:
                old,new = con
                if old == cur[0]:
                    if new not in vis:
                        len_new = len(data_tables)
                        data_tables.append(new)
                        out[0].append((new,len_new))
                        other_out += rec_walk((new,len_new),vis+[new])

            out = [out]+other_out
            if len(out[0][0]) == 0:
                out = []
            return out
        plan = rec_walk((self.context.target_table,0),[self.context.target_table])

        if not self.save_folder is None:
            agg_plan = (data_tables,plan)
            pickle.dump(agg_plan,open(plan_path,"wb"))

        return data_tables,plan
        
        
    #generate Dataset
    def gen_data_(self):
        data_tables, plan = self.generate_agg_plan_()

        outputs = ([],[])
        
        context = self.context
        numpy_tables = self.tables

        total_number = len(numpy_tables[context.target_table]["target"])
        done_number = 0
        times = []

        if not self.limits is None:
            done_number = self.limits[0]
        

        if not self.save_folder is None:
            gen_path = "{}/gen_data.p".format(self.save_folder)
            if os.path.isfile(gen_path) and os.access(gen_path, os.R_OK):
                saved_gen = pickle.load(open(gen_path,"rb"))
                done_number = saved_gen[0]
                times = saved_gen[1]
                outputs = saved_gen[2]

        


        for c,c_y in enumerate(numpy_tables[context.target_table]["target"]):
            if c < done_number:
                continue

            if not self.limits is None:
                if c >= self.limits[1]:
                    break
            start = time.time()

            outputs[1].append(c_y)
            out_data_ids = [[] for i in data_tables]
            out_data = [[] for i in data_tables]
            out_ids = [[] for i in data_tables]

            out_data_ids[0] = [c]

            for step in plan:
                cur = step[1]
                for prev in step[0]:
                    cur_id_name, prev_id_name = context.connected[(cur[0],prev[0])][0]

                    id_counter = -1
                    for cur_entry in out_data_ids[cur[1]]:
                        id_counter += 1
                        if cur_entry == -1:
                            prev_entries = [-1]
                        else:
                            cur_id = numpy_tables[cur[0]][cur_id_name][cur_entry]
                            prev_entries = list(np.where(numpy_tables[prev[0]][prev_id_name] == cur_id)[0])
                            if len(prev_entries) == 0:
                                prev_entries = [-1]
                        out_data_ids[prev[1]] += prev_entries
                        out_ids[prev[1]] += [id_counter for i in prev_entries]

            for c,data_ids in enumerate(out_data_ids):
                out_data[c] = numpy_tables[data_tables[c]]["value"][data_ids]
                out_ids[c] = np.array(out_ids[c])

            outputs[0].append( (out_data,out_ids) )

            done_number += 1
            end = time.time()
            times.append(end - start)
            if self.verbose > 0:
                if not self.limits is None:
                    done_print = done_number - self.limits[0]
                    total_print = self.limits[1] - self.limits[0]
                    print("data: {}/{} in {:6.5f}s/entry approximated time left: {:6.4f}h".format(done_print,total_print,np.mean(times),((total_print - done_print) * np.mean(times)) / (60*60)),end="\r")
                else:
                    print("data: {}/{} in {:6.5f}s/entry approximated time left: {:6.4f}h".format(done_number,total_number,np.mean(times),((total_number - done_number) * np.mean(times)) / (60*60)),end="\r")

            if not self.save_folder is None:
                if done_number % self.save_step == 0:
                    saved_gen = (done_number,times,outputs)
                    pickle.dump(saved_gen,open(gen_path,"wb"))

        if self.verbose > 0:
            print()
            
        plan2 = []
        for step in plan[::-1]:
            new_step = ([],step[1][1],step[2][1])
            for entry in step[0]:
                new_step[0].append(entry[1])
            plan2.append(new_step)

        outputs = (outputs[0],np.array(outputs[1]))
        plan= plan2
        return outputs,plan,np.sum(times)
    
        
        
    #preprocess values
    def preprocess_tables_(self):
        if not self.save_folder is None:
            if self.numpy_folder is None:
                table_path = "{}/numpy_tables.p".format(self.save_folder)
            else:
                table_path = "{}/numpy_tables.p".format(self.numpy_folder)
            if os.path.isfile(table_path) and os.access(table_path, os.R_OK):
                numpy_tables = pickle.load(open(table_path,"rb"))
                return numpy_tables


        numpy_tables = dict()
        transformers = dict()
        kept_labels = dict()
        context = self.context
        
        for table in context.tables:
            
            table_entries = 0
            
            transformers[table] = dict()
            kept_labels[table] = dict()
            numpy_tables[table] = dict()

            table_ids = set([context.pkeys[table]])
            if table in context.fkeys:
                table_ids = table_ids.union(context.fkeys[table])

            table_values = []

            arrs = {col: [] for col in context.cols[table]}
            for x in context.orng_tables[table]:
                for dom in arrs:
                    arrs[dom].append(x[dom].value)

            for dom in arrs:
                arrs[dom] = np.array(arrs[dom])[:,np.newaxis]
                
                if dom in table_ids:
                    numpy_tables[table][dom] = arrs[dom].astype('U')
                    table_entries = arrs[dom].shape[0]
                
                elif not np.issubdtype(arrs[dom].dtype, np.number):
                    if len(np.unique(arrs[dom])) > self.max_unique:
                        uniques,counts = np.unique(arrs[dom],return_counts=True)
                        keeps = np.flip(uniques[np.argsort(counts)])[:self.max_unique]
                        kept_labels[table][dom] = keeps
                        arrs[dom][np.isin(arrs[dom],keeps,invert=True)] = 'other'

                    cur_trans = LabelBinarizer()
                    cur_trans.fit(arrs[dom])
                    transformers[table][dom] = cur_trans
                    arrs[dom] = cur_trans.transform(arrs[dom])
                    if table == context.target_table and dom == context.target_att:
                        numpy_tables[table]["target"] = arrs[dom]
                    else:
                        table_values.append(arrs[dom])
                else:
                    cur_trans = MinMaxScaler()
                    cur_trans.fit(arrs[dom])
                    transformers[table][dom] = cur_trans
                    arrs[dom] = cur_trans.transform(arrs[dom])
                    if table == context.target_table and dom == context.target_att:
                        numpy_tables[table]["target"] = arrs[dom]
                    else:
                        table_values.append(arrs[dom])

            if len(table_values) >= 1:
                table_values = np.concatenate(table_values,axis=1)
            else:
                table_values = np.zeros((table_entries,0))
            numpy_tables[table]["value"] = table_values
            numpy_tables[table]["value"] = np.concatenate([numpy_tables[table]["value"],
                                                      np.zeros((1,numpy_tables[table]["value"].shape[1]))])
            
        #self.transformers = transformers
        #self.kept_labels = kept_labels
        if not self.save_folder is None:
            pickle.dump(numpy_tables,open(table_path,"wb"))
            
        return numpy_tables

    
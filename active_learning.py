import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import pairwise_distances, accuracy_score, f1_score 


class ActiveLearner(object):
    
    def __init__(self, X_initial, y_initial, model=None):
        
        self.X_initial = X_initial
        self.y_initial = y_initial
        
        # Choose default model and communicate to user if no model instance is supplied
        if model is None:
            self.model = GaussianNB()
            print(f"Using {self.model} by default") 
        else:
            self.model = model
        
        # Check if model supports incremental learning using 'partail_fit' callable via sci-kit learn API
        if 'partial_fit' not in dir(self.model):
            raise AttributeError(f"Model {self.model} does not support incremental learning and thus cannot be used in active learning settings.")
            
        # Learn on initial instances supplied by user
        self.model = self.model.partial_fit(X_initial, y_initial, classes=np.unique(y_initial))
        self.hyperparameters = {}
        
    def __extract_hyperparamters(self, strategy, kwargs):
        if strategy == 'regroup':
            self.hyperparameters['v'] = kwargs.get('unlearnable_group_discount_factor', 0.9)
            
        elif strategy == 'information_density':
            print(kwargs)
            self.hyperparameters['beta'] = kwargs.get('relative_density_importance', 1)
            
    def initiate_interactive_training(self, X, y, strategy, X_benchmark=None, y_benchmark=None, **kwargs):
        
        f1_hist, acc_hist = [], []
        n = 0
        
        # Extract hyperparameters used for regroup query strategy
        self.__extract_hyperparamters(strategy, kwargs)
        
        if strategy == 'information_density':
            dist_out = pairwise_distances(X, metric="canberra", n_jobs = -1)
            avg_similarity_vector = (dist_out.sum(axis = 1) / len(X))
            avg_disimilarity_vector = 1.0 / avg_similarity_vector
        
        while len(X) > 0:
               
            # Predict using the current model
            pred_prob_vector = self.model.predict_proba(X)[:,1] #We take probability output for class '1'
            
            # Strategy implemented is ReGroup
            if strategy == 'regroup':
                # Discounting for 'unlearnable' group based on the count 'n' they have been not-selected by user
                pred_prob_vector *= (self.hyperparameters['v'] ** n)
                
                # In ReGroup: the optimal query is the one with positive true label (picked by user in interactive setting) with maximum probability
                optimal_query = pred_prob_vector.argmax()
                
            
            # Strategy Implemented is Uncertainty Based Query Strategy (UBQS)
            elif strategy == 'UBQS':
                ''' In UBQS, we pick the query instance model is MOST uncertain about.
                    Uncertainty can be defined or quantified in multiple ways.
                    In this implementation, we use entropy based definition.
                    Therefore, for binary classification task, we pick the instance which is closest to 0.5 predicted probability '''
                
                optimal_query = np.abs(pred_prob_vector - 0.5).argmin()
                
            elif strategy == 'information_density':
                optimal_query = (np.abs(pred_prob_vector - 0.5) * (avg_disimilarity_vector ** self.hyperparameters['beta'])).argmin()
                
            elif strategy == 'passive_learning':
                if n == 0:
                    print("This strategy queries instaces at random and is used to benchmark other strategies")
                optimal_query = np.random.choice(range(len(pred_prob_vector)))
                
            else:
                supported_startegies = ['regroup: ReGroup (http://aiweb.cs.washington.edu/ai/pubs/amershiCHI2012_ReGroup.pdf)', 
                                        'UBQS: Uncertainty Based Query Strategy (entropy based)', 
                                        'information_density: Information density with UBQS'
                                       ]
                
                raise AttributeError(f"Only following query strategies supported currently \n{', '.join(supported_startegies)}")
            
            # Extract the optimal query 
            X_optimal = X[optimal_query, :].reshape(1,-1)
            y_optimal = y[optimal_query].reshape(1)

            # Delete the extracted optimal query to use X, y dynamically within while loop
            X = np.delete(X, optimal_query, axis = 0).reshape(-1, X.shape[1])
            y = np.delete(y, optimal_query, axis = 0)
            
            if strategy == 'information_density':
                avg_disimilarity_vector = np.delete(avg_disimilarity_vector, optimal_query, axis = 0)


            # Train over the optimal X and y based on selected query strategy
            self.model = self.model.partial_fit(X_optimal, y_optimal, classes=np.unique(self.y_initial))

            n += 1
            
            f1_hist.append(f1_score(self.model.predict(X_benchmark), y_benchmark))
            acc_hist.append(accuracy_score(self.model.predict(X_benchmark), y_benchmark))
        
        return f1_hist, acc_hist
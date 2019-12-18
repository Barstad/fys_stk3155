import numpy as np

class Node:
    def __init__(self, prediction):
        self.prediction = prediction
        self.parent = None
        self.feature_index = 0
        self.threshold = 0
        self.left = None
        self.right = None
        self.is_leaf = False
        self.impurity = np.inf

class DecisionTreeRegression:
    def __init__(self, max_depth=None, use_features = None):
        """
        max_depth : int
            max depth of tree
        
        use_features:
            features to use on each split decision
            supported : 
                'sqrt' - select int(sqrt(m)) number of features
                where m is X.shape[1]
        
        """
        if use_features is not None:
            if use_features != 'sqrt':
                raise ValueError("Supported input to use_features is 'sqrt'")
        
        self.use_features =  use_features
        self.max_depth = max_depth
        self.nodes = []

    def fit(self, X, y):
        self.n_features_ = X.shape[1]
        self.tree_ = self._grow_tree(X, y)

    def predict(self, X):
        return [self._predict(inputs) for inputs in X]
    
    def _std_dev(self, count, sum_, sum_of_squared):
        """
        Standard deviation calculation. Mean of square minus square of mean.

        """ 
        val = np.sqrt((sum_of_squared/count) - (sum_/count)**2)
        return val

    def _best_split(self, X, y):
        # Num samples
        m = y.size
        # Leaf node
        if m <= 1:
            return None, None, None
        
        # initial variance
        best_var = np.inf
        best_idx, best_thr = None, None
        
        # Selecting random set of features to include in the split search
        if self.use_features == 'sqrt':
            sample_size = int(np.sqrt(self.n_features_))
            candidates = set(np.random.choice(self.n_features_, sample_size, False).tolist())
        else:
            candidates = set([i for i in range(self.n_features_)])
        
        # Loop over features
        # for idx in range(self.n_features_):
        #     if idx not in candidates:
        #         continue
        for idx in candidates:
            
            # select data for feature and sort label and feature by feature values
            feature = X[:, idx]
            sort_index = np.argsort(feature)
            feature = feature[sort_index]
            label = y[sort_index]
            
            # Initial values for variance calculation. Allowing incremental update, 
            # to reduce algoritm complexity.
            rhs_cnt, rhs_sum, rhs_sum_squared = m, label.sum(), (label**2).sum()
            lhs_cnt, lhs_sum, lhs_sum_squared = 0, 0.0, 0.0
            
            # Loop through samples
            for i in range(1, m):
                
                # current label
                c = label[i-1]
                
                # increment counts
                lhs_cnt += 1
                rhs_cnt -= 1
                # increment sums
                lhs_sum += c
                rhs_sum -= c
                lhs_sum_squared += c**2
                rhs_sum_squared -= c**2
                
                # Calculate var for left and right split
                var_left = self._std_dev(lhs_cnt, lhs_sum, lhs_sum_squared)
                var_right = self._std_dev(rhs_cnt, rhs_sum, rhs_sum_squared)
                                
                # Calculate weighted var
                w_var = var_left*lhs_cnt + var_right*rhs_cnt
                
                # Make sure that repeating values are kept to one side of the split
                if feature[i] == feature[i - 1]:
                    continue

                # Update best split
                if w_var<best_var:
                    best_var = w_var
                    best_idx = idx
                    best_thr = (feature[i] + feature[i-1]) / 2
                    # best_thr = feature[i-1]
                    variances = (var_left, var_right)
                    counts = (lhs_cnt, rhs_cnt)
            
        return best_idx, best_thr, best_var

    def _grow_tree(self, X, y, parent = None, depth=0):
        # Predict average of the group   
        prediction = np.mean(y)
        # Create node with prediction value
        node = Node(prediction=prediction)
        # Set parent on nodes after root
        if depth > 0:
            node.parent = parent
        
#         if depth < self.max_depth and node.parent.is_leaf == False:
        if depth < self.max_depth:
            idx, thr, best_var = self._best_split(X, y)
            if idx is not None:
                # Split on the criterion from best_split
                indices_left = X[:, idx] < thr
                X_left, y_left = X[indices_left], y[indices_left]
                X_right, y_right = X[~indices_left], y[~indices_left]
                node.feature_index = idx
                node.threshold = thr
                node.impurity = best_var
                # Impurity threshold
                if depth > 0:
                    # Check for improvement in impurity
                    if node.impurity > node.parent.impurity:
                        # if no improvement, set parent as leaf
                        node.parent.is_leaf = True
                        return node.parent

                # Recursively create branches on each node, increment depth
                node.left = self._grow_tree(X_left, y_left, node, depth + 1)
                node.right = self._grow_tree(X_right, y_right, node, depth + 1)
                
                #Link to parent
                node.left.parent = node
                node.right.parent = node
                node.depth = depth
        else:
            node.is_leaf = True

        return node

    def _predict(self, inputs):
        # Single instance prediction
        node = self.tree_
        while node.left:
            if inputs[node.feature_index] < node.threshold:
                node = node.left
            else:
                node = node.right
        return node.prediction
    
    
class RandomForest():
    def __init__(self, n_trees, tree_depth=10):
        """
        n_trees : int
            number of trees to make up the forest
        tree_depth : int
            depth of each tree
        """
        np.random.seed(12)
        self.n_trees = n_trees
        self.tree_depth = tree_depth
        
    def fit(self, X, y):
        # Fitting n_trees number of trees
        self.trees = [self._create_tree(X, y) for i in range(self.n_trees)]

    def _create_tree(self, X, y):
        # bootstrap dataset
        idxs = np.random.choice(len(y), len(y), True)
        # Create decision tree that uses a random subset of features at each split
        tree = DecisionTreeRegression(max_depth = self.tree_depth, use_features = 'sqrt')
        tree.fit(X[idxs], y[idxs])
        return tree
        
    def predict(self, x):
        # Precition as average of the individual tree predictions
        return np.mean([t.predict(x) for t in self.trees], axis=0)
    
    
class GBM():
    def __init__(self, n_estimators, learning_rate = 0.1, tree_depth = 5):
        """
        Gradient boosting algoritm
            Implementation of gradient boosting based on mean squared error loss,
            using decision trees.
        
        n_estimators : int
            Number of trees to fit recursively
        
        learning rate : float
            Size of gradient steps
        
        tree_depth : int
            depth of each tree
        
        
        """
        self.n_estimators = n_estimators
        self.tree_depth = tree_depth
        self.learning_rate = learning_rate
        
    def fit(self, X, y):
        # Store trees
        self.trees = []
        # Initial estimate
        self.y_hat = np.array([y.mean()]*len(y))
        self.actual = y
        self.counter = 0
        # fit trees
        self._create_tree(X)
        
    def predict(self, X):
        # Base line
        pred = self.actual.mean()
        # Predict the residuals to update the baseline prediction
        for tree in self.trees:
            pred = pred - np.array(tree.predict(X)) * self.learning_rate
        return pred
        
    def _create_tree(self, X):
        # Keep track of number of trees
        if self.counter >= self.n_estimators:
            return
        self.counter+= 1
        
        residual = self.y_hat - self.actual
        tree = DecisionTreeRegression(max_depth = self.tree_depth, use_features = None)
        # Fit tree to residual
        tree.fit(X, residual)
        self.trees.append(tree)
        pred = np.array(tree.predict(X))
        # Update global prediction with residual prediction
        self.y_hat -= pred * self.learning_rate
        # Recursively update
        self._create_tree(X)
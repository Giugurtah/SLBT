import numpy as np
from .base import BaseSLBT
from .base import Node

from ._utils.criteria import _impurity, _gpi, _get_sizes
from ._tree.split import score
from ._utils.utils import _contingency_matrix, _stratified_contingency
from ._tree._homogeneity.base import get_homogeneity_strategy
from .reporting import TreeReporter

class SLBT(BaseSLBT):
    #*------Public methods---------
    #Method to fit the SLBT model
    def fit(self, X, y, x_s = None):
        print("---------------------------------------------------------------------------------")
        print("---------------------------------------------------------------------------------")
        print("---------------------------------------------------------------------------------")
        if x_s is None:
            x_s = np.zeros(len(y), dtype=int)
            strategy = get_homogeneity_strategy("AB")
            self.homogeneity = "AB"
        else:
            strategy = get_homogeneity_strategy(self.homogeneity)

        self.targhet_dist = [np.unique(y), np.unique(y, return_counts=True)[1]/len(y)]

        self.root_N = len(y)

        # Reporter inizialization
        self.reporter = TreeReporter(homogeneity=self.homogeneity, decimals=4)

        # Tree growth
        self.root = self._grow_tree(strategy, X, y, x_s)

        # Tree pruning
        #self.root, changed = self._prune_tree(self.root)

        self._calculate_tree_partial_impurity_reduction()

        # Reporter updating
        #if changed and self.reporter is not None:
        #    self._rebuild_report()

    def prune_after_vp(self, nodeID):
        if self.root is None:
            return
        
        # 1. Raccogli tutti i nodi in una lista
        all_nodes = []
        self._collect_nodes(self.root, all_nodes)
        
        # 2. Ordina per impurity_decrease crescente
        all_nodes.sort(key=lambda n: n.impurity_decrease)
        
        max_iterations = len(all_nodes) * 2  # Limite per evitare loop infiniti
        iterations = 0
        changed = True

        while changed and iterations < max_iterations:
            changed = False
            iterations += 1
            for i in range(len(all_nodes)-1):
                current_node = all_nodes[i]
                next_node = all_nodes[i+1]
                if(current_node.impurity_decrease > next_node.impurity_decrease):
                    all_nodes[i] = next_node
                    all_nodes[i+1] = current_node
                    changed = True

        # 3. Cerca nodo di taglio
        search = True

        virtual_leaves = [self.root]
        virtual_leaves_set = {self.root}

        reached = False
        i = 0

        while reached is False:
            current_node = all_nodes[i]

            if current_node.position != nodeID: 
                i += 1
                if current_node in virtual_leaves_set and current_node._is_leaf_node() is False:
                        virtual_leaves.remove(current_node)
                        virtual_leaves_set.remove(current_node)

                        virtual_leaves.append(current_node.left)
                        virtual_leaves_set.add(current_node.left)
                            
                        virtual_leaves.append(current_node.right)
                        virtual_leaves_set.add(current_node.right)
            else:
                reached = True

        # 4. Pota i rami in eccesso
        for node in virtual_leaves:
            if node._is_leaf_node() is False:
                node.feature = None
                node.treshold = None
                node.left = None
                node.right = None
                node.LIFT_1 = None
                node.LIFT_2 = None
                node.gpi = None
                node.ppi = None

                highest_index = 0
                highest_presence = 0
                for i in range(len(node.distribution)):
                    if node.distribution[i] > highest_presence:
                        highest_presence = node.distribution[i]
                        highest_index = i

                node.value = node.labels[highest_index]
                node.GCR = self._get_gcr(node.distribution, node.labels)

        self._rebuild_report()
        return

    #Method to predict the labels of a given dataset
    def predict(self, X):
        predictions = []

        for i in range(len(X)):
            print("Riga ispezionata:", i)
            predictions.append(self._traverse_tree(X.iloc[i], self.root))
                
        return np.array(predictions)

    #*------Private methods---------
    #Growing functions
    def _grow_tree(self, strategy, X, y, x_s=None, root_impurity=1, root_feats=1, depth=0, pos=1):
        print("BEGINNING NODE EVALUATION AT DEPTH:", depth, " ID:", pos) #TODO da cancellare
        
        #*Preprocessing of the dataset
        X = self._drop_constant_columns(X)
        n_samples, n_feats, n_labels, impurity, distribution = _get_sizes(X, y)
        print("The subset currently contains", n_samples, "samples,", n_feats, "feats and", n_labels, "labels") #TODO da cancellare

        if(root_feats==1):
            root_feats = len(y)
            root_impurity = impurity
            impurity_decrease = 0
            tree_partial_impurity_reduction = 0
        else:
            impurity_decrease = (root_impurity - impurity*len(y)/root_feats)/root_impurity
            tree_partial_impurity_reduction = 0

        #Check the stopping criteria before the best split search
        node = self._check_criteria_before(y, pos, impurity, distribution, depth, n_labels, n_samples, n_feats, impurity_decrease, tree_partial_impurity_reduction)
        if(node is not None):
            return node

        #Evaluation of gpi for all features
        gpi, gpi_i = _gpi(X, y, x_s) 


        # Find the best predictor for the current node  
        best_feature, best_treshold, best_ppi, best_gpi, alpha, beta = self._find_best_predictor(X, y, x_s, gpi_i, gpi)
        thresholds = strategy.get_treshold_values(best_treshold, np.unique(X[best_feature]), x_s)

        #Check the stopping criteria after the best split search
        node = self._check_criteria_after(y, pos, impurity, distribution, depth, best_gpi, best_ppi, impurity_decrease, tree_partial_impurity_reduction)
        if(node is not None):
            return node
        
        #Split the dataset 
        indexL, indexR = strategy.split(X[best_feature], x_s, thresholds)
        
        #Calculate LIFT values
        lift1, lift2 = strategy.compute_lift(beta, distribution)
        
        print("best_treshold:", thresholds) #TODO da cancellare
        print("LIFT_1:", lift1) #TODO da cancellare
        print("LIFT_2:", lift2) #TODO da cancellare
        print("\n")

        # Create the child nodes
        left = self._grow_tree(strategy, X.loc[indexL, :], y[indexL], x_s[indexL], root_impurity, root_feats, depth+1, 2*pos)
        right = self._grow_tree(strategy, X.loc[indexR, :], y[indexR], x_s[indexR], root_impurity, root_feats, depth+1, 2*pos+1)

        # Create the current node
        node = Node(
            gpi=best_gpi,
            ppi=best_ppi,
            position=pos,
            feature=best_feature,
            treshold=thresholds,
            left=left,
            right=right,
            impurity=impurity,
            impurity_decrease=impurity_decrease,
            tree_partial_impurity_reduction=tree_partial_impurity_reduction,
            distribution=distribution,
            N=len(y),
            labels=np.unique(y),
            LIFT_1=lift1,
            LIFT_2=lift2,
            GCR=None,
            strat_labels=np.unique(x_s) if x_s is not None else None
        )

        # Add the current node to the reporter
        if getattr(self, "reporter", None) is not None:
            self.reporter.add_node(node, is_leaf=False)

        # Return the current node
        return node

    def _check_criteria_before(self, y, pos, impurity, distribution, depth, n_labels, n_samples, n_feats, impurity_decrease, tree_partial_impurity_reduction):
        if( depth>=self.max_depth or 
            n_labels==1 or 
            n_samples<self.min_samples_split or 
            n_feats==0 or 
            impurity<self.min_impurity
            ):
            
            # Create a leaf node
            leaf_value = y.mode()[0]
            gcr = self._get_gcr(distribution, np.unique(y))
        
            print("(Before) Leaf Node. Target associated value:", leaf_value) #TODO da cancellare
            print("\n")

            leaf_node = Node(
                position=pos,
                value=leaf_value,
                impurity=impurity,
                distribution=distribution,
                impurity_decrease=impurity_decrease,
                tree_partial_impurity_reduction=tree_partial_impurity_reduction,
                N=len(y),
                labels=np.unique(y),
                GCR=gcr,
            )

            if getattr(self, "reporter", None) is not None:
                self.reporter.add_node(leaf_node, is_leaf=True)

            return leaf_node
        return None
    def _check_criteria_after(self, y, pos, impurity, distribution, depth, best_gpi, best_ppi, impurity_decrease, tree_partial_impurity_reduction):
        if( best_gpi<self.min_gpi or 
            best_ppi<self.min_ppi or 
            best_ppi == 0):

            # Create a leaf node
            leaf_value = y.mode()[0]
            gcr = self._get_gcr(distribution, np.unique(y))
            
            print("(After) Leaf Node reached. Target associated value:", leaf_value) #TODO da cancellare
            print("\n")

            leaf_node = Node(
                position=pos,
                value=leaf_value,
                impurity=impurity,
                impurity_decrease=impurity_decrease,
                tree_partial_impurity_reduction=tree_partial_impurity_reduction,
                distribution=distribution,
                N=len(y),
                labels=np.unique(y),
                GCR=gcr,
            )

            if getattr(self, "reporter", None) is not None:
                self.reporter.add_node(leaf_node, is_leaf=True)

            return leaf_node
        return None

    def _drop_constant_columns(self, X):
        # Function to drop constant columns from the dataset
        nunique = X.nunique(dropna=False)
        cols_to_drop = nunique[nunique <= 1].index
        return X.drop(columns=cols_to_drop)

    #*Function to find the best predictor for the current node
    def _find_best_predictor(self, X, y, x_s, gpi_order, gpi_vals):
        # Initialization
        best = {
            "feature": None,
            "threshold": None,
            "ppi": -np.inf,
            "gpi": -np.inf,
            "alpha": None,
            "beta": None,
        }

        for pos, i in enumerate(gpi_order):
            # Current feature information
            current_feature = X[i]
            n_mod = len(np.unique(current_feature))
            print("Calcolo best split sulla variabile ", i)

            # Building of contingency matrix F and call to the model  
            Fs_noN = _stratified_contingency(current_feature, y, x_s, norm=False)
            Fs = _stratified_contingency(current_feature, y, x_s, norm=True)

            ppi, S, alpha, beta = score(Fs_noN, Fs, self.homogeneity)
            print("best ppi: ", best["ppi"], " current ppi: ", ppi )
            # Update best split if necessary
            if ppi > best["ppi"]:
                best["ppi"] = ppi
                best["feature"] = str(current_feature.name)
                best["threshold"] = S
                best["alpha"] = alpha
                best["beta"] = beta
                best["gpi"] = gpi_vals[pos]

        print("Best split found on feature:", best["feature"]) #TODO da cancellare
        print("With threshold:\n", best["threshold"]) #TODO da cancellare
        print("With PPI value:", best["ppi"]) #TODO da cancellare
        print("With GPI value:", best["gpi"]) #TODO da cancellare
        print("Alpha:\n", best["alpha"]) #TODO da cancellare
        print("Beta:\n", best["beta"]) #TODO da cancellare
        return  best["feature"], best["threshold"], best["ppi"], best["gpi"], best["alpha"], best["beta"]

    #*---------Update functions---------
    def _get_gcr(self, distribution, labels):
        gcr = [0 for _ in range(len(labels))]
        for i in range(len(labels)):
            for j in range(len(self.targhet_dist[0])):
                if(self.targhet_dist[0][j] == labels[i]):
                    gcr[i] = (distribution[i]/self.targhet_dist[1][j])
        return gcr

    def _traverse_tree(self, x, node):
        if node._is_leaf_node():
            print("Giunto alla foglia. Valore=", node.value)
            return node.value
        
        if x[node.feature] in node.treshold:
            return self._traverse_tree(x, node.left)
        else:
            return self._traverse_tree(x, node.right)

    def _prune_tree(self, node: Node) -> tuple[Node, bool]:
        """
        Pruning bottom-up:
        se un nodo interno ha due figli foglia con lo stesso valore di y,
        lo si trasforma in foglia e si eliminano i figli.

        Returns
        -------
        (new_node, changed)
            new_node : Node prunato (può essere lo stesso oggetto modificato)
            changed  : True se in questo sottoalbero è stato fatto almeno un pruning
        """

        if node is None:
            return None, False

        # check if the node is a leaf
        if node._is_leaf_node():
            return node, False

        # recursive pruning
        left, changed_left = self._prune_tree(node.left)
        right, changed_right = self._prune_tree(node.right)

        node.left = left
        node.right = right

        changed_here = False

        # check if both the child nodes are leaves with the same predicted value
        if (
            node.left is not None
            and node.right is not None
            and node.left._is_leaf_node()
            and node.right._is_leaf_node()
            and node.left.value == node.right.value
        ):
            # if true the current node becomes a leaf
            node.value = node.left.value

            # the features pertaining an internal node are dropped
            node.gpi = None
            node.ppi = None
            node.feature = None
            node.treshold = None
            node.LIFT_1 = None
            node.LIFT_2 = None

            # GCR is evaluated because the current node is now a leaf
            node.GCR = self._get_gcr(node.distribution, node.labels)

            # any reference to the children nodes are dropped
            node.left = None
            node.right = None

            changed_here = True

        changed = changed_left or changed_right or changed_here
        return node, changed

    def _calculate_tree_partial_impurity_reduction(self):
        """
        Calcola tree_partial_impurity_reduction per ogni nodo.
        
        Costruisce progressivamente l'insieme delle foglie virtuali:
        - Parte con la radice come unica foglia virtuale
        - Ad ogni nodo considerato, se è foglia virtuale:
        
        * Lo rimuove dalle foglie virtuali
        * Aggiunge i suoi figli (se esistono)
        """
        if self.root is None:
            return
        
        # 1. Raccogli tutti i nodi in una lista
        all_nodes = []
        self._collect_nodes(self.root, all_nodes)
        
        # 2. Ordina per impurity_decrease crescente
        all_nodes.sort(key=lambda n: n.impurity_decrease)
        
        # 3. Verifica e correggi l'ordinamento
        # Un nodo figlio non può mai precedere il suo padre nella lista
        # perché impurity_decrease è cumulativo dalla radice
        max_iterations = len(all_nodes) * 2  # Limite per evitare loop infiniti
        iterations = 0
        changed = True

        while changed and iterations < max_iterations:
            changed = False
            iterations += 1
            for i in range(len(all_nodes)-1):
                current_node = all_nodes[i]
                next_node = all_nodes[i+1]
                if(current_node.impurity_decrease > next_node.impurity_decrease):
                    all_nodes[i] = next_node
                    all_nodes[i+1] = current_node
                    changed = True

        # 5. Variabili per i calcoli
        self.root.tree_partial_impurity_reduction = 0.0
        root_N = self.root.N
        previous_part_imp_red = 0
        search = True

        virtual_leaves = []
        virtual_leaves_set = {self.root}
        virtual_leaves_set.remove(self.root)

        virtual_leaves.append(self.root.left)
        virtual_leaves_set.add(self.root.left)
                    
        virtual_leaves.append(self.root.right)
        virtual_leaves_set.add(self.root.right)

        # 6. Processa ogni nodo in ordine
        for current_node in all_nodes[1:]:
            part_imp_red = 0
            for leaf in virtual_leaves:
                    part_imp_red += leaf.impurity_decrease*leaf.N/root_N

            if part_imp_red - previous_part_imp_red <0.01 and search is True:
                print("suggested trovato in pos: ", current_node.position)
                current_node.suggested_pruning = True
                search = False

            if current_node in virtual_leaves_set and current_node._is_leaf_node() is False:
                virtual_leaves.remove(current_node)
                virtual_leaves_set.remove(current_node)

                virtual_leaves.append(current_node.left)
                virtual_leaves_set.add(current_node.left)
                    
                virtual_leaves.append(current_node.right)
                virtual_leaves_set.add(current_node.right)

                current_node.tree_partial_impurity_reduction = part_imp_red
                previous_part_imp_red = part_imp_red
            else:
                current_node.tree_partial_impurity_reduction = part_imp_red

        for current_node in all_nodes:
            print("position: ", current_node.position, " is leaf: ", current_node._is_leaf_node(), "part_imp_red: ", current_node.tree_partial_impurity_reduction)
                
    def _collect_nodes(self, node, nodes_list):
        """
        Raccoglie tutti i nodi dell'albero in una lista (DFS).
        """
        if node is None:
            return
        
        nodes_list.append(node)
        
        if not node._is_leaf_node():
            self._collect_nodes(node.left, nodes_list)
            self._collect_nodes(node.right, nodes_list)

    def _rebuild_report(self):
        """
        Re-builds the report if the tree has been pruned
        """
        if self.reporter is None:
            return

        from .reporting import TreeReporter

        decimals = getattr(self.reporter, "decimals", 4)
        homogeneity = self.homogeneity

        new_reporter = TreeReporter(homogeneity=homogeneity, decimals=decimals)

        def _traverse(node: Node):
            if node is None:
                return
            is_leaf = node._is_leaf_node()
            new_reporter.add_node(node, is_leaf=is_leaf)
            if not is_leaf:
                _traverse(node.left)
                _traverse(node.right)

        _traverse(self.root)
        self.reporter = new_reporter
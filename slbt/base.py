import pandas as pd

#*------BaseSLBT class---------
class BaseSLBT:
    def __init__(self, 
                 min_ppi = 0.0,
                 min_gpi=0.0, 
                 min_impurity=0.0,
                 compound_feats=False, 
                 min_samples_split=1, 
                 max_depth=100,
                 feats_viewed=10, 
                 homogeneity="none",
                 ):

        #* Storing parameters
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth
        self.min_gpi = min_gpi
        self.min_ppi = min_ppi
        self.min_impurity = min_impurity

        #* Initializing attributes
        self.feats_viewed = feats_viewed
        self.homogeneity = homogeneity

        #* Dataframe to store results
        self.reporter = None

        #* For tree stucture
        self.targhet_dist = None
        self.depth = None
        self.l_c = None
        self.r_c = None
        self.root_N = None
        
        self.root=None

#*------Node class---------
class Node:
    def __init__(self, 
                 gpi=None, 
                 ppi=None, 
                 position=None, 
                 impurity=None,
                 impurity_decrease=None,
                 tree_partial_impurity_reduction=None,
                 feature=None, 
                 treshold=None, 
                 left=None, 
                 right=None,
                 LIFT_1=None,
                 LIFT_2=None,
                 GCR=None,
                 distribution=None,
                 N=None,
                 labels=None,
                 strat_labels=None,
                 suggested_pruning=None,
                 *,value=None):
        
        self.gpi = gpi
        self.ppi = ppi
        self.position = position
        self.feature = feature
        self.treshold = treshold
        self.impurity = impurity
        self.impurity_decrease = impurity_decrease
        self.tree_partial_impurity_reduction = tree_partial_impurity_reduction
        self.suggested_pruning = suggested_pruning
        self.left = left
        self.right = right
        self.value = value
        self.distribution = distribution
        self.N = N
        self.labels = labels
        self.LIFT_1 = LIFT_1
        self.LIFT_2 = LIFT_2
        self.GCR = GCR
        self.strat_labels = strat_labels

    def _is_leaf_node(self):
        return self.value is not None
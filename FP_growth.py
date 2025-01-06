from collections import Counter
from functools import reduce
from itertools import combinations
from copy import deepcopy

class TreeNode:
    def __init__(self, item_name, count=1, parent=None):
        self.item_name = item_name  
        self.count = count          
        self.parent = parent        # parent node
        self.children = {}          # child nodes {item_name: TreeNode}
        self.node_link = None       # node link to next same item node

class HeaderTable:
    def __init__(self):
        self.header_table = {}  # {item_name: [support_count, node_link]}
        self.item_order = {}    # store the order of frequent items in initial FP-tree

    def add_node_link(self, item_name, node):
        """Update node links in header_table when adding new node to FP-tree"""

        if self.header_table[item_name][1] is None:
            self.header_table[item_name][1] = node
        else:
            # traverse to the last node
            current = self.header_table[item_name][1]
            while current.node_link:
                current = current.node_link
            current.node_link = node

    def increment_support(self, item_name, count=1):
        """Initialize header_table with support counts of frequent items"""
        if item_name not in self.header_table:
            self.header_table[item_name] = [0, None]
        self.header_table[item_name][0] += count

class FPTree:
    def __init__(self):
        self.root = TreeNode("null", 0)  # create root node
        self.header_table = HeaderTable()
        
    def insert_tree(self, transaction, parent_node, count=1):
        """Insert a transaction into FP-tree"""
        if not transaction:
            return
            
        item = transaction[0]  # get first item
        
        # check if child node exists
        if item in parent_node.children:
            # if exists, increment count
            parent_node.children[item].count += count
        else:
            # create new node
            new_node = TreeNode(item, count, parent_node)
            parent_node.children[item] = new_node
            # update node link in header_table
            self.header_table.add_node_link(item, new_node)
            
        # recursively insert remaining items
        if len(transaction) > 1:
            self.insert_tree(transaction[1:], parent_node.children[item], count)
            
    def create_tree(self, transactions, min_sup):
        """
        Construct initial FP-tree with two scans of transaction database
        Note: Conditional FP-trees are built from conditional pattern bases later
        """
        # First scan: count frequent items
        # reduce merges all transactions, Counter counts frequency of each item
        item_counts = Counter(reduce(lambda x, y: x + y, transactions))
                
        # filter infrequent items and sort by support count in descending order
        frequent_items = {k: v for k, v in sorted(
            [(k,v) for k,v in item_counts.items() if v >= min_sup],
            key=lambda x: (-x[1], x[0])
        )}
        
        # store order of frequent items in initial FP-tree
        for i, item in enumerate(frequent_items.keys()):
            self.header_table.item_order[item] = i
        
        # initialize header_table with support counts
        for item, count in frequent_items.items():
            self.header_table.increment_support(item, count)
            
        # Second scan: construct FP-tree
        for transaction in transactions:
            # reorder items in transaction
            # filter infrequent items and sort by support count
            frequent_trans = [item for item in transaction if item in frequent_items]
            frequent_trans.sort(key=lambda x: (-frequent_items[x], x))
            
            # insert ordered transaction into FP-tree
            if frequent_trans:
                self.insert_tree(frequent_trans, self.root)

    def get_prefix_path(self):
        """
        Decompose FP-tree into single prefix-path P and multipath Q parts
        Returns: (list of (item_name, support), subtree with branching node as root)
        """
        prefix_items = []  # Store (item_name, support) pairs
        current = self.root
        
        # Find first branching node and collect prefix items
        while len(current.children) == 1:
            child = list(current.children.values())[0]
            prefix_items.append((child.item_name, child.count))
            current = child
        
        # If single prefix-path exists
        if prefix_items:
            # Create deep copy of tree for multipath part Q
            multipath_tree = deepcopy(self)
            
            # Set branching node as root and reset its properties
            current_multi = multipath_tree.root
            for _ in range(len(prefix_items)):
                current_multi = list(current_multi.children.values())[0]
            multipath_tree.root = current_multi
            multipath_tree.root.item_name = "null"
            multipath_tree.root.count = 0
            multipath_tree.root.parent = None

            # Update header table for multipath part Q
            prefix_item_names = {item for item, _ in prefix_items}
            multipath_tree.header_table.header_table = {
                item: info for item, info in multipath_tree.header_table.header_table.items()
                if item not in prefix_item_names
            }
            
            return prefix_items, multipath_tree
        
        # If no single prefix-path exists, return original tree
        return None, self

def mine_single_prefix_path(prefix_items):
    """
    Mine frequent patterns from single prefix-path items
    Generate all combinations of items and use branching node support as pattern support
    
    Args:
        prefix_items: list of (item_name, support) pairs from prefix path
    """
    patterns = []
    # Generate all combinations of items
    for i in range(1, len(prefix_items) + 1):
        for comb in combinations(prefix_items, i):
            # Get item names and minimum support from combination
            pattern_items = [item for item, _ in comb]
            min_support = min(support for _, support in comb)
            patterns.append((pattern_items, min_support))
    return patterns


def get_conditional_pattern_base(header_table, item):
    """
    Extract conditional pattern base for an item
    Returns patterns in format like {(fcam:2), (cb:1)}
    """
    patterns = []
    node = header_table.header_table[item][1]
    
    while node:
        # Get prefix path excluding the item itself
        prefix = []
        current = node.parent
        while current.item_name != "null":
            prefix.append(current.item_name)
            current = current.parent
            
        if prefix:
            # Store pattern as (prefix_string, support)
            # prefix is reversed to maintain correct order
            prefix.reverse()
            pattern_str = ''.join(prefix)  # Convert path to string format
            patterns.append((pattern_str, node.count))
        node = node.node_link
        
    return patterns

def construct_conditional_fptree(patterns, min_sup, original_order):
    """
    Construct conditional FP-tree from conditional pattern base
    Input patterns format: {(fcam:2), (cb:1)}
    """
    # Convert patterns to transaction format
    transactions = []
    for pattern_str, count in patterns:
        # Convert pattern string to list of items
        transaction = list(pattern_str)
        # Add same transaction multiple times based on count
        transactions.extend([transaction] * count)
    
    # Create and build conditional FP-tree
    cond_tree = FPTree()
    
    # Count item frequencies in expanded transactions
    item_count = {}
    for trans in transactions:
        for item in trans:
            item_count[item] = item_count.get(item, 0) + 1
            
    # Filter infrequent items
    frequent_items = {k: v for k, v in item_count.items() if v >= min_sup}
    
    # Inherit item order from original FP-tree
    cond_tree.header_table.item_order = original_order

    # Initialize header table
    for item, count in frequent_items.items():
        cond_tree.header_table.increment_support(item, count)

    # Insert each transaction into conditional FP-tree
    for transaction in transactions:
        # Filter infrequent items and sort by original order
        filtered_trans = [item for item in transaction if item in frequent_items]
        filtered_trans.sort(key=lambda x: original_order[x])
        if filtered_trans:
            cond_tree.insert_tree(filtered_trans, cond_tree.root, 1) 
            
    return cond_tree


def fp_growth(tree, alpha, min_sup):
    """
    FP-Growth algorithm main function
    Decompose FP-tree into single prefix-path P and multipath Q parts
    Final result: freq_pattern_set(P) ∪ freq_pattern_set(Q) ∪ (freq_pattern_set(P) × freq_pattern_set(Q))
    
    Args:
        tree: Current FP-tree being processed
        alpha: Current item prefix
        min_sup: Minimum support threshold
    """
    patterns = []
    
    # Decompose FP-tree into single prefix-path P and multipath Q
    prefix_path, multipath_tree = tree.get_prefix_path()
    
    if prefix_path:
        # 1. Process single prefix-path part P
        prefix_patterns = mine_single_prefix_path(prefix_path)
        
        # 2. Process multipath part Q (if exists)
        if multipath_tree.root.children:
            multipath_patterns = []
            header_table = multipath_tree.header_table.header_table
            # Process items in order of original FP-tree
            items = sorted(header_table.keys(), 
                         key=lambda x: -multipath_tree.header_table.item_order[x])
            
            # Build conditional pattern-base and conditional FP-tree for each item in Q
            for item in items:
                new_alpha = [item] + alpha
                support = header_table[item][0]
                multipath_patterns.append((new_alpha, support))
                
                # Build item's conditional pattern-base
                cond_patterns = get_conditional_pattern_base(multipath_tree.header_table, item)
                # Build item's conditional FP-tree
                cond_tree = construct_conditional_fptree(cond_patterns, min_sup, 
                                                       multipath_tree.header_table.item_order)
                
                # Recursively process conditional FP-tree
                if cond_tree.root.children:
                    multipath_patterns.extend(fp_growth(cond_tree, new_alpha, min_sup))
            
            # 3. Merge results from P and Q:
            # a) Add patterns from P
            patterns.extend([(p[0] + alpha, p[1]) for p in prefix_patterns])
            # b) Add patterns from Q
            patterns.extend(multipath_patterns)
            # c) Add combined patterns (P × Q)
            for p_pattern, p_support in prefix_patterns:
                for q_pattern, q_support in multipath_patterns:
                    combined_pattern = p_pattern + q_pattern
                    combined_support = min(p_support, q_support)
                    patterns.append((combined_pattern + alpha, combined_support))
        else:
            # If Q is empty, return only patterns from P
            patterns.extend([(p[0] + alpha, p[1]) for p in prefix_patterns])
            
    else:
        # If no single prefix-path exists, use standard FP-Growth processing
        header_table = tree.header_table.header_table
        items = sorted(header_table.keys(), 
                      key=lambda x: tree.header_table.item_order[x])
                      
        for item in items:
            new_alpha = [item] + alpha
            support = header_table[item][0]
            patterns.append((new_alpha, support))
            
            # Build conditional pattern-base and conditional FP-tree
            cond_patterns = get_conditional_pattern_base(tree.header_table, item)
            cond_tree = construct_conditional_fptree(cond_patterns, min_sup, 
                                                   tree.header_table.item_order)
            
            # Recursively process conditional FP-tree
            if cond_tree.root.children:
                patterns.extend(fp_growth(cond_tree, new_alpha, min_sup))
                
    return patterns

def mine_fptree(transactions, min_sup):
    """
    Main function to mine frequent patterns using FP-Growth algorithm
    
    Args:
        transactions: list of transactions
        min_sup: minimum support threshold
        
    Returns:
        list of tuples (pattern, support)
    """
    fp_tree = FPTree()
    # construct initial FP-tree with two database scans
    fp_tree.create_tree(transactions, min_sup)

    if not fp_tree.root.children:
        return []
    # recursively mine frequent patterns from FP-tree
    return fp_growth(fp_tree, [], min_sup)


if __name__ == "__main__":
    # example transaction database
    transactions = [
        ['A', 'B'],
        ['B', 'C', 'D'],
        ['A', 'C', 'D', 'E'],
        ['A', 'D', 'E'],
        ['A', 'B', 'C'],
        ['A', 'B', 'C', 'D'], 
        ['B', 'C'],
        ['A', 'B', 'C'],
        ['A', 'B', 'D'],
        ['B', 'C', 'E']
    ]

    # transactions = [
    #     ['I1', 'I2', 'I5'],
    #     ['I2', 'I4'],
    #     ['I2', 'I3'],
    #     ['I1', 'I2', 'I4'],
    #     ['I1', 'I3'],
    #     ['I2', 'I3'],
    #     ['I1', 'I3'],
    #     ['I1', 'I2', 'I3', 'I5'],
    #     ['I1', 'I2', 'I3']
    # ]

    # transactions = [
    #     ['a', 'b', 'c'],
    #     ['a', 'b', 'c'],
    #     ['a', 'b', 'd'],
    #     ['a', 'b', 'e']
    # ]

    # transactions = [
    #     ['M','O','N','K','E','Y'], 
    #     ['D','O','N','K','E','Y'], 
    #     ['M','A','K','E'], 
    #     ['M','U','C','K','Y'], 
    #     ['C','O','O','K','I','E']
    #     ]


    # transactions = [
    #     ['a', 'b', 'c', 'e'],
    #     ['a', 'b', 'c', 'e'],
    #     ['a', 'b', 'c', 'd', 'f'],
    #     ['a', 'b', 'c', 'd', 'f'],
    #     ['a', 'b', 'c', 'd', 'f'],
    #     ['a', 'b', 'c', 'd', 'e'],
    #     ['a', 'b', 'c'],
    #     ['a', 'b'],
    #     ['a'],
    #     ['a']
    # ]

    min_sup = 4
    
    patterns = mine_fptree(transactions, min_sup)
    # print(patterns)


    print("Frequent patterns with support:")
    for pattern, support in sorted(patterns, key=lambda x: (-x[1], x[0])):
        print(f"{pattern}: {support}")
from collections import Counter
from functools import reduce
from itertools import combinations

class TreeNode:
    def __init__(self, item_name, count=1, parent=None):
        self.item_name = item_name  
        self.count = count          
        self.parent = parent        # 父节点
        self.children = {}          # 子节点字典 {item_name: TreeNode}
        self.node_link = None       # 相同项节点链接

class HeaderTable:
    def __init__(self):
        self.header_table = {}  # {item_name: [support_count, node_link]}
        self.item_order = {}    # 存储初始FP树中1频繁项集的初始顺序

    # FP树新加节点时，更新header_table的链表        
    def add_node_link(self, item_name, node):
        if item_name not in self.header_table:
            self.header_table[item_name] = [0, None]

        if self.header_table[item_name][1] is None:
            self.header_table[item_name][1] = node
        else:
            # 遍历到最后一个节点
            current = self.header_table[item_name][1]
            while current.node_link:
                current = current.node_link
            current.node_link = node

    # 初始化header_table时，保存当前FP子树的1频繁项集的支持度
    def increment_support(self, item_name, count=1):
        if item_name not in self.header_table:
            self.header_table[item_name] = [0, None]
        self.header_table[item_name][0] += count

class FPTree:
    def __init__(self):
        self.root = TreeNode("null", 0)  # 创建根节点
        self.header_table = HeaderTable()
        
    def insert_tree(self, transaction, parent_node, count=1):
        if not transaction:
            return
            
        item = transaction[0]  # 获取第一个项
        
        # 检查是否已存在该子节点
        if item in parent_node.children:
            # 如果存在节点，只更新计数
            parent_node.children[item].count += count
        else:
            # 创建新节点
            new_node = TreeNode(item, count, parent_node)
            parent_node.children[item] = new_node
            # 更新header_table的链表
            self.header_table.add_node_link(item, new_node)
            
        # 递归把剩余事务插入到FP树
        if len(transaction) > 1:
            self.insert_tree(transaction[1:], parent_node.children[item], count)
            
    # 只在构建初始FP树时调用
    # FP频繁子树是由后续的frequent pattern base频繁模式基构建的
    def create_tree(self, transactions, min_sup):
        # 第一次扫描:计数频繁1项集
        # reduce把所有事务合并成一个列表，Counter计数每个项的出现次数
        item_counts = Counter(reduce(lambda x, y: x + y, transactions))
                
        # 过滤非频繁项并按支持度降序排序
        frequent_items = {k: v for k, v in sorted(
            [(k,v) for k,v in item_counts.items() if v >= min_sup],
            key=lambda x: (-x[1], x[0])
        )}
        
        # 保存初始FP树中1频繁项集的初始顺序
        for i, item in enumerate(frequent_items.keys()):
            self.header_table.item_order[item] = i
        
        # 初始化header_table
        # 添加header_table支持度 链表为空
        for item, count in frequent_items.items():
            self.header_table.increment_support(item, count)
            
        # 第二次扫描:构建FP树
        for transaction in transactions:
            # 重新排列每条事务
            # 过滤非频繁项并按header_table的顺序降序排列
            frequent_trans = [item for item in transaction if item in frequent_items]
            frequent_trans.sort(key=lambda x: (-frequent_items[x], x))
            
            # 插入重排后的事务，构建初始FP树
            if frequent_trans:
                self.insert_tree(frequent_trans, self.root)
                
    def has_single_path(self):
        # 判断频繁子树是否为单一路径
        # 如果是单一路径，可以直接生成所有频繁模式
        current = self.root
        while current.children:
            if len(current.children) > 1:
                return False
            current = list(current.children.values())[0]
        return True
        
    def get_single_path(self):
        # 获取单一路径上的所有节点
        path = []
        current = self.root
        while current.children:
            current = list(current.children.values())[0]
            path.append(current)
        return path

def get_conditional_pattern_base(header_table, item):
    # 从树的底部逐个item获取条件模式基
    patterns = []
    # 当前item的链表的第一个节点
    node = header_table.header_table[item][1]
    
    # 遍历当前item的链表的所有节点
    while node:
        path = []
        # 一个条件模式基的路径的支持度是当前叶子节点item的支持度
        support = node.count
        current = node.parent
        while current.item_name != "null":
            path.append(current.item_name)
            current = current.parent
        if path:
            path.reverse()
            patterns.append((path, support))
        node = node.node_link
        
    return patterns

def construct_conditional_fptree(patterns, min_sup, original_order):
    # 由对应item的条件模式基构建频繁子树
    cond_tree = FPTree()
    
    # 统计每个项的支持度
    item_count = {}
    for pattern, count in patterns:
        for item in pattern:
            item_count[item] = item_count.get(item, 0) + count
            
    # 过滤非频繁项
    frequent_items = {k: v for k, v in item_count.items() if v >= min_sup}
    
    # 继承原始FP树中1频繁项集的顺序
    cond_tree.header_table.item_order = original_order

    for item, count in frequent_items.items():
        cond_tree.header_table.increment_support(item, count)
    

    for pattern, count in patterns:
        filtered_pattern = [item for item in pattern if item in frequent_items]
        # 按原始FP树中1频繁项集的顺序排序
        filtered_pattern.sort(key=lambda x: original_order[x])
        if filtered_pattern:
            cond_tree.insert_tree(filtered_pattern, cond_tree.root, count)
            
    return cond_tree

def fp_growth(tree, alpha, min_sup):
    """FP-Growth算法主函数"""
    patterns = []
    
    if tree.has_single_path():
        path_nodes = tree.get_single_path()
        # FP子树是单一路径，直接挖掘所有频繁模式
        for i in range(1, len(path_nodes) + 1):
            for comb in combinations(path_nodes, i):
                min_support = min(node.count for node in comb)
                pattern_items = [node.item_name for node in comb] + alpha
                patterns.append((pattern_items, min_support))
    else:
        # 按原始FP树中1频繁项集的顺序
        # 支持度从低到高遍历所有项
        header_table = tree.header_table.header_table
        items = sorted(header_table.keys(), 
                      key=lambda x: tree.header_table.item_order[x])
                      
        for item in items:
            # alpha 已经遍历过的item前缀
            new_alpha = [item] + alpha
            # item在当前FP子树中的支持度
            support = header_table[item][0]
            patterns.append((new_alpha, support))
            
            # 获取当前item的条件模式基
            cond_patterns = get_conditional_pattern_base(tree.header_table, item)
            # 由item的条件模式基构建频繁子树
            cond_tree = construct_conditional_fptree(cond_patterns, min_sup, tree.header_table.item_order)
            
            # 如果频繁子树不为空，递归挖掘频繁模式
            if cond_tree.root.children:
                patterns.extend(fp_growth(cond_tree, new_alpha, min_sup))
                
    return patterns

def mine_fptree(transactions, min_sup):

    fp_tree = FPTree()
    # 通过两次遍历构建初始FP树
    fp_tree.create_tree(transactions, min_sup)

    if not fp_tree.root.children:
        return []
            # 递归挖掘FP树的频繁模式
    return fp_growth(fp_tree, [], min_sup)

# 测试代码
if __name__ == "__main__":

    # 示例数据
    
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
    min_sup = 2
    
    patterns = mine_fptree(transactions, min_sup)
    
    print("频繁模式及其支持度:")
    for pattern, support in sorted(patterns, key=lambda x: (-x[1], x[0])):
        print(f"{pattern}: {support}")
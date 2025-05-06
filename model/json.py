import json
from .database_util import TreeNode

def parse_explain_result(explain_result, query_id, encoding):
    plan_data = json.loads(explain_result)['QUERY PLAN'][0]['Plan']
    return recursive_parse(plan_data, query_id, encoding, parent=None)

def recursive_parse(node, query_id, encoding, parent=None):
    # 提取节点基础信息
    node_type = node['Node Type']
    filters = []
    if 'Filter' in node:
        filters.append(node['Filter'])
    if 'Index Cond' in node:
        filters.append(node['Index Cond'])
    
    # 提取join信息
    join_cond = format_join(node)
    
    # 构建树节点，使用Actual Rows作为card
    tree_node = TreeNode(
        nodeType=node_type,
        typeId=encoding.encode_type(node_type),
        filt=filters,
        card=float(node.get('Actual Rows', 0)),  # 使用Actual Rows并转为浮点数
        join=encoding.encode_join(join_cond) if join_cond else 0,
        join_str=join_cond,
        filterDict=encoding.encode_filters(filters, node.get('Alias'))
    )
    
    # 处理表信息
    if 'Relation Name' in node:
        tree_node.table = node['Relation Name']
        tree_node.table_id = encoding.encode_table(node['Relation Name'])
    tree_node.query_id = query_id
    
    # 递归处理子节点
    if 'Plans' in node:
        for child in node['Plans']:
            child_node = recursive_parse(child, query_id, encoding, parent=tree_node)
            child_node.parent = tree_node
            tree_node.addChild(child_node)
    
    return tree_node

def format_join(node):
    join_keys = ['Hash Cond', 'Join Filter', 'Index Cond']
    for key in join_keys:
        if key in node:
            cond = node[key]
            tables = sorted([c.split('.')[0] for c in cond.split(' = ')])
            return f"{tables[0]}⋈{tables[1]}"
    return None
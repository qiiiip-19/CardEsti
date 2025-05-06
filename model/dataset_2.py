import torch
from torch.utils.data import Dataset
import numpy as np
import json
import pandas as pd
import sys, os
from collections import deque

# Import existing utils and the new parsing functions
from .database_util import TreeNode, filterDict2Hist # Keep existing imports if needed by node2feature etc.
from .database_util import floyd_warshall_rewrite, pad_1d_unsqueeze, pad_2d_unsqueeze, pad_rel_pos_unsqueeze, pad_attn_bias_unsqueeze # Keep batching utils
from .json import parse_explain_result # Import your new parsing function

class PlanTreeDataset(Dataset):
    # Modified __init__ signature:
    # - input_data: Your list of JSON objects [{'query': ..., 'query_id': ..., 'explain_result': ...}, ...]
    # - label_map: A dictionary or DataFrame mapping query_id to {'cost': ..., 'cardinality': ...}
    # - Other parameters remain similar
    # 移除label_map参数，直接在内部处理标签
    def __init__(self, input_data: list, encoding, hist_file, card_norm, cost_norm, to_predict, table_sample):
        self.table_sample = table_sample
        self.encoding = encoding
        self.hist_file = hist_file
        self.to_predict = to_predict
        self.length = len(input_data)
        self.query_ids = [item['query_id'] for item in input_data]

        # 解析根节点并提取Actual Rows作为card
        self.root_nodes = []
        self.cards = []
        for item in input_data:
            query_id = item['query_id']
            explain_str = item['explain_result']
            root_node = parse_explain_result(explain_str, query_id, self.encoding)
            self.root_nodes.append(root_node)
            self.cards.append(root_node.card)  # 直接使用根节点的card属性

        # 固定cost为0
        self.costs = [0.0] * len(input_data)

        # 归一化处理
        # Normalize labels
        # For test data, self.cards will be mostly 0s if 'Actual Rows' is null.
        # card_norm.normalize_labels will still run, producing normalized 0s. This is fine.
        self.card_labels = torch.from_numpy(card_norm.normalize_labels(np.array(self.cards, dtype=float)))
        self.cost_labels = torch.from_numpy(cost_norm.normalize_labels(np.array(self.costs, dtype=float)))

        # 设置预测目标和标签
        if to_predict == 'cost':
            self.gts = self.costs
            self.labels = self.cost_labels
        elif to_predict == 'card':
            self.gts = self.cards
            self.labels = self.card_labels
        elif to_predict == 'both':
            self.gts = self.cards  # 根据需求选择主标签
            self.labels = self.card_labels
        else:
            raise ValueError(f'Unknown to_predict type: {to_predict}')

        # 预计算批次数据（保持不变）
        self.collated_dicts = []
        for root_node in self.root_nodes:
            node_dict = self.node2dict(root_node)
            collated = self.pre_collate(node_dict)
            self.collated_dicts.append(collated)

    # This method now takes the already parsed root_node
    def node2dict(self, root_node):
        # This function remains crucial as it calls topo_sort and calculate_height
        # topo_sort implicitly iterates through the tree and collects features via node2feature
        adj_list, num_child, features = self.topo_sort(root_node)
        heights = self.calculate_height(adj_list, len(features))

        return {
            'features' : torch.FloatTensor(np.array(features)),
            'heights' : torch.LongTensor(heights),
            'adjacency_list' : torch.LongTensor(np.array(adj_list)),
        }

    # This function performs the topological sort and collects features
    def topo_sort(self, root_node):
        adj_list = [] # from parent to children
        num_child = []
        features = []
        node_map = {} # Map node objects to integer IDs for adjacency list
        current_id = 0

        # Queue for BFS traversal, storing (node_object, assigned_id)
        toVisit = deque()
        node_map[root_node] = current_id
        toVisit.append(root_node)
        current_id += 1

        # This list will store nodes in the order of their assigned IDs (0 to N-1)
        # which is also the BFS discovery order from the root.
        processed_nodes = [] # Keep track of nodes in the order they are processed

        while toVisit:
            node = toVisit.popleft()
            processed_nodes.append(node)
            node_idx = node_map[node]

            # --- Feature Calculation Trigger ---
            # Calculate feature *here* just before adding to the list
            # Assumes node2feature exists and works with TreeNode attributes
            if node.feature is None: # Calculate feature if not already done (it shouldn't be by json.py)
                 node.feature = node2feature(node, self.encoding, self.hist_file, self.table_sample)
            features.append(node.feature)
            # ---------------------------------

            num_child.append(len(node.children))
            for child in node.children:
                 if child not in node_map: # Avoid cycles if any, though plans should be DAGs
                    node_map[child] = current_id
                    toVisit.append(child)
                    adj_list.append((node_idx, current_id)) # Use integer IDs
                    current_id += 1
                 else: # Handle case where node might be revisited (though unlikely in plan trees)
                    adj_list.append((node_idx, node_map[child]))


        # Ensure features are in the correct order corresponding to node IDs 0 to N-1
        # Reorder features based on the processed_nodes list and their assigned IDs
        ordered_features = [None] * len(processed_nodes)
        for node in processed_nodes:
            node_id = node_map[node]
            ordered_features[node_id] = node.feature

        # Check if all features were assigned
        if None in ordered_features:
             raise RuntimeError("Feature ordering failed, some nodes might not have been processed correctly.")


        return adj_list, num_child, ordered_features


    # calculate_height remains the same as it works on the adjacency list
    def calculate_height(self, adj_list, tree_size):
        if tree_size == 1:
            return np.array([0])
        if not adj_list: # Handle case with single node but tree_size > 1 (shouldn't happen)
             return np.zeros(tree_size, dtype=int)


        adj_list = np.array(adj_list)
        node_ids = np.arange(tree_size, dtype=int)
        node_order = np.zeros(tree_size, dtype=int)
        uneval_nodes = np.ones(tree_size, dtype=bool)

        # Ensure adj_list is not empty before accessing indices
        if adj_list.size == 0:
             # If no edges, all nodes are at height 0 (or handle as error)
             return np.zeros(tree_size, dtype=int)


        parent_nodes = adj_list[:,0]
        child_nodes = adj_list[:,1]

        n = 0
        while uneval_nodes.any():
            # Find roots (nodes that are not children) or nodes whose parents are already evaluated
            # A node can be evaluated if it's not in child_nodes OR if all its parents are evaluated
            # This is complex with the current loop structure. Let's rethink.

            # Alternative: Standard height calculation (longest path from root)
            # Requires knowing the root(s). Assume node 0 is the root.
            # Or, simpler: height = level in BFS/topo sort (distance from root)
            # The current topo sort implicitly gives levels. Let's try that.

            # Let's stick to the original logic for now, assuming it calculates depth correctly.
            # The original logic calculates the order in which nodes can be evaluated bottom-up.
            # node_order[i] = n means node i can be evaluated at step n.
            # This seems more like evaluation order than height/depth.
            # Let's redefine height as distance from the root (node 0).

            # --- Recalculating Height as Depth from Root (Node 0) ---
            if tree_size == 0: return np.array([])
            heights = -np.ones(tree_size, dtype=int) # Initialize heights to -1 (unvisited)
            queue = deque([(0, 0)]) # (node_id, height) - Start with root (ID 0) at height 0
            heights[0] = 0
            max_h = 0

            adj_dict = {i: [] for i in range(tree_size)}
            for u, v in adj_list:
                adj_dict[u].append(v)

            processed_in_bfs = 0
            while queue:
                u, h = queue.popleft()
                processed_in_bfs += 1
                max_h = max(max_h, h)
                if u in adj_dict:
                    for v in adj_dict[u]:
                        if heights[v] == -1: # If not visited
                            heights[v] = h + 1
                            queue.append((v, h + 1))
                        # If visited, don't update (shortest path - BFS level)

            # Check if all nodes were reached
            if processed_in_bfs != tree_size or np.any(heights == -1):
                 print(f"Warning: Not all nodes reachable from root 0 in height calculation. Processed: {processed_in_bfs}/{tree_size}")
                 # Handle unreachable nodes, e.g., assign max height or raise error
                 # Assign max_h + 1 to unreachable nodes for now
                 heights[heights == -1] = max_h + 1


            return heights
            # --- End Recalculated Height ---

            # --- Original Height Logic (Kept for reference) ---
            # uneval_mask = uneval_nodes[child_nodes]
            # unready_parents = parent_nodes[uneval_mask]
            # node2eval = uneval_nodes & ~np.isin(node_ids, unready_parents)
            # if not np.any(node2eval):
            #      # If no nodes can be evaluated, but some are unevaluated, there might be a cycle or disconnected components.
            #      print(f"Warning: Stuck in height calculation at step {n}. Unevaluated nodes: {np.where(uneval_nodes)[0]}")
            #      # Assign remaining nodes a default order or handle error
            #      node_order[uneval_nodes] = n
            #      break # Exit loop to prevent infinite loop
            # node_order[node2eval] = n
            # uneval_nodes[node2eval] = False
            # n += 1
            # return node_order
             # --- End Original Height Logic ---


    # pre_collate remains the same, it works on the dictionary output by node2dict
    def pre_collate(self, the_dict, max_node = 30, rel_pos_max = 20):
        # Ensure features are present and are a Tensor
        if not isinstance(the_dict['features'], torch.Tensor):
             features_np = np.array(the_dict['features'])
             # Check for None values if node2feature failed for some nodes
             if np.isscalar(features_np) and features_np is None:
                 raise ValueError("Features contain None values. Check node2feature.")
             # Check if features_np is empty or contains objects/None before converting
             if features_np.size == 0 or features_np.dtype == object:
                  raise ValueError(f"Features array is problematic: size={features_np.size}, dtype={features_np.dtype}")

             x = torch.FloatTensor(features_np)
        else:
             x = the_dict['features']

        # Handle case where x might be empty if node2dict failed
        if x.nelement() == 0:
             # Decide how to handle empty features, maybe return None or raise error
             # For now, let's create a dummy tensor based on expected size if possible
             # This requires knowing the feature dimension, which might vary.
             # Let's assume node2feature always returns a fixed size, find it from encoding/model?
             # Hardcoding a fallback is risky. Raising error is safer.
             raise ValueError("Input features to pre_collate are empty.")


        N = x.shape[0] # Use shape[0] which is number of nodes (features)
        x = pad_2d_unsqueeze(x, max_node) # Pad features


        attn_bias = torch.zeros([N + 1, N + 1], dtype=torch.float) # N+1 for super token

        # Ensure adjacency list is a Tensor and has the right shape
        adj_list_tensor = the_dict['adjacency_list']
        if not isinstance(adj_list_tensor, torch.Tensor):
             adj_list_tensor = torch.LongTensor(np.array(adj_list_tensor))

        # Handle empty adjacency list (single node graph)
        if adj_list_tensor.nelement() == 0:
             if N == 1: # Correct case for single node
                 shortest_path_result = np.array([[0]])
                 adj = torch.zeros([N, N], dtype=torch.bool) # N=1 -> [[False]]
             else: # Error case: multiple nodes but no edges? Or N=0?
                 raise ValueError(f"Inconsistent state: {N} nodes but empty adjacency list.")
        else:
             # Ensure adj_list has 2 columns
             if adj_list_tensor.dim() != 2 or adj_list_tensor.shape[1] != 2:
                  raise ValueError(f"Adjacency list has unexpected shape: {adj_list_tensor.shape}")

             edge_index = adj_list_tensor.t()
             # Check edge indices are within bounds [0, N-1]
             if edge_index.numel() > 0 and (edge_index.max() >= N or edge_index.min() < 0) :
                  raise ValueError(f"Edge index out of bounds (Max: {edge_index.max()}, N: {N})")

             adj = torch.zeros([N, N], dtype=torch.bool)
             # Only index if edge_index is not empty
             if edge_index.numel() > 0:
                adj[edge_index[0,:], edge_index[1,:]] = True

             # Ensure floyd_warshall_rewrite handles N=0 or N=1 correctly if needed
             if N > 0:
                shortest_path_result = floyd_warshall_rewrite(adj.numpy())
             else:
                shortest_path_result = np.empty((0,0)) # Handle N=0 case


        # Ensure shortest_path_result is valid before converting
        if N > 0:
            rel_pos = torch.from_numpy(shortest_path_result).long()
            # Apply relational position bias
            attn_bias[1:, 1:][rel_pos >= rel_pos_max] = float('-inf')
            rel_pos = pad_rel_pos_unsqueeze(rel_pos, max_node)
        else: # Handle N=0 case
             rel_pos = torch.zeros((0,0), dtype=torch.long)
             rel_pos = pad_rel_pos_unsqueeze(rel_pos, max_node) # Should result in padded zeros

        # Pad attention bias
        attn_bias = pad_attn_bias_unsqueeze(attn_bias, max_node + 1)

        # Pad heights
        heights_tensor = the_dict['heights']
        if not isinstance(heights_tensor, torch.Tensor):
            heights_tensor = torch.LongTensor(heights_tensor)
        heights = pad_1d_unsqueeze(heights_tensor, max_node)

        return {
            'x' : x,
            'attn_bias': attn_bias,
            'rel_pos': rel_pos,
            'heights': heights
        }


    # __len__ remains the same
    def __len__(self):
        return self.length

    # __getitem__ now returns the pre-collated dict and labels
    def __getitem__(self, idx):
        query_id = self.query_ids[idx]
        cost_label = self.cost_labels[idx]
        card_label = self.card_labels[idx]
        # Return the pre-calculated collated dictionary
        collated_dict = self.collated_dicts[idx]

        # Return format expected by the collator function (dict, (cost_label, card_label))
        return collated_dict, (cost_label, card_label)


# Default sizes for feature components if not available
# These should match the model's FeatureEmbed expectations
DEFAULT_HIST_BINS = 50 # From model.py FeatureEmbed default
EXPECTED_HIST_FEATURES_LEN = (DEFAULT_HIST_BINS -1) * 3 # For 3 filters
EXPECTED_SAMPLE_FEATURES_LEN = 1000 # From model.py FeatureEmbed default


# --- Helper Function (Ensure node2feature is defined correctly) ---
# This function MUST exist, either in this file or imported,
# and it must accept (node, encoding, hist_file, table_sample)
# and return a feature numpy array.

def node2feature(node: TreeNode, encoding, hist_file, table_sample, use_hist=False, use_sample = False):
    # This is a placeholder based on the original code's structure.
    # Ensure this matches the actual implementation needed.
    # It calculates features based on node attributes populated by recursive_parse.

    # Example structure based on original:
    # type, join, filter123, mask123, hist123, table, sample
    # Dimensions: 1, 1, 9, 3, (hist_bins-1)*3, 1, sample_len

    # Type and Join IDs (already encoded in TreeNode by recursive_parse)
    type_join = np.array([node.typeId, node.join])

    # Filter features (cols, ops, vals) - padded to 3 filters
    num_filter = len(node.filterDict['colId'])
    # Ensure filterDict values are lists of numbers/encodings
    filter_vals_dict = node.filterDict
    # Pad each list (colId, opId, val) if necessary, BEFORE stacking
    padded_filter_components = []
    for key in ['colId', 'opId', 'val']: # Maintain order
        vals_list = filter_vals_dict.get(key, [])
        arr = np.array(vals_list)
        pad_len = 3 - len(arr)
        if pad_len > 0:
            # Pad with 0 for 'val', or a specific NA index for IDs if available in encoding
            # For simplicity, using 0 for IDs if NA index isn't readily defined for padding here.
            # encoding.col2idx['NA'] or encoding.op2idx['NA'] could be used if appropriate.
            pad_value = 0
            if key == 'colId' and 'NA' in encoding.col2idx:
                pad_value = encoding.col2idx['NA']
            elif key == 'opId' and 'NA' in encoding.op2idx:
                pad_value = encoding.op2idx['NA']
            padded_arr = np.pad(arr, (0, pad_len), 'constant', constant_values=pad_value)
        else:
            padded_arr = arr[:3] # Truncate if more than 3
        padded_filter_components.append(padded_arr)

    # Check if padded_filter_vals has 3 elements (colId, opId, val)
    while len(padded_filter_components) < 3:
        padded_filter_components.append(np.zeros(3)) # Add zero arrays if filterDict was incomplete

    filts = np.array(padded_filter_components).flatten() # Should be 9 elements

    # Filter mask
    mask = np.zeros(3)
    mask[:num_filter] = 1

    # Histogram features
    # Assuming filterDict2Hist returns a flat numpy array of size (bin_num - 1) * 3
    if use_hist and hist_file is not None:
        # Pass the original hist_file if use_hist is True
        hists = filterDict2Hist(hist_file, node.filterDict, encoding)
    else:
        hists = np.zeros(EXPECTED_HIST_FEATURES_LEN)

    # hists = filterDict2Hist(hist_file, node.filterDict, encoding)
    # Ensure hists has the correct expected size, pad if necessary

    # Table ID (already encoded)
    table = np.array([node.table_id])

    # Sample bitmap
    if use_sample and table_sample is not None:
        sample_data = np.zeros(EXPECTED_SAMPLE_FEATURES_LEN) # Default empty sample
        if node.query_id is not None and node.query_id in table_sample:
            if node.table != 'NA' and node.table in table_sample[node.query_id]:
                # Ensure the sample from table_sample is already a numpy array of correct length
                retrieved_sample = table_sample[node.query_id][node.table]
                if isinstance(retrieved_sample, np.ndarray) and len(retrieved_sample) == EXPECTED_SAMPLE_FEATURES_LEN:
                    sample_data = retrieved_sample
                else:
                    print(f"Warning: Sample for query {node.query_id}, table {node.table} is not in expected format. Using zeros.")
    else:
        sample_data = np.zeros(EXPECTED_SAMPLE_FEATURES_LEN)

    # Concatenate all features
    # Ensure dimensions match model's expected input for FeatureEmbed
    feature_vector = np.concatenate((type_join, filts, mask, hists, table, sample_data))

    # Add check for NaN or Inf values
    if np.isnan(feature_vector).any() or np.isinf(feature_vector).any():
        print(f"Warning: NaN or Inf found in feature vector for node type {node.nodeType}, query_id {node.query_id}")
        # Handle appropriately: replace with zeros, raise error, etc.
        feature_vector = np.nan_to_num(feature_vector, nan=0.0, posinf=0.0, neginf=0.0)

    if len(feature_vector) != 1165:
        raise ValueError(f"Feature vector length is {len(feature_vector)}, expected 1165. Check component lengths:"
                         f"typeId:1, joinId:1, filts:9, mask:3, hists:{len(hists_final)}, table_sample:{len(table_and_sample_features)}")


    return feature_vector

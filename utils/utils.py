from __future__ import print_function

import os
import pickle as pkl
import sys
import json

import networkx as nx
import numpy as np
import scipy.sparse as sp

import dill as pickle

import torch
from utils.constants import GNN_MSG_KEY, GNN_NODE_FEAT_IN_KEY, GNN_NODE_FEAT_OUT_KEY, GNN_EDGE_FEAT_KEY, GNN_AGG_MSG_KEY


interesting_args = ["heap_dep_bypass", "filepath_r", "filepath"]

# def getInterestingArg(args):
#     for key in args:
#         if key in interesting_args:

def care_APIs():
    iapis = ['ntduplicateobject', 'deviceiocontrol', 'movefilewithprogresstransactedw', 'openservicea', 'ntquerysysteminformation', 'ntsetvaluekey', 'wnetgetprovidernamew', 'ntsetinformationfile', 'ntcreateprocessex', 'ntcreatekey', 'rtlcreateuserprocess', 'movefilewithprogressw', 'cryptexportkey', 'openservicew', 'ntopenprocess', 'controlservice', 'cryptencrypt', 'ntterminateprocess', 'ntclose', 'getadaptersaddresses', 'crypthashdata', 'regqueryvalueexw', 'getclipboarddata', 'process32nextw', 'regsetvalueexa', 'createservicea', 'regopenkeyexw', 'ntdelayexecution', 'ntdeviceiocontrolfile', 'setclipboardviewer', 'ntallocatevirtualmemory', 'readprocessmemory', 'regopenkeyexa', 'shellexecuteexw', 'ntwritefile', 'ldrgetdllhandle', 'cryptgenkey', 'createservicew', 'getcomputernamew', 'regqueryvalueexa', 'ntopenfile', 'internetreadfile', 'obtainuseragentstring', 'urldownloadtocachefilew', 'getusernamea', 'ntcreatefile', 'addclipboardformatlistener', 'getcomputernamea', 'ntloaddriver', 'ntcreateprocess', 'ntprotectvirtualmemory', 'enumservicesstatusa', 'regsetvalueexw', 'internetsetoptiona', 'setwindowshookexa', 'ldrgetprocedureaddress', 'setwindowshookexw', 'enumservicesstatusw', 'process32firstw', 'setfileattributesw', 'internetopena', 'ldrloaddll', 'ntcreateuserprocess', 'internetopenw', 'createprocessinternalw', 'urldownloadtofilew']
    return [iapi.lower() for iapi in iapis]

def reset_graph_features(g):
    keys = [GNN_NODE_FEAT_IN_KEY, GNN_AGG_MSG_KEY, GNN_MSG_KEY, GNN_NODE_FEAT_OUT_KEY]
    for key in keys:
        if key in g.ndata:
            del g.ndata[key]
    if GNN_EDGE_FEAT_KEY in g.edata:
        del g.edata[GNN_EDGE_FEAT_KEY] 


def compute_node_degrees(g):
    """
    Given a graph, compute the degree of each node
    :param g: DGL graph
    :return: node_degrees: a tensor with the degree of each node
             node_degrees_ids: a labeled version of node_degrees (usable for 1-hot encoding)
    """
    fc = lambda i: g.in_degrees(i).item()
    node_degrees = list(map(fc, range(g.number_of_nodes())))
    unique_deg = list(set(node_degrees))
    mapping = dict(zip(unique_deg, list(range(len(unique_deg)))))
    node_degree_ids = [mapping[deg] for deg in node_degrees]
    return torch.LongTensor(node_degrees), torch.LongTensor(node_degree_ids)


def save_txt(obj, ofpath):
    """
    Save an object as text

    Args:
        obj (list): list to be converted to string to save to text
        ofpath (str): path where to store the file
    """
    with open(ofpath, 'w+') as ofh:
        ofh.write('\n'.join(obj))


def save_pickle(obj, ofpath):
    """
    Save an object as pickle

    Args:
        graph (DGLGraph): graph to be saved
        ofpath (str): path where to store the file
    """
    with open(ofpath, 'wb') as ofh:
        pickle.dump(obj, ofh)


def load_pickle(ifpath):
    """
    Load an object from pickle

    Args:
        ifpath (str): path from where a graph is loaded
    """
    with open(ifpath, 'rb') as ifh:
        return pickle.load(ifh)




def indices_to_one_hot(data, out_vec_size):
    """
    Convert an iterable of indices to one-hot encoded labels.
    """
    targets = np.array(data).reshape(-1)
    return np.eye(out_vec_size)[targets].reshape(-1)


def label_encode_onehot(labels):
    classes = set(labels)
    classes_dict = {c: np.identity(len(classes))[
        i, :] for i, c in enumerate(classes)}
    labels_onehot = np.array(
        list(map(classes_dict.get, labels)), dtype=np.int32)
    return labels_onehot



def parse_index_file(filename):
    """Parse index file."""
    index = []
    for line in open(filename):
        index.append(int(line.strip()))
    return index


def sample_mask(idx, l):
    """Create mask."""
    mask = np.zeros(l)
    mask[idx] = 1
    return np.array(mask, dtype=np.bool)


def preprocess_features(features):
    """Row-normalize feature matrix and convert to tuple representation"""
    rowsum = np.array(features.sum(1))
    
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    features = r_mat_inv.dot(features)
    return features


def normalize_adj(adj, symmetric=True):
    if symmetric:
        d = sp.diags(np.power(np.array(adj.sum(1)), -0.5).flatten(), 0)
        a_norm = adj.dot(d).transpose().dot(d).tocsr()
    else:
        d = sp.diags(np.power(np.array(adj.sum(1)), -1).flatten(), 0)
        a_norm = d.dot(adj).tocsr()
    return a_norm


def preprocess_adj(adj, symmetric=True):
    adj = adj + sp.eye(adj.shape[0])
    adj = normalize_adj(adj, symmetric)
    return adj




'''
TF-IDF utils
'''
# adapted from https://github.com/JonathanRaiman/pytreebank

def normalize_string(string):
    """
    Standardize input strings by making
    non-ascii spaces be ascii, and by converting
    treebank-style brackets/parenthesis be characters
    once more.
    Arguments:
    ----------
        string : str, characters to be standardized.
    Returns:
    --------
        str : standardized
    """
    return string.replace("\xa0", " ")\
                 .replace("\\", "")\
                 .replace("-LRB-", "(")\
                 .replace("-RRB-", ")")\
                 .replace("-LCB-", "{")\
                 .replace("-RCB-", "}")\
                 .replace("-LSB-", "[")\
                 .replace("-RSB-", "]")
                    
def attribute_text_label(node, current_word):
    """
    Tries to recover the label inside a string
    of the form '(3 hello)' where 3 is the label,
    and hello is the string. Label is not assigned
    if the string does not follow the expected
    format.
    Arguments:
    ----------
        node : LabeledTree, current node that should
            possibly receive a label.
        current_word : str, input string.
    """
    node.text = normalize_string(current_word)
    node.text = node.text.strip(" ")
    node.udepth = 1
    if len(node.text) > 0 and node.text[0].isdigit():
        split_sent = node.text.split(" ", 1)
        label = split_sent[0]
        if len(split_sent) > 1:
            text = split_sent[1]
            node.text = text

        if all(c.isdigit() for c in label):
            node.label = int(label)
        else:
            text = label + " " + text
            node.text = text

    if len(node.text) == 0:
        node.text = None


def create_tree_from_string(line):
    """
    Parse and convert a string representation
    of an example into a LabeledTree datastructure.
    Arguments:
    ----------
        line : str, string version of the tree.
    Returns:
    --------
        LabeledTree : parsed tree.
    """
    depth         = 0
    current_word  = ""
    root          = None
    current_node  = root

    for char in line:
        if char == '(':
            if current_node is not None and len(current_word) > 0:
                attribute_text_label(current_node, current_word)
                current_word = ""
            depth += 1
            if depth > 1:
                # replace current head node by this node:
                child = LabeledTree(depth=depth)
                current_node.add_child(child)
                current_node = child
                root.add_general_child(child)
            else:
                root = LabeledTree(depth=depth)
                root.add_general_child(root)
                current_node = root

        elif char == ')':
            # assign current word:
            if len(current_word) > 0:
                attribute_text_label(current_node, current_word)
                current_word = ""

            # go up a level:
            depth -= 1
            if current_node.parent != None:
                current_node.parent.udepth = max(current_node.udepth+1, current_node.parent.udepth)
            current_node = current_node.parent
        else:
            # add to current read word
            current_word += char
    if depth != 0:
        raise ParseError("Not an equal amount of closing and opening parentheses")

    return root


class LabeledTree(object):
    SCORE_MAPPING = [-12.5,-6.25,0.0,6.25,12.5]

    def __init__(self,
                 depth=0,
                 text=None,
                 label=None,
                 children=None,
                 parent=None,
                 udepth=1):
        self.label    = label
        self.children = children if children != None else []
        self.general_children = []
        self.text = text
        self.parent   = parent
        self.depth    = depth
        self.udepth   = udepth

    def uproot(tree):
        """
        Take a subranch of a tree and deep-copy the children
        of this subbranch into a new LabeledTree
        """
        uprooted = tree.copy()
        uprooted.parent = None
        for child in tree.all_children():
            uprooted.add_general_child(child)
        return uprooted

    def shrink_tree(tree, final_depth):
        if tree.udepth <= final_depth:
            return tree
        for branch in tree.general_children:
            if branch.udepth == final_depth:
                return branch.uproot()

    def shrunk_trees(tree, final_depth):
        if tree.udepth <= final_depth:
            yield tree
        for branch in tree.general_children:
            if branch.udepth == final_depth:
                yield branch.uproot()

    def copy(self):
        """
        Deep Copy of a LabeledTree
        """
        return LabeledTree(
            udepth = self.udepth,
            depth = self.depth,
            text = self.text,
            label = self.label,
            children = self.children.copy() if self.children != None else [],
            parent = self.parent)

    def add_child(self, child):
        """
        Adds a branch to the current tree.
        """
        self.children.append(child)
        child.parent = self
        self.udepth = max([child.udepth for child in self.children]) + 1

    def add_general_child(self, child):
        self.general_children.append(child)

    def all_children(self):
        if len(self.children) > 0:
            for child in self.children:
                for subchild in child.all_children():
                    yield subchild
            yield self
        else:
            yield self

    def lowercase(self):
        """
        Lowercase all strings in this tree.
        Works recursively and in-place.
        """
        if len(self.children) > 0:
            for child in self.children:
                child.lowercase()
        else:
            self.text = self.text.lower()

    def to_dict(self, index=0):
        """
        Dict format for use in Javascript / Jason Chuang's display technology.
        """
        index += 1
        rep = {}
        rep["index"] = index
        rep["leaf"] = len(self.children) == 0
        rep["depth"] = self.udepth
        rep["scoreDistr"] = [0.0] * len(LabeledTree.SCORE_MAPPING)
        # dirac distribution at correct label
        if self.label is not None:
            rep["scoreDistr"][self.label] = 1.0
            mapping = LabeledTree.SCORE_MAPPING[:]
            rep["rating"] = mapping[self.label] - min(mapping)
        # if you are using this method for printing predictions
        # from a model, the the dot product with the model's output
        # distribution should be taken with this list:
        rep["numChildren"] = len(self.children)
        text = self.text if self.text != None else ""
        seen_tokens = 0
        witnessed_pixels = 0
        for i, child in enumerate(self.children):
            if i > 0:
                text += " "
            child_key = "child%d" % (i)
            (rep[child_key], index) = child.to_dict(index)
            text += rep[child_key]["text"]
            seen_tokens += rep[child_key]["tokens"]
            witnessed_pixels += rep[child_key]["pixels"]

        rep["text"] = text
        rep["tokens"] = 1 if (self.text != None and len(self.text) > 0) else seen_tokens
        rep["pixels"] = witnessed_pixels + 3 if len(self.children) > 0 else text_size(self.text)
        return (rep, index)

    def to_json(self):
        rep, _ = self.to_dict()
        return json.dumps(rep)

    def display(self):
        from IPython.display import Javascript, display

        display(Javascript("createTrees(["+self.to_json()+"])"))
        display(Javascript("updateTrees()"))

    def to_lines(self):
        if len(self.children) > 0:
            left_lines, right_lines = self.children[0].to_lines(), self.children[1].to_lines()
            self_line = [left_lines[0] + " " + right_lines[0]]
            return self_line + left_lines + right_lines
        else:
            return [self.text]

    def to_labeled_lines(self):
        if len(self.children) > 0:
            left_lines, right_lines = self.children[0].to_labeled_lines(), self.children[1].to_labeled_lines()
            self_line = [(self.label, left_lines[0][1] + " " + right_lines[0][1])]
            return self_line + left_lines + right_lines
        else:
            return [(self.label, self.text)]

    def __str__(self):
        """
        String representation of a tree as visible in original corpus.
        print(tree)
        #=> '(2 (2 not) (3 good))'
        Outputs
        -------
            str: the String representation of the tree.
        """
        if len(self.children) > 0:
            rep = "(%d " % self.label
            for child in self.children:
                rep += str(child)
            return rep + ")"
        else:
            text = self.text\
                .replace("(", "-LRB-")\
                .replace(")", "-RRB-")\
                .replace("{", "-LCB-")\
                .replace("}", "-RCB-")\
                .replace("[", "-LSB-")\
                .replace("]", "-RSB-")

            return ("(%d %s) " % (self.label, text))

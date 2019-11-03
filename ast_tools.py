"""
AST Tools - Copyright  (c) 2019 Timothy Watts
All rights reserved.

Collection of useful tools to inspect Pythons AST.

This file is part of Sneklang and is released under the "GNU Affero General Public License ". 
Please see the LICENSE file that should have been included as part of this package.

"""


def parse_ast(node):
    # check if this is a node or list
    if isinstance(node, list):
        result = []
        for child_node in node: # A list of nodes, really
            result += [parse_ast(child_node)]
        return result

    # A node it seems
    if '_ast' == getattr(node, '__module__', False):
        result = {}
        for k in node.__dict__:
            result[k] = parse_ast(getattr(node, k))
        # The original class would be nice if we want to reconstruct the tree
        return node.__class__, result

    # Who knows what it is, just return it.
    return node

def deserialize(node):
    """ Returns an ast instance from an expanded dict. """
    if isinstance(node, tuple):
        klass, kws = node
        return klass(**deserialize(kws))
    elif isinstance(node, dict):
        d = {}
        for k, v in node.items():
            d[k] = deserialize(v)
        return d
    elif isinstance(node, list):
        return [deserialize(n) for n in node]
    else:
        return node

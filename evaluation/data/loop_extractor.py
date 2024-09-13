import tree_sitter

def get_for_loops(node):
    """
    Recursively traverses the AST and collects 'for' loop nodes along with their code as text.
    """
    if node.type == 'for_statement':  # If the node is a 'for' loop
        return [(node, node.text)]
   
    nodes = []
    for child in node.children:  # Traverse through children nodes
        nodes += get_for_loops(child)
   
    return nodes


def parse_code_and_get_for_loops(code_str, parser):
    tree = parser.parse(bytes(code_str, "utf8"))
    root_node = tree.root_node
   
    for_loops = get_for_loops(root_node)
   
    return [loop[1] for loop in for_loops]



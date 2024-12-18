
def build_graph(g, output):
    """Recursively add key graph elements to the Graph object based on output from Forward mode. 
    This is a helper function for generate_graph function.

    :param g: The graph object from graphviz used to make computation graph
    :type g: graphviz.graphs.Graph
    """
    # Base case: the node has no parent
    if output.parent == []:
        return
    # Recursive case
    for i, p in enumerate(output.parent):
        if len(output.op) == 1: # the operation has only one element (e.g., [sin()])
            g.node(output.v_index)
            g.node(p.v_index)
            if i == 0: # add operation sign only if it is the first time the parent appears
                g.edge(p.v_index, output.v_index, output.op[0])
            else:
                g.edge(p.v_index, output.v_index)
        else: # the operation has only one element (e.g., ['*', 3])
            g.node(output.v_index)
            g.node(p.v_index)
            g.node(str(output.op[1])) # add node for scalar (e.g., 3)
            g.edge(p.v_index, output.v_index, output.op[0])
            g.edge(str(output.op[1]), output.v_index) # add edge for the scalar node
        build_graph(g, p)

def generate_graph(x, g):
    """Generates the computation graph (in .png format) given input variables, 
    and output Nodes from Forward mode

    :param x: The input variable values
    :type x: A list of input variable values
    :param g: A Forward object
    :type g: A Forward object
    >>> x = [3, 4, 5]
    >>> def f2(x1, x2, x3):
            return [np.sin(x1)-x2-3/x3, np.cos(x2)*x1/x3]
    >>> g2 = Forward(f2, *x)
    >>> generate_graph(x, g2)
    """
    import graphviz as gv
    # Initialize a Graph object in graphviz and set the style of the Graph object
    graph = gv.Graph(format='png')
    graph.attr(rankdir="LR", size="30, 30")
    # Initialize the index of the output function
    findex = 0
    # Iterate through the output Node(s) from the input function
    output_lst = g.output if isinstance(g.output, list) else [g.output]
    for out in output_lst:
        # Call the helper function to recursively add key graph elements to the Graph object
        build_graph(g=graph, output=out)
        findex += 1
        # Add nodes and edges to the graphviz Graph object
        graph.node(f'f{findex}')
        graph.edge(out.v_index, f'f{findex}')
    
    # Add input variables nodes and edges
    xlen = len(x)
    for i in range(xlen):
        graph.node(f'x{i+1}')
        graph.edge(f'x{i+1}', f'v{i-xlen+1}')
    
    # Save the computation graph to file
    graph.render(filename="./computationGraph")
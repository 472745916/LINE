def line_loss(y_true, y_pred):
    return -K.mean(K.log(K.sigmoid(y_true*y_pred)))

def create_model(numNodes, embedding_size, order='second'):

    v_i = Input(shape=(1,))
    v_j = Input(shape=(1,))

    first_emb = Embedding(numNodes, embedding_size, name='first_emb')
    second_emb = Embedding(numNodes, embedding_size, name='second_emb')
    context_emb = Embedding(numNodes, embedding_size, name='context_emb')

    v_i_emb = first_emb(v_i)
    v_j_emb = first_emb(v_j)

    v_i_emb_second = second_emb(v_i)
    v_j_context_emb = context_emb(v_j)

    first = Lambda(lambda x: tf.reduce_sum(
        x[0]*x[1], axis=-1, keep_dims=False), name='first_order')([v_i_emb, v_j_emb])
    second = Lambda(lambda x: tf.reduce_sum(
        x[0]*x[1], axis=-1, keep_dims=False), name='second_order')([v_i_emb_second, v_j_context_emb])

    if order == 'first':
        output_list = [first]
    elif order == 'second':
        output_list = [second]
    else:
        output_list = [first, second]

    model = Model(inputs=[v_i, v_j], outputs=output_list)
def _gen_sampling_table(self):
    # create sampling table for vertex
    power = 0.75
    numNodes = self.node_size
    node_degree = np.zeros(numNodes)  # out degree
    node2idx = self.node2idx
    for edge in self.graph.edges():
        node_degree[node2idx[edge[0]]]  += self.graph[edge[0]][edge[1]].get('weight', 1.0)
    total_sum = sum([math.pow(node_degree[i], power) for i in range(numNodes)])
    norm_prob = [float(math.pow(node_degree[j], power)) /  total_sum for j in range(numNodes)]
    self.node_accept, self.node_alias = create_alias_table(norm_prob)
    # create sampling table for edge    numEdges = self.graph.number_of_edges()
    total_sum = sum([self.graph[edge[0]][edge[1]].get('weight', 1.0) for edge in self.graph.edges()])
    norm_prob = [self.graph[edge[0]][edge[1]].get('weight', 1.0) * numEdges / total_sum for edge in self.graph.edges()]
    self.edge_accept, self.edge_alias = create_alias_table(norm_prob)
def calc_cell_paper(NN_row: np.ndarray, meta_data: pd.DataFrame, variable: str) -> pd.Series:
    """ This function calculates the posterior probability that a cell belongs to a class of the
    given variable, based on the classes of its k nearest neighbours. This function operates on
    a 1-dimensional array of length k, which can be a row of a larger dataframe.

    The posterior probability that a cell :math `i` with expression :math: `x_i` orignates from class
    :math: `c`, depending on it nearest neighbours :math: `b_i` is calculated as follows:
    .. math::

        p(c|x_i) = \frac{\sum_{i \in b_i} W_c Y_{b_i \in c}}{\sum_{i \in b_i} W_c} \\
        W_c = \frac{n_{Y=c}}{n}

    Where :math: `W_c` is the prior probability of class :math: `c` and :math: `Y_i` are the class
    labels of the nearest neighbours.

    Arguments:
        NN_row: array of length k, with k being the number of nearest neighbours. Values are the indices of
            the nearest neighbours in the meta_data
        meta_data: Contains the variable of interest. Is sliced with the indices given by NN_row to
            get varaible classes of the nearest neighbours.
        variable: variable for which to calculate the posterior class probabilities.

    Returns: A series with for each class the posterior probabiltiy that the cell came from that class and
        a `pred_class` entry that indicated the class assigned to the cell based on the maximal posterior probability.
    """

    # Get classes of the neighbours
    NN_vars = meta_data.loc[NN_row, variable]

    # Count how often classes occur among neighbours
    NN_vars_counts = NN_vars.value_counts()

    # For classes that don't occur the count is 0, ensure we have vector the contains all classes
    NN_v_counts = pd.Series(0, index=meta_data[variable].unique())
    NN_v_counts[NN_vars_counts.index] = NN_vars_counts

    # Prior probability that a class occurs among all cells
    Wc = meta_data.groupby(variable).count().iloc[:, 0] / meta_data.shape[0]

    # Sum of prior probibilities of classes that are in the nearest neighbours
    sum_Wc = sum([Wc[variable] for variable in NN_vars])

    # Calculate posterior probability and add an entry for the class assignement based on the
    # maximum posterior prob.
    post_p = NN_v_counts * Wc / sum_Wc
    post_p["pred_class"] = post_p.idxmax()

    return post_p

calc_cell_paper

def calc_inverse_weights(NN_row: np.ndarray, meta_data: pd.DataFrame, variable: str) -> pd.Series:
    """ This function calculates the posterior probability that a cell belongs to a class of the
    given variable, based on the classes of its k nearest neighbours. The neighbor counts are weighted
    by the inverse of how often this class occurs among all cells. This function operates on
    a 1-dimensional array of length k, which can be a row of a larger dataframe.

    The posterior probability that a cell :math: `i` with expression :math: `x_i` orignates from class
    :math: `c`, depending on it nearest neighbours :math: `b_i` is calculated as follows:
    .. math::

        p(c|x_i) = \frac{\sum_{i \in b_i} W_c Y_{b_i \in c}}{\sum_{c=1}^{C}\sum_{i \in b_i} W_c Y_{b_i \in c}} \\
        W_c = \frac{1}{\sum_{n=1}^{N} Y_{n\in c}}

    Where :math: `W_c` is the prior probability of class :math: `c` and :math: `Y_i` are the class
    labels of the nearest neighbours. And :math: `Y` equals :math:`1` if neighbour :math: `b_i` or cell :math: `n`
    has class :math: `c` and :math: `0` otherwise.

    Arguments:
        NN_row: array of length k, with k being the number of nearest neighbours. Values are the indices of
            the nearest neighbours in the meta_data
        meta_data: Contains the variable of interest. Is sliced with the indices given by NN_row to
            get varaible classes of the nearest neighbours.
        variable: variable for which to calculate the posterior class probabilities.

    Returns: A series with for each class the posterior probabiltiy that the cell came from that class and
        a `pred_class` entry that indicated the class assigned to the cell based on the maximal posterior probability.
    """

    # Get classes of the neighbours
    NN_vars = meta_data.loc[NN_row, variable]

    # Count how often classes occur among neighbours
    NN_vars_counts = NN_vars.value_counts()

    # For classes that don't occur the count is 0, ensure we have vector the contains all classes
    NN_v_counts = pd.Series(0, index=meta_data[variable].unique())
    NN_v_counts[NN_vars_counts.index] = NN_vars_counts

    # Count how often class occurs among all cells and take the inverse of that as priors
    Wc = meta_data.groupby(variable).count().iloc[:, 0]
    Wc = 1 / Wc

    # Calculate posterior probability and add an entry for the class assignement based on the
    # maximum posterior prob.
    post_p = NN_v_counts * Wc / sum(NN_v_counts * Wc)
    post_p["pred_class"] = post_p.idxmax()

    return post_p


def calc_adriano(NN_row: np.ndarray, meta_data: pd.DataFrame, variable: str, k: int = 100) -> pd.Series:
    """ This function calculates the posterior probability that a cell belongs to a class of the
    given variable, based on the classes of its k nearest neighbours. This function operates on
    a 1-dimensional array of length k, which can be a row of a larger dataframe.

    Arguments:
        NN_row: array of length k, with k being the number of nearest neighbours. Values are the indices of
            the nearest neighbours in the meta_data
        meta_data: Contains the variable of interest. Is sliced with the indices given by NN_row to
            get varaible classes of the nearest neighbours.
        variable: variable for which to calculate the posterior class probabilities.

    Returns: A series with for each class the posterior probabiltiy that the cell came from that class and
        a `pred_class` entry that indicated the class assigned to the cell based on the maximal posterior probability.
    """

    # Get classes of the neighbours
    NN_vars = meta_data.loc[NN_row, variable]

    # Count how often classes occur among neighbours
    NN_vars_counts = NN_vars.value_counts()

    # For classes that don't occur the count is 0, ensure we have vector the contains all classes
    NN_v_counts = pd.Series(0, index=meta_data[variable].unique())
    NN_v_counts[NN_vars_counts.index] = NN_vars_counts

    # Divide count by the number of neighbours
    likelihood = NN_v_counts / sum(NN_v_counts)

    # Prior probability that a class occurs among all cells
    prior = meta_data.groupby(variable).count().iloc[:, 0] / meta_data.shape[0]
    inverse_prior = 1 / prior

    # Calculate evidence
    evidence = sum(likelihood * inverse_prior)

    # Calculate posterior probability and add an entry for the class assignement based on the
    # maximum posterior prob.
    posterior = likelihood * inverse_prior / evidence
    posterior["pred_class"] = posterior.idxmax()

    return posterior


def calc_post_p(expr: pd.DataFrame, variable: str, meta_data: pd.DataFrame, method: str = "cell_paper", k: int = 100,
                pred_class: bool = False) -> pd.DataFrame:
    """ Calculate the posterior probability that a cell comes from class of the given variable given the class
    of its k nearest neighbours.
    See respective methods for how the posterior is calculated

    The class to class probabilities are calculates by averaging over the posterior probabilities of
    all cells that the same true class label.

    ..math::

        \frac{\sum_{x_i|Y_i=c}p(c|x_i)}{n_{Y_i=c}}

    Arguments:
        expr: measured marker expression, rows are cells and columns are markers
        variable: variable for which to calculate the posterior class probabilities.
        meta_data: Contains the variable of interest. Is sliced with the indices given by a row in A to
            get varaible classes of the nearest neighbours.
        method: which method for calculating the posterior probability should be used. `cell_paper` for same method
            as in the Cell Paper, `sklearn` for the sklearn predict_proba method (then `kNN` should be given) and
            `inverse_weight` to weight the counts by inverse of the class frequency. See method functions for the
            specifics. `adriano` is ...

    Returns:
        post_p_matrix: N by C matrix, where C is the number of classes in the given variable. Where entry
            `post_p_matrix`_ij is the posterior probability that cell i came from class j given its k-nearest
            neigbours. Also one column with name `pred_class` is the class assigned to the cell based on the highest
            posterior probability.
        class_to_class: C by C matrix, so a class to class probability where the diagonal elements indicate
            how self-contained a class is. Obtained by averagering all posterior probabilities of cells that
            are assigned the same class.
    """

    # Train kNN classifier
    kNN = KNeighborsClassifier(n_neighbors=k, weights="distance")
    kNN.fit(expr, meta_data[variable])

    # Get the 100 nearest neighbours. NN_df is N by 100 (N cells and 100 neighbours).
    # Values are indices of the nearest neighbours
    A = kNN.kneighbors(expr, k, return_distance=False)
    NN_df = pd.DataFrame(A)

    # Get the posterior probability for each cell for each class and class assignements
    if method == "cell_paper":
        post_p_matrix = NN_df.apply(calc_cell_paper, args=(meta_data, variable), axis=1)
    elif method == "sklearn":
        post_p_matrix = pd.DataFrame(kNN.predict_proba(expr), index=expr.index,
                                     columns=sorted(meta_data[variable].unique()))
        post_p_matrix["pred_class"] = kNN.predict(expr)
    elif method == "inverse_weight":
        post_p_matrix = NN_df.apply(calc_inverse_weights, args=(meta_data, variable), axis=1)
    elif method == "adriano":
        post_p_matrix = NN_df.apply(calc_adriano, args=(meta_data, variable), axis=1)

    # Average over all cells with the same class
    post_p_matrix["class"] = meta_data[variable]
    if pred_class == True:
        temp = post_p_matrix.drop("class", inplace=False, axis=1)
        class_to_class = temp.groupby("pred_class").mean()
    else:
        temp = post_p_matrix.drop("pred_class", inplace=False, axis=1)
        class_to_class = temp.groupby("class").mean()

    # Some classes do not show up in the rows of the class to class matrix
    # Because this class is not pedicted for any cell (not the highest posterior prob. for any cell)
    # So here we add these classes as rows all probabilities zero to get a square matrix
    missing_classes = (set(post_p_matrix.columns)) - set(class_to_class.index) - {"class", "pred_class"}
    for class_name in missing_classes:
        class_to_class = class_to_class.append(pd.Series([0] * class_to_class.shape[1],
                                                         name=class_name, index=class_to_class.columns))

    # Order columns same as rows so diagonal elements are the self-containment of the class
    class_to_class = class_to_class[class_to_class.index]

    return post_p_matrix, class_to_class
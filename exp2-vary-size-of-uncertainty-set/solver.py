import numpy as np 
import cvxpy as cp
import pandas as pd
import dsp                                     # disciplined saddle point programming
from typing import Tuple

from cvxpy.transforms.suppfunc import SuppFunc # Import SuppFunc to transform support function constraints

def solve_robust_prob(
        A: np.ndarray,                          # feature matrix
        b: np.ndarray,                          # response vector
        unmasked_A_df: pd.DataFrame,            # dataframe with complete (unmasked) features; used for extra constraints
        use_city_center_constraint: bool,       # Flag to indicate if a city-center based constraint should be enforced
        city_center: np.ndarray,                # 2-element array (longitude, latitude) of the city's center
        city_center_diag_matrix: np.ndarray,    # 2×2 diagonal matrix converting degrees to kilometers (squared factors)
        use_grid = None                         # Optional grid information for square uncertainty sets; if None, grid is not used
)-> Tuple[np.ndarray, float]:
    """
    Solves the robust OLS problem, with the specified robust constraints.

    :param A: the feature matrix, with possibly masked entries. shape is (n, d)
    :param b: the response vector
    :param unmasked_A_df: the DF containing the unmasked A
    :param use_city_center_constraint: indicates whether to use robust constraint which has the distance from the city center
    :param city_center: numpy array of shape (2,) that has the coordinates of the city center
    :param city_center_diag_matrix: numpy array of shape (2, 2) that defines the distance metric around the city center
    :param use_grid: specifies whether to use the square robust uncertainty sets from the grid of the map of London
    """
    n, d = np.shape(A)
    assert np.shape(b) == (n,)

    assert np.shape(city_center) == (2,)
    assert np.shape(city_center_diag_matrix) == (2, 2)
    
    x = cp.Variable(d)              # decision variable for regression coefficients
    intercept = cp.Variable(1)      # decision variable for intercept
    p = cp.Variable(n)              # slack variables (p_i for each sample)
    constraints = [p >= 0]          # errors must be non-negative
    for i in range(n):
        a = A[i]                    # extract the i-th sample's feature vector 
        if not np.any(np.isnan(a)): 
            # standard absolute residual constraint for fully observed data
            # |a^T x + intercept - b[i]| <= p[i]
            constraints.append(cp.abs(cp.sum(cp.multiply(a, x)) + intercept - b[i]) <= p[i])
        else:
            # For rows with missing data (assumed here to have exactly 2 missing entries, e.g., longitude and latitude):
            nan_indices, observed_indices = distinguish_indices(a)
            assert len(observed_indices) > 0
            assert len(nan_indices) == 2
            assert nan_indices[0] < nan_indices[1]

            y_loc1, y_loc2 = dsp.LocalVariable(d), dsp.LocalVariable(d)
            G_constraints1, G_constraints2 = [], []

            # observed entries constraint
            # for all those entries that are not masked, they must be equal
            G_constraints1.append(y_loc1[observed_indices] == a[observed_indices])
            G_constraints2.append(y_loc2[observed_indices] == a[observed_indices])

            # If grid-based uncertainty sets are provided (i.e., use_grid is not None):
            if use_grid is not None:
                # Unpack the grid information: grid_map, grid_center, longitude step, latitude step
                grid_map, grid_center, lng_step, lat_step = use_grid
                # Get the grid cell indices for sample i from the grid_map
                lng_low, lat_low = grid_map[i]
                # Convert grid cell indices to geographic coordinates by scaling and offsetting by grid_center
                lng_low = lng_low * lng_step + grid_center[0]
                lat_low = lat_low * lat_step + grid_center[1]

                # Enforce that the uncertain (masked) longitude value lies within the corresponding grid cell
                # this defines the Box Uncertainty set
                G_constraints1 += [y_loc1[nan_indices[0]] <= lng_low + lng_step, y_loc1[nan_indices[0]] >= lng_low]
                # Similarly, enforce that the uncertain latitude value lies within the grid cell
                G_constraints1 += [y_loc1[nan_indices[1]] <= lat_low + lat_step, y_loc1[nan_indices[1]] >= lat_low]
                # The same grid constraints are added for the second local variable (y_loc2)
                G_constraints2 += [y_loc2[nan_indices[0]] <= lng_low + lng_step, y_loc2[nan_indices[0]] >= lng_low]
                G_constraints2 += [y_loc2[nan_indices[1]] <= lat_low + lat_step, y_loc2[nan_indices[1]] >= lat_low]
    
            # city center constraint constraint
            # note that we square the distance
            if use_city_center_constraint:
                 # For sample i, get the distance from the city center from the unmasked DataFrame
                distance_from_center = unmasked_A_df.iloc[i]["dist"]
                
                # enforce that the squared (weighted) distance between the imputed location and the city center
                # is within the squared distance
                G_constraints1.append(
                    cp.quad_form(y_loc1[nan_indices] - city_center, city_center_diag_matrix) <= distance_from_center ** 2
                )
                G_constraints2.append(
                    cp.quad_form(y_loc2[nan_indices] - city_center, city_center_diag_matrix) <= distance_from_center ** 2
                )
            # Construct the bilinear (saddle) inner product between the regression coefficients 
            # and the local variable.
            # f1 corresponds to the pairing with y_loc1, f2 with y_loc2.
            f1 = dsp.saddle_inner(x, y_loc1)
            f2 = dsp.saddle_inner(-x, y_loc2)

            # use cvxpy suppfunc transform to compute the support function of the uncertainty sets
            # defined by the local variables and constraints.
            # suppfunc (y_loc1, G_constraints1)(x) returns the worst case value of y_loc1^T * x under
            # G constraints1
            # in convex analysis, the support function of a convex set C in the direction x is defined as:
            # sigma_c(x) = sup_{y \in C}(y^T theta)
            # this quantity tells you the largest value that the linear function y^T theta can take when y is
            # restricted to C.

            # in our code some samples have missing feature entries, and we don't know their exact values - but
            # we know they lie in some uncertainty sets defined by constraints.
            # to handle this uncertainty, we create two local variables y_loc1 and y_loc2, which represent
            # possible imputations of the missing values. t
            # the constraints G_constraints1  and G_constraints2 define the uncertainty set C for these missing entries
            
            # the fllowing code:
            # SuppFUnc... computes the support function of the uncertainty set defined by y_loc1 and the constraints
            # in G_constraints1, in the direction of the regression coefficients x. In other words, it returns
            # G1 = sup_{y in C1} y^T x # largest possible value
            # G2 = sup_{y in C2} y^T (-x) # this is equivalent to taking the inf_{y in C_2} y^T x
            G1 = SuppFunc(y_loc1, G_constraints1)(x)
            G2 = SuppFunc(y_loc2, G_constraints2)(-x)

            # for one particular sample, the measured features are incomplete.
            # instead of having a fixed value for y, (the missing entries), we know only that y lies in some set C.
            # by taking the support function in the direction x (or -x), we compute the worst case contribution of
            # the missing features to the prediction x^Ty

            # add robust contraints for the i-th sample
            # these ensure that even in the worst case over the uncertainty sets
            # the absolute error is bounded by the slack variable p[i]
            # p[i] is a slack variable, so we basically penalize for exceeding p[i]
            
            # this is quite like the assignment - we need to put constraints on |prediction - price_vector| <= p[i] for the i-th sample
            # the upper bound constraint here is that the residual (sup_{y in C1} [x^T y] + intercept - b[i]) does not exceed p[i] (slack)
            constraints.append(G1 + intercept - b[i] <= p[i])
            # - (x^T y + intercept - b[i]) <= p[i]
            # -x^Ty - intercept + b[i]
            # SInce G2= sup _{y in C2} (-x^T y)
            # the constraint is sup_{y in C2} [-x^T y] - intercept + b[i]) does not exceed p[i] (slack)
            constraints.append(G2 - intercept + b[i] <= p[i])

    # define the objective function: minimize the average of the squared residuals
    obj = cp.Minimize(cp.sum_squares(p) / n)
    # formulate the opt problem with the objective and accumulated constraints
    prob = cp.Problem(obj, constraints)
    # Solve the problem using the ECOS solver, with verbosity enabled to show progress and details
    prob.solve(solver = cp.ECOS, verbose=True)
    return x.value, intercept.value[0]

def distinguish_indices(a: np.ndarray)-> Tuple[np.ndarray, np.ndarray]:
    """
    Finds the indices where the vector a has Nan entries vs observed entries, and then returns each.

    :param a: A numpy array of shape (n,).
    :return: A Tuple where the first entry is a numpy array of the indices with Nan entries, and the second
             entry is a numpy array of the indices with non-Nan entries.
    """
    nan_indices = np.nonzero(np.isnan(a))[0]
    observed_indices = np.nonzero(np.invert(np.isnan(a)))[0]
    return nan_indices, observed_indices

import pandas
from scipy.stats import norm
from scipy.stats import beta
import numpy as np
import math


def nearest(value, nearest, func, type_fun=float):
    return type_fun(func(float(value) / nearest) * nearest)

def nearest_span(span, round_to):
    """Round the span to the nearest round_to value"""
    span = (nearest(span[0], round_to, math.floor, int), 
            nearest(span[1], round_to, math.floor, int))
    return span

def rotate_points(dataframe, x_key, y_key, angle_deg, rot_point):
    """
    Rotate points around a rotation point by given angle.
    """

    data = dataframe.copy().reset_index(drop=False)

    angle = angle_deg * math.pi / 180.0

    # translate
    data[x_key] = data[x_key] - rot_point[0]
    data[y_key] = data[y_key] - rot_point[1]

    # rotate
    rot_matrix = np.matrix([[math.cos(angle), -math.sin(angle)],
                            [math.sin(angle), math.cos(angle)]])
    rotated = np.matrix(data[[x_key, y_key]]) * rot_matrix
    data[x_key] = rotated[:, 0]
    data[y_key] = rotated[:, 1]

    # translate back
    data[x_key] = data[x_key] + rot_point[0]
    data[y_key] = data[y_key] + rot_point[1]

    return data

def get_peaks_valleys(moving_avg, span):
    working_data = moving_avg[(moving_avg['pos'] > span[0]) & 
                              (moving_avg['pos'] < span[1])]

    maxes = []
    while len(working_data) > 0:
        local_max = working_data.loc[working_data['count'].idxmax()]
        maxes.append((local_max.pos, local_max['count']))

        drop_window = 200
        drop_rows = working_data[(working_data['pos'] > local_max.pos - drop_window/2) & 
                                 (working_data['pos'] < local_max.pos + drop_window/2)]
        working_data = working_data.drop(drop_rows.index)

    working_data = moving_avg[(moving_avg['pos'] > span[0]) & 
                              (moving_avg['pos'] < span[1])]
    mins = []
    while len(working_data) > 0:
        local_min = working_data.loc[working_data['count'].idxmin()]
        mins.append((local_min.pos, local_min['count']))

        drop_window = 200
        drop_rows = working_data[(working_data['pos'] > local_min.pos - drop_window/2) & 
                                 (working_data['pos'] < local_min.pos + drop_window/2)]
        working_data = working_data.drop(drop_rows.index)

    peaks_and_valleys = maxes + mins
    peaks_and_valleys = sorted(peaks_and_valleys, key=lambda k: k[0])

    # TODO: calculate an estimated amplitude and periodicity
    
    return peaks_and_valleys

def moving_average(data, N, key='count', pos_key='pos'):

    c_min, c_max = int(min(data[pos_key])), int(max(data[pos_key]))

    moving_average = {pos_key: [], 'count': []}
    cum_sum = [0]
    i = 1
    for pos in range(c_min, c_max):
        
        count = data[data[pos_key] == pos][key].values
        if len(count) == 0: count = 0
        else: count = count[0]

        cum_sum.append(cum_sum[i-1] + count)
        
        if i >= N:
            moving_ave = float(cum_sum[i] - cum_sum[i-N])/N

            # center the pos by subtracting N/2
            moving_average[pos_key].append(pos-(N/2))
            moving_average['count'].append(moving_ave)

        i += 1

    data = pandas.DataFrame(moving_average)
    return data


def moving_convolution(data, kernel, key='count'):

    values = {'pos': [], key: []}

    if len(data) == 0: return pandas.DataFrame(values)

    c_min, c_max = int(min(data['pos'])), int(max(data['pos']))
    i = 1
    N = len(kernel)

    for pos in range(c_min, c_max):
        
        count = data[data['pos'] == pos][key].values

        values['pos'].append(pos)
        
        i += 1

        kernel_span = pos + kernel.x[0], pos + kernel.x[len(kernel)-1]

        selected = data[(data['pos'] >= kernel_span[0]) & (data['pos'] <= kernel_span[1])]
        
        kernel['moving_pos'] = kernel['x'] + pos

        kernel_merged = pandas.merge(kernel, selected, left_on='moving_pos', right_on='pos')
        kernel_merged['value'] = kernel_merged[key] * kernel_merged['p']

        value = sum(kernel_merged['value'])

        values[key].append(value)

    data = pandas.DataFrame(values)
    return data


def fill_sparse_counts(bin_counts):
    """
    fill in sparse counts with zero
    """
    plot_span = min(bin_counts.pos), max(bin_counts.pos)
    
    all_pos = pandas.DataFrame({'pos': range(plot_span[0], plot_span[1])})

    df_counts = pandas.merge(all_pos, bin_counts, on = 'pos', how = 'left')
    df_counts = df_counts.fillna(0)

    return df_counts[['pos', 'count']]


def derivative(data, key='count'):
    deriv = {'pos': [], 'slope': []}
    for i in range(0, len(data)-1):
        slope = data.loc[i+1][key]-data.loc[i][key]
        deriv['pos'].append(data.loc[i].pos)
        deriv['slope'].append(slope)
    deriv = pandas.DataFrame(deriv)
    # deriv = moving_average(deriv, N=smooth, key='slope')
    return deriv


def nuc_kernel(n, penalty_weight=0.2):
    """Create a kernel scoring the organization of a bin, penalizing the edges"""
    beta_k = beta_kernel(n=n)
    norm_k = normal_kernel(n=n)

    kernel = beta_k.copy()
    kernel.p = -beta_k.p*penalty_weight + norm_k.p*(1-penalty_weight)
    
    return norm_k # kernel


def beta_kernel(n):
    a = 1e-1
    b = 1e-1
    rv = beta(a, b)

    x_step = 1.0/(n+1)
    x_range = (x_step, 1.0-x_step)

    x = np.arange(x_range[0], x_range[1], x_step)
    y = rv.pdf(x)

    y /= sum(y)

    x = x / x_step - n/2 - 1

    kernel = pandas.DataFrame({'x': x, 'p': y})

    return kernel

def normal_kernel(n=50, quantile=1e-5):
    """
    create a kernel matrix based on normal
    
    n - length of the kernel
    quantile - cutoff for two tails to create kernel array
    """

    # choose quantile to decide where in the normal curve
    # we want to generate our kernel from
    q1, q2 = norm.ppf(quantile), norm.ppf(1 - quantile)

    # create kernel
    x = np.linspace(q1, q2, n)
    p = norm.pdf(x, scale=1)
    
    # scale to size of array we want
    x = x * (n-1)/q2/2
    x = map(lambda entry: int(round(entry)), x)
    
    # normalize to sum to 1
    p /= sum(p)

    return pandas.DataFrame({'x': x, 'p': p})


def find_max_mins(data, scale = 1.0, N=101, deriv_scale=1, ret_debug=False, key='count', fill_sparse=True):
    """
    Find the max and mins of a dataset by smoothing the data.

    params:
        data - periodic-like data, defaults to nucleosome count data
        scale - scalar to multiply the moving average calculated after smoothing the raw data 
        N - number of samples for moving average
        N_1 - number of samples to smooth moving average after initial moving average calculation
        deriv_scale - scale derivative for visual inspection
    """

    if len(data) > 0 and fill_sparse:
       data = fill_sparse_counts(data)

    kernel = normal_kernel(n=N)

    smoothed_data = moving_convolution(data, kernel, key=key)
    smoothed_data[key] = smoothed_data[key] * scale

    deriv = derivative(smoothed_data, key=key)
    # deriv = moving_average(deriv, N=20, key='slope')
    deriv[key] = deriv['slope']*deriv_scale
    
    deriv_2 = derivative(deriv, key=key)
    # deriv = moving_average(deriv, N=smooth, key='slope')
    deriv_2[key] = deriv_2['slope']*deriv_scale

    # find where slopes are 0
    j = 0
    local_max_mins = {'pos': [], key: [], 'max_min': []}
    for i in range(0, len(deriv)-1, 1):

        d_1 = deriv.loc[i]
        d_2 = deriv.loc[i+1]
        row = d_1

        if d_1[key] * d_2[key] <= 0:

            count = float((smoothed_data[smoothed_data['pos'] == row.pos])[key])

            second_deriv = deriv_2[deriv_2['pos'] == row.pos][key]
            if len(second_deriv) == 0: continue

            # use second deriv to determine if a max or minimum
            if float(second_deriv) > 0: max_min = '-'
            else: max_min = '+'

            local_max_mins['max_min'].append(max_min)
            local_max_mins[key].append(count)
            local_max_mins['pos'].append(row.pos)

            j += 1

    local_max_mins = pandas.DataFrame(local_max_mins)

    scores = {'pos': [], 'score': []}
    for i in range(0, len(local_max_mins), 1):

        row_1 = local_max_mins.loc[i]

        # # calculate the score for max critical points
        # if row_1.max_min == "+":

        #     score = 0
        #     n = 0
            
        #     # if there is a left critical point, get its score
        #     if i > 0:
        #         row_0 = local_max_mins.loc[i-1]
        #         score += score_pair_critical(row_0, row_1)
        #         n += 1

        #     # if there is a right critical point, get its score
        #     if i < len(local_max_mins)-1:
        #         row_2 = local_max_mins.loc[i+1]
        #         score = score_pair_critical(row_1, row_2)
        #         n += 1
                
        #     if n > 0:
        #         score /= n

        scores['pos'].append(row_1.pos)
        scores['score'].append(row_1[key])

    scores = pandas.DataFrame(scores)

    return smoothed_data, deriv, deriv_2, local_max_mins, scores


def score_pair_critical(row_1, row_2, scale=1):
    """calculate the score of a peak of nuc fragment reads"""
    count_diff = (row_2['count'] - row_1['count'])

    score = abs(count_diff*scale)
    return score


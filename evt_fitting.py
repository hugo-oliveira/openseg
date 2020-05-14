import sys

import libmr

# ---------------------------------------------------------------------------------
def weibull_tailfitting(mean, distances, num_classes, tailsize=20):

    weibull_model = {}
    for i in range(num_classes):
        weibull_model[i] = {}

        weibull_model[i]['distances'] = distances[i]
        weibull_model[i]['mean_vec'] = mean[i]
        weibull_model[i]['weibull_model'] = []

        mr = libmr.MR()
        
        tail_to_fit = sorted(distances[i])[-tailsize:]
        mr.fit_high(tail_to_fit, len(tail_to_fit))
        weibull_model[i]['weibull_model'] += [mr]
        
        sys.stdout.flush()
        
    return weibull_model


# ---------------------------------------------------------------------------------
def query_weibull(weibull_model, label_index):

    category_weibull = []
    category_weibull += [weibull_model[label_index]['mean_vec']]
    category_weibull += [weibull_model[label_index]['distances']]
    category_weibull += [weibull_model[label_index]['weibull_model'][0]]

    return category_weibull

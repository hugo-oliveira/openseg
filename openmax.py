import sys
import os
import numpy as np
import scipy as sp
import scipy.spatial.distance as spd
from evt_fitting import weibull_tailfitting, query_weibull

from joblib import Parallel, delayed

import libmr

# ---------------------------------------------------------------------------------
def compute_distance(image_feat, mean_vec, distance_type='eucos'):
    
    dist = 0.0
    
    if distance_type == 'euclidean':
        
        dist = np.square(image_feat - mean_vec)
        dist = np.sqrt(dist.sum()) / 200.
        
    elif distance_type == 'cosine':
        
        dist = 1.0 - ((image_feat * mean_vec).sum() / (np.linalg.norm(image_feat, ord=2) * np.linalg.norm(mean_vec, ord=2)))
    
    elif distance_type == 'eucos':
        
        dist_euc = np.square(image_feat - mean_vec)
        dist_euc = np.sqrt(dist_euc.sum()) / 200.
        
        dist_cos = 1.0 - ((image_feat * mean_vec).sum() / (np.linalg.norm(image_feat, ord=2) * np.linalg.norm(mean_vec, ord=2)))
        
        dist = dist_euc + dist_cos
        
    return dist

# ---------------------------------------------------------------------------------
def compute_openmax_probability(openmax_fc8, openmax_score_u, num_classes):
    
    prob_scores, prob_unknowns = [], []
    for category in range(num_classes):
        prob_scores += [sp.exp(openmax_fc8[category])]

    total_denominator = sp.sum(sp.exp(openmax_fc8)) + sp.exp(sp.sum(openmax_score_u))
    prob_scores = [prob_scores / total_denominator]
    prob_unknowns = [sp.exp(sp.sum(openmax_score_u)) / total_denominator]

    prob_scores = sp.asarray(prob_scores)
    prob_unknowns = sp.asarray(prob_unknowns)

    scores = sp.mean(prob_scores, axis=0)
    unknowns = sp.mean(prob_unknowns, axis=0)
    modified_scores = scores.tolist() + [unknowns]
    assert len(modified_scores) == num_classes + 1, 'Error in openmax!'
    return modified_scores


# ---------------------------------------------------------------------------------
def process(i, n_pixels, weibull_model, pixel_feat, softmax_feat, alpharank, num_classes, distance_type, openmax_probs):

    ranked_list = softmax_feat.argsort().ravel()[::-1]
    
    alpha_weights = [((alpharank + 1) - z) / float(alpharank) for z in range(1, alpharank + 1)]
    ranked_alpha = sp.zeros(num_classes)
    for z in range(len(alpha_weights)):
        ranked_alpha[ranked_list[z]] = alpha_weights[z]
    
    # Now recalibrate each fc8 score for each channel and for each class to include probability of unknown
    openmax_fc8, openmax_score_u = [], []
    
    for j in range(num_classes):
        # get distance between current channel and mean vector
        category_weibull = query_weibull(weibull_model, j)
        channel_distance = compute_distance(pixel_feat, category_weibull[0], distance_type=distance_type)
        
        # obtain w_score for the distance and compute probability of the distance
        # being unknown wrt to mean training vector and channel distances for
        # category and channel under consideration
        wscore = category_weibull[2].w_score(channel_distance)
        modified_fc8_score = pixel_feat[j] * (1 - wscore * ranked_alpha[j])
        openmax_fc8 += [modified_fc8_score]
        openmax_score_u += [pixel_feat[j] - modified_fc8_score]
    
    openmax_fc8 = sp.asarray(openmax_fc8)
    openmax_score_u = sp.asarray(openmax_score_u)
    
    # Pass the recalibrated fc8 scores for the image into openmax
    openmax_probs[i] = compute_openmax_probability(openmax_fc8, openmax_score_u, num_classes)

# ---------------------------------------------------------------------------------
def recalibrate_scores(weibull_model, test_features, test_softmax, num_classes, alpharank=10, distance_type='eucos'):
    
    test_features = test_features.reshape(test_features.shape[0] * test_features.shape[1] * test_features.shape[2],
                                          test_features.shape[3])
    test_softmax = test_softmax.reshape(test_softmax.shape[0] * test_softmax.shape[1] * test_softmax.shape[2],
                                        test_softmax.shape[3])
#     openmax_probs = np.empty([test_features.shape[0], num_classes+1], dtype=np.float)
    openmax_probs = [np.zeros([num_classes + 1], dtype=np.float) for i in range(test_features.shape[0])]
    
    n_jobs = 10
    
    for i in range(len(test_features)):
        process(i, test_features.shape[0], weibull_model, test_features[i], test_softmax[i], alpharank, num_classes, distance_type, openmax_probs)
#     Parallel(n_jobs=n_jobs)(delayed(process)(i, test_features.shape[0], weibull_model, test_features[i], test_softmax[i], alpharank, num_classes, distance_type, openmax_probs) for i in range(len(test_features)))

    openmax_probs = np.asarray(openmax_probs)
    return openmax_probs  # , sp.asarray(softmax_probab)


# ---------------------------------------------------------------------------------
def openmax(test_features, test_softmax, mean, distances, num_classes, distance_type, weibull_tailsize, alpha_rank):
    
    weibull_model = weibull_tailfitting(mean, distances, num_classes, tailsize=weibull_tailsize)

    openmax = recalibrate_scores(weibull_model, test_features, test_softmax, num_classes, alpharank=alpha_rank, distance_type=distance_type)

    return openmax

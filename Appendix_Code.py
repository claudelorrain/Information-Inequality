# Please pip install the below packages or ensure your IDE has them imported before proceeding!
# ADVICE: The results section generates many images and tables in sequence. It would be best to comment
# out everything except the figure(s) you want to generate since some graphs and figures take over 60 minutes to generate data!

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import networkx as nx
import numpy.random
import pandas as pd
from statsmodels.tsa.stattools import adfuller
from statsmodels.api import OLS
from stargazer.stargazer import Stargazer
import patsy
np.set_printoptions(threshold=np.inf)


# ============================== Statistical Tests ===============================
def adf_test(timeseries):
    if np.all(timeseries) == timeseries[1]:
        return 0
    else:
        dftest = adfuller(timeseries, autolag="AIC")
        p_value = dftest[1]
        return p_value

def wage_curve_test(wages, unemployment_rates):
    raw_data = np.transpose(np.vstack((wages, unemployment_rates)))
    filtered_data = raw_data[~np.any(raw_data == 0, axis=1)]
    np.log(filtered_data)
    df = pd.DataFrame(filtered_data, columns=['Log_Wage', 'Log_U_t'])

    y, X = patsy.dmatrices('Log_Wage ~ Log_U_t', data=df, return_type='dataframe')
    model1 = OLS(y, X)
    results1 = model1.fit(cov_type='HAC', cov_kwds={'maxlags':1})
    return results1.pvalues.loc['Log_U_t']


# ================================= Heatmap Generation ======================================

def heatmap(data, row_labels, col_labels, ax=None,
            cbar_kw=None, cbarlabel="", **kwargs):

    if ax is None:
        ax = plt.gca()

    if cbar_kw is None:
        cbar_kw = {}

    # Plot the heatmap
    im = ax.imshow(data, **kwargs)

    # Create colorbar
    cbar = ax.figure.colorbar(im, ax=ax, **cbar_kw)
    cbar.ax.set_ylabel(cbarlabel, rotation=-90, va="bottom")

    ax.set_xticks(np.arange(data.shape[1]), labels=col_labels)
    ax.set_yticks(np.arange(data.shape[0]), labels=row_labels)
    ax.tick_params(top=True, bottom=False,
                   labeltop=True, labelbottom=False)
    plt.setp(ax.get_xticklabels(), rotation=0, ha="right",
             rotation_mode="anchor")
    ax.spines[:].set_visible(False)

    ax.set_xticks(np.arange(data.shape[1]+1)-.5, minor=True)
    ax.set_yticks(np.arange(data.shape[0]+1)-.5, minor=True)
    ax.grid(which="minor", color="w", linestyle='-', linewidth=3)
    ax.tick_params(which="minor", bottom=False, left=False)

    return im, cbar

def annotate_heatmap(im, data=None, valfmt="{x:.2f}",
                     textcolors=("black", "white"),
                     threshold=None, **textkw):

    if not isinstance(data, (list, np.ndarray)):
        data = im.get_array()

    # Normalize the threshold to the images color range.
    if threshold is not None:
        threshold = im.norm(threshold)
    else:
        threshold = im.norm(data.max())/2.

    # Set default alignment to center, but allow it to be
    # overwritten by textkw.
    kw = dict(horizontalalignment="center",
              verticalalignment="center")
    kw.update(textkw)

    # Get the formatter in case a string is supplied
    if isinstance(valfmt, str):
        valfmt = matplotlib.ticker.StrMethodFormatter(valfmt)

    # Loop over the data and create a `Text` for each "pixel".
    # Change the text's color depending on the data.
    texts = []
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            kw.update(color=textcolors[int(im.norm(data[i, j]) > threshold)])
            text = im.axes.text(j, i, valfmt(data[i, j], None), **kw)
            texts.append(text)

    return texts

# =============================== Pre-Simulation Graphs =====================================
def sample_networks():
    # Code to generate sample network and degree distribution in figures 2 and 3

    # 30 worker random BA(5) network
    nx.draw(nx.barabasi_albert_graph(30, 5, seed=None, initial_graph=None), with_labels=True)
    plt.show()

    # Degree distribution graph
    number_workers_sample = 200
    degree_dists = np.zeros(shape=(100, number_workers_sample))

    for i in range(100):
        B = nx.barabasi_albert_graph(number_workers_sample, np.random.randint(3, 8), seed=None, initial_graph=None)
        arr_B = nx.to_numpy_array(B)
        degree_dists[i] = np.sort(np.matmul(arr_B, np.ones(number_workers_sample)))[::-1]
        print(degree_dists[i])

    maximum = np.ceil(np.max(np.mean(degree_dists, axis=0))).astype(int)
    plt.hist(np.mean(degree_dists, axis=0), bins=maximum, histtype='bar', edgecolor='black', linewidth=1.2,
             color='orange')
    plt.gca().set(ylabel='Frequency', xlabel='Degree')
    plt.show()

# ================================= Subroutines ===========================================
def gini(array):
    array = array.flatten()
    if np.amin(array) < 0:
        array -= np.amin(array)
    array = array + 0.00000001
    array = np.sort(array)
    index = np.arange(1,array.shape[0]+1)
    n = array.shape[0]
    return np.divide((np.sum((2*index-n-1) * array)),(n*np.sum(array)))

def make_queue(probabilities, employment_alloc_prev):
    # Given the information network and previous hiring, return a queuing.
    if queue_type == "networked":
        queue = np.zeros(shape=[number_workers, number_firms+1])  # LxF+1 0s. Any workers who select F+1 are not queueing at any firm.

        employment_alloc_prev = employment_alloc_prev.astype(int)
        rng = numpy.random.default_rng()

        choices = np.array(
            [np.where(employment_alloc_prev[i]<0,
                      rng.choice(np.where(employment_alloc_prev<0,np.random.randint(number_firms, size=number_workers), employment_alloc_prev),
                                 n_choices, replace=True, p=probabilities[i]),
                      rng.choice(employment_alloc_prev, n_choices, replace=True, p=probabilities[i])) for i in
             range(number_workers)])  # Choose two with replacement

        np.put_along_axis(queue, choices, 1, axis=1)
        queue = np.transpose(queue[:, :-1])  # Now take transpose and drop those who didn't queue at any firm -- easier for firm calculations. So queue is now FxL.

    elif queue_type == "uniform":
        queue = np.zeros(shape=[number_workers, number_firms])  # LxF 0s

        rng = numpy.random.default_rng()
        # Completely random choices
        choices = np.array(
            [rng.choice(number_firms, n_choices, replace=True) for i in
             range(number_workers)])  # Choose two with replacement

        np.put_along_axis(queue, choices, 1, axis=1)
        queue = np.transpose(queue)  # Now take transpose -- easier for firm calculations. So queue is now FxL.

    return queue


def set_wages(queue, beta, reservation_wages, all_competences):
    # After workers decide which firms to queue at, firms offer wages to workers in their queue as a return for
    # their skill-competence match with requirement, workers accept if the wage is greater than reservation.
    # Queue FxN
    # Wages FxN
    # Acceptances FxN

    # Wage setting

    queue_competences = np.multiply(all_competences, queue)

    wages = np.multiply(np.transpose(np.tile(beta, reps=(number_workers, 1))), np.power(queue_competences, a))
    high = np.zeros_like(wages)
    np.put_along_axis(high, np.argmax(wages, axis=0).reshape(1, number_workers), 1, axis=0)
    highest_wages = wages * high

    acceptances = (highest_wages > np.tile(reservation_wages, reps=(number_firms, 1))).astype(int)

    final_wages = np.multiply(acceptances, wages)

    return acceptances, final_wages


def select(beta, price, quantities, wages):
    # Firms that make negative profits are removed (their betas)
    # Replaced with a new entrant (new beta)

    beta_selected = beta

    profits = price * quantities - np.sum(wages, axis=1)

    indices = np.asarray(profits < 0).nonzero()[0]  # Indices where profits are negative
    number_bust = len(indices)
    beta_selected[indices] = np.random.uniform(1.0, 10.0, size=number_bust)  # Random beta

    return indices, beta_selected

def update(acceptances, queue, beta):
    # Firm updates betas (return to competence)
    # Calculate proportion of queued workers who accepted wage offer in previous round p
    total_queued = np.sum(queue, axis=1)

    # Prop is proportion who accepted offer
    prop = np.zeros(number_firms)  # Set to 0 if no one queued
    nonzero_queued = total_queued.nonzero()[0]
    prop[nonzero_queued] = (np.sum(acceptances[nonzero_queued], axis=1)) / total_queued[
        nonzero_queued]  # Proportion of queue who accepted
    prop[np.where(total_queued == 1)] = 0
    prop[nonzero_queued] = prop[nonzero_queued] - rho

    prop_diff = np.zeros_like(prop)
    prop_diff[np.where(prop < 0)] = 1  # Increase firm return (too few hired from queue)
    prop_diff[np.where(prop > 0)] = -1  # Reduce firm return (too many hired from queue)
    adjust = np.zeros(number_firms)
    for i in range(number_firms):
        adjust[i] = 1 + (prop_diff[i] * abs(np.random.normal(0, sigma_beta)))

    beta_next = np.multiply(beta, adjust)

    employment_alloc = np.matmul(np.transpose(np.arange(1, number_firms + 1)), acceptances)-1

    return beta_next, employment_alloc

def run_simulation():

    # Initialise variables
    requirements = requirements_initial
    alpha = alpha_initial
    beta = beta_initial
    skills = skills_initial
    reservation_wages = reservation_wages_initial
    B = seed_B

    # Setup aggregate lists
    total_unemployment = np.zeros(number_periods)
    total_wages = np.zeros(number_periods)
    wage_distribution = np.zeros(shape=[number_periods, number_workers])
    real_gdp = np.zeros(number_periods)
    prices = np.zeros(number_periods)
    active_competence_path = np.zeros(number_periods)
    beta_path = np.zeros(shape=[number_periods, number_firms])
    firm_destruction = np.zeros(number_periods)


    # Random initial job/unemployment allocations
    acceptances = np.zeros(shape=[number_firms + 1, number_workers])  # FxL 0s
    np.put_along_axis(acceptances, np.random.randint(0, number_firms+1, (1, number_workers)), 1, axis=0)
    acceptances = acceptances[:-1, ]
    employment_alloc = np.matmul(np.transpose(np.arange(1, number_firms + 1)), acceptances)-1

    # Draw the network if needed
    # nx.draw(B)

    # Initialise network and queuing probabilities
    arr_B = nx.to_numpy_array(B)
    scaled_B=arr_B

    # Ensure no unconnected workers
    for i in range(number_workers):
        if np.all(scaled_B[i]==0):
            scaled_B[i][np.random.randint(number_workers)] = 1

    scaled_sum = scaled_B.sum(axis=1)[:, None]  # These are row sums.
    probabilities = np.divide(scaled_B * (1 - loyalty), scaled_sum, out=np.zeros_like(scaled_B),
                              where=scaled_sum != 0)  # So probability dist over workers are row-wise.

    np.fill_diagonal(probabilities, loyalty)

    # Efficiency
    all_competences = np.zeros(shape=[number_firms, number_workers]).astype(float)

    for i in range(number_firms): # Have to use a for loop, don't see any way around it
        all_competences[i] = skills[2] * np.exp(-0.5 * ((requirements[i] - skills[0]) / skills[1]) ** 2)

    # Maximum possible competence C*
    maximum_total_competence = np.sum(np.max(all_competences, axis=0))

    # ------------------------ Main Loop -----------------------------
    for i in range(number_periods):
        beta_path[i] = beta # Record firm requirements

        # Queueing according to B, previous acceptances and loyalty
        queue = make_queue(probabilities, employment_alloc)

        # Set and record wages
        acceptances, wages = set_wages(queue, beta, reservation_wages, all_competences)
        w_unemployed_ind = np.nonzero(np.sum(acceptances, axis=0) == 0)[0]
        total_unemployment[i] = len(w_unemployed_ind)
        wage_distribution[i] = np.sum(wages, axis=0)
        wage_distribution[i][w_unemployed_ind] = reservation_wages[w_unemployed_ind]
        total_wages[i] = np.sum(wage_distribution[i])

        # Set and record production and price
        active_competences = np.multiply(all_competences, acceptances)
        active_competence_path[i] = np.sum(active_competences)/maximum_total_competence
        quantities = np.multiply(alpha, np.power(np.sum(active_competences, axis=1), a))
        real_gdp[i] = np.sum(quantities)
        price = total_wages[i]/real_gdp[i]
        prices[i] = price

        # Select and record loss-making firms
        indices, beta_selected = select(beta, price, quantities, wages)
        firm_destruction[i] = len(indices)

        # Update wage rules
        beta, employment_alloc = update(acceptances, queue, beta_selected)
        # Employment alloc is calculated in update() from only current acceptances, not selected information, so used for next period queueing also.

    return total_unemployment, total_wages, wage_distribution, real_gdp, prices, active_competence_path, firm_destruction, beta_path, all_competences


# ===================================== Results ==========================================

# Figures 2 and 3
sample_networks()

# ---------------------------- 4.3.1 Business cycles -----------------------
#  Figures 5a and 5b
# The exact figures will vary slightly since a new random network and distribution of skills and requirements is generated.
# However, the methodology is precisely the same as the one that generated the figures in the paper.

number_firms = 50
number_workers = 250
number_periods = 500
n_choices = 5
reservation_wages_initial = abs(np.random.normal(loc=1.0, scale=0, size=number_workers))
loyalty = 1/number_firms

# Skills come parameterised by [mean, variance, scale (stretch up or down)]
skills_initial = np.stack((np.random.uniform(0, 1, size=number_workers),  # Mean
                               np.random.uniform(0.1, 0.1, size=number_workers),  # Variance
                               np.random.uniform(1, 1, size=number_workers)))  # Scale
requirements_initial = np.random.uniform(0,1,size=number_firms)

# Technology
alpha_initial = abs(np.linspace(start=10, stop=10, num=number_firms))
a = 0.5

# Return
beta_initial = np.random.uniform(1.0,10.0,size=number_firms)
sigma_beta = 0.1
rho = 0.5

seed_B = nx.barabasi_albert_graph(number_workers,5, seed=None, initial_graph=None)
nx.draw(seed_B)
queue_type = "uniform"
(upt_path5a, wages_path5a, wage_dist_path5a, gdp_path5a, price_path5a, active_competence_path_final5a, dest_path5a,
 beta_path_final5a, all_competences_calc5a) = run_simulation()
queue_type = "networked"
(upt_path5b, wages_path5b, wage_dist_path5b, gdp_path5b, price_path5b, active_competence_path_final5b, dest_path5b,
 beta_path_final5b, all_competences_calc5b) = run_simulation()

# UNIFORM
fig5a, axs5a = plt.subplots(2, 2)
axs5a[0,0].plot(np.arange(number_periods), gdp_path5a, marker=".")
axs5a[0,0].set_title("GDP")
axs5a[0,1].plot(np.arange(number_periods), upt_path5a/number_workers, marker=".")
axs5a[0,1].set_title("Unemployment rate U(t)")
axs5a[1,0].plot(np.arange(number_periods), np.array([gini(i) for i in wage_dist_path5a]), marker=".")
axs5a[1,0].set_title("Gini coefficient G(t)")
axs5a[1,1].plot(np.arange(number_periods), active_competence_path_final5a, marker=".")
axs5a[1,1].set_title("Efficiency E(t)")


fig5a.set_tight_layout(True)
plt.savefig("5a.jpg")

# NETWORKED
fig5b, axs5b = plt.subplots(2, 2)
axs5b[0,0].plot(np.arange(number_periods), gdp_path5b, marker=".")
axs5b[0,0].set_title("GDP")
axs5b[0,1].plot(np.arange(number_periods), upt_path5b/number_workers, marker=".")
axs5b[0,1].set_title("Unemployment rate U(t)")
axs5b[1,0].plot(np.arange(number_periods), np.array([gini(i) for i in wage_dist_path5b]), marker=".")
axs5b[1,0].set_title("Gini coefficient G(t)")
axs5b[1,1].plot(np.arange(number_periods), active_competence_path_final5b, marker=".")
axs5b[1,1].set_title("Efficiency E(t)")
fig5b.set_tight_layout(True)
plt.savefig("5b.jpg")


# Figures 6a and 6b -- same network, skills, and requirements as above, only sigma_beta changed
sigma_beta = 0.7

queue_type = "uniform"
(upt_path6a, wages_path6a, wage_dist_path6a, gdp_path6a, price_path6a, active_competence_path_final6a, dest_path6a,
 beta_path_final6a, all_competences_calc6a) = run_simulation()
queue_type = "networked"
(upt_path6b, wages_path6b, wage_dist_path6b, gdp_path6b, price_path6b, active_competence_path_final6b, dest_path6b,
 beta_path_final6b, all_competences_calc6b) = run_simulation()

# UNIFORM
fig6a, axs6a = plt.subplots(2, 2)
axs6a[0,0].plot(np.arange(number_periods), gdp_path6a, marker=".")
axs6a[0,0].set_title("GDP")
axs6a[0,1].plot(np.arange(number_periods), upt_path6a/number_workers, marker=".")
axs6a[0,1].set_title("Unemployment rate U(t)")
axs6a[1,0].plot(np.arange(number_periods), np.array([gini(i) for i in wage_dist_path6a]), marker=".")
axs6a[1,0].set_title("Gini coefficient G(t)")
axs6a[1,1].plot(np.arange(number_periods), active_competence_path_final6a, marker=".")
axs6a[1,1].set_title("Efficiency E(t)")

fig6a.set_tight_layout(True)
plt.savefig("6a.jpg")

# NETWORKED
fig6b, axs6b = plt.subplots(2, 2)
axs6b[0,0].plot(np.arange(number_periods), gdp_path6b, marker=".")
axs6b[0,0].set_title("GDP")
axs6b[0,1].plot(np.arange(number_periods), upt_path6b/number_workers, marker=".")
axs6b[0,1].set_title("Unemployment rate U(t)")
axs6b[1,0].plot(np.arange(number_periods), np.array([gini(i) for i in wage_dist_path6b]), marker=".")
axs6b[1,0].set_title("Gini coefficient G(t)")
axs6b[1,1].plot(np.arange(number_periods), active_competence_path_final6b, marker=".")
axs6b[1,1].set_title("Efficiency E(t)")

fig6b.set_tight_layout(True)
plt.savefig("6b.jpg")


# Figures 7a and 7b: ADF Test for stationarity of each of the 4 series

number_firms = 25
number_workers = 125
number_periods = 500
reservation_wages_initial = abs(np.random.normal(loc=1.0, scale=0, size=number_workers))
loyalty = 1/number_firms
alpha_initial = abs(np.linspace(start=10, stop=10, num=number_firms))
a = 0.5
rho = 0.5

# Return sensitivity list
sigma_beta_list = np.arange(0,1.05,0.05)
repetitions = 15
# Lists of
gdp_p_list_7a = np.transpose(np.vstack((sigma_beta_list, np.zeros(shape=(repetitions, len(sigma_beta_list))))))
upt_p_list_7a = np.transpose(np.vstack((sigma_beta_list, np.zeros(shape=(repetitions, len(sigma_beta_list))))))
gini_p_list_7a = np.transpose(np.vstack((sigma_beta_list, np.zeros(shape=(repetitions, len(sigma_beta_list))))))
efficiency_p_list_7a = np.transpose(np.vstack((sigma_beta_list, np.zeros(shape=(repetitions, len(sigma_beta_list))))))

queue_type = "networked" # 7a or b depending on uniform or networked
for i in range(len(sigma_beta_list)):
    for j in range(repetitions):
        print(i, ", ", j)
        n_choices = j+1
        sigma_beta = sigma_beta_list[i]
        seed_B = nx.barabasi_albert_graph(number_workers, 5, seed=None, initial_graph=None)
        skills_initial = np.stack((np.random.uniform(0, 1, size=number_workers),  # Mean
                                   np.random.uniform(0.1, 0.1, size=number_workers),  # Variance
                                   np.random.uniform(1, 1, size=number_workers)))  # Scale
        requirements_initial = np.random.uniform(0,1,size=number_firms)
        beta_initial = np.random.uniform(1.0, 10.0, size=number_firms)


        (upt_path7a, wages_path7a, wage_dist_path7a, gdp_path7a, price_path7a, active_competence_path_final7a, dest_path7a,
         beta_path_final7a, all_competences_calc7a) = run_simulation()
        gini_path_7a = np.array([gini(i) for i in wage_dist_path7a])

        p_value_gdp = adf_test(gdp_path7a)
        p_value_upt = adf_test(upt_path7a/number_workers)
        p_value_gini = adf_test(gini_path_7a)
        p_value_efficiency = adf_test(active_competence_path_final7a)

        gdp_p_list_7a[i,j+1] = p_value_gdp
        upt_p_list_7a[i,j+1] = p_value_upt
        gini_p_list_7a[i,j+1] = p_value_gini
        efficiency_p_list_7a[i,j+1] = p_value_efficiency

significance_level = 0.05
gdp_p_list_7a_prop = np.sum(gdp_p_list_7a[:, 1:] <= significance_level, axis=1)/repetitions
upt_p_list_7a_prop = np.sum(upt_p_list_7a[:, 1:] <= significance_level, axis=1)/repetitions
gini_p_list_7a_prop = np.sum(gini_p_list_7a[:, 1:] <= significance_level, axis=1)/repetitions
efficiency_p_list_7a_prop = np.sum(efficiency_p_list_7a[:, 1:] <= significance_level, axis=1)/repetitions


fig7a, axs7a = plt.subplots(2, 2)
axs7a[0,0].plot(sigma_beta_list,gdp_p_list_7a_prop, marker=".", color='r')
axs7a[0,0].set_title("GDP")
axs7a[0,1].plot(sigma_beta_list, upt_p_list_7a_prop, marker=".",color='r')
axs7a[0,1].set_title("Unemployment rate U(t)")
axs7a[1,0].plot(sigma_beta_list, gini_p_list_7a_prop, marker=".",color='r')
axs7a[1,0].set_title("Gini coefficient G(t)")
axs7a[1,1].plot(sigma_beta_list, efficiency_p_list_7a_prop, marker=".",color='r')
axs7a[1,1].set_title("Efficiency E(t)")

fig7a.set_tight_layout(True)
plt.savefig("7b.jpg") # or change to 7a

# ------------------------ 4.3.2 Wage curve -----------------------------
# Figures 8a and 8b

number_firms = 50
number_workers = 250
number_periods = 500
n_choices = 5
reservation_wages_initial = abs(np.random.normal(loc=1.0, scale=0, size=number_workers))
loyalty = 1/number_firms

# Skills come parameterised by [mean, variance, scale (stretch up or down)]
skills_initial = np.stack((np.random.uniform(0, 1, size=number_workers),  # Mean
                               np.random.uniform(0.1, 0.1, size=number_workers),  # Variance
                               np.random.uniform(1, 1, size=number_workers)))  # Scale
requirements_initial = np.random.uniform(0,1,size=number_firms)

# Technology
alpha_initial = abs(np.linspace(start=10, stop=10, num=number_firms))
a = 0.5

# Return
beta_initial = np.random.uniform(1.0,10.0,size=number_firms)
sigma_beta = 0.15
rho = 0.5

seed_B = nx.barabasi_albert_graph(number_workers,5, seed=None, initial_graph=None)
queue_type = "uniform"
(upt_path8a, wages_path8a, wage_dist_path8a, gdp_path8a, price_path8a, active_competence_path_final8a, dest_path8a,
 beta_path_final8a, all_competences_calc8a) = run_simulation()
queue_type = "networked"
(upt_path8b, wages_path8b, wage_dist_path8b, gdp_path8b, price_path8b, active_competence_path_final8b, dest_path8b,
 beta_path_final8b, all_competences_calc8b) = run_simulation()

# UNIFORM
coefa = np.polyfit(upt_path8a/number_workers, wages_path8a/number_workers,1)
poly1d_fn = np.poly1d(coefa)
# poly1d_fn is now a function which takes in x and returns an estimate for y

fig8a, axs8a = plt.subplots()
axs8a.scatter(upt_path8a/number_workers, wages_path8a/number_workers, marker='.')
axs8a.plot(upt_path8a/number_workers, poly1d_fn(upt_path8a/number_workers), linestyle='dashed', color='black')
axs8a.set_ylabel("Mean wage", fontsize=14)
axs8a.set_xlabel("Unemployment rate", fontsize=14)

fig8a.set_tight_layout(True)
plt.savefig("8a.jpg")

# NETWORKED
coefb = np.polyfit(upt_path8b/number_workers, wages_path8b/number_workers,1)
poly1d_fn = np.poly1d(coefb)
fig8b, axs8b = plt.subplots()
axs8b.scatter(upt_path8b/number_workers, wages_path8b/number_workers, marker=".")
axs8b.plot(upt_path8b/number_workers, poly1d_fn(upt_path8b/number_workers), linestyle='dashed', color='black')
axs8b.set_ylabel("Mean wage", fontsize=14)
axs8b.set_xlabel("Unemployment rate", fontsize=14)

fig8b.set_tight_layout(True)
plt.savefig("8b.jpg")

# Table 2

number_firms = 20
number_periods = 500
loyalty = 1/number_firms
alpha_initial = abs(np.linspace(start=10, stop=10, num=number_firms))
a = 0.5
rho = 0.5
queue_type = "networked"
sigma_beta = 0.15
repetitions = 5

choices_list = np.arange(1, 11)
workers_list = number_firms*np.arange(1,6)

# List of p values
p_list_9a = np.zeros(shape=(int(len(choices_list)*len(workers_list)), repetitions))

# For 1-10 choices, for 20-100 workers, for 5 repetitions
for i in range(len(choices_list)):
    for j in range(len(workers_list)):
        for k in range(repetitions):
            print(i, ", ", j, ",", k)
            n_choices = choices_list[i]
            number_workers = workers_list[j]
            reservation_wages_initial = abs(np.random.normal(loc=1.0, scale=0, size=number_workers))
            seed_B = nx.barabasi_albert_graph(number_workers, 5, seed=None, initial_graph=None)
            skills_initial = np.stack((np.random.uniform(0, 1, size=number_workers),  # Mean
                                       np.random.uniform(0.1, 0.1, size=number_workers),  # Variance
                                       np.random.uniform(1, 1, size=number_workers)))  # Scale
            requirements_initial = np.random.uniform(0,1,size=number_firms)
            beta_initial = np.random.uniform(1.0, 10.0, size=number_firms)


            (upt_path9a, wages_path9a, wage_dist_path9a, gdp_path9a, price_path9a, active_competence_path_final9a, dest_path9a,
             beta_path_final9a, all_competences_calc9a) = run_simulation()

            p_value = wage_curve_test(wages_path9a/number_workers, upt_path9a/number_workers)
            p_list_9a[i*len(workers_list)+j,k] = p_value
    
significance_level = 0.05

p_list_9a_prop = np.sum(p_list_9a <= significance_level, axis=1)/repetitions
total_list = np.vstack((np.repeat(choices_list, len(workers_list)), np.tile(workers_list, len(choices_list)), p_list_9a_prop))
print(np.transpose(total_list))

# ------------------ 4.3.3 Upt, efficiency, and gini ---------------------

# Figures 9a, 9b, 10, 10b, 11a, 11b
number_firms = 20
number_periods = 500
loyalty = 1/number_firms
alpha_initial = abs(np.linspace(start=10, stop=10, num=number_firms))
a = 0.5
rho = 0.5
sigma_beta = 0.15
repetitions = 10  # Number of Monte Carlo runs at each combination

choices_list = np.arange(1, 11)  # f_n choices
workers_list = (number_firms*np.arange(1,6, 0.5)).astype(int)  # N/F Ratios

# Lists of median rates
median_upt_rates = np.zeros(shape=(int(len(choices_list)*len(workers_list)), repetitions))
median_ginis = np.zeros(shape=(int(len(choices_list)*len(workers_list)), repetitions))
median_efficiencies = np.zeros(shape=(int(len(choices_list)*len(workers_list)), repetitions))

queue_type = "networked" # a or b depending on uniform (a) or networked (b)

# For 1-10 choices, for 20-100 workers, for 5 repetitions
for i in range(len(choices_list)):
    for j in range(len(workers_list)):
        for k in range(repetitions):
            print("Choice: ", i, "N_Workers: ", workers_list[j], "Repetition: ", k)
            n_choices = choices_list[i]
            number_workers = workers_list[j]
            reservation_wages_initial = abs(np.random.normal(loc=1.0, scale=0, size=number_workers))
            seed_B = nx.barabasi_albert_graph(number_workers, 5, seed=None, initial_graph=None)
            skills_initial = np.stack((np.random.uniform(0, 1, size=number_workers),  # Mean
                                       np.random.uniform(0.1, 0.1, size=number_workers),  # Variance
                                       np.random.uniform(1, 1, size=number_workers)))  # Scale
            requirements_initial = np.random.uniform(0,1,size=number_firms)
            beta_initial = np.random.uniform(1.0, 10.0, size=number_firms)


            (upt_path9a, wages_path9a, wage_dist_path9a, gdp_path9a, price_path9a, active_competence_path_final9a, dest_path9a,
             beta_path_final9a, all_competences_calc9a) = run_simulation()
            gini_path_9a = np.array([gini(i) for i in wage_dist_path9a])

            median_upt_rates[i*len(workers_list)+j, k] = np.median(upt_path9a/number_workers)
            median_ginis[i*len(workers_list)+j, k] = np.median(gini_path_9a)
            median_efficiencies[i*len(workers_list)+j, k] = np.median(active_competence_path_final9a)


med_median_upt_rates = np.median(median_upt_rates, axis=1)
med_median_ginis = np.median(median_ginis, axis=1)
med_median_efficiencies = np.median(median_efficiencies, axis=1)

med_median_upt_rates_reshaped = med_median_upt_rates.reshape(len(choices_list), len(workers_list))
fig9a, axs9a = plt.subplots()
im9a, cbar9a = heatmap(med_median_upt_rates_reshaped, choices_list, workers_list/20, ax=axs9a,
                   cmap="YlGn", cbarlabel="Median Unemployment U(t)")
texts9a = annotate_heatmap(im9a, valfmt="{x:.2f}")
axs9a.set_xlabel("Concentration N/F", fontsize=13)
axs9a.set_ylabel("Choices f_n", fontsize=13)
axs9a.xaxis.set_label_position('top')
fig9a.tight_layout()
plt.savefig("9b.jpg")

fig10a, axs10a = plt.subplots()
med_median_ginis_reshaped = med_median_ginis.reshape(len(choices_list), len(workers_list))
im10a, cbar10a = heatmap(med_median_ginis_reshaped, choices_list, workers_list/20, ax=axs10a,
                   cmap="Reds", cbarlabel="Median Gini G(t)")
texts10a = annotate_heatmap(im10a, valfmt="{x:.2f}")
axs10a.set_xlabel("Concentration N/F", fontsize=13)
axs10a.set_ylabel("Choices f_n", fontsize=13)
axs10a.xaxis.set_label_position('top')
fig10a.tight_layout()
plt.savefig("10b.jpg")

fig11a, axs11a = plt.subplots()
med_median_efficiencies_reshaped = med_median_efficiencies.reshape(len(choices_list), len(workers_list))
im11a, cbar11a = heatmap(med_median_efficiencies_reshaped, choices_list, workers_list/20, ax=axs11a,
                   cmap="Blues", cbarlabel="Median Efficiency E(t)")
texts11a = annotate_heatmap(im11a, valfmt="{x:.2f}")
axs11a.set_xlabel("Concentration N/F", fontsize=13)
axs11a.set_ylabel("Choices f_n", fontsize=13)
axs11a.xaxis.set_label_position('top')
fig11a.tight_layout()
plt.savefig("11b.jpg")


# --------------------- 4.4 Microeconomic inference -----------------------
# Table 3 -- Generate the data csv file

number_firms = 50
number_workers = 250
number_periods = 500
loyalty = 1/number_firms
alpha_initial = abs(np.linspace(start=10, stop=10, num=number_firms))
reservation_wages_initial = abs(np.random.normal(loc=1.0, scale=0, size=number_workers))
a = 0.5
rho = 0.5
sigma_beta = 0.15
repetitions = 50  # Number of Monte Carlo runs at each combination

choices_list = np.array([1, 2, 4, 8, 16, 32])  # f_n choices

# Raw data lists
median_wages_big_list = np.zeros(shape=(len(choices_list), repetitions, number_workers))  # 3D array
connections_big_list = np.zeros(shape=(len(choices_list), repetitions, number_workers))
twostep_connections_big_list = np.zeros(shape=(len(choices_list), repetitions, number_workers))
skill_fit_big_list = np.zeros(shape=(len(choices_list), repetitions, number_workers))
n_choices_big_list = np.zeros(shape=(len(choices_list), repetitions, number_workers))

queue_type = "networked"

for i in range(len(choices_list)):
    for k in range(repetitions):
        print("Choice: ", i, "Repetition: ", k)
        n_choices = choices_list[i]
        seed_B = nx.barabasi_albert_graph(number_workers, 5, seed=None, initial_graph=None)
        arr_B_g = nx.to_numpy_array(seed_B)
        arr_B_twostep_g = np.matmul(arr_B_g, arr_B_g)  # Two-step loops
        np.fill_diagonal(arr_B_twostep_g, 0)

        skills_initial = np.stack((np.random.uniform(0, 1, size=number_workers),  # Mean
                                   np.random.uniform(0.1, 0.1, size=number_workers),  # Variance
                                   np.random.uniform(1, 1, size=number_workers)))  # Scale
        requirements_initial = np.random.uniform(0,1,size=number_firms)
        beta_initial = np.random.uniform(1.0, 10.0, size=number_firms)


        (upt_path_t3, wages_path_t3, wage_dist_path_t3, gdp_path_t3, price_path_t3, active_competence_path_final_t3, dest_path_t3,
         beta_path_final_t3, all_competences_calc_t3) = run_simulation()

        median_wages_big_list[i, k] = np.median(wage_dist_path_t3[249:499], axis=0)
        connections_big_list[i,k] = np.sum(arr_B_g, axis=1)
        twostep_connections_big_list[i,k] = np.sum(arr_B_twostep_g, axis=1)
        skill_fit_big_list[i,k] = np.sum(all_competences_calc_t3, axis=0)/number_firms
        n_choices_big_list[i,k] = np.repeat(n_choices, number_workers)

median_wages_big_list_r = median_wages_big_list.reshape(len(choices_list), repetitions*number_workers)
connections_big_list_r = connections_big_list.reshape(len(choices_list), repetitions*number_workers)
twostep_connections_big_list_r = twostep_connections_big_list.reshape(len(choices_list), repetitions*number_workers)
skill_fit_big_list_r = skill_fit_big_list.reshape(len(choices_list), repetitions*number_workers)
n_choices_big_list_r = n_choices_big_list.reshape(len(choices_list), repetitions*number_workers)

raw_data = np.transpose(np.vstack((median_wages_big_list_r.ravel(), connections_big_list_r.ravel(), twostep_connections_big_list_r.ravel(), skill_fit_big_list_r.ravel(), n_choices_big_list_r.ravel())))
df = pd.DataFrame(raw_data, columns=['Median_Wage', 'One_Step_Connections', 'Two_Step_Connections', 'Skill_Fit', 'Number_Choices'])
df.to_csv('FILE_PATH.csv', sep=',')

# Table 3: Analysis of csv file

# Figure 12

df=pd.read_csv('FILE_PATH.csv',sep=',')
df['Median_Wage'] = np.log(df['Median_Wage'])
df = df.rename(columns={'Median_Wage':'Log_Median_Wage'})

df2 = df.loc[df['Number_Choices'] ==1]
y2, X2 = patsy.dmatrices('Log_Median_Wage ~ One_Step_Connections + Skill_Fit', data=df2, return_type='dataframe')
model2 = OLS(y2,X2)
results2 = model2.fit(cov_type='HC1')
df2.plot.scatter(x='One_Step_Connections', y='Log_Median_Wage')

df3 = df.loc[df['Number_Choices'] ==2]
y3, X3 = patsy.dmatrices('Log_Median_Wage ~ One_Step_Connections + Skill_Fit', data=df3, return_type='dataframe')
model3 = OLS(y3,X3)
results3 = model3.fit(cov_type='HC1')
df2.plot.scatter(x='One_Step_Connections', y='Log_Median_Wage')

df4 = df.loc[df['Number_Choices'] ==4]
y4, X4 = patsy.dmatrices('Log_Median_Wage ~ One_Step_Connections + Skill_Fit', data=df4, return_type='dataframe')
model4 = OLS(y4,X4)
results4 = model4.fit(cov_type='HC1')
df4.plot.scatter(x='One_Step_Connections', y='Log_Median_Wage')

df5 = df.loc[df['Number_Choices'] ==8]
y5, X5 = patsy.dmatrices('Log_Median_Wage ~ One_Step_Connections + Skill_Fit', data=df5, return_type='dataframe')
model5 = OLS(y5,X5)
results5 = model5.fit(cov_type='HC1')
df2.plot.scatter(x='One_Step_Connections', y='Log_Median_Wage')

df6 = df.loc[df['Number_Choices'] ==16]
y6, X6 = patsy.dmatrices('Log_Median_Wage ~ One_Step_Connections + Skill_Fit', data=df6, return_type='dataframe')
model6 = OLS(y6,X6)
results6 = model6.fit(cov_type='HC1')
df2.plot.scatter(x='One_Step_Connections', y='Log_Median_Wage')

df7 = df.loc[df['Number_Choices'] ==32]
y7, X7 = patsy.dmatrices('Log_Median_Wage ~ One_Step_Connections + Skill_Fit', data=df7, return_type='dataframe')
model7 = OLS(y7,X7)
results7 = model7.fit(cov_type='HC1')

star = Stargazer([results2, results3, results4, results5, results6, results7])
star.significance_levels([0.00001, 0.000001, 0.0000001])

fig12, axs12 = plt.subplots(2,2)
axs12[0,0].scatter(df5.loc[df5['Skill_Fit']<0.15]['One_Step_Connections'], df5.loc[df5['Skill_Fit']<0.15]['Log_Median_Wage'], marker='.', s=10)
axs12[0,1].scatter(df5.loc[df5['Skill_Fit']>0.3]['One_Step_Connections'], df5.loc[df5['Skill_Fit']>0.3]['Log_Median_Wage'], marker='.', s=10)
axs12[1,0].scatter(df6.loc[df6['Skill_Fit']<0.15]['One_Step_Connections'], df6.loc[df6['Skill_Fit']<0.15]['Log_Median_Wage'], marker='.', s=10)
axs12[1,1].scatter(df6.loc[df6['Skill_Fit']>0.3]['One_Step_Connections'], df6.loc[df6['Skill_Fit']>0.3]['Log_Median_Wage'], marker='.', s=10)
axs12[0,0].set_xlabel("One Step Connections")
axs12[1,0].set_xlabel("One Step Connections")
axs12[0,1].set_xlabel("One Step Connections")
axs12[1,1].set_xlabel("One Step Connections")
axs12[0,0].set_ylabel("Log Median Wage")
axs12[1,0].set_ylabel("Log Median Wage")
axs12[0,1].set_ylabel("Log Median Wage")
axs12[1,1].set_ylabel("Log Median Wage")

fig12.set_tight_layout(True)

# Log-log and table 3


df=pd.read_csv('FILE_PATH.csv',sep=',')
df['Median_Wage'] = np.log(df['Median_Wage'])
df = df.rename(columns={'Median_Wage':'Log_Median_Wage'})

df['One_Step_Connections'] = np.log(df['One_Step_Connections'])
df = df.rename(columns={'One_Step_Connections':'Log_One_Step_Connections'})

df2 = df.loc[df['Number_Choices'] ==1]
y2, X2 = patsy.dmatrices('Log_Median_Wage ~ Log_One_Step_Connections + Skill_Fit', data=df2, return_type='dataframe')
model2 = OLS(y2,X2)
results2 = model2.fit(cov_type='HC1')

df3 = df.loc[df['Number_Choices'] ==2]
y3, X3 = patsy.dmatrices('Log_Median_Wage ~ Log_One_Step_Connections + Skill_Fit', data=df3, return_type='dataframe')
model3 = OLS(y3,X3)
results3 = model3.fit(cov_type='HC1')

df4 = df.loc[df['Number_Choices'] ==4]
y4, X4 = patsy.dmatrices('Log_Median_Wage ~ Log_One_Step_Connections + Skill_Fit', data=df4, return_type='dataframe')
model4 = OLS(y4,X4)
results4 = model4.fit(cov_type='HC1')

df5 = df.loc[df['Number_Choices'] ==8]
y5, X5 = patsy.dmatrices('Log_Median_Wage ~ Log_One_Step_Connections + Skill_Fit', data=df5, return_type='dataframe')
model5 = OLS(y5,X5)
results5 = model5.fit(cov_type='HC1')

df6 = df.loc[df['Number_Choices'] ==16]
y6, X6 = patsy.dmatrices('Log_Median_Wage ~ Log_One_Step_Connections + Skill_Fit', data=df6, return_type='dataframe')
model6 = OLS(y6,X6)
results6 = model6.fit(cov_type='HC1')

df7 = df.loc[df['Number_Choices'] ==32]
y7, X7 = patsy.dmatrices('Log_Median_Wage ~ Log_One_Step_Connections + Skill_Fit', data=df7, return_type='dataframe')
model7 = OLS(y7,X7)
results7 = model7.fit(cov_type='HC1')
star = Stargazer([results2, results3, results4, results5, results6, results7])
star.show_degrees_of_freedom(False)
print(star.render_latex())


plt.show()
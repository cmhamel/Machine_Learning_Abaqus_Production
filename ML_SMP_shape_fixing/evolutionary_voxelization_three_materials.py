import os
import sys
import random
import numpy as np
import subprocess
import matplotlib.pyplot as plt
import math
import pickle
import csv
from collections import defaultdict

from deap import base
from deap import creator
from deap import tools
from deap import algorithms

from scoop import futures

from matplotlib import rcParams
rcParams.update({'font.size': 16})
rcParams.update({'figure.autolayout': True})


GENERATE_SCRIPT = 'objet_SMP_GUI.py'
GENERATE_SCRIPT_GUI = 'objet_SMP_GUI.py'

FREQ = 1
CXPB = 0.6
MUTPB = 0.3
NGEN = 125

MU = 24
LAMBDA = 45

CHECK_POINT = "checkpoint_objet_three_material_shape_fixing_sinusoid_larger_amplitude.pkl"

creator.create("FitnessFunc", base.Fitness, weights=(-1.0, ))
creator.create("Individual", list, fitness=creator.FitnessFunc)


def ea_mu_plus_lambda(population, toolbox, checkpoint,
                      mu, lambda_,
                      cxpb, mutpb, ngen,
                      stats=None,
                      halloffame=None,
                      verbose=__debug__):

    if checkpoint:
        # A file name has been give, then load the data from the file
        with open(checkpoint, "rb") as cp_file:
            cp = pickle.load(cp_file)

        population = cp["population"]
        start_gen = cp["generation"] + 1
        halloffame = cp["halloffame"]
        logbook = cp["logbook"]
        random.setstate(cp["rndstate"])
        best_specimens = cp["bestspecimens"]

        print(logbook)

    else:
        # start a new evolution since no cp_file was given
        # population = toolbox.population(n=36)
        start_gen = 1
        # halloffame = tools.HallOfFame(maxsize=1)
        logbook = tools.Logbook()
        best_specimens = []

    logbook.header = ['gen', 'nevals'] + (stats.fields if stats else [])

    # Evaluate the individuals with an invalid fitness
    invalid_ind = [ind for ind in population if not ind.fitness.valid]
    fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
    for ind, fit in zip(invalid_ind, fitnesses):
        ind.fitness.values = fit

    if halloffame is not None:
        halloffame.update(population)

    record = stats.compile(population) if stats is not None else {}
    logbook.record(gen=0, nevals=len(invalid_ind), **record)
    if verbose:
        print(logbook.stream)

    # Begin the generational process
    for gen in range(start_gen, ngen + 1):
        # clear the directory of all sim files
        clear_directory()

        # Vary the population
        offspring = algorithms.varOr(population, toolbox, lambda_, cxpb, mutpb)

        # Evaluate the individuals with an invalid fitness
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

        # Update the hall of fame with the generated individuals
        if halloffame is not None:
            halloffame.update(offspring)

        # Select the next generation population
        population[:] = toolbox.select(population + offspring, mu)

        # Append the best solution from this gen to the best specimens
        # array
        #
        best_specimens.append(tools.selBest(population, k=1)[0])

        # Update the statistics with the new population
        record = stats.compile(population) if stats is not None else {}
        logbook.record(gen=gen, nevals=len(invalid_ind), **record)
        if verbose:
            print(logbook.stream)

        if gen % FREQ == 0:
            # fill the dictionary
            cp = dict(population=population,
                      generation=gen,
                      halloffame=halloffame,
                      logbook=logbook,
                      rndstate=random.getstate(),
                      bestspecimens=best_specimens)

            with open(CHECK_POINT, "wb") as cp_file:
                pickle.dump(cp, cp_file)

    return population, logbook


def evaluate_design(individual, gui=False, evaluatebest=False):
    platform = sys.platform

    if gui:
        abaqus_str = 'abaqus cae'

        stats = True

        if platform == 'linux':
            abaqus_str = abaqus_str + ' script=' + GENERATE_SCRIPT_GUI + ' -mesa'

        else:
            abaqus_str = abaqus_str + ' script=' + GENERATE_SCRIPT_GUI

    else:
        abaqus_str = 'abaqus cae noGUI=' + GENERATE_SCRIPT

        stats = False

    boo = True
    ind_num = 1
    genome_name = 'temp_genome_individual'

    dir_files = os.listdir('.')
    split_files = []

    for fil in dir_files:
        split_files.append(fil.split('.')[0])

    while boo:

        if genome_name + str(ind_num) in split_files:
            ind_num = ind_num + 1
        else:
            boo = False

    genome_name = genome_name + str(ind_num)

    with open(genome_name + '.txt', 'w') as f:
        f.write('**Heading\n')
        f.write('**Number of voxels in each direction, nx, ny\n')
        ny = 5
        nx = int(len(individual) / ny)
        f.write(str(nx) + ', ' + str(ny) + '\n')
        for j in range(len(individual)):

            f.write('%s ' % (individual[j]))

    job_name = 'job_' + genome_name

    abaqus_str = abaqus_str + ' -- ' + genome_name + '.txt ' + job_name

    fnull = open(os.devnull, 'wb')
    subprocess.call(abaqus_str,
                    shell=True,
                    stderr=fnull,
                    stdout=fnull)
    #
    # now evaluate the fitness of this design
    #
    # if evaluatebest:
    #     f_str = 'best-output.csv'
    # else:
    #     f_str = 'output-' + job_name + '.csv'
    #
    f_str = 'output-' + job_name + '.csv'
    #
    coord = True
    #
    failed = False
    #
    columns = defaultdict(list)
    try:
        with open(f_str) as f:
            reader = csv.DictReader(f)
            for row in reader:
                for (k, v) in row.items():
                    columns[k].append(v)

    except FileNotFoundError:
        failed = True

    f.close()

    errors_1, errors_2 = [], []
    xs = columns['x_coordinate']
    # ys = columns['y_coordinate']

    temp_xs = np.array(xs, dtype=np.float64)
    # temp_ys = np.array(ys, dtype=np.float64)

    permutation = temp_xs.argsort()

    sorted_xs = temp_xs[permutation]

    if failed:
        errors_1.append(np.inf)
        errors_2.append(np.inf)
    else:
        for key in columns.keys():
            if 'y_displacement' in key:
                uy = np.array(columns[key], dtype=np.float64)
                sorted_uy = uy[permutation]
                if evaluatebest:
                    analytic_uy_1, analytic_uy_2 = target_shape_function(sorted_xs)

                    title = key.split('_')[-1]

                    plt.figure(1)
                    plt.plot(sorted_xs, analytic_uy_1,
                             color='k', label='Desired Shape 1')
                    # plt.plot(sorted_xs, analytic_uy_2,
                    #          color='r', label='Desired Shape 2')
                    plt.plot(sorted_xs, sorted_uy,
                             color='b', label='Achieved Shape')
                    plt.xlabel('X Coordinates (mm)')
                    plt.ylabel('Y Displacement (mm)')
                    plt.xlim((0.0, 70.0))
                    plt.ylim((min(analytic_uy_1) * (5.0 / 4.0), 5.0))
                    ax = plt.subplot(111)
                    ax.legend(loc='lower left')
                    plt.title('Time Step ' + title)
                    plt.savefig(key + '.png')
                    plt.clf()

                error_1, error_2 = target_shape_error(temp_xs, uy, stats)
                errors_1.append(error_1)
                errors_2.append(error_2)

    return min(errors_1), #min(errors_2)


def target_shape_function(xs):
    analytic_uy_1 = -8.0 * (np.ones(len(xs)) - np.cos(np.pi*xs/80.0))

    analytic_uy_2 = -10.0 * np.square(2*xs/80.0)

    return analytic_uy_1, analytic_uy_2


def target_shape_error(xs, uy, stats=False):

    analytic_uy_1, analytic_uy_2 = target_shape_function(xs)

    error_1, error_2 = 0.0, 0.0

    for i in range(len(xs)):
        error_1 = error_1 + (analytic_uy_1[i] - uy[i])**2
        error_2 = error_2 + (analytic_uy_2[i] - uy[i])**2

    error_1 = (1.0/len(xs)) * math.sqrt(error_1)
    error_2 = (1.0/len(xs)) * math.sqrt(error_2)

    return error_1, error_2


def evaluate_best(checkpoint):

    if checkpoint:
        # A file name has been give, then load the data from the file
        with open(checkpoint, "rb") as cp_file:
            cp = pickle.load(cp_file)

        population = cp["population"]
        start_gen = cp["generation"]
        halloffame = cp["halloffame"]
        logbook = cp["logbook"]
        random.setstate(cp["rndstate"])
        best_specimens = cp["bestspecimens"]

    print(logbook)
    print(halloffame)

    error = evaluate_design(halloffame[0], gui=True, evaluatebest=True)

    generations = logbook.select("gen")
    fit_mins = logbook.select("min")
    fit_avgs = logbook.select("avg")

    plt.figure(1)
    plt.plot(generations, fit_mins,
             color='b', label='Minimum Fitness')
    plt.plot(generations, fit_avgs,
             color='r', label='Average Fitness')
    plt.xlabel('Generation')
    plt.ylabel('Fitness Function Value')
    ax = plt.subplot(111)
    ax.legend(loc='best')
    plt.savefig('fitness_history.png')
    plt.clf()


def post_process_evolution(checkpoint):

    clear_directory()

    if checkpoint:
        # A file name has been give, then load the data from the file
        with open(checkpoint, "rb") as cp_file:
            cp = pickle.load(cp_file)

        population = cp["population"]
        start_gen = cp["generation"]
        halloffame = cp["halloffame"]
        logbook = cp["logbook"]
        random.setstate(cp["rndstate"])
        best_specimens = cp["bestspecimens"]

    for best in best_specimens:
        toolbox.evaluate(best)

    evaluate_best(checkpoint)

    print(len(population))
    print(len(logbook))
    print(len(best_specimens))


def clear_directory():
    files = os.listdir('.')
    os.system('del *.csv')
    extensions = ['png', 'rec', 'odb', 'sta', 'msg', 'rpy',
                  'txt', 'dat', 'log', 'inp', 'com', 'prt',
                  'sim', 'ipm', 'mdl', 'stt', '1', '023',
                  '.csv']

    for f in files:
        try:
            e = f.split(".")[1]
        except IndexError:
            e = ""

        if e in extensions:
            try:
                os.remove(f)
            except OSError:
                pass


def make_user_subroutines():
    os.system('abaqus make library=utrs.for >NUL')


if __name__ == '__main__':

    make_user_subroutines()

    # check_point = CHECK_POINT
    check_point = None
    toolbox = base.Toolbox()

    toolbox.register("map", futures.map)

    toolbox.register("attr_material",
                     random.randint, 0, 2)

    toolbox.register("individual",
                     tools.initRepeat,
                     creator.Individual,
                     toolbox.attr_material, 350)

    toolbox.register("population",
                     tools.initRepeat,
                     list,
                     toolbox.individual)

    toolbox.register("evaluate", evaluate_design)
    toolbox.register("mate", tools.cxTwoPoint)
    toolbox.register("mutate", tools.mutUniformInt, low=0, up=2, indpb=0.15)
    toolbox.register("select", tools.selTournament, tournsize=3)
    # toolbox.register("select", tools.selNSGA2)
    clear_directory()

    pop = toolbox.population(n=MU)
    hof = tools.HallOfFame(maxsize=1)

    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)#, axis=0)
    stats.register("std", np.std)#, axis=0)
    stats.register("min", np.min)#, axis=0)
    stats.register("max", np.max)#, axis=0)

    # pop, log = ea_mu_plus_lambda(pop, toolbox, check_point,
    #                              mu=MU, lambda_=LAMBDA,
    #                              cxpb=CXPB, mutpb=MUTPB, ngen=NGEN,
    #                              stats=stats,
    #                              halloffame=hof)

    evaluate_best(CHECK_POINT)

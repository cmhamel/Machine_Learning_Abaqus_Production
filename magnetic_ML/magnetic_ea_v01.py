import os
import sys
import random
import time
import numpy as np
import subprocess
import matplotlib.pyplot as plt
import math
import pickle

from deap import base
from deap import creator
from deap import tools
from deap import algorithms

from scoop import futures


GENERATE_SCRIPT = 'abaqus_script_no_gui.py'
GENERATE_SCRIPT_GUI = 'abaqus_script.py'

N_X = 5
N_Y = 5

FREQ = 1
CXPB = 0.6
MUTPB = 0.3
NGEN = 50
ERROR = 5.e-6

MU = 16
LAMBDA = 40

CHECK_POINT = "magnetic_sinusoid_attempt_2_direction.pkl"


creator.create("FitnessFunc", base.Fitness, weights=(-1.0,))
creator.create("Individual", list, fitness=creator.FitnessFunc)


def clear_directory():
    files = os.listdir('.')

    extensions = ['png', 'rec', 'odb', 'sta', 'msg', 'rpy',
                  'txt', 'dat', 'log', 'inp', 'com', 'prt',
                  'sim', 'ipm', 'mdl', 'stt', '1', '023']

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
        #
        # start a new evolution since no cp_file was given
        # population = toolbox.population(n=36)
        #
        start_gen = 1
        #
        # halloffame = tools.HallOfFame(maxsize=1)
        #
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
        #
        # Append the best solution from this gen to the best specimens
        # array
        #
        best_specimens.append(tools.selBest(population, k=1)[0])
        #
        # Update the statistics with the new population
        #
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

        evaluated_fitnesses = []
        for ind in population:
            fit1 = ind.fitness.values[0]

            evaluated_fitnesses.append(fit1)

        if min(evaluated_fitnesses) < ERROR:
            print('Early survival of the fittest')
            return population, logbook

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
        f.write(str(N_X) + ', ' + str(N_Y) + '\n')

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
    f_str = 'output-' + job_name + '.txt'
    #
    xs, ys, zs = [], [], []
    ux, uy, uz = [], [], []
    #
    coord = True
    failed = False
    #
    try:
        with open(f_str) as f:
            line = f.readline()
            while line:
                if 'Displacements' in line:
                    coord = False
                if 'Simulation Failed' in line:
                    failed = True
                    break
                if '*' in line:
                    line = f.readline()

                x_y_z = line.split(',')
                try:
                    if coord:
                        xs.append(float(x_y_z[1]))
                        ys.append(float(x_y_z[2]))
                        zs.append(float(x_y_z[3]))
                    else:
                        ux.append(float(x_y_z[1]))
                        uy.append(float(x_y_z[2]))
                        uz.append(float(x_y_z[3]))
                except IndexError:
                    failed = True
                    if coord:
                        xs.append(np.inf)
                        ys.append(np.inf)
                        zs.append(np.inf)
                    else:
                        ux.append(np.inf)
                        uy.append(np.inf)
                        uz.append(np.inf)

                line = f.readline()

    except FileNotFoundError:
        failed = True

    temp_xs = np.array(xs, dtype=np.float64)
    temp_ys = np.array(ys, dtype=np.float64)
    temp_zs = np.array(zs, dtype=np.float64)

    if failed:
        error = np.inf
    else:
        error = target_shape_error(temp_xs, temp_ys, temp_zs, ux, uy, uz, stats)

    return error,


def target_shape_error(xs, ys, zs, ux, uy, uz, stats=False):
    analytic_ux = np.zeros(len(xs))
    analytic_uy = -0.005 * (np.ones(len(xs)) - np.cos(np.pi * xs / 0.025))
    analytic_uz = np.zeros(len(xs))

    # analytic_uy = -5.0 * (2.0 * xs / 50.0)**2

    error = 0.0

    for i in range(len(xs)):
        error = error + (analytic_uy[i] - uy[i]) ** 2 #+ \
                # (analytic_ux[i] - ux[i]) ** 2 + \
                # (analytic_uz[i] - uz[i]) ** 2

    error = (1.0 / len(xs)) * math.sqrt(error)

    if stats:
        test = analytic_uy
        plt.plot(xs, uy, 'ro')
        plt.plot(xs, test, 'bo')
        plt.xlabel('X Coordinates')
        plt.ylabel('Y Displacement')
        plt.savefig('Y_displacement.png')
        plt.clf()

        test = ux
        plt.plot(xs, test, 'bo')
        plt.xlabel('X Coordinates')
        plt.ylabel('X Displacement')
        plt.savefig('X_displacement.png')
        plt.clf()

        plt.plot(xs, analytic_uy - uy)
        plt.xlabel('X coordinates')
        plt.ylabel('Absolute Error')
        plt.savefig('Error.png')
        plt.clf()

        print('error function = ' + str(error))

        abs_err = np.linalg.norm(analytic_uy - uy)
        rel_err = abs_err / np.linalg.norm(analytic_uy)
        percent_err = rel_err * 100.0

        print('Absolute norm error = ' + str(abs_err / len(xs)))
        print('Relative norm error = ' + str(rel_err))
        print('Percent error = ' + str(percent_err))

    return error


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

    # for best in best_specimens:
    #     toolbox.evaluate(best)

    evaluate_best(checkpoint)

    print(len(population))
    print(len(logbook))
    print(len(best_specimens))


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

    print(len(halloffame))
    print(logbook)
    print(halloffame)

    error1 = evaluate_design(halloffame[0], gui=True)

    print(error1)


if __name__ == '__main__':

    toolbox = base.Toolbox()

    toolbox.register("map", futures.map)

    toolbox.register("attr_int",
                     random.randint, 1, 4)

    toolbox.register("individual",
                     tools.initRepeat,
                     creator.Individual,
                     toolbox.attr_int, N_X * N_Y)

    toolbox.register("population",
                     tools.initRepeat,
                     list,
                     toolbox.individual)

    toolbox.register("evaluate", evaluate_design)
    toolbox.register("mate", tools.cxTwoPoint)
    toolbox.register("mutate", tools.mutUniformInt, low=1, up=2, indpb=0.1)
    toolbox.register("select", tools.selTournament, tournsize=3)

    pop = toolbox.population(n=MU)
    hof = tools.HallOfFame(maxsize=1)

    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("std", np.std)
    stats.register("min", np.min)
    stats.register("max", np.max)

    clear_directory()

    # check_point = CHECK_POINT
    #
    # post_process_evolution(check_point)

    check_point = None
    #
    pop, log = ea_mu_plus_lambda(pop, toolbox, check_point,
                                 mu=MU, lambda_=LAMBDA,
                                 cxpb=CXPB, mutpb=MUTPB, ngen=NGEN,
                                 stats=stats,
                                 halloffame=hof)



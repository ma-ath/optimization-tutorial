from cocoex import Observer, Suite, ExperimentRepeater
from cocoex.utilities import MiniPrint
import cocopp

from src.algorithms import DifferentialEvolution
from src.utils import setup_logger, CocoProblemWrapper


logger = setup_logger(console_level=None, file_level=None)
algorithm = DifferentialEvolution()
budget_multiplier = 3

# Prepare the benchmark suite
suite = Suite("bbob", "", "")
output_folder = f'{algorithm.name}_{budget_multiplier}D_on_{suite.name}'
observer = Observer(suite.name, "result_folder: " + output_folder)
repeater = ExperimentRepeater(budget_multiplier)
minimal_print = MiniPrint()

# Run the benchmark
logger.info(f'Starting benchmark: {algorithm.name} on {suite.name} suite')
while not repeater.done():                  # while budget is left and successes are few
    for problem in suite:                   # loop takes 2-3 minutes x budget_multiplier
        if repeater.done(problem):
            logger.debug(f'Skipping problem {problem} as budget is exhausted or enough successes achieved.')
            continue                                # skip this problem
        problem.observe_with(observer)              # generate data for cocopp
        coco_problem = CocoProblemWrapper(problem)  # wrap the problem for compatibility with our code

        logger.info(f'Starting optimization on problem {problem.id} with dimension {problem.dimension}')
        result = algorithm.optimize(coco_problem,
                                    function_dimension=problem.dimension,
                                    max_population_size=50,
                                    stop_fitness=1e-6,
                                    budget=1000 * problem.dimension)
        x_opt = result['x_opt']
        problem(x_opt)                          # make sure the returned solution is evaluated
        repeater.track(problem)                 # track evaluations and final_target_hit
        minimal_print(problem)                  # show progress

# Generate the performance report
cocopp.main(observer.result_folder + ' bfgs!')
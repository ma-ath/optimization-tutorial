if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(prog='coco_benchmark.py', description='Run COCO benchmark with specified algorithms.')
    parser.add_argument('--run', action=argparse.BooleanOptionalAction, help='Flag to run the benchmark')
    parser.add_argument('--view', action=argparse.BooleanOptionalAction, help='Flag to view the benchmark results')
    parser.add_argument('--results-folder', type=str, default='', help='Path to the results folder for viewing')

    args = parser.parse_args()

    if args.run:
        from typing import Any
        from cocoex import Observer, Suite, ExperimentRepeater
        from cocoex.utilities import MiniPrint

        from src.algorithms._algorithm import Algorithm
        from src.algorithms import DifferentialEvolution, GaussianEvolutionStrategy, CMAES, RandomSearch
        from src.utils import setup_logger, CocoProblemWrapper


        logger = setup_logger(console_level=None, file_level='DEBUG')
        budget_multiplier = 3

        algorithms: dict[Algorithm, dict[str, Any]] = {
            DifferentialEvolution(): {
                "max_population_size": 50,
            },
            GaussianEvolutionStrategy(): {
                "max_population_size": 50,
            },
            RandomSearch(): {
                "max_population_size": 300,
            },
            CMAES(): {
                "max_population_size": 10,
            },
        }

        for algorithm, kwargs in algorithms.items():
            # Prepare the benchmark suite
            suite = Suite("bbob", "", "")
            output_folder = f'{algorithm.name.replace(" ", "")}_bm{budget_multiplier}_{suite.name.decode()}'
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
                                                budget = 10000 * problem.dimension,
                                                stop_fitness = 1e-6,
                                                function_dimension = problem.dimension,
                                                **kwargs)
                    x_opt = result['x_opt']
                    problem(x_opt)                          # make sure the returned solution is evaluated
                    repeater.track(problem)                 # track evaluations and final_target_hit
                    minimal_print(problem)                  # show progress
        logger.info(f'Finished all benchmarks.')

    if args.view:
        assert args.results_folder, "Please provide the --results-folder argument to view results."
        import os
        from cocopp import main

    # Build full paths for all result subfolders
    folders = [
        os.path.join(args.results_folder, f)
        for f in os.listdir(args.results_folder)
        if os.path.isdir(os.path.join(args.results_folder, f))
    ]
    main(folders + ['bfgs!'])

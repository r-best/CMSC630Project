import os
import yaml
import logging
import traceback
import numpy as np
from pathos.pools import ProcessPool

from cmsc630 import Image


OPERATIONS = ['equalize', 'quantize', 'filter', 'gaussiannoise', 'saltnpeppernoise']
PARALLEL_UNSAFE_OPS = ['filter']
COLOR_MAP = {'red': 0, 'green': 1, 'blue': 2, 'rgb': 3, 'gray': 4}


def applyToBatch(batch, operation, parallel_safe):
    """Takes in a batch of Images and a function to execute on them.
    Executes the function on the entire batch in parallel if `parallel_safe`
    is true, sequentially if false.
    
    IMPORTANT: `parallel_safe` can always be set to true to allow faster
    processing of the batch UNLESS THE OPERATION IMPLEMENTS ITS OWN INTERNAL
    PARALLELIZATION. Python doesn't like it when child processes spawn more
    child processes, so parallelized operations must be executed sequentially
    over the batch.

    Arguments:
        batch (Image[]): List of Image objects to operate on
        operation (lambda Image: Image): Function that takes
            in an Image object, performs an operation on it, and
            returns an Image
        parallel_safe (boolean): Whether or not this operation is
            safe to parallelize across the batch
    
    Returns:
        (Image[]): The batch after applying the operation to every Image
    """
    print("Applying operation...")
    if operation is None:
        logging.info("Nothing to do")
        return batch
    
    if parallel_safe:
        return ProcessPool().map(operation, batch)
    else:
        return list(map(operation, batch))


def parseFilter(filterList):
    """Used to parse the matrix from the config file's `filter` step.
    Filters should be provided as a list of rows, each row being a 
    whitespace-separated string of numbers. The matrix must be square
    with an odd height and width so that it has a clear centerpoint.

    Examples:
    ```yml
    filter:
      - -1 0 1              [[-1 0 1]
      - -1 0 1      =>       [-1 0 1]   # Prewitt Filter (Border Detection)
      - -1 0 1               [-1 0 1]]
    ```

    Arguments:
        filterlist (list): Array of strings, each string representing a
            row of the filter as a set of numbers split by whitespace
    
    Returns:
        (ndarray): The filterList parsed into a numpy matrix
    """
    filter_mat = None
    for line in filterList:
        try:
            line = np.array([float(x) for x in line.split()])
            if line.shape[0] != len(filterList):
                raise Exception("Filter must be square, pad with zeroes if you need a non-square filter")

            if filter_mat is None:
                filter_mat = line
            else:
                filter_mat = np.vstack((filter_mat,line))
        except ValueError:
            logging.fatal("Invalid configuration: filter must contain only numbers"); exit()
        except Exception as e:
            logging.fatal(e); exit()
    return filter_mat


def main():
    logging.basicConfig(format="%(levelname)s: %(message)s")

    with open("./config.yml", "r") as fp:
        conf = yaml.load(fp)

    if 'inputDir' not in conf:
        logging.fatal("Invalid configuration, 'inputDir' parameter must be specified")
        exit()
    if 'outputDir' not in conf:
        logging.fatal("Invalid configuration, 'outputDir' parameter must be specified")
        exit()
    if 'steps' not in conf or conf['steps'] is None:
        logging.warn("No steps are defined in the config file, no action to take")
        exit()
    
    output_dir = os.path.join(os.getcwd(), conf['outputDir'])
    if os.path.exists(output_dir) and not os.path.isdir(output_dir):
        logging.fatal("Invalid configuration, output directory is a file")
        exit()
    
    if 'outputColor' not in conf:
        save_color = Image.COLOR_RGB
    elif conf['outputColor'] not in COLOR_MAP.keys():
        logging.fatal("Invalid configuration, 'outputColor' must be one of 'red', 'blue', 'green', 'rgb', or 'gray'")
        exit()
    else:
        save_color = COLOR_MAP[conf['outputColor']]

    images = Image.fromDir(conf['inputDir'])
    operations = []

    for item in conf['steps']:
        if 'name' not in item:
            logging.fatal("Invalid configuration, step must have a name")
            exit()
        if item['name'] not in OPERATIONS:
            logging.fatal("Invalid configuration, step name must be one of equalize, quantize, filter, gaussiannoise, or saltnpeppernoise")
            exit()

        try:
            if item['color'] not in COLOR_MAP.keys():
                raise ValueError("color must be one of 'red', 'blue', 'green', 'rgb', or 'gray'")

            color = COLOR_MAP[item['color']]
            op = None

            if item['name'] == 'equalize':
                op = lambda img: img.equalize(
                    color=color
                )
            elif item['name'] =='quantize':
                delta = item['delta']
                technique = item['technique']
                if not isinstance(delta, int) or delta < 1:
                    raise ValueError("delta must be a positive integer")
                if technique not in (Image.QUANT_UNIFORM, Image.QUANT_MEAN, Image.QUANT_MEDIAN):
                    raise ValueError(f"technique must be one of '{Image.QUANT_UNIFORM}', '{Image.QUANT_MEAN}', or '{Image.QUANT_MEDIAN}'")
                op = lambda img: img.quantize(
                    delta=delta,
                    technique=technique,
                    color=color
                )
            elif item['name'] =='filter':
                filter = parseFilter(item['filter'])
                strat = item['strategy']
                border = item['border']
                if strat not in (Image.FILTER_STRAT_LINEAR, Image.FILTER_STRAT_MEAN, Image.FILTER_STRAT_MEDIAN):
                    raise ValueError(f"technique must be one of '{Image.FILTER_STRAT_LINEAR}', '{Image.FILTER_STRAT_MEAN}', or '{Image.FILTER_STRAT_MEDIAN}'")
                if border not in (Image.FILTER_BORDER_CROP, Image.FILTER_BORDER_EXTEND, Image.FILTER_BORDER_IGNORE, Image.FILTER_BORDER_PAD):
                    raise ValueError(f"border must be one of '{Image.FILTER_BORDER_CROP}', '{Image.FILTER_BORDER_EXTEND}', '{Image.FILTER_BORDER_IGNORE}', '{Image.FILTER_BORDER_PAD}'")
                op = lambda img: img.filter(
                    filter=filter,
                    strategy=strat,
                    border=border,
                    color=color
                )
            elif item['name'] =='gaussiannoise':
                rate = item['rate']
                mean = item['mean']
                stddev = item['stddev']
                if not isinstance(rate, float) or rate < 0 or rate > 1:
                    raise ValueError("rate must be a decimal value between 0 and 1")
                if (not isinstance(mean, int) or mean < 0) and mean is not None:
                    raise ValueError("mean must be a positive number or null")
                if (not isinstance(stddev, int) or stddev < 0) and stddev is not None:
                    raise ValueError("standard deviation must be a positive number or null")
                op = lambda img: img.makeGaussianNoise(
                    rate=rate,
                    mean=mean,
                    stddev=stddev,
                    color=color
                )
            elif item['name'] =='saltnpeppernoise':
                rate = item['rate']
                if not isinstance(rate, float) or rate < 0 or rate > 1:
                    raise ValueError("rate must be a decimal value between 0 and 1")
                op = lambda img: img.makeSaltnPepperNoise(
                    rate=rate,
                    color=color
                )
            else: raise ValueError(f"operation name '{item['name']}' not recognized")

            # Add operation to list of operations to perform
            operations.append((op, item['name'] not in PARALLEL_UNSAFE_OPS))
        except KeyError as e:
            logging.fatal(f"Invalid configuration in '{item['name']}' step: must contain parameter '{e.args[0]}'"); exit()
        except ValueError as e:
            logging.fatal(f"Invalid configuration in '{item['name']}' step: {e}"); exit()
        except Exception as e:
            logging.fatal(traceback.format_exc()); exit()
    
    # Perform all the operations on the batch in order
    for op in operations:
        images = applyToBatch(images, op[0], op[1])
    
    for image in images:
        image.saveToFile(output_dir, color=save_color)


if __name__ == '__main__':
    main()
    print("Done!")

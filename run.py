import yaml
import traceback
import logging
import numpy as np
from time import time
from pprint import pprint

from cmsc630 import Image


OPERATIONS = ['equalize', 'quantize', 'filter', 'gaussiannoise', 'saltnpeppernoise']
COLOR_MAP = {'red': 0, 'green': 1, 'blue': 2, 'rgb': 3, 'gray': 4}


def applyToBatch(batch, operation):
    """
    """
    if operation is None:
        logging.info("Nothing to do")
        return batch

    for image in batch:
        image = operation(image)
    return batch


def parseFilter(filterList):
    """
    """
    x = None
    for line in filterList:
        try:
            line = np.array([float(x) for x in line.split()])
            if line.shape[0] != len(filterList):
                raise Exception("Filter must be square, pad with zeroes if you need a non-square filter")

            if x is None:
                x = line
            else:
                x = np.vstack((x,line))
        except ValueError:
            logging.fatal("Invalid configuration: filter must contain only numbers"); exit()
        except Exception as e:
            logging.fatal(e); exit()
    return x


def main():
    logging.basicConfig(format="%(levelname)s: %(message)s")

    with open("./config.yml", "r") as fp:
        conf = yaml.load(fp)

    pprint(conf)

    images = Image.fromDir('./test')
    operations = []

    for item in conf['steps']:
        if 'name' not in item:
            logging.fatal("ERROR: Invalid configuration, step must have a name")
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
            operations.append(op)
        except KeyError as e:
            logging.fatal(f"Invalid configuration in '{item['name']}' step: must contain parameter '{e.args[0]}'"); exit()
        except ValueError as e:
            logging.fatal(f"Invalid configuration in '{item['name']}' step: {e}"); exit()
        except Exception as e:
            logging.fatal(traceback.format_exc()); exit()
    
    for op in operations:
        images = applyToBatch(images, op)

if __name__ == '__main__':
    main()
    print("Done!")

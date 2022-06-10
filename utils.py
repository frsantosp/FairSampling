from texttable import Texttable
import numpy as np

def tab_printer(args):
    """
    Function to print the logs in a nice tabular format.
    :param args: Parameters used for the model.
    """
    if not(isinstance(args, dict) ):
        args = vars(args)
    keys = sorted(args.keys())
    t = Texttable() 
    t.add_rows([["Parameter", "Value"]] +  [[k.replace("_"," ").capitalize(), args[k]] for k in keys])
    print(t.draw())

def print_results(results):
    summary = dict()
    for key in results[0].keys():
        measure = list()
        for result in results:
            measure.append(result[key])
        summary[key] = (np.round(np.mean(measure),4),np.round(np.var(measure),4))    
    tab_printer(summary)     
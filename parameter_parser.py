import argparse  

def parameter_parser():

    parser = argparse.ArgumentParser(description = "Run sampling.")
    
    parser.add_argument("--ego-exist",
                       type = int,
                       default = 1,
                    help = "ego network exist") 
    parser.add_argument("--random-target",
                       type = int,
                       default = 1,
                    help = "randomly select target set.") 
    
    parser.add_argument("--path",
                    nargs = "?",
                    default = "Tagged/",
                help = "The network data folder path")

    parser.add_argument("--sample-path", nargs = "?", 
                        default = "./samples_v2/",
                        help= "path to generated samples folder")

    parser.add_argument("--log-path",
                    nargs = "?",
                    default =  "./logs/",
                help = "path to input logs directory")

    parser.add_argument("--chunksize",
                        type = int,
                        default = 10**7,
                    help = "the size of each chunk of the original graph to be loaded in each iteration. Defualt is 10^7")
    
    
    parser.add_argument("--sample-size",
                        type = int,
                        default = 100,
                    help = "Number of node to sample. Default is 400.")
    
    parser.add_argument("--iter-number",
                        type = int,
                        default = 1000,
                    help = "Number of iteration for MH approach. Default is 1000.")

    parser.add_argument("--sample-number",
                        type = int,
                        default = 1,
                    help = "Number of samples graph to generate. Default is 1.")

#     parser.add_argument("--p1",
#                         type = float,
#                         default = 0.5,
#                     help = "sample parameter 1. Defualt is 0.5")

#     parser.add_argument("--p2",
#                         type = float,
#                         default = 0.5,
#                     help = "sample parameter 2. Defualt is 0.5")


    parser.add_argument("--log-file",
                        nargs = "?",
                        default = "log.txt",
                    help = "log file name. Default is log.txt")
    
    parser.add_argument("--protected",
                        nargs = "?",
                        default = "gender",
                    help = "Name of the protected atribute. Default is gender")
 
    parser.add_argument("--sep",
                    nargs = "?",
                    default = ",",
                help = "edge list column speration char. Defualt is ,")
    
    parser.add_argument("--sampler",
                    nargs = "?",
                    default = "DFS",
                help = "Sampling method.")
    
    
    parser.add_argument("--undirected", 
                       type = int,
                       default = 0,
                    help = "The graph is undirected. Default is false.")

    parser.add_argument('--file', type=open, action=LoadFromFile)

    return parser.parse_args()

class LoadFromFile (argparse.Action):
    def __call__ (self, parser, namespace, values, option_string = None):
        with values as f:
            parser.parse_args(f.read().split(), namespace)
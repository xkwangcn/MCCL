from train_seprate.train_sep import Solver_Sep

from options import Options

def main():

    args = Options().parse()
    type = args.main_mode
    if type == 'train_sep':
        solver = Solver_Sep(args)
        solver.run()
if __name__ == '__main__':
    main()

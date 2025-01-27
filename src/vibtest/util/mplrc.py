"""Set some global matplotlib parameters, to remain consistent in all the plots."""

import matplotlib.pyplot as plt


def load_rcparams(style='running') -> None:

    # Running figures
    if style == 'running':
        plt.rcParams['font.size'] = 11

    # Report figures
    if style == 'print':
        plt.rcParams['mathtext.fontset'] = 'stix'
        plt.rcParams['font.family'] = 'serif'
        plt.rcParams['font.serif'] = ['STIX Two Text'] + plt.rcParams['font.serif']
        plt.rcParams['figure.figsize'] = (6.34,3.34)
        plt.rcParams['font.size'] = 11
        plt.rcParams['figure.dpi'] = 200

    # Presentation figures
    if style == 'print':
        plt.rcParams['mathtext.fontset'] = 'stix'
        plt.rcParams['font.family'] = 'sans-serif'
        plt.rcParams['font.sans-serif'] = ['Noto Sans'] + plt.rcParams['font.sans-serif']
        plt.rcParams['font.size'] = 14


if __name__ == '__main__':
    load_rcparams()

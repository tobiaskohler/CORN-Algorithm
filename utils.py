# write a helper function which is called cprint() that takes a string and a color as arguments and prints the string in that color

def cprint(string, color):
    colors = {
        'red': '\033[31m',
        'green': '\033[32m',
        'yellow': '\033[33m',
        'blue': '\033[34m',
        'magenta': '\033[35m',
        'cyan': '\033[36m',
        'white': '\033[37m',
        'reset': '\033[0m'
    }
    print(colors[color] + string + colors['reset'])
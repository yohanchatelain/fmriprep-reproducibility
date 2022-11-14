class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


sep_h1 = "="*10
sep_h2 = "*"*10
sep_h3 = '_'*50

verbose = False

def enable_verbose_mode():
    global verbose
    verbose = True
    
def print_result(target, ratio, alpha, name=None):
    _name = f'{bcolors.BOLD}{name:<9}{bcolors.ENDC} ' if name else ''
    if ratio < alpha:
        msg = f"PASS [{ratio*100:>6.3f}%]"
        msg_color = f'{_name}{bcolors.BOLD}{bcolors.OKGREEN}{msg:^7}{bcolors.ENDC}  {target.get_filename()}'
    else:
        msg = f"FAIL [{ratio*100:>6.3f}%]"
        msg_color = f'{_name}{bcolors.BOLD}{bcolors.FAIL}{msg:^7}{bcolors.ENDC}  {target.get_filename()}'
    print(msg_color)


def print_name_method(name):
    print(f'{bcolors.BOLD}{name}{bcolors.ENDC}')


def print_debug(msg, verbose=False):
    if verbose:
        print(msg)


def print_sep(msg, sep):
    print(f'{sep} {msg} {sep}')


def print_sep1(msg):
    print_sep(msg, sep_h1)


def print_sep2(msg):
    print_sep(msg, sep_h2)


def print_sep3(msg):
    print_sep(msg, sep_h3)

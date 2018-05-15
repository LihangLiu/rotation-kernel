
def print_module(m):
    for name in dir(m):
        if name.startswith('__'):
            continue
        pointer = getattr(m, name)
        print('[{0}] = {1}'.format(name, pointer))

def write_2_file(content, txt_file, mode='a', new_line=True):
    with open(txt_file, mode) as f:
        f.write(content)
        f.write('\n')


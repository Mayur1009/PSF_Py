# Function to calculate neighbour and their index in original list.

from re import match

'''
The is done by converting the list to string and the using the match() function provided in python package "re".

Parameters: 
            data : array
                The data find neighbours.
            
            w : int
                Size of window.
                 
Returns:
            neighbour : array
                Return array of Neighbours
            
            n_i : array
                Returns array of index of neighbours relative to data.
 
'''


def _neighbours(data, w):
    t = ''.join(str(i) for i in data)
    pattern = data[-w:]
    p = ''.join(str(i) for i in pattern)
    # print(f'p = {p}')
    neighbour = []
    while len(t) is not 0:
        if match(p, t) and len(t) > w:
            neighbour.append(int(t[w]))
            # print(f'neighbour = {neighbour}')
            t = t[w:]
        else:
            t = t[1:]
    return neighbour


def _neighbour_index(data, w):
    t = ''.join(str(i) for i in data)
    pattern = data[-w:]
    p = ''.join(str(i) for i in pattern)
    # print(f'p = {p}')
    n_i = []
    i = 0
    while len(t) is not 0:
        if match(p, t) and len(t) > w:
            i += w
            # print(f'i = {i}')
            n_i.append(i)
            t = t[w:]
        else:
            t = t[1:]
            i += 1
            # print(f'i = {i}')
    return n_i

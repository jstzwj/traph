


class dtype(object):
    def __init__(self, name):
        self.type_name = name

float = dtype('float')
double = dtype('double')
half = dtype('half')
uint8 = dtype('uint8')
int16 = dtype('int16')
int32 = dtype('int32')
int64 = dtype('int64')


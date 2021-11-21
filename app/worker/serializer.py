import json
import numpy as np
from pprint import pprint

ARRAY_TYPE = "<class 'numpy.ndarray'>"


class MyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            print('[SERIALIZATION] serializing np.array ...')
            res = {
                '__type__': ARRAY_TYPE,
                'data': obj.tolist()
            }
            # print('[SERIALIZATION] Object to serialize: ')
            # pprint(res)
            return res
        else:
            return super(MyEncoder, self).default(obj)


def my_decoder(obj):
    if '__type__' in obj and obj['__type__'] == ARRAY_TYPE:
        print('[SERIALIZATION] deserializing np.array ...')
        res = np.asarray(obj)
        # print('[SERIALIZATION] resulted np.array:')
        # print(res)
        return res
    return obj


# Encoder function
def dumps(obj):
    print('[SERIALIZATION] serializing object: ')
    print(obj)
    res = json.dumps(obj, cls=MyEncoder)
    print('[SERIALIZATION] serialized object: ')
    print(res)
    return res


# Decoder function
def loads(obj):
    print('[SERIALIZATION] deserializing object: ')
    print(obj)
    res = json.loads(obj, object_hook=my_decoder)
    print('[SERIALIZATION] deserialized object: ')
    print(res)


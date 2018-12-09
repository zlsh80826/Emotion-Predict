import cntk as C

"""
Read ctf format to minibatch source, input function
"""

def deserialize(func, ctf_path, model, randomize=True, repeat=True, is_test=False):
    if not is_test:
        mb_source = C.io.MinibatchSource(
            C.io.CTFDeserializer(
                ctf_path,
                C.io.StreamDefs(
                    token = C.io.StreamDef('token', shape=model.word_dim, is_sparse=True),
                    emotion  = C.io.StreamDef('emotion', shape=model.num_emotions, is_sparse=True))),
            randomize=randomize,
            max_sweeps=C.io.INFINITELY_REPEAT if repeat else 1)

        input_map = {
            argument_by_name(func, 'token'): mb_source.streams.token,
            argument_by_name(func, 'emotion'): mb_source.streams.emotion
        }
    else:
        mb_source = C.io.MinibatchSource(
            C.io.CTFDeserializer(
                ctf_path,
                C.io.StreamDefs(
                    token = C.io.StreamDef('token', shape=model.word_dim, is_sparse=True))),
            randomize=randomize,
            max_sweeps=C.io.INFINITELY_REPEAT if repeat else 1)

        input_map = {
            argument_by_name(func, 'token'): mb_source.streams.token
        }    
    return mb_source, input_map

"""
Helper function used to map the variable
"""
def argument_by_name(func, name):
    found = [arg for arg in func.arguments if arg.name == name]
    if len(found) == 0:
        raise ValueError('no matching names in arguments')
    elif len(found) > 1:
        raise ValueError('multiple matching names in arguments')
    else:
        return found[0]

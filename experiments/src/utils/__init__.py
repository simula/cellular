import os

def append_id( filename, identifier ):
    name, ext = os.path.splitext(filename)
    return "%s_%s%s" % (name, identifier, ext)
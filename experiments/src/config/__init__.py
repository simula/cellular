import json

class Config(dict):

    @classmethod
    def from_json(cls, path):
        with open(path, "r") as f:
            return cls(json.load(f))

    def __init__(self, init={ }):
        super().__init__()
        for key, val in init.items():
            if type(val) is dict:
                self.__setattr__(key, Config(val))
            else:
                self.__setattr__(key, val)

    def __setitem__(self, key, value):
        return super(Config, self).__setitem__(key, value)

    def __getitem__(self, key):
        return super(Config, self).__getitem__(key)

    def __delitem__(self, key):
        return super(Config, self).__delitem__(key)

    __getattr__ = __getitem__
    __setattr__ = __setitem__

    def save(self, path):
        with open(path, "w") as f:
            json.dump(self, f, indent=4)
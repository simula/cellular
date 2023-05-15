import sys

old_std = { }
tees = {}

def log_std(path, std):
    logfile = open(path, "w")
    if std not in old_std:
        old_std[std] = getattr(sys, std)
        tees[std] = Tee(old_std[std])
    tees[std].set_log_file(logfile)
    setattr(sys, std, tees[std])
    return logfile

class Tee():
    def __init__(self, _fd1):
        self.fd1 = _fd1

    def __del__(self):
        if self.fd1 != sys.stdout and self.fd1 != sys.stderr:
            self.fd1.close()
        if self.fd2 != sys.stdout and self.fd2 != sys.stderr:
            self.fd2.close()

    def set_log_file(self, _fd2):
        self.fd2 = _fd2

    def write(self, text):
        self.fd1.write(text)
        self.fd2.write(text)

    def flush(self):
        self.fd1.flush()
        self.fd2.flush()
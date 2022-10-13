def to_device(device, *args):
    return [x.to(device) for x in args]
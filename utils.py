def concat_examples_trm(batch, device=None):
    if len(batch) == 0:
        raise ValueError('batch is empty')

    source = [b[0] for b in batch]
    target = [b[1] for b in batch]
    return source, target

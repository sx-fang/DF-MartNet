"""Integer bookkeeping for GPU-independent global sampling budgets.

This module has no numerical-library imports. Configured counts are global;
only the derived per-process shares depend on the actual process count.
"""
import operator


SAMPLE_BUDGET_PROTOCOL = 'global-counts-v1-20261018'


def _integer(value, name, minimum):
    if isinstance(value, bool):
        raise ValueError(f'{name} must be an integer >= {minimum}')
    try:
        value = operator.index(value)
    except TypeError as err:
        raise ValueError(f'{name} must be an integer >= {minimum}') from err
    if value < minimum:
        raise ValueError(f'{name} must be >= {minimum}; got {value}')
    return value


def split_global_budget(total, world_size, *, minimum=0, name='sample budget'):
    """Return exact, balanced shares, with remainders assigned to low ranks.

    A required local minimum is a guard, never permission to round up the
    global count. Insufficient budgets require fewer participating processes
    or a separately authorized change of the global configuration.
    """
    total = _integer(total, name, 1)
    world_size = _integer(world_size, 'world_size', 1)
    minimum = _integer(minimum, 'minimum', 0)
    base, remainder = divmod(total, world_size)
    if base < minimum:
        raise ValueError(
            f'{name}={total} cannot supply at least {minimum} per process '
            f'with world_size={world_size}; global budgets will not be inflated')
    return tuple(base + (rank < remainder) for rank in range(world_size))


def budget_entry(total, world_size, rank, *, minimum=0, name='sample budget'):
    """Serializable global/local/all-rank evidence for one configured count."""
    counts = split_global_budget(total, world_size, minimum=minimum, name=name)
    rank = _integer(rank, 'rank', 0)
    if rank >= world_size:
        raise ValueError('rank must be less than world_size')
    return {'global': total, 'local': counts[rank], 'per_rank': list(counts)}


def chunk_shares(counts, cursor, capacity):
    """Synchronized MC round: exhausted ranks contribute zero, not extra paths.

    ``cursor`` counts the per-rank slots already consumed. Advancing it by
    ``capacity`` after a successful round visits every allocated slot once.
    """
    cursor = _integer(cursor, 'cursor', 0)
    capacity = _integer(capacity, 'capacity', 1)
    return tuple(min(capacity, max(0, count - cursor)) for count in counts)

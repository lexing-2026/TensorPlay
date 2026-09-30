"""Which operations are run through to rather than written out.

An operation is in here when there is nothing to be gained from writing out
what it does: it already has code written for it that is better than a loop,
or what it does cannot be expressed as one.  The names are matched against
what the operation calls itself, so a name here is the operation's own name
rather than anything derived from where it is used.
"""

#: Operations that are run through to, by name.
tp_fallback_ops: set = set()

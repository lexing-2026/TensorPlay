"""A short fixed-width spelling for every dtype.

Used where a dtype has to line up with others: a log line, a column of a
table, a key.  The long names are for reading, these are for lining up.
"""

import tensorplay as tp


# Used for testing and logging
dtype_abbrs = {dt: dt.abbr for dt in tp._C._get_all_dtypes()}

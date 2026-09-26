"""Every rule about how a kernel should be configured, in one place.

Two kinds of rule, asked the same way and kept in one table: which
configurations of a written template are worth measuring, asked at compile time;
and what a kernel the compiler wrote should be configured with, asked when it
is written.  They are in one table because a program that registered one and not
the other would be configured by which file a rule was written in.
"""

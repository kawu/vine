BPfun
=====

An experimental MWE identification system based on the
[backprop](https://backprop.jle.im/index.html) library.


Installation
============

First you will need to download and install the [Haskell Tool Stack][stack].
Then run the following commands:

    git clone https://github.com/kawu/bpfun.git
    cd bpfun
    stack install

The above command builds the `bpfun-exe` command-line tool and (on Linux) puts
it in the `~/.local/bin/` directory by default.


[stack]: http://docs.haskellstack.org "Haskell Tool Stack"
[backprop]: https://backprop.jle.im/index.html "Backpropagation library"

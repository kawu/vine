BPfun
=====

A collection of examples showing how to implement a simple encoder-decoder NMT
system using [backprop](https://backprop.jle.im/index.html).


Installation
============

First you will need to download and install the [Haskell Tool Stack][stack].
Then run the following commands:

    git clone https://github.com/kawu/bpfun.git
    cd bpfun
    stack ghci --only-main

You can then proceed to play with the examples:

```console
> import Net.Ex1
> import Numeric.Backprop
>
> gradBP f 3
6
```

More examples to come.

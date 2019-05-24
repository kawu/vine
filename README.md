Vine
=====

Vine is a neural graph-based [verbal MWE identification][parseme-st] tool built
on top of the [backprop][backprop] library.


Installation
============

First you will need to download and install the [Haskell Tool Stack][stack].
Then run the following commands:

    git clone https://github.com/kawu/vine.git
    cd vine
    stack install

The above command builds the `vine` command-line tool and (on Linux) puts it in
the `~/.local/bin/` directory by default.


[stack]: http://docs.haskellstack.org "Haskell Tool Stack"
[backprop]: https://backprop.jle.im/index.html "Backpropagation library"
[parseme-st]: http://multiword.sourceforge.net/PHITE.php?sitesig=CONF&page=CONF_04_LAW-MWE-CxG_2018___lb__COLING__rb__&subpage=CONF_40_Shared_Task "PARSEME Shared Task"

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


Data format
==============

Vine works with the PARSEME [.cupt][cupt] (extended CoNLL-U) format.


Prerequisites
=============

Before you can use the tool to identify verbal MWEs (VMWEs) in a given
language, you will need to train the appropriate identification models.
To this end, you will need:

  * A training file with VMWE annotations in the [.cupt][cupt] format
  * A file with the word embedding vectors correspondig to the individual words
    in the training file

TODO: You can find an example training set in the `example` directory.

Configuration
-------------

You need to specify the training configuration to train a model.  The
configuration includes the SGD parameters and the type of the objective
function to optimize.  The default configuration can be found in the
`config/config.dhall` file.  It is written in the [Dhall][dhall] programming
language.


Training
========

Vine requires training one model per MWE category (`VID` -- verbal idiom,
`LVC.full` -- light-verb construction, etc.).  See `TODO` for a description of
the different categories employed in the PARSEME annotation framework.

Given:

  * Training file `train.cupt` in the [.cupt][cupt] format
  * File with the corresponding training word emeddings `train.embs`
  * Training configuration `config.dhall`

You can use the following command to train a model for the identification of
verbal idioms:

    vine train -i train.cupt -e train.embs --mwe VID --sgd config.dhall -o VID/VID.bin

where the `-o` option specifies the output model file.  The tool will write the
intermediate model files in the form `VID.bin.N` in the target directory `VID`,
where `N` is the number of the iteration (epoch).


Runtime options
---------------

You can use the [GHC runtime system options][ghc-rts] to speed up training.
For instance, you can make use of multiple cores using `-N`, or increase the
allocation area size using `-A`. For example, to train the model using four
(TODO: threads or cores?) threads and 256M allocation area size, run:

    vine train -i train.cupt -e train.embs --mwe VID --sgd config.dhall -o VID/VID.bin +RTS -A256M -N4


Tagging
=======

TODO


Ensemble averaging
------------------

TODO


PARSEME-ST
==========

Coming soon.


[stack]: http://docs.haskellstack.org "Haskell Tool Stack"
[backprop]: https://backprop.jle.im/index.html "Backpropagation library"
[parseme-st]: http://multiword.sourceforge.net/PHITE.php?sitesig=CONF&page=CONF_04_LAW-MWE-CxG_2018___lb__COLING__rb__&subpage=CONF_40_Shared_Task "PARSEME Shared Task"
[cupt]: http://multiword.sourceforge.net/PHITE.php?sitesig=CONF&page=CONF_04_LAW-MWE-CxG_2018___lb__COLING__rb__&subpage=CONF_45_Format_specification "PARSEME .cupt format"
[ghc-rts]: http://www.haskell.org/ghc/docs/latest/html/users_guide/runtime-control.html "GHC runtime system options"
